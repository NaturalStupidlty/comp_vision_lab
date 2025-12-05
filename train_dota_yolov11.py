#!/usr/bin/env python3
"""
End-to-end DOTA to Ultralytics YOLOv11 training script.

This script converts the DOTA dataset's oriented bounding boxes into YOLO-style
axis-aligned annotations, writes a dataset YAML, trains a YOLOv11 model, runs
validation, and performs a few sample inferences with visualization.

Expected DOTA layout (under --data_root):
    DOTA/
        train/
            images/         # .png/.jpg
            labelTxt*/      # labelTxt, labelTxt-v1.0, or labelTxt-v1.5 (zip or txt)
        val/
            images/
            labelTxt*/
        test/
            images/         # no labels

Output YOLO layout (under --yolo_data_dir):
    dota_yolo/
        images/{train,val,test}/
        labels/{train,val}/
        dota.yaml

Usage example:
python train_dota_yolov11.py \
    --data_root /path/to/DOTA \
    --yolo_data_dir /path/to/dota_yolo \
    --model yolo11s.pt \
    --epochs 50 \
    --batch_size 8 \
    --img_size 1024 \
    --lr 0.001 \
    --output_dir ./outputs \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# DOTA uses '###' for ignore regions; skip them when building labels.
IGNORED_CLASSES = {"###"}
# Known image extensions in DOTA.
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv11 on DOTA with automatic format conversion.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root DOTA directory.")
    parser.add_argument(
        "--yolo_data_dir",
        type=str,
        default="dota_yolo",
        help="Output directory for YOLO-formatted dataset (images/ & labels/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Initial YOLOv11 weights (pre-trained on COCO), e.g., yolo11s.pt or yolo11m.pt.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--img_size", type=int, default=1024, help="Training image size (imgsz).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate (lr0).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to store checkpoints, logs, validation metrics, and inference visualizations.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device string for Ultralytics (e.g., cuda or cpu).")
    parser.add_argument(
        "--label_subdir",
        type=str,
        default=None,
        help="Optional label directory name to force (e.g., labelTxt-v1.0). If omitted, auto-detect.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Score threshold for sample inference visualization after training.",
    )
    parser.add_argument(
        "--max_images_per_split",
        type=int,
        default=None,
        help="Optional cap on number of images to convert per split (train/val/test) for quick dry-runs.",
    )
    parser.add_argument(
        "--run_inference_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to run full-image inference on after training.",
    )
    parser.add_argument(
        "--max_infer_images",
        type=int,
        default=20,
        help="Maximum number of images to run in the post-training inference sweep.",
    )
    parser.add_argument(
        "--tile_output_dir",
        type=str,
        default=None,
        help="Optional output dir for tiled YOLO dataset (patch-based). If set, the converted dataset will be tiled.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="Tile size for optional patch-based tiling (height/width in pixels).",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=200,
        help="Overlap between tiles (in pixels) when tiling.",
    )
    parser.add_argument(
        "--tile_keep_fraction",
        type=float,
        default=0.1,
        help="Minimum fraction of a bbox area that must fall inside a tile to keep it when tiling.",
    )
    parser.add_argument(
        "--tile_max_images",
        type=int,
        default=None,
        help="Optional cap on number of images per split to tile (for quick tests). Use None for full tiling.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def find_label_source(split_root: Path, label_subdir: Optional[str] = None) -> Tuple[str, Path]:
    """
    Find a label source for a split.

    Returns a tuple of (mode, path) where mode is "zip" or "dir".
    Preference order: user-specified label_subdir, labelTxt, labelTxt-v1.0, labelTxt-v1.5.
    """
    candidates: List[Path] = []
    if label_subdir:
        candidates.append(split_root / label_subdir)
    candidates.extend([split_root / "labelTxt", split_root / "labelTxt-v1.0", split_root / "labelTxt-v1.5"])

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        # Direct zip file supplied.
        if cand.is_file() and cand.suffix == ".zip":
            return "zip", cand

        if not cand.exists():
            continue

        zip_path = cand / "labelTxt.zip"
        if zip_path.exists():
            return "zip", zip_path

        txt_files = list(cand.glob("*.txt"))
        if txt_files:
            return "dir", cand

    raise FileNotFoundError(f"Could not locate labelTxt directory or zip under {split_root}")


class DOTALabelReader:
    """Utility to read DOTA label files from a directory or a zip archive."""

    def __init__(self, split_root: Path, label_subdir: Optional[str] = None):
        self.mode, self.path = find_label_source(split_root, label_subdir)
        self.zip_file: Optional[zipfile.ZipFile] = None
        self.members: Dict[str, Union[str, Path]] = {}

        if self.mode == "zip":
            self.zip_file = zipfile.ZipFile(self.path)
            for member in self.zip_file.namelist():
                if member.lower().endswith(".txt"):
                    self.members[Path(member).name] = member
        else:
            for txt in Path(self.path).glob("*.txt"):
                self.members[txt.name] = txt

    def __enter__(self) -> "DOTALabelReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.zip_file:
            self.zip_file.close()

    def iter_label_items(self) -> Iterable[Tuple[str, List[str]]]:
        """Yield (stem, lines) for each annotation file."""
        for name, member in self.members.items():
            stem = Path(name).stem
            lines = self.read_lines(stem)
            yield stem, lines

    def read_lines(self, stem: str) -> List[str]:
        """Read label lines for a given image stem."""
        file_name = f"{stem}.txt"
        member = self.members.get(file_name)
        if member is None:
            return []
        if self.mode == "zip" and self.zip_file:
            with self.zip_file.open(member) as f:
                return f.read().decode("utf-8").splitlines()
        elif self.mode == "dir":
            with open(member, "r", encoding="utf-8") as f:
                return f.read().splitlines()
        return []

    @property
    def stems(self) -> List[str]:
        return [Path(name).stem for name in self.members.keys()]


def find_image(image_dir: Path, stem: str) -> Optional[Path]:
    """Locate an image by stem, checking common DOTA extensions."""
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_dota_line(line: str) -> Optional[Tuple[List[float], str]]:
    """
    Parse a single DOTA annotation line.

    The expected format is: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    Lines with metadata (e.g., 'imagesource:' or 'gsd:') are ignored.
    """
    if not line or ":" in line:
        return None
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    coords = list(map(float, parts[:8]))
    class_name = parts[8]
    if class_name in IGNORED_CLASSES:
        return None
    return coords, class_name


def quad_to_xyxy(coords: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert oriented quadrilateral coordinates to axis-aligned bounds."""
    xs = coords[0::2]
    ys = coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def yolo_normalize(
    x_min: float, y_min: float, x_max: float, y_max: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert pixel xyxy box to YOLO normalized cx, cy, w, h."""
    cx = ((x_min + x_max) / 2.0) / img_w
    cy = ((y_min + y_max) / 2.0) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return cx, cy, w, h


def collect_class_names(data_root: Path, splits: Sequence[str], label_subdir: Optional[str]) -> List[str]:
    """Build a sorted list of unique class names across provided splits."""
    classes = set()
    for split in splits:
        split_root = data_root / split
        if not split_root.exists():
            continue
        try:
            with DOTALabelReader(split_root, label_subdir) as reader:
                for _, lines in reader.iter_label_items():
                    for line in lines:
                        parsed = parse_dota_line(line)
                        if not parsed:
                            continue
                        _, class_name = parsed
                        classes.add(class_name)
        except FileNotFoundError:
            continue
    return sorted(classes)


def copy_or_symlink(src: Path, dst: Path) -> None:
    """Symlink an image when possible; fallback to copy if the FS disallows symlinks."""
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def convert_split_to_yolo(
    data_root: Path,
    yolo_root: Path,
    split: str,
    class_to_id: Dict[str, int],
    label_subdir: Optional[str],
    max_images: Optional[int],
) -> None:
    """
    Convert one split (train/val) of DOTA into YOLO format.

    - Reads oriented boxes from DOTA labels.
    - Converts to axis-aligned boxes, then YOLO normalized cx, cy, w, h.
    - Writes labels under labels/{split}/ and symlinks/copies images under images/{split}/.
    """
    split_root = data_root / split
    image_dir = split_root / "images"
    if not image_dir.exists():
        print(f"[WARN] Missing images for split '{split}' at {image_dir}, skipping.")
        return

    try:
        reader = DOTALabelReader(split_root, label_subdir)
    except FileNotFoundError:
        print(f"[WARN] No labels found for split '{split}', skipping label conversion.")
        reader = None

    dest_img_dir = yolo_root / "images" / split
    dest_lbl_dir = yolo_root / "labels" / split
    ensure_dir(dest_img_dir)
    ensure_dir(dest_lbl_dir)

    processed = 0

    if reader is None:
        # Only copy/symlink images (test split typically).
        for img_path in sorted(image_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            dst = dest_img_dir / img_path.name
            copy_or_symlink(img_path, dst)
            processed += 1
            if max_images and processed >= max_images:
                break
        return

    with reader:
        for stem, lines in reader.iter_label_items():
            img_path = find_image(image_dir, stem)
            if img_path is None:
                print(f"[WARN] No image found for label {stem}, skipping.")
                continue

            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to read image size for {img_path}: {exc}")
                continue

            yolo_entries: List[str] = []
            for line in lines:
                parsed = parse_dota_line(line)
                if not parsed:
                    continue
                coords, class_name = parsed
                x_min, y_min, x_max, y_max = quad_to_xyxy(coords)
                if x_max <= x_min or y_max <= y_min:
                    continue
                cx, cy, w, h = yolo_normalize(x_min, y_min, x_max, y_max, img_w, img_h)
                class_id = class_to_id[class_name]
                yolo_entries.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Write YOLO label file (empty file for images without objects).
            label_out = dest_lbl_dir / f"{stem}.txt"
            with open(label_out, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_entries))

            # Link/copy image.
            dst_img = dest_img_dir / img_path.name
            copy_or_symlink(img_path, dst_img)

            processed += 1
            if max_images and processed >= max_images:
                break


def write_dataset_yaml(yolo_root: Path, class_names: List[str]) -> Path:
    """Write the Ultralytics dataset YAML file."""
    yaml_path = yolo_root / "dota.yaml"
    data = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    import yaml  # Local import to avoid dependency if user only wants conversion.

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path


def build_yolo_dataset(
    data_root: Path, yolo_data_dir: Path, label_subdir: Optional[str], max_images_per_split: Optional[int]
) -> Path:
    """Convert the DOTA dataset to YOLO format and return the YAML path."""
    ensure_dir(yolo_data_dir)
    class_names = collect_class_names(data_root, splits=["train", "val"], label_subdir=label_subdir)
    if not class_names:
        raise RuntimeError("No classes found in DOTA annotations. Check label paths.")

    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    print(f"[INFO] Found {len(class_names)} classes: {class_names}")

    for split in ["train", "val", "test"]:
        convert_split_to_yolo(
            data_root,
            yolo_data_dir,
            split,
            class_to_id,
            label_subdir,
            max_images=max_images_per_split,
        )

    yaml_path = write_dataset_yaml(yolo_data_dir, class_names)
    print(f"[INFO] Wrote dataset YAML to {yaml_path}")
    return yaml_path


def generate_windows(size: Tuple[int, int], tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """Generate sliding windows that cover an image with given size."""
    w, h = size
    stride = max(1, tile_size - overlap)
    x_starts = list(range(0, max(w - tile_size, 0) + 1, stride))
    y_starts = list(range(0, max(h - tile_size, 0) + 1, stride))
    if not x_starts:
        x_starts = [0]
    if not y_starts:
        y_starts = [0]
    if x_starts[-1] + tile_size < w:
        x_starts.append(max(0, w - tile_size))
    if y_starts[-1] + tile_size < h:
        y_starts.append(max(0, h - tile_size))
    windows = []
    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            windows.append((x0, y0, x1, y1))
    return windows


def yolo_to_xyxy(
    labels: List[List[float]], img_w: int, img_h: int
) -> List[Tuple[int, float, float, float, float]]:
    """Convert YOLO normalized labels to pixel xyxy."""
    boxes = []
    for row in labels:
        if len(row) < 5:
            continue
        cls = int(row[0])
        cx, cy, w, h = row[1:5]
        x_min = (cx - w / 2.0) * img_w
        x_max = (cx + w / 2.0) * img_w
        y_min = (cy - h / 2.0) * img_h
        y_max = (cy + h / 2.0) * img_h
        boxes.append((cls, x_min, y_min, x_max, y_max))
    return boxes


def xyxy_to_yolo(
    cls: int, x_min: float, y_min: float, x_max: float, y_max: float, tile_w: int, tile_h: int
) -> str:
    """Convert pixel xyxy to YOLO normalized string."""
    cx = ((x_min + x_max) / 2.0) / tile_w
    cy = ((y_min + y_max) / 2.0) / tile_h
    w = (x_max - x_min) / tile_w
    h = (y_max - y_min) / tile_h
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def tile_yolo_dataset(
    src_root: Path,
    dst_root: Path,
    class_names: List[str],
    tile_size: int = 1024,
    overlap: int = 200,
    keep_fraction: float = 0.1,
    max_images_per_split: Optional[int] = None,
) -> Path:
    """
    Tile a YOLO-formatted dataset into fixed-size patches with overlap.

    Args:
        src_root: Source YOLO dataset root (images/train, labels/train, etc.).
        dst_root: Destination root for tiled dataset.
        class_names: List of class names.
        tile_size: Patch size in pixels.
        overlap: Overlap between tiles in pixels.
        keep_fraction: Minimum fraction of a bbox that must lie inside a tile to keep it.
    """
    ensure_dir(dst_root / "images" / "train")
    ensure_dir(dst_root / "images" / "val")
    ensure_dir(dst_root / "images" / "test")
    ensure_dir(dst_root / "labels" / "train")
    ensure_dir(dst_root / "labels" / "val")

    for split in ["train", "val", "test"]:
        src_img_dir = src_root / "images" / split
        src_lbl_dir = src_root / "labels" / split
        if not src_img_dir.exists():
            continue
        dst_img_dir = dst_root / "images" / split
        dst_lbl_dir = dst_root / "labels" / split
        ensure_dir(dst_img_dir)
        if split != "test":
            ensure_dir(dst_lbl_dir)

        img_paths = [p for p in sorted(src_img_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        processed = 0
        for img_path in img_paths:
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
                    im = im.convert("RGB")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to read image {img_path}: {exc}")
                continue

            labels: List[List[float]] = []
            if split != "test":
                lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        for line in f.read().splitlines():
                            if not line.strip():
                                continue
                            parts = line.strip().split()
                            labels.append([float(x) for x in parts])

            boxes_xyxy = yolo_to_xyxy(labels, img_w, img_h)
            windows = generate_windows((img_w, img_h), tile_size, overlap)

            for (x0, y0, x1, y1) in windows:
                tile = im.crop((x0, y0, x1, y1))
                tile_w, tile_h = tile.size
                tile_labels: List[str] = []
                for cls, bx0, by0, bx1, by1 in boxes_xyxy:
                    inter_w = min(bx1, x1) - max(bx0, x0)
                    inter_h = min(by1, y1) - max(by0, y0)
                    if inter_w <= 0 or inter_h <= 0:
                        continue
                    inter_area = inter_w * inter_h
                    box_area = max((bx1 - bx0), 0) * max((by1 - by0), 0)
                    if box_area <= 0:
                        continue
                    if inter_area / box_area < keep_fraction:
                        continue
                    nx0 = max(bx0, x0) - x0
                    ny0 = max(by0, y0) - y0
                    nx1 = min(bx1, x1) - x0
                    ny1 = min(by1, y1) - y0
                    if nx1 <= nx0 or ny1 <= ny0:
                        continue
                    tile_labels.append(xyxy_to_yolo(int(cls), nx0, ny0, nx1, ny1, tile_w, tile_h))

                tile_name = f"{img_path.stem}_x{x0}_y{y0}{img_path.suffix}"
                tile.save(dst_img_dir / tile_name)
                if split != "test":
                    with open(dst_lbl_dir / f"{Path(tile_name).stem}.txt", "w", encoding="utf-8") as f:
                        f.write("\n".join(tile_labels))

            processed += 1
            if max_images_per_split and processed >= max_images_per_split:
                break

    tiled_yaml = write_dataset_yaml(dst_root, class_names)
    print(f"[INFO] Tiled dataset YAML written to {tiled_yaml}")
    return tiled_yaml


def detect_objects_in_image(
    model: YOLO,
    image_path: Union[str, np.ndarray],
    device: str = "cuda",
    score_threshold: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
    display: bool = False,
) -> List[Tuple[float, float, float, float, int, float]]:
    """
    Run YOLOv11 inference on an image and return detected boxes.

    Returns:
        List of (x_min, y_min, x_max, y_max, class_id, score) tuples in pixel coordinates.
    """
    # Ultralytics expects BGR when numpy arrays are passed; flip channels if likely RGB.
    source = image_path
    base_image_for_draw: Optional[Image.Image] = None
    if isinstance(image_path, np.ndarray):
        arr = image_path
        if arr.ndim == 3 and arr.shape[2] == 3:
            source = arr[:, :, ::-1]  # RGB -> BGR
        else:
            source = arr
        base_image_for_draw = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    else:
        source = image_path
        base_image_for_draw = Image.open(image_path).convert("RGB")

    results = model.predict(source=source, device=device, conf=score_threshold, verbose=False)
    if not results:
        return []

    dets: List[Tuple[float, float, float, float, int, float]] = []
    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None:
        return dets

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
        dets.append((float(x1), float(y1), float(x2), float(y2), int(c), float(s)))

    if save_path is not None or display:
        draw = ImageDraw.Draw(base_image_for_draw)
        font = ImageFont.load_default()
        names = getattr(model, "names", None) or {}
        for x1, y1, x2, y2, cid, score in dets:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            label = f"{names.get(cid, cid)} {score:.2f}"
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
            draw.text((x1, y1 - text_h), label, fill="white", font=font)

        if save_path is not None:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            base_image_for_draw.save(save_path)

        if display:
            plt.figure(figsize=(10, 10))
            plt.imshow(base_image_for_draw)
            plt.axis("off")
            plt.show()

    return dets


def perform_training(args: argparse.Namespace, dataset_yaml: Path) -> Tuple[YOLO, Path]:
    """Train YOLOv11 on the converted DOTA dataset and return the trained model and best weights path."""
    ensure_dir(Path(args.output_dir))
    model = YOLO(args.model)
    train_save_dir = Path(args.output_dir) / "yolo11_dota"

    print("[INFO] Starting training...")
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        lr0=args.lr,
        device=args.device,
        project=str(Path(args.output_dir)),
        name="yolo11_dota",
        exist_ok=True,
    )

    best_weights = train_save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found at {best_weights}")

    print(f"[INFO] Training complete. Best weights at {best_weights}")
    trained_model = YOLO(best_weights)
    return trained_model, best_weights


def run_validation(model: YOLO, dataset_yaml: Path, args: argparse.Namespace) -> Dict[str, float]:
    """Run validation on the trained model and persist metrics to disk."""
    print("[INFO] Running validation...")
    metrics = model.val(data=str(dataset_yaml), imgsz=args.img_size, batch=args.batch_size, device=args.device)
    metrics_dict = metrics.results_dict

    metrics_path = Path(args.output_dir) / "validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"[INFO] Validation metrics saved to {metrics_path}")
    return metrics_dict


def run_dataset_inference(
    model: YOLO, yolo_data_dir: Path, split: str, max_images: Optional[int], args: argparse.Namespace
) -> None:
    """
    Run inference on a dataset split and save visualized predictions using Ultralytics' built-in saving.

    Images with rectangles and txt outputs (xyxy) are stored under:
        output_dir/predict_<split>/
    """
    image_dir = yolo_data_dir / "images" / split
    if not image_dir.exists():
        print(f"[WARN] No image directory for split '{split}' at {image_dir}, skipping inference sweep.")
        return

    all_images = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not all_images:
        print(f"[WARN] No images found in {image_dir}, skipping inference sweep.")
        return

    sources = all_images[:max_images] if max_images else all_images
    save_dir = Path(args.output_dir)
    ensure_dir(save_dir)

    print(f"[INFO] Running inference on {len(sources)} images from '{split}' split...")
    # Ultralytics will save annotated images and txt files automatically.
    model.predict(
        source=[str(p) for p in sources],
        imgsz=args.img_size,
        device=args.device,
        conf=args.score_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(save_dir),
        name=f"predict_{split}",
        exist_ok=True,
        verbose=False,
    )
    print(f"[INFO] Saved predictions with rectangles to {save_dir / f'predict_{split}'}")


def run_sample_inference(model: YOLO, data_root: Path, args: argparse.Namespace) -> None:
    """Run a few sample predictions on test/val images and save visualizations."""
    for split in ["test", "val", "train"]:
        candidate_dir = data_root / split / "images"
        if candidate_dir.exists():
            image_dir = candidate_dir
            break
    else:
        print("[WARN] No images found for sample inference.")
        return

    image_paths = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not image_paths:
        print(f"[WARN] No images found in {image_dir} for sample inference.")
        return

    inference_vis_dir = Path(args.output_dir) / "inference_vis"
    ensure_dir(inference_vis_dir)

    sample_images = image_paths[:3]
    for img_path in sample_images:
        save_path = inference_vis_dir / f"{img_path.stem}_pred{img_path.suffix}"
        detections = detect_objects_in_image(
            model=model,
            image_path=str(img_path),
            device=args.device,
            score_threshold=args.score_threshold,
            save_path=save_path,
            display=False,
        )
        #print(f"[INFO] Detections for {img_path}: {detections}")
    print(f"[INFO] Saved inference visualizations to {inference_vis_dir}")


def maybe_skip_conversion(yolo_data_dir: Path, max_images_per_split: Optional[int]) -> bool:
    """Return True if the YOLO dataset already exists with labels and images and no limiting flag is set."""
    if max_images_per_split:
        return False
    required = [
        yolo_data_dir / "images" / "train",
        yolo_data_dir / "images" / "val",
        yolo_data_dir / "labels" / "train",
        yolo_data_dir / "labels" / "val",
    ]
    return all(p.exists() and any(p.iterdir()) for p in required) and (yolo_data_dir / "dota.yaml").exists()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    yolo_data_dir = Path(args.yolo_data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not data_root.exists():
        print(f"[ERROR] data_root does not exist: {data_root}")
        sys.exit(1)

    if maybe_skip_conversion(yolo_data_dir, args.max_images_per_split):
        print(f"[INFO] YOLO dataset already present at {yolo_data_dir}, skipping conversion.")
        dataset_yaml = yolo_data_dir / "dota.yaml"
    else:
        dataset_yaml = build_yolo_dataset(
            data_root, yolo_data_dir, args.label_subdir, args.max_images_per_split
        )

    # Optional tiling stage for large images.
    if args.tile_output_dir:
        tile_dir = Path(args.tile_output_dir).expanduser().resolve()
        import yaml

        with open(dataset_yaml, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        class_names = data_cfg["names"]
        print(f"[INFO] Tiling YOLO dataset from {yolo_data_dir} to {tile_dir} ...")
        dataset_yaml = tile_yolo_dataset(
            src_root=yolo_data_dir,
            dst_root=tile_dir,
            class_names=class_names,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            keep_fraction=args.tile_keep_fraction,
            max_images_per_split=args.tile_max_images,
        )
        yolo_data_dir = tile_dir

    trained_model, best_weights = perform_training(args, dataset_yaml)
    metrics = run_validation(trained_model, dataset_yaml, args)
    print(f"[INFO] Validation metrics: {metrics}")

    # Keep a stable copy of best weights for downstream use.
    stable_best = output_dir / "dota_yolo11_best.pt"
    ensure_dir(stable_best.parent)
    shutil.copy2(best_weights, stable_best)
    print(f"[INFO] Copied best weights to {stable_best}")

    run_sample_inference(trained_model, data_root, args)
    run_dataset_inference(
        trained_model,
        yolo_data_dir=yolo_data_dir,
        split=args.run_inference_split,
        max_images=args.max_infer_images,
        args=args,
    )


if __name__ == "__main__":
    main()
