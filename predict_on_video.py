from ultralytics import YOLO
model = YOLO("outputs_a100_long/dota_yolo11_best.pt")
model.predict("video.mp4", save=True, project="runs/predict", name="dota_video",
              device=0, save_txt=True, save_conf=True)
