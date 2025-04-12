from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")

# Detectar en video 
model.predict(source="./VIDEOS/video3.mp4", show=True, conf=0.3)
