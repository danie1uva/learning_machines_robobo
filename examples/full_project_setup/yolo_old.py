from ultralytics import YOLO
import cv2
import torch

# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

model = YOLO("yolo11n.pt")
results = model.predict(source = "0", show=True, device = "cuda")
print(results)