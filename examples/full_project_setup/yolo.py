from ultralytics import YOLO
import torch

# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Load YOLO model
model = YOLO("yolo11n.pt")

# Predict on a single image
image_path = "testimage.jpg"  # Replace with the path to your image
results = model.predict(source=image_path, show=True, device="cuda")

# Print the results
print(results)