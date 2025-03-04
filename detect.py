import torch
import cv2
import numpy as np

MODEL_PATH = "models/yolov5_best.pt"  # Your trained YOLOv5 model

def detect_cells(image_path):
    # Load YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
    
    # Perform detection
    results = model(image_path)
    
    # Read the original image as a tensor
    image = cv2.imread(image_path)  # BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)

    # Extract detected bounding boxes
    detections = results.xyxy[0]  # [x1, y1, x2, y2, confidence, class]

    cell_tensors = []  # List to store cropped cell images as tensors

    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])  # Extract bounding box coordinates

        # Crop the detected cell from the image tensor
        cropped_tensor = image_tensor[:, y1:y2, x1:x2]  # Keep (C, H, W) format

        # Normalize pixel values to [0, 1] (optional)
        cropped_tensor /= 255.0

        cell_tensors.append(cropped_tensor)

    return cell_tensors  # List of cropped cell images as tensors
