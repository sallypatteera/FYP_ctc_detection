# Run YOLOv5s for cell detection
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Load YOLOv5 model once
# MODEL_PATH = "models/bestyolo2.pt"
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, verbose=False)

# Detect cells in an image
def detect_cells(image, model):
    """Runs YOLOv5 detection on an image and returns bounding boxes."""
    results = model(image)

    # Extract bounding boxes from YOLO output
    detected_cells = []
    for *box, conf, cls in results.xyxy[0].tolist():
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            detected_cells.append((x1, y1, x2, y2))

    return detected_cells

# Crop cells according to the bounding boxes
def crop_images_from_bboxes(image: np.ndarray, bboxes: list) -> list:
    """
    Crops bounding boxes from an image.
    
    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        bboxes (list): List of bounding boxes in the format [(x1, y1, x2, y2), ...].
        
    Returns:
        list: List of cropped images as NumPy arrays.
    """
    height, width, _ = image.shape
    cropped_images = []
    
    for (x1, y1, x2, y2) in bboxes:
        # Ensure coordinates are within image bounds
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        
        cropped_img = image[y1:y2, x1:x2]
        if cropped_img.size > 0:
            cropped_images.append(cropped_img)
    
    # print(f"Detected {len(cropped_images)} cells.")
    # print("Cropped images:", cropped_images)
    
    return cropped_images
    

# Execute all the code in this file
def run_detection(image_path, model):
    image = cv2.imread(image_path)  # Read image only once
    detected_cells = detect_cells(image, model)
    cropped_cells = crop_images_from_bboxes(image, detected_cells)
    return cropped_cells

# test
# image = "images/tile_x007_y001.jpg"
# run_detection(image)