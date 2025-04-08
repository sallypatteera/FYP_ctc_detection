# Run YOLOv5s for cell detection
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Load YOLOv5 model once
MODEL_PATH = "models/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, verbose=False)

# Detect cells in an image
def detect_cells(image, model):
    """Runs YOLOv5 detection on an image and returns bounding boxes."""
    results = model(image)

    # Extract bounding boxes from YOLO output
    detected_cells = []
    for *box, conf, cls in results.xyxy[0].tolist():
        # only keep cancer cell class
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            detected_cells.append((x1, y1, x2, y2))

    return detected_cells

def is_square(w, h, threshold=0.7):
    aspect_ratio = min(w / h, h / w)
    return aspect_ratio >= threshold

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
        # Check if the bounding box is square
        w, h = x2 - x1, y2 - y1
        if not is_square(w, h):
            # Skip non-square bounding boxes
            continue
        
        cropped_img = image[y1:y2, x1:x2]
        if cropped_img.size > 0:
            cropped_images.append(cropped_img)
            # save the cropped image to the folder "results"
            cv2.imwrite(f"results/cropped_{len(cropped_images)}.jpg", cropped_img)
            plt.imshow(cropped_img)
            plt.title("cropped image")
            plt.show()
    
    # print(f"🔸🔸 Detected {len(cropped_images)} cells.")
    # print("Cropped images:", cropped_images)
    
    return cropped_images
    

# Execute all the code in this file
def run_detection(image_path, model):
    image = cv2.imread(image_path)  # Read image only once
    detected_cells = detect_cells(image, model)
    cropped_cells = crop_images_from_bboxes(image, detected_cells)

    if not cropped_cells:
        print("No valid cells detected.")
        return None
    return cropped_cells

# test
image = '/Users/patteerasupvithayanond/Documents/FYP dataset/fullsize/H2452/H2452_0304_2/brightfield/tile_x001_y002.jpg'
run_detection(image, model)