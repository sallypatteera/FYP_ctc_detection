# Main Interface
import argparse
import os
import glob
import torch
from tensorflow.keras.models import load_model
import warnings

from src.yolo_detection import run_detection
# from src.sam_filtercrop import execute_filter_crop
from src.effnet_classification import execute_classification

# load model
# Load YOLOv5 model once
warnings.filterwarnings("ignore", category=FutureWarning)
MODEL_PATH = "models/bestyolo2.pt"
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, verbose=False)

# Load the EfficientNet model
MODEL_PATH = "models/efficientnetb0_model_0404_Best.h5"
effnet_model = load_model(MODEL_PATH)


def process_image(image_path):
    print(f"\n🔹 Processing: {image_path}")

    print("🔹 Running YOLOv5 detection...")
    detected_cells = run_detection(image_path, model=yolo_model)
    print(f"    Detected {len(detected_cells)} cells.")

    # print("Cropping images from bounding boxes...") # actually it's filtering
    # filtered_cropped_cells = execute_filter_crop(detected_cells) # array of multiple cropped images
    # print(f"Valid {len(filtered_cropped_cells)/len(detected_cells)} cells for classifying.")
    # print()

    print("🔹 Running DL model on valid cropped cells...")
    execute_classification(detected_cells, effnet_model)
    print("===========================================")

    



def main(image_folder):
    if os.path.isdir(image_folder):
        image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
        if not image_files:
            print()
            print("No image found in the specified folder.")
            return
        print()
        print(f"Found {len(image_files)} images in the folder.")

        for image_path in image_files:
            process_image(image_path)
    # handle single image
    elif os.path.isfile(image_folder):
        process_image(image_folder)
    else:
        print("Invalid image path or folder path provided.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Line Classification")
    parser.add_argument("image_folder", type=str, help="Path to the image or folder containing images to be processed.")
    args = parser.parse_args()
    
    main(args.image_folder)