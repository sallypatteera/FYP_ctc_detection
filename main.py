# Main Interface
import argparse
import os
import glob
import torch
from tensorflow.keras.models import load_model
import warnings

from src.yolo_detection import run_detection
from src.mobilenet_filtercrop import execute_filtering, filter_valid_conf
from src.effnet_classification import execute_classification

# load model
# Load YOLOv5 model once
warnings.filterwarnings("ignore", category=FutureWarning)
YOLO_SUBCLASS_MODEL_PATH = "models/best_yolo_subclass.pt"
YOLO_MODEL_PATH = "models/best_yolo_wbc_cc.pt"
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_SUBCLASS_MODEL_PATH, verbose=False)

# Load the EfficientNet model
EFFNET_MODEL_PATH = "models/efficientnetb0_model_0404_Best.h5"
effnet_model = load_model(EFFNET_MODEL_PATH)

# Load the trained MobileNetV2 filtering model
FILTER_MODEL_PATH = "models/mobilenetv2_filter_best_0904.h5"
mobile_model = load_model(FILTER_MODEL_PATH)


def process_image(image_path):
    print(f"\n🔹 Processing: {image_path}")

    print("🔹 Running YOLOv5 detection...")
    detected_cells, detected_conf = run_detection(image_path, model=yolo_model)

    if detected_cells is not None:
        print(f"🔸🔸 Detected {len(detected_cells)} cells.\n")

        print("🔹 Cropping images from bounding boxes...")  # actually it's filtering
        filtered_cropped_cells, filtered_valid_indices = execute_filtering(detected_cells, mobile_model)
        if filtered_cropped_cells is not None:

            print("🔹 Running EfficientNetB0 classifier on valid cropped cells...")
            filtered_valid_conf = filter_valid_conf(detected_conf, filtered_valid_indices)
            maxpred = execute_classification(filtered_cropped_cells, effnet_model, filtered_valid_conf)
            print(f"🔸🔸 This image contains {len(detected_cells)} {maxpred} cells.")
            print("\n===========================================")
        else:
            print("🔸🔸 No valid cells found for classification.\n")
            print("===========================================")
            return
    else:
        print("🔸🔸 No cells detected.\n")
        print("===========================================")
        return


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