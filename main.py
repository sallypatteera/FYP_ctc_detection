# Main Interface
import argparse
import os
import glob
import torch
from tensorflow.keras.models import load_model
import warnings

from src.yolo_detection_subclass import run_detection
# from src.sam_filtercrop import execute_filter_crop
from src.mobilenet_filtercrop_subclass import execute_filtering, filter_valid_conf
from src.effnet_classification_subclass import execute_classification

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
FILTER_MODEL_PATH = "models/mobilenetv2_filter_best.h5"
mobile_model = load_model(FILTER_MODEL_PATH)


def process_image(image_path):
    print(f"\n🔹 Processing: {image_path}")

    print("🔹 Running YOLOv5 detection...")
    # detected_cells = run_detection(image_path, model=yolo_model)
    detected_cells, detected_conf = run_detection(image_path, model=yolo_model)

    if detected_cells is not None:
        print(f"🔸🔸 Detected {len(detected_cells)} cells.")

        print("Cropping images from bounding boxes...")  # actually it's filtering
        # filtered_cropped_cells = execute_filtering(detected_cells, mobile_model)  # array of multiple cropped images
        filtered_cropped_cells, filtered_valid_indices = execute_filtering(detected_cells, mobile_model)
        if filtered_cropped_cells is not None:
            print(f"🔸🔸 Valid {len(filtered_cropped_cells)} out of {len(detected_cells)} cells for classifying.")

            print("🔹 Running DL model on valid cropped cells...")
            filtered_valid_conf = filter_valid_conf(detected_conf, filtered_valid_indices)
            # execute_classification(filtered_cropped_cells, effnet_model)
            execute_classification(filtered_cropped_cells, effnet_model, filtered_valid_conf)
            print("===========================================")
        else:
            print("🔸🔸 No valid cells found for classification.")
            print("===========================================")
            return
    else:
        print("🔸🔸 No cells detected.")
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