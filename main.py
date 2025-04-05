# Main Interface
import argparse
import os
import glob
from src.detection import run_detection
from src.dl_model import predict_image


def process_image(image_path):
    print(f"\nProcessing: {image_path}")

    print("Running YOLOv5 detection...")
    detected_cells = run_detection(image_path)
    print(f"Detected {len(detected_cells)} cells from {image_path}.")

    print("Running DL model on cropped cells...")
    dl_predictons = predict_image(detected_cells)



def main(image_folder):
    if os.path.isdir(image_folder):
        image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
        if not image_files:
            print("No image found in the specified folder.")
            return
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