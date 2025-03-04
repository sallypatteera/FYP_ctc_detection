# Main interface
import argparse
import os
from detect import detect_cells
from classify import classify_cell

def get_image_paths(input_path):
    """Returns a list of image file paths from a single file or directory."""
    if os.path.isdir(input_path):  # If input is a folder
        image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:  # If input is a single file
        image_files = [input_path]
    
    return image_files

def main():
    parser = argparse.ArgumentParser(description="Cancer Cell Detection and Classification CLI")
    parser.add_argument("input_path", type=str, help="Path to an image file or folder")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")

    args = parser.parse_args()

    # Get list of image files from folder or single file
    image_paths = get_image_paths(args.input_path)

    print(f"[INFO] Processing {len(image_paths)} images...")

    for image_path in image_paths:
        print(f"[INFO] Detecting cancer cells in {image_path}...")
        detected_cells = detect_cells(image_path)

        print("[INFO] Classifying detected cells...")
        for i, cell_tensor in enumerate(detected_cells):
            classification = classify_cell(cell_tensor)  
            print(f"Cell {i + 1} in {image_path}: {classification}")

if __name__ == "__main__":
    main()
