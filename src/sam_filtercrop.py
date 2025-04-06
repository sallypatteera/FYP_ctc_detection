# filter only valid cropped images from yolo detection using SAM

# Set up
import os
CHECKPOINT_PATH = os.path.join("models/weights", "sam_vit_h_4b8939.pth")
print("CHEKCPOINT_PATH:", CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import supervision as sv
# from skimage import measure
# from scipy import ndimage as ndi

# Helper functions: contrast enhancement
# Function for contrast enhancement using CLAHE
def enhance_contrast(img):
    # Ensure image is in 8-bit unsigned integer format
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Convert to grayscale if the image is in color
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    return enhanced_img


# Function to apply FFT bandpass filtering
def apply_fft_bandpass(img, low_cutoff, high_cutoff):
    """
    Apply FFT bandpass filtering to an image.
    :param img: Grayscale input image
    :param low_cutoff: Low frequency cutoff (normalized, 0-1)
    :param high_cutoff: High frequency cutoff (normalized, 0-1)
    :return: Filtered image in the spatial domain
    """
    # Perform FFT and shift zero frequency to the center
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Create a bandpass mask
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)

    for u in range(rows):
        for v in range(cols):
            dist = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            if low_cutoff * crow < dist < high_cutoff * crow:
                mask[u, v] = 1

    # Apply the mask
    fshift_filtered = fshift * mask

    # Inverse FFT to transform back to spatial domain
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)  # Get magnitude

    return img_filtered

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=1):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

# Helper functions: calculate morphological features
def calculate_diameter(mask):
    """Calculate the diameter of the mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Calculate the maximum distance between any two points in the contour
        max_distance = 0
        for i in range(len(largest_contour)):
            for j in range(i + 1, len(largest_contour)):
                pt1 = tuple(largest_contour[i][0])
                pt2 = tuple(largest_contour[j][0])
                distance = cv2.norm(np.array(pt1) - np.array(pt2))
                max_distance = max(max_distance, distance)
        return max_distance
    return 0  # Return 0 if no contours found

def calculate_circularity(mask):
    """Calculate the circularity of the mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            return circularity
    return 0  # Return 0 if no contours found

# Filtering valid cropped images by segmenting the cells and calculating morphological features
# if morphological features are not valid (not in the specified range), discard the cropped image
def filter_crop(single_crop):
  # enhance image
  blurred_img = apply_gaussian_blur(single_crop, kernel_size=(5,5), sigma=1)
  enhanced_img = enhance_contrast(blurred_img)
  image_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

  # define constants
  min_diameter = 0.4*max(image_rgb.shape)
  max_diameter = max(image_rgb.shape)
  circularity_threshold = 0.5

  # get mask
  sam_result = mask_generator.generate(image_rgb)
  masks = [
      mask['segmentation']
      for mask in sorted(sam_result, key=lambda x: x['area'], reverse=True)
  ]

  # get only the biggest mask
  the_mask = masks[0]
  diameter = calculate_diameter(the_mask)
  circularity = calculate_circularity(the_mask)

  # filtering
  if min_diameter < diameter < max_diameter and circularity > circularity_threshold:
    print("valid crop")
    return single_crop
    # plt.imshow(image_rgb)
    # plt.title("crop")
    # plt.show()
    # return "valid"
  else:
    print("invalid crop")
    return None
    # plt.imshow(image_rgb)
    # plt.title("crop")
    # plt.show()
    # return "invalid"

def execute_filter_crop(cropped_cells):
    """
    Execute the filter_crop function on a list of cropped cells.
    
    Parameters:
        cropped_cells (list): List of cropped images as NumPy arrays.
        
    Returns:
        list: List of valid cropped images.
    """
    filtered_crops = []
    for crop in cropped_cells:
        result = filter_crop(crop)
        if result is not None:
            filtered_crops.append(result)
    return filtered_crops

# test
# import cv2
# import os

# folder = "/Users/patteerasupvithayanond/Documents/FYP_ctc_detection/results"
# image_np = []
# for image in os.listdir(folder):
#     img = cv2.imread(os.path.join(folder, image))
#     if img is not None:
#         image_np.append(img)

# execute_filter_crop(image_np) 