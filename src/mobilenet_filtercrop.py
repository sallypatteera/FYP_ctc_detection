import tensorflow as tf
import os
from tensorflow.keras.models import load_model

# Load the trained MobileNetV2 filtering model
FILTER_MODEL_PATH = "models/mobilenetv2_filter_best.h5"
filter_model = load_model(FILTER_MODEL_PATH)

# Preprocess a single cell crop (resize, normalize)
def preprocess_crop_filter(crop):
    img = tf.image.resize(crop, (224, 224))  # change to your training size
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Return only the crops that are predicted as valid (prob > threshold)
def filter_valid_crops(crop_list, model, threshold=0.5):
    preprocessed = tf.stack([preprocess_crop_filter(crop) for crop in crop_list], axis=0)
    preds = model.predict(preprocessed, verbose=0).flatten()
    
    # Filter valid crops
    valid_crops = [crop for crop, p in zip(crop_list, preds) if p > threshold]
    
    print(f"Filtered {len(valid_crops)} valid crops out of {len(crop_list)} total.")
    return valid_crops

def execute_filtering(crop_list, model):
    filtered_valid_crops = filter_valid_crops(crop_list, model)
    if not filtered_valid_crops:
        return None
    return filtered_valid_crops

# helper function
def filter_valid_conf(conf, valid_indices):
    filtered_valid_conf = [conf[i] for i in valid_indices]
    return filtered_valid_conf

# # test
# import cv2

# folder = "/Users/patteerasupvithayanond/Documents/FYP_ctc_detection/results"
# image_np = []
# for image in os.listdir(folder):
#     img = cv2.imread(os.path.join(folder, image))
#     if img is not None:
#         image_np.append(img)

# execute_filtering(image_np, filter_model) 