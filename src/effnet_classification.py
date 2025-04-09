# Load and run prediction using the trained EfficientNet model
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os

# suppress TF warnings and info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # Load the EfficientNet model
MODEL_PATH = "models/efficientnetb0_model_0404_Best.h5"
model = load_model(MODEL_PATH)

# class labels
CLASS_NAMES = ['A549', 'H2452', 'HEYA8', 'T24']

# preprocess a single crop
def preprocess_crop(crop):
    img = tf.image.resize(crop, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = tf.image.rgb_to_grayscale(img)
    return img

# predict a batch of crops and return class names
def predict_cell_batch(crop_list, model):
    # Preprocess the crops
    batch = tf.stack([preprocess_crop(crop) for crop in crop_list], axis=0)
    # Make predictions
    preds = model.predict(batch, verbose=0)
    # Round the predictions to two decimal places
    preds = np.round(preds, 2)
    return preds

"""
yolo's validation confidence = 0.955
effnet's validation confidence = 0.96

total confidence = 0.955 + 0.96 = 1.915

weight_yolo = 0.955 / 1.915 = 0.498
weight_effnet = 0.96 / 1.915 = 0.502

~around 50% for each model
"""

# Combine model predictions
def combine_predictions(yolo_conf, effnet_prob):
    weight_yolo = 0.498
    weight_effnet = 0.502

    # Combine probabilities
    if len(yolo_conf) != len(effnet_prob):
        raise ValueError("Length of YOLO confidence and EfficientNet probabilities must match.")
    weight_yolo_array = np.full_like(yolo_conf, weight_yolo)
    weight_effnet_array = np.full_like(effnet_prob, weight_effnet)
    combined_prob = yolo_conf * weight_yolo_array + effnet_prob * weight_effnet_array

    final_class_indices = tf.argmax(combined_prob, axis=1).numpy()
    final_class_names = [CLASS_NAMES[i] for i in final_class_indices]
    return final_class_names

# count class occurrences
def show_pred_classes(final_class_names):
    class_counts = {}
    for name in final_class_names:
        if name in class_counts:
            class_counts[name] += 1
        else:
            class_counts[name] = 1
    
    # print(f"Class counts: {class_counts}")
    maxpred = max(class_counts, key=class_counts.get)
    # print("🔸🔸 Classification results:")
    # print(f"🔸🔸 This image contains {len(final_class_names)} {maxpred} cells.")
    return maxpred

# execute all the code in this file
def execute_classification(crop_list, model, filtered_valid_conf):
    effnet_prob = predict_cell_batch(crop_list, model)
    final_class_names = combine_predictions(filtered_valid_conf, effnet_prob)
    maxpred = show_pred_classes(final_class_names)
    return maxpred


# --------------------------------------
# test the model with a single image
# folder = "/Users/patteerasupvithayanond/Documents/FYP_ctc_detection/results"
# # loop through the images in the folder and read in numpy arrays and give a list or numpy arrays
# import cv2

# image_np = []
# for image in os.listdir(folder):
#     img = cv2.imread(os.path.join(folder, image))
#     if img is not None:
#         image_np.append(img)

# predictions = predict_cell_batch(image_np, model)
# show_pred_classes(predictions)
# execute_classification(image_np, model)
