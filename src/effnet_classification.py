# Load and run prediction using the trained EfficientNet model
import tensorflow as tf
import numpy as np
# from tensorflow.keras.models import load_model
import os

# suppress TF warnings and info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # Load the EfficientNet model
# MODEL_PATH = "models/efficientnetb0_model_0404_Best.h5"
# model = load_model(MODEL_PATH)

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
    # Get class indices
    class_indices = tf.argmax(preds, axis=1).numpy()
    # Map indices to class names
    class_names = [CLASS_NAMES[i] for i in class_indices]

    # Print predictions
    # for i, pred in enumerate(class_names):
    #     print(f"cell_{i}: {pred}")
    # print("Predictions:", class_names)

    return class_names

# count class occurrences
def show_pred_classes(class_names):
    class_counts = {}
    for name in class_names:
        if name in class_counts:
            class_counts[name] += 1
        else:
            class_counts[name] = 1
    
    # print(f"Class counts: {class_counts}")
    maxpred = max(class_counts, key=class_counts.get)
    print("🔸🔸 Classification results:")
    print(f"🔸🔸 This image contains {len(class_names)} {maxpred} cells.")

# execute all the code in this file
def execute_classification(crop_list, model):
    predictions = predict_cell_batch(crop_list, model)
    show_pred_classes(predictions)


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

# predictions = predict_cell_batch(image_np)
# show_pred_classes(predictions)

# idea: display input image with bounding boxes, and predictions
