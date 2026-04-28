import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
IMG_SIZE = 128
THRESHOLD = 0.28328282 
MODEL_PATH = "model/breast_cancer_cnn_final.keras"
TEST_FOLDER = "test_images"

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!\n")


def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Error loading:", img_path)
        return None, None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    prediction = model.predict(img, verbose=0)[0][0]
    result = "MALIGNANT" if prediction > THRESHOLD else "BENIGN"
    return result, prediction


# Testing part 


print("Running predictions...\n")
for img_name in os.listdir(TEST_FOLDER):
    img_path = os.path.join(TEST_FOLDER, img_name)
    result, pred = predict_image(img_path)
    if result is None:
        continue
    print(f"{img_name} → {result} ({pred:.4f})")
print("\nDone.")
