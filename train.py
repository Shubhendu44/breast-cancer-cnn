import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc

train_path = "data/train"
val_path   = "data/val"
categories = ["benign", "malignant"]
IMG_SIZE = 128
LIMIT = 5000


data = []
labels = []

count = 0

for category in categories:
    path = os.path.join(train_path, category)
    label = categories.index(category)
    for img in os.listdir(path):
        if count >= LIMIT:
            break
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        data.append(image)
        labels.append(label)
        count += 1
print("Training images loaded:", len(data))
data = np.array(data)
labels = np.array(labels)

val_data = []
val_labels = []

for category in categories:
    path = os.path.join(val_path, category)
    label = categories.index(category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        val_data.append(image)
        val_labels.append(label)

print("Validation images loaded:", len(val_data))

val_data = np.array(val_data)
val_labels = np.array(val_labels)

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
history = model.fit(
    data,
    labels,
    epochs=20,
    batch_size=16,
    validation_data=(val_data, val_labels),
    callbacks=[early_stop]
)
loss, accuracy = model.evaluate(val_data, val_labels)
print("Validation Accuracy:", accuracy)

y_probs = model.predict(val_data).ravel()

precision, recall, thresholds = precision_recall_curve(val_labels, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("Best Threshold:", best_threshold)


y_pred = (y_probs > best_threshold).astype(int)
cm = confusion_matrix(val_labels, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n")
print(classification_report(val_labels, y_pred))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.clf()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.clf()


plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.clf()

fpr, tpr, _ = roc_curve(val_labels, y_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
plt.clf()

print("All plots saved in /outputs")
model.save("model/breast-cancer_cnn.keras")
print("Model saved as breast-cancer_cnn.keras")