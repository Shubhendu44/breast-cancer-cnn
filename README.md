#  Breast Cancer Detection using CNN
A deep learning project that classifies breast ultrasound images into **Benign** and **Malignant** categories using a Convolutional Neural Network (CNN).

The model is trained on a dataset of MRI images and achieves ~92% accuracy on the test set.

##  Overview
Breast cancer is one of the most common cancers worldwide. Early detection plays a crucial role in improving survival rates.  
This project builds an **end-to-end deep learning pipeline** using CNNs to automatically classify ultrasound images, assisting in medical diagnosis.

---

##  Features

- Binary classification: **Benign vs Malignant**
- Custom-built CNN architecture (from scratch)
- Image preprocessing and normalization
- Early stopping to prevent overfitting
- Precision-Recall based **threshold optimization**
- Performance evaluation using:
  - Confusion Matrix
  - Classification Report
  - ROC Curve
- Visualization of:
  - Accuracy & Loss curves
  - Precision-Recall curve
- Clean testing pipeline for unseen images

##  Project Structure
breast-cancer-cnn/
│
├── data/ # (Not included - see dataset section)
│ ├── train/
│ └── val/
│
├── model/ # Saved trained model (excluded from repo)
│
├── outputs/ # Generated graphs
│ ├── accuracy.png
│ ├── loss.png
│ ├── pr_curve.png
│ └── roc.png
│
├── test_images/ # Sample test images
│
├── train.py # Model training script
├── test.py # Prediction script
├── requirements.txt
├── README.md
└── .gitignore


---

##  Model Architecture
The CNN model is built using TensorFlow/Keras:
- Conv2D (32 filters) + ReLU
- MaxPooling
- Conv2D (64 filters) + ReLU
- MaxPooling
- Conv2D (128 filters) + ReLU
- MaxPooling
- Flatten layer
- Dense (128 neurons) + ReLU
- Dropout (0.5)
- Output layer (Sigmoid)

##  Dataset

- **Breast Ultrasound Images Dataset**
- Source: Kaggle

>  Dataset is not included due to size limitations.

###  How to Use Dataset

1. Download dataset from Kaggle
2. Extract and arrange like:
data/
├── train/
│ ├── benign/
│ └── malignant/
│
├── val/
├── benign/
└── malignant/

##  Image Preprocessing

- Resized to **128 × 128**
- Normalized to range **[0, 1]**
- Invalid images skipped
- Limited dataset size to prevent memory issues

##  Training

### Run Training:
python train.py

Includes:
Model training with validation
Early stopping
Evaluation metrics
Threshold optimization (F1-score based)
Graph generation:
Accuracy curve
Loss curve
Precision-Recall curve
ROC curve

## ## Results
Validation Accuracy: ~92%
Balanced precision and recall
Optimized threshold improves classification performance


## Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
Scikit-learn
##  Key Highlights
Built CNN from scratch
Implemented threshold tuning using PR curve
End-to-end ML pipeline:
Data → Model → Evaluation → Prediction
Clean, reproducible codebase

##  Future Improvements
Use Transfer Learning (ResNet, EfficientNet)
Add data augmentation
Deploy as web application (Flask/Streamlit)
Integrate Grad-CAM for model explainability
Use larger and more diverse datasets

