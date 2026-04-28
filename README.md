#  Breast Cancer Detection using CNN

A deep learning project that classifies breast ultrasound images into **Benign** and **Malignant** categories using a Convolutional Neural Network (CNN).
--
The model is trained on a dataset of MRI images and achieves ~92% accuracy on the test set.

##  Overview
Breast cancer is one of the most common cancers worldwide, and early detection significantly improves survival rates.  
This project implements an **end-to-end deep learning pipeline** that takes ultrasound images as input and predicts whether the tumor is benign or malignant.

##  Features
- Binary classification: **Benign vs Malignant**
- Custom CNN architecture built from scratch
- Image preprocessing and normalization
- Early stopping to prevent overfitting
- Precision-Recall based **threshold optimization**
- Performance evaluation using:
  - Confusion Matrix
  - Classification Report
  - ROC Curve
- Visualization of:
  - Accuracy curve
  - Loss curve
  - Precision-Recall curve
- Clean prediction pipeline for unseen images

---

## 📂 Project Structure

```
breast-cancer-cnn/
│
├── data/                 (Not included - see dataset section)
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   │
│   └── val/
│       ├── benign/
│       └── malignant/
│
├── model/                (Saved trained model - excluded from repo)
│
├── accuracy.png
│── loss.png
│── pr_curve.png
│── roc.png
│
├── test_images/
│   ├── benign (12).png
│   └── malignant (14).png
│
├── train.py
├── test.py
├── requirements.txt
├── README.md
└── .gitignore
```

##  Model Architecture
```
The CNN model consists of:
- Conv2D (32 filters, ReLU) → MaxPooling  
- Conv2D (64 filters, ReLU) → MaxPooling  
- Conv2D (128 filters, ReLU) → MaxPooling  
- Flatten layer  
- Dense (128 neurons, ReLU)  
- Dropout (0.5)  
- Output layer (Sigmoid activation)
```
##  Dataset

Dataset: Breast Ultrasound Images Dataset  
Source: Kaggle  
 Dataset is not included due to size limitations.
###  Dataset Setup
Download and organize the dataset as:
```
data/
├── train/
│   ├── benign/
│   └── malignant/
│
└── val/
    ├── benign/
    └── malignant/
```
##  Image Preprocessing

- Images resized to **128 × 128**
- Pixel values normalized to **[0, 1]**
- Invalid images are skipped
- Dataset size limited to prevent memory issues

---

##  Training
Run the training script:
python train.py

### Training includes:

- Model training with validation
- Early stopping
- Evaluation metrics
- Threshold optimization (based on F1-score)
- Generation of plots:
  - Accuracy curve
  - Loss curve
  - Precision-Recall curve
  - ROC curve

---

##  Results

- Validation Accuracy: **~92%**
- Balanced precision and recall
- Improved classification using optimized threshold

Example Confusion Matrix:

[[450  50]
 [ 23 377]]

---

##  Prediction

Run prediction on test images:

python predict.py

### Example Output:

benign (12).png → BENIGN (0.0000)  
malignant (14).png → MALIGNANT (0.9970)

- Outputs predicted class
- Displays confidence score
- Uses optimized threshold

---

##  Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

##  Key Highlights
- CNN built completely from scratch  
- Threshold tuning using Precision-Recall curve  
- End-to-end ML pipeline implemented  
- Clean and reproducible project structure  

---

##  Future Improvements

- Apply Transfer Learning (ResNet, EfficientNet)
- Add data augmentation
- Deploy using Flask or Streamlit
- Add Grad-CAM visualization
- Use larger and more diverse datasets

---

##  Notes

- Dataset is excluded due to size
- Model file is not included in repository
- Place model inside `model/` folder before testing

---


Deep Learning project for breast cancer detection using CNN.
