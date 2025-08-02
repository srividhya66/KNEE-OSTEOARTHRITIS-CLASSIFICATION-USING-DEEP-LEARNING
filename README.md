# 🦵 Knee Osteoarthritis (OA) Classification using EfficientNetB7

This project implements a deep learning-based approach to classify knee X-ray images for Osteoarthritis (OA) severity using the **EfficientNetB7** architecture in TensorFlow. It leverages transfer learning and mixed precision training for improved performance and efficiency.

---

## 📁 Dataset Structure

Make sure your dataset is organized in the following structure, split into `train`, `val`, and `test` directories:


Each subfolder represents a class label based on **Kellgren–Lawrence (KL) grading** (0 to 4).

---

## 🧠 Model Architecture

- **Base Model**: `EfficientNetB7` (pre-trained on ImageNet)
- **Head Layers**:
  - GlobalAveragePooling2D
  - Dropout (rate=0.5)
  - Dense (512 units, ReLU activation)
  - Output: Dense (5 units, Softmax)

**Loss Function**: `categorical_crossentropy`  
**Optimizer**: `Adam (learning rate = 1e-4)`  
**Metrics**: `accuracy`, `precision`, `recall`, `F1-score`

---

## 🧪 Training Pipeline

- Data Augmentation using `ImageDataGenerator`
  - Rescaling
  - Rotation, shift, shear, zoom
- Callbacks:
  - `EarlyStopping`
  - `ModelCheckpoint`
  - `ReduceLROnPlateau`
- Uses `tf.keras.mixed_precision` to speed up training

---

## 📈 Evaluation

After training, the model is evaluated using:

- ✅ Accuracy
- ✅ Confusion Matrix
- ✅ Classification Report
- ✅ Optionally: ROC AUC Score

> 📊 Evaluation results will appear in the notebook. You can visualize predictions and misclassifications too.

---

## 💻 How to Run

1. Clone this repo:

```bash
git clone https://github.com/srividhya66/knee-oa-classifier.git
cd knee-oa-classifier
📊 Sample Results
(Add visualizations like training curves, confusion matrix, and Grad-CAM here if available)

📦 Dependencies
Python ≥ 3.7

TensorFlow ≥ 2.8

NumPy, Matplotlib, Seaborn

scikit-learn

Pillow

Torch & torchvision (for possible ViT comparison)

🔮 Future Enhancements
 Implement Grad-CAM for explainability

 Compare with Vision Transformers (ViT)

 Integrate with a web dashboard (e.g., Streamlit)

 Automate data preprocessing pipeline

 Deploy as a REST API with FastAPI or Flask

