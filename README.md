# PneumoNet — AI-Powered Pneumonia Detection

PneumoNet is a deep learning system that classifies chest X-rays as **NORMAL** or **PNEUMONIA** using transfer learning, with built-in explainability through Grad-CAM heatmaps and confidence-based decision flagging.

## 🎯 Overview

Pneumonia remains one of the leading causes of death worldwide, and chest X-ray interpretation is a routine but time-intensive task for radiologists. PneumoNet was built to explore how transfer learning and explainable AI can assist in flagging potential pneumonia cases — while being transparent about model confidence and limitations.

This project was built end-to-end: from raw dataset to a deployed, interactive web application.

## ✨ Key Features

- **Transfer Learning** — Fine-tuned ResNet18 (pretrained on ImageNet) for binary classification
- **Class Imbalance Handling** — Weighted loss function to address the imbalance between NORMAL (1,341) and PNEUMONIA (3,875) training samples
- **Data Augmentation** — Random flips, rotations, color jitter, and affine transforms to improve generalization
- **Grad-CAM Explainability** — Visual heatmaps showing exactly which regions of the X-ray influenced the model's decision
- **Confidence Thresholding** — Predictions below 70% confidence are flagged as "Inconclusive" rather than forcing a diagnosis
- **Interactive Web App** — Built with Streamlit for real-time inference and visualization

## 📊 Model Performance

| Metric | NORMAL | PNEUMONIA |
|---|---|---|
| Precision | 0.81 | 0.94 |
| Recall | 0.91 | 0.87 |
| F1-Score | 0.86 | 0.91 |

**Overall Test Accuracy: 89%** (624 test images)

### Why this tradeoff matters

An earlier version of this model achieved 100% PNEUMONIA recall but only 56% NORMAL recall — meaning it was biased toward over-flagging healthy patients. After diagnosing this as a class imbalance problem and applying a weighted loss function, NORMAL recall improved to 91% while PNEUMONIA recall remains strong at 87%. This reflects a deliberate, medically-informed precision-recall tradeoff: missing a real pneumonia case is more dangerous than a false alarm, but excessive false positives reduce clinical trust in the system.

## 🏗️ Architecture

```
Input X-Ray (224x224)
        │
        ▼
  ResNet18 Backbone (pretrained, fine-tuned)
        │
        ▼
  Fully Connected Layer (2 classes)
        │
        ▼
  Softmax → Confidence Score
        │
        ├──> < 70% confidence → "Inconclusive"
        └──> ≥ 70% confidence → NORMAL / PNEUMONIA
        │
        ▼
  Grad-CAM on final conv layer → Heatmap overlay
```

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, Torchvision
- **Explainability:** pytorch-grad-cam
- **Web App:** Streamlit
- **Training Environment:** Google Colab (T4 GPU)
- **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Paul Mooney, Kaggle

## 📁 Project Structure

```
PneumoNet/
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── models/
│   └── pneumonet_best.pth  # Trained model weights
├── data/                   # Dataset (not included — see Setup)
├── requirements.txt
└── README.md
```

## ⚙️ Setup & Installation

1. **Clone the repository**
```bash
git clone https://github.com/kanishka1804/PneumoNet.git
cd PneumoNet
```

2. **Create environment**
```bash
conda create -n pneumonet python=3.10 -y
conda activate pneumonet
pip install -r requirements.txt
```

3. **Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it under `data/` with the structure:
```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

4. **Run the app**
```bash
streamlit run app.py
```

## 🔍 How Grad-CAM Works

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of an X-ray that most influenced the model's prediction. This was added specifically to make the model's decisions interpretable — a critical requirement for any AI system used in a medical context, where a "black box" prediction alone is not clinically useful.

## ⚠️ Limitations & Future Work

- Trained on a single public dataset (Kaggle) — performance may vary on X-rays from different equipment or populations
- Binary classification only (NORMAL vs PNEUMONIA) — does not distinguish bacterial vs viral pneumonia
- Future improvements: ensemble models, multi-class classification, calibration analysis, and validation on an external dataset

## ⚠️ Disclaimer

PneumoNet is an educational project built to demonstrate deep learning and explainable AI techniques. It is **not a certified medical device** and should never be used as a substitute for professional medical diagnosis.