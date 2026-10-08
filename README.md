<<<<<<< HEAD
# 🫁 PneumoScan — Pneumonia Detection from Chest X-Rays
## Beginner's Complete Guide

---

## What This Does

This project uses a **Convolutional Neural Network (CNN)** to analyze chest X-ray images
and classify them as either **NORMAL** or **PNEUMONIA**.

It includes:
- `train.py`   → Trains the CNN model on your Kaggle dataset
- `app.py`     → Launches a local web interface for predictions
- `templates/` → The web UI (HTML/CSS/JS)

---

## How CNNs Work (Quick Explanation)

A CNN (Convolutional Neural Network) is a type of AI model inspired by the human visual cortex.
It scans images in small patches (called "filters") to detect:
- Layer 1: edges and corners
- Layer 2: textures and curves
- Layer 3+: complex shapes and patterns (e.g., lung opacity)

We use **Transfer Learning** with **MobileNetV2**:
- MobileNetV2 was pre-trained on 1.2 million photos
- We add a small "classification head" on top
- This lets us get great accuracy even with a limited dataset

---

## Step-by-Step Setup

### Step 1 — Download the Dataset from Kaggle

1. Go to: https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images
2. Click "Download" (ZIP file ~2 GB)
3. Extract it — you should get a folder named `chest_xray/`
4. Place the `chest_xray/` in the same folder as `train.py`

Expected structure:
```
pneumonia_detector/
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/      (images)
│   │   └── PNEUMONIA/   (images)
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── train.py
├── app.py
├── requirements.txt
└── templates/
    └── index.html
```

### Step 2 — Install Python Dependencies

Make sure you have Python 3.9+ installed. Then run:

```bash
pip install -r requirements.txt
```

This installs TensorFlow, Flask, Pillow, and other required libraries.

> 💡 If you have a GPU (NVIDIA), install `tensorflow-gpu` instead of `tensorflow`
> for much faster training!

### Step 3 — Train the Model

```bash
python train.py
```

This will:
- Load and augment the training images
- Train MobileNetV2 + custom head (2 phases)
- Save the best model as `pneumonia_model.h5`
- Generate `training_history.png` and `confusion_matrix.png`

Training takes ~10-30 minutes on a CPU, ~5 minutes on a GPU.

Expected output:
```
📂 Loading dataset...
✅ Classes: {'NORMAL': 0, 'PNEUMONIA': 1}
✅ Training samples  : 4192
...
🎉 Training complete! Model saved to: ./pneumonia_model.h5
```

### Step 4 — Launch the Web App

```bash
python app.py
```

Then open your browser and go to:
```
http://localhost:5000
```

You'll see the PneumoScan interface where you can upload any chest X-ray
and get an instant prediction!

---

## Understanding the Results

| Field       | Meaning |
|-------------|---------|
| **Label**   | NORMAL or PNEUMONIA |
| **Confidence** | How certain the model is (e.g. 94.5%) |
| **Risk Level** | High (>70%), Moderate (40-70%), Low (<40%) |
| **Probability bars** | Visual split between NORMAL and PNEUMONIA |

---

## Model Performance (Expected)

With the standard Kaggle dataset, you should achieve approximately:
- Accuracy: ~80-95%
- AUC: ~0.94
- Recall (sensitivity): ~95%+ (important: we want to catch all pneumonia cases!)

---

## ⚠️ Important Disclaimer

This tool is for **educational and research purposes only**.
It is **NOT** a certified medical device and must **NOT** be used
for actual clinical diagnosis. Always consult a licensed physician.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Could not load model` | Run `python train.py` first |
| `Out of memory` | Reduce `BATCH_SIZE` in `train.py` (try 16 or 8) |
| Low accuracy | Try increasing `EPOCHS_FINE` or reducing `Dropout` |
| Slow training | Use Google Colab (free GPU) or reduce image size to 160×160 |
| Port 5000 in use | Change `port=5000` to `port=8080` in `app.py` |
=======
# CNN-pneumonia-detection
>>>>>>> 61d4c87ca7601fe3187997673ef9e555d7c078d5
