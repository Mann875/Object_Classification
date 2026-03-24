# Object Classification
# Flower Classification via Transfer Learning — TensorFlow & PyTorch

The project fine-tunes a pre-trained **ResNet50** CNN on a 5-class flower image dataset using both TensorFlow/Keras and PyTorch, comparing their results side by side. Implemented and trained on **Google Colab** with GPU acceleration.

---

## Dataset Overview

**Flowers Recognition** dataset sourced from [Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).

| Property | Value |
|:---------|:------|
| Total Images | 4,317 |
| Classes | 5 |
| Images per Class | ~800 |
| Input Resolution | 150×150 (loaded) → resized to 224×224 for ResNet50 |
| Train/Test Split | 75% / 25% (stratified) |
| Training Samples | 3,237 |
| Test Samples | 1,080 |

### Classes & Label Encoding

| Class | Encoded Label |
|:------|:-------------:|
| Daisy | 0 |
| Dandelion | 1 |
| Rose | 2 |
| Sunflower | 3 |
| Tulip | 4 |

---

## Data Pipeline

| Step | Description |
|:-----|:------------|
| **Image Loading** | OpenCV reads each image in BGR colour (`cv2.IMREAD_COLOR`) and resizes to 150×150 |
| **Label Assignment** | `assign_label()` tags each image with its flower class string; `LabelEncoder` converts strings to integers 0–4 |
| **One-Hot Encoding** | `to_categorical(Y_int, num_classes=5)` produces the final label matrix |
| **Normalisation** | Pixel values divided by `255.0` to scale to [0, 1] |
| **Train/Test Split** | `train_test_split` with `test_size=0.25`, `random_state=42`, and `stratify=Y_int` to preserve class balance |
| **Resize for ResNet50** | `tf.image.resize` upscales images from 150×150 → 224×224 to match ImageNet input requirements |

---

## Model Architecture — Transfer Learning with ResNet50

Both implementations share the same core strategy: load **ResNet50** pre-trained on ImageNet, **freeze all base layers**, and attach a custom classification head for the 5 flower classes.

### TensorFlow / Keras

```
ResNet50 Base  (frozen — ImageNet weights)
       ↓
GlobalAveragePooling2D      →  reduces (7×7×2048) to (2048,)
       ↓
Dense(512, activation='relu')
       ↓
Dropout(0.5)
       ↓
Dense(5, activation='softmax')   →  5-class output
```

| Setting | Value |
|:--------|:------|
| Optimizer | `Adam(lr=1e-4)` |
| Loss | `categorical_crossentropy` |
| Metric | `accuracy` |

---

### PyTorch

```
ResNet50 Base  (frozen — IMAGENET1K_V1 weights)
       ↓  (original fc layer replaced)
Linear(2048 → 512)
       ↓
ReLU()
       ↓
Dropout(0.5)
       ↓
Linear(512 → 5)   →  5-class output
```

| Setting | Value |
|:--------|:------|
| Loss | `CrossEntropyLoss` |
| Optimizer | `Adam(lr=1e-4)` — applied only to unfrozen `fc.*` parameters |
| Tensor Format | NHWC → NCHW via `.permute(0, 3, 1, 2)` |
| Device | Auto-detected (`cuda` if available, else `cpu`) |
| Batch Size | 32 via `DataLoader` + `TensorDataset` |

---

## Training

Both models are trained for **20 epochs** with **batch size 32**. Per-epoch metrics tracked:

Tracked mertic like Training loss, Training Accuracy, Validation Loss, Validation Accuracy for both TensorFlow and PyTorch. 

---

## Evaluation & Visualisations

After training, each model is evaluated on the held-out test set with the following outputs:

| Output | Description |
|:-------|:------------|
| **Classification Report** | Per-class precision, recall, and F1-score via `sklearn.metrics.classification_report` |
| **Confusion Matrix** | Plotted with `ConfusionMatrixDisplay` (Blues colormap) for both TF and PyTorch |
| **Loss & Accuracy Curves** | Side-by-side train vs. validation plots over 20 epochs per framework |
| **Accuracy Bar Chart** | Direct TensorFlow vs. PyTorch final test accuracy comparison |

---


## Getting Started

### Prerequisites

```bash
pip install tensorflow torch torchvision torchaudio opencv-python tqdm scikit-learn
```

### Running on Google Colab

1. Upload the **Flowers Recognition** dataset to Google Drive at `MyDrive/flowers/` with one subfolder per class
2. Open `Untitled.ipynb` in Colab and mount Drive when prompted
3. Run all cells in order
4. **GPU runtime is strongly recommended** → Runtime → Change runtime type → T4 GPU

---

## Tech Stack

| Library | Purpose |
|:--------|:--------|
| `tensorflow` / `keras` | ResNet50 loading, model building & training (TF path) |
| `torch` / `torchvision` | ResNet50 loading, model building & training (PyTorch path) |
| `opencv-python` | Image reading and resizing |
| `numpy` | Array manipulation and tensor format conversion |
| `sklearn` | Label encoding, train/test split, classification metrics |
| `matplotlib` / `seaborn` | Loss curves, accuracy plots, confusion matrices |
| `tqdm` | Progress bars during image loading |
| `google.colab` | Google Drive mounting |

---

## 🧠 Key Takeaways

- **Transfer learning** from ImageNet enables strong performance on a small dataset (~4,300 images) by training only the lightweight classification head
- **Freezing the ResNet50 base** dramatically reduces trainable parameters and training time while preserving powerful low-level visual features learned on ImageNet
- **Dual-framework implementation** makes this notebook a useful side-by-side reference for TensorFlow vs. PyTorch workflow comparison
- **Dropout (0.5)** in the custom head is the primary regularisation technique guarding against overfitting on the small dataset
- **Stratified splitting** ensures proportional class representation across both train and test sets

---
