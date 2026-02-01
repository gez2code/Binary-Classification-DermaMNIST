# DermaMNIST Binary Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gez2code/Binary-Classification-DermaMNIST/blob/main/Binary_Classification_DermaMNIST.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)

Binary classification of skin lesions (Malignant vs Benign) using deep learning on the DermaMNIST dataset.

**Course:** Deep Learning | **Semester:** Autumn 25/26 | **University:** UNICAM

---

## 🎯 Objective

Maximize **recall (sensitivity)** for malignant lesion detection to minimize missed cancers in medical screening scenarios.

**Why Recall?** In medical diagnosis:
- **False Negative** (missed cancer) → Potentially fatal
- **False Positive** (unnecessary biopsy) → Inconvenient but safe

---

## 🚀 Quick Start (Recommended: Google Colab)

> ⚠️ **For Reviewers:** Please use Google Colab for reproducibility. All configurations are hardcoded. In General it is possible to run the script from top to buttom.
Limitation in RAM may occure due to workload.

### Step-by-Step Instructions

1. **Click the "Open in Colab" badge above**
2. **Enable GPU Runtime:**
   - Go to `Runtime` → `Change runtime type` → Select `T4 GPU`
3. **Run all cells sequentially** (`Runtime` → `Run all`)
4. **Total estimated runtime:** ~45-60 minutes (with T4 GPU)

### Default Configuration (Hardcoded for Reproducibility)

```python
# ═══════════════════════════════════════════════════════════════════════════
#                         CONFIGURATION (DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════════════════

# ENVIRONMENT
USE_COLAB = True       # Use Google Colab environment
USE_WANDB = False      # Disabled - no account required for reviewers

# REPRODUCIBILITY (Fixed seed for exact replication)
SEED = 42

# PROJECT
PROJECT_NAME = 'DermaMNIST_Binary_Study'
```

> 💡 **Note:** `USE_WANDB = False` ensures no external account is needed. All metrics are printed and plotted directly in the notebook.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [MedMNIST](https://medmnist.com/) - DermaMNIST subset |
| **Size** | 10,015 images |
| **Split** | Train: 7,007 / Val: 1,003 / Test: 2,005 |
| **Dimensions** | 28×28 RGB |
| **Classes** | Malignant (1) vs Benign (0) |
| **Imbalance** | ~4:1 (Benign:Malignant) |

### Original Classes → Binary Mapping

| Original Class | Binary Label | Clinical Rationale |
|----------------|--------------|-------------------|
| Melanoma (mel) | **Malignant (1)** | Most dangerous skin cancer |
| Basal cell carcinoma (bcc) | **Malignant (1)** | Most common skin cancer |
| Actinic keratoses (akiec) | **Malignant (1)** | Pre-cancerous, can become SCC |
| Vascular lesions (vasc) | **Malignant (1)** | Can indicate malignancy |
| Melanocytic nevi (nv) | Benign (0) | Common moles |
| Benign keratosis (bkl) | Benign (0) | Non-cancerous growth |
| Dermatofibroma (df) | Benign (0) | Benign fibrous nodule |

---

## 📓 Notebook Structure & Runtime Estimates

Below is a detailed breakdown of each section in the notebook with expected runtimes on a **T4 GPU**.

### Overview Table

| Section | Description | Runtime | GPU Required |
|---------|-------------|---------|--------------|
| 1. Setup | Install packages, imports | ~2 min | No |
| 2. Data Loading | Download & preprocess DermaMNIST | ~1 min | No |
| 3. Data Exploration | Visualize class distribution | ~30 sec | No |
| 4. Phase 1 | Architecture screening (14 models) | ~25-30 min | **Yes** |
| 5. Phase 2 | Hyperparameter tuning (8 experiments) | ~12-15 min | **Yes** |
| 6. Phase 3 | Final evaluation on test set | ~2 min | Yes |
| 7. Results | Plots, confusion matrix, analysis | ~1 min | No |
| **Total** | | **~45-60 min** | |

---

### Detailed Section Descriptions

#### 1️⃣ Setup & Configuration (~2 minutes)

```
📦 What it does:
   - Installs required packages (medmnist, tensorflow, etc.)
   - Sets random seeds for reproducibility
   - Configures GPU memory growth
   - Defines helper functions

⚙️ Key outputs:
   - TensorFlow version confirmation
   - GPU availability check
   - "Setup complete" message
```

#### 2️⃣ Data Loading & Preprocessing (~1 minute)

```
📦 What it does:
   - Downloads DermaMNIST from MedMNIST repository
   - Converts 7-class labels to binary (malignant/benign)
   - Normalizes pixel values to [0, 1]
   - Creates train/validation/test splits

⚙️ Key outputs:
   - Dataset shapes: (7007, 28, 28, 3), (1003, ...), (2005, ...)
   - Class distribution statistics
   - "Data loaded successfully" message
```

#### 3️⃣ Data Exploration (~30 seconds)

```
📦 What it does:
   - Visualizes sample images from each class
   - Plots class distribution (pie chart, bar chart)
   - Calculates class imbalance ratio

⚙️ Key outputs:
   - Sample image grid
   - Class distribution: ~80.5% Benign, ~19.5% Malignant
   - Imbalance ratio: 4.1:1
```

#### 4️⃣ Phase 1: Architecture Screening (~25-30 minutes)

```
📦 What it does:
   - Trains 14 different model architectures
   - Uses consistent training protocol (class weights, early stopping)
   - Evaluates on validation set
   - Selects winner based on F2 score + lowest FN

⚙️ Models trained:
   ├── Custom CNN: shallow, deep, wide (3 models, ~3 min each)
   ├── Hybrid CNN: with/without CBAM (2 models, ~3 min each)
   ├── MobileNetV2: frozen, partial, full (3 models, ~2 min each)
   ├── EfficientNetB0: frozen, partial, full (3 models, ~2 min each)
   └── DenseNet121: frozen, partial, full (3 models, ~2 min each)

⚙️ Key outputs:
   - Training curves for each model
   - Validation metrics table
   - Phase 1 Winner announcement (expected: Hybrid_NoCBAM)
```

#### 5️⃣ Phase 2: Hyperparameter Tuning (~12-15 minutes)

```
📦 What it does:
   - Takes Phase 1 winner (Hybrid_NoCBAM)
   - Tests 8 hyperparameter configurations
   - Optimizes dropout, learning rate, class weights

⚙️ Experiments:
   ├── Dropout: 0.4, 0.6 (2 experiments)
   ├── Learning rate: 0.001, 0.0003 (2 experiments)
   ├── Class weights: W4 (1:4), W5 (1:5) (2 experiments)
   └── Filter sizes: 32, 128 (2 experiments)

⚙️ Key outputs:
   - Tuning results comparison table
   - Phase 2 Winner announcement (expected: Hybrid_DR06)
```

#### 6️⃣ Phase 3: Final Evaluation (~2 minutes)

```
📦 What it does:
   - Loads best model from Phase 2
   - Calibrates decision threshold on validation set
   - Evaluates on held-out TEST set (first time!)
   - Generates final metrics

⚙️ Key outputs:
   - Optimal threshold value
   - Test set metrics: Recall, Precision, F2, AUC
   - Confusion matrix
```

#### 7️⃣ Results & Analysis (~1 minute)

```
📦 What it does:
   - Plots final confusion matrix
   - Generates classification report
   - Compares all models visually
   - Provides clinical interpretation

⚙️ Key outputs:
   - Confusion matrix visualization
   - ROC curve
   - Summary statistics
   - Clinical interpretation text
```

---

## 📈 Expected Results

### Final Test Set Performance (Hybrid_DR06, dropout=0.6)

| Metric | Value |
|--------|-------|
| **Recall (Sensitivity)** | 91.58% (359/392 malignant detected) |
| **Precision** | 37.95% |
| **F2 Score** | 0.714 |
| **AUC** | 0.877 |
| **False Negatives** | 33 (missed cancers) |
| **False Positives** | 587 (unnecessary referrals) |

### Confusion Matrix

```
                    Predicted
                 Benign  Malignant
Actual Benign     1026      587
Actual Malignant    33      359
```

### Clinical Interpretation

- ✅ **359 of 392 cancers detected** (91.6% sensitivity)
- ❌ **33 cancers missed** (8.4% false negative rate)
- ⚠️ **587 false alarms** (36% of benign cases)

**Suitable for:** Screening (triage) applications where missing a cancer is worse than a false alarm.

**Not suitable for:** Final diagnosis (requires dermatologist confirmation).

---

## 🔬 Methodology

### Three-Phase Experimental Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Architecture Screening (14 experiments, ~25 min)              │
│  ├── Custom CNN: 3 variants (shallow, deep, wide)                       │
│  ├── Hybrid CNN: 2 variants (with/without CBAM)                         │
│  ├── MobileNetV2: 3 variants (frozen, partial, full)                    │
│  ├── EfficientNetB0: 3 variants (frozen, partial, full)                 │
│  └── DenseNet121: 3 variants (frozen, partial, full)                    │
│                                     │                                   │
│                                     ▼                                   │
│  PHASE 2: Hyperparameter Tuning (8 experiments, ~15 min)                │
│  ├── Dropout: 0.4, 0.6                                                  │
│  ├── Learning rate: 0.001, 0.0003                                       │
│  ├── Class weights: W4, W5                                              │
│  └── Filter sizes: 32, 128                                              │
│                                     │                                   │
│                                     ▼                                   │
│  PHASE 3: Final Evaluation (~2 min)                                     │
│  ├── Threshold calibration on validation set                            │
│  ├── Test set evaluation (held out until now!)                          │
│  └── Clinical interpretation                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| **Primary Metric** | F2 Score | Weights recall 2× more than precision |
| **Class Weights** | {0: 1.0, 1: 3.0} | Addresses 4:1 class imbalance |
| **Early Stopping** | patience=10 | Prevents overfitting, monitors val_f2 |
| **Threshold** | 0.55 (calibrated) | Optimized for recall ≥90% |
| **Seed** | 42 | Ensures reproducibility |

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: medmnist` | Run `!pip install medmnist` |
| `CUDA out of memory` | Restart runtime, or reduce batch_size to 32 |
| Very slow training | Check GPU is enabled: `Runtime` → `Change runtime type` → `T4 GPU` |
| `ResourceExhaustedError` | Restart runtime to clear GPU memory |
| Different results | Ensure SEED=42 and same TensorFlow version |

### Verify GPU is Enabled

Run this cell to check:
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
# Should show: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 📁 Project Structure

```
Binary-Classification-DermaMNIST/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── Binary_Classification_DermaMNIST.ipynb # Main notebook
└── DermaMNIST_Binary_Study/               # Created during training
    ├── Phase1_Baselines/                  # Architecture screening results
    ├── Phase2_Tuning/                     # Hyperparameter tuning results
    └── Phase3_Final/                      # Final model and results
```

---

## 📚 References

- **Dataset**: Yang, J., et al. (2023). MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. [Scientific Data](https://doi.org/10.1038/s41597-022-01721-8)
- **MobileNetV2**: Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. [CVPR 2018](https://arxiv.org/abs/1801.04381)
- **DenseNet**: Huang, G., et al. (2017). Densely Connected Convolutional Networks. [CVPR 2017](https://arxiv.org/abs/1608.06993)
- **EfficientNet**: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling. [ICML 2019](https://arxiv.org/abs/1905.11946)
- **CBAM**: Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. [ECCV 2018](https://arxiv.org/abs/1807.06521)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✍️ Author

**Abraham Gezehei**  
📧 abraham.gezehei@studenti.unicam.it  
🎓 Università di Camerino (UNICAM)  
📅 Deep Learning Course, Autumn Semester 25/26

**Supervisor:** Prof. Michela Quadrini

---

## 🙏 Acknowledgments

- [MedMNIST](https://medmnist.com/) for providing the standardized medical imaging dataset
- [TensorFlow](https://tensorflow.org/) team for the deep learning framework
- [Google Colab](https://colab.research.google.com/) for free GPU access
