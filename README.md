# DermaMNIST Binary Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gez2code/Binary-Classification-DermaMNIST/blob/main/Binary_Classification_DermaMNIST.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)

Binary classification of skin lesions (Malignant vs Benign) using deep learning with transfer learning approaches.

---

## 🎯 Objective

Maximize **recall (sensitivity)** for malignant lesion detection to minimize missed cancers in medical screening scenarios.

**Why Recall?** In medical diagnosis:
- **False Negative** (missed cancer) → Potentially fatal
- **False Positive** (unnecessary biopsy) → Inconvenient but safe

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
| Actinic keratoses (akiec) | **Malignant (1)** | Pre-cancerous, can become SCC |
| Basal cell carcinoma (bcc) | **Malignant (1)** | Most common skin cancer |
| Melanoma (mel) | **Malignant (1)** | Most dangerous skin cancer |
| Benign keratosis (bkl) | Benign (0) | Non-cancerous growth |
| Dermatofibroma (df) | Benign (0) | Benign fibrous nodule |
| Melanocytic nevi (nv) | Benign (0) | Common moles |
| Vascular lesions (vasc) | Benign (0) | Benign blood vessel growths |

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. Go to `Runtime` → `Change runtime type` → Select `GPU` (T4)
3. Run all cells sequentially

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/gez2code/Binary-Classification-DermaMNIST.git
cd Binary-Classification-DermaMNIST

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook Binary_Classification_DermaMNIST.ipynb
```

---

## ⚙️ Configuration

All configuration options are located in the **Configuration cell** at the top of the notebook.

### Quick Setup Table

| Your Environment | `USE_COLAB` | `USE_WANDB` | `USE_DRIVE` |
|------------------|-------------|-------------|-------------|
| **Google Colab** (recommended) | `True` | `True` or `False` | `True` or `False` |
| **Local Jupyter/VS Code** | `False` | `True` or `False` | `False` |

### Configuration Options

```python
# ═══════════════════════════════════════════════════════════════════════
#                         CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# ENVIRONMENT
USE_COLAB = True      # True = Google Colab, False = Local machine
USE_WANDB = True      # True = Enable experiment tracking, False = Disable
USE_DRIVE = True      # True = Save to Google Drive, False = Save locally

# REPRODUCIBILITY  
SEED = 42             # Random seed (don't change for reproducibility)

# PROJECT
PROJECT_NAME = 'DermaMNIST_Binary_Study'
```

### Weights & Biases Setup (Optional)

If you set `USE_WANDB = True`:
1. Create a free account at [wandb.ai](https://wandb.ai)
2. Add your API key to Colab Secrets (key: `WANDB_API_KEY`)
3. The notebook will auto-login

---

## 🔬 Methodology

### Three-Phase Experimental Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Architecture Screening (15 experiments)                       │
│  ├── Custom CNN: 3 variants (shallow, deep, wide)                       │
│  ├── MobileNetV2: 3 variants (frozen, partial, full)                    │
│  ├── EfficientNetB0: 3 variants (frozen, partial, full)                 │
│  ├── DenseNet121: 3 variants (frozen, partial, full)                    │
│  └── Hybrid CNN + CBAM: Attention mechanism                             │
│                                     │                                   │
│                                     ▼                                   │
│  PHASE 2: Hyperparameter Tuning [Winner from Phase 1]                   │
│  ├── Freeze depth optimization                                          │
│  ├── Dropout regularization                                             │
│  └── Learning rate tuning                                               │
│                                     │                                   │
│                                     ▼                                   │
│  PHASE 3: Final Evaluation                                              │
│  ├── Threshold calibration on validation set                            │
│  ├── Test set evaluation (held out until now!)                          │
│  └── Clinical interpretation                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Architecture Overview

| Architecture | Parameters | Input Size | Description |
|--------------|------------|------------|-------------|
| Custom CNN | ~50K-200K | 28×28 | Native resolution, baseline |
| Hybrid CNN + CBAM | ~300K | 28×28 | Attention mechanism |
| MobileNetV2 | 3.4M | 56×56 | Lightweight transfer learning |
| EfficientNetB0 | 5.3M | 56×56 | Efficient compound scaling |
| DenseNet121 | 8M | 56×56 | Dense connections |

> ❌ **Removed:** ResNet50 (25M), VGG16 (138M) - too large for 28×28 images

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **F2 Score as monitor metric** | Weights recall 2x more than precision |
| **Threshold calibration** | Boost recall while maintaining precision ≥40% |
| **Class weighting (W3)** | `{0: 1.0, 1: 3.0}` to handle imbalance |
| **Test set isolation** | No peeking until final evaluation |

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

## 📈 Results

Results depend on your specific run. The notebook automatically:
- Selects the best model from Phase 1 based on validation F2 score
- Fine-tunes hyperparameters in Phase 2
- Calibrates the decision threshold for optimal recall
- Evaluates on the held-out test set

### Expected Performance Range

| Metric | Typical Range |
|--------|---------------|
| **Recall (Sensitivity)** | 80-90% |
| **Precision** | 35-50% |
| **F2 Score** | 0.65-0.75 |
| **AUC** | 0.80-0.90 |

### Clinical Interpretation

The model prioritizes **minimizing missed cancers** (false negatives) at the cost of some false positives:
- ✅ **High recall**: Catches most malignant lesions
- ⚠️ **Lower precision**: Some unnecessary biopsies
- 🎯 **Clinical value**: Better to over-refer than miss cancer

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: medmnist` | Run `pip install medmnist` |
| `CUDA out of memory` | Reduce `batch_size` to 32 or 16 |
| `wandb: permission denied` | Set `USE_WANDB = False` or run `wandb login` |
| `Drive mount failed` | Set `USE_DRIVE = False` for local saving |
| Very slow training | Ensure GPU is enabled: `tf.config.list_physical_devices('GPU')` |

### Getting Help

1. Check existing [Issues](https://github.com/gez2code/Binary-Classification-DermaMNIST/issues)
2. Open a new issue with:
   - Error message
   - Environment (Colab/Local, OS, Python version)
   - Steps to reproduce

---

## 📚 References

- **Dataset**: Yang, J., et al. (2023). MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. [arXiv:2110.14795](https://arxiv.org/abs/2110.14795)
- **MobileNetV2**: Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. [CVPR 2018](https://arxiv.org/abs/1801.04381)
- **DenseNet**: Huang, G., et al. (2017). Densely Connected Convolutional Networks. [CVPR 2017](https://arxiv.org/abs/1608.06993)
- **EfficientNet**: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling. [ICML 2019](https://arxiv.org/abs/1905.11946)
- **CBAM**: Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. [ECCV 2018](https://arxiv.org/abs/1807.06521)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MedMNIST](https://medmnist.com/) for providing the standardized medical imaging dataset
- [TensorFlow](https://tensorflow.org/) team for the deep learning framework
- [Weights & Biases](https://wandb.ai/) for experiment tracking tools

---

## ✍️ Author

Created as part of a deep learning study on medical image classification.

**Questions?** Open an issue or reach out!
