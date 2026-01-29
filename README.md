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
| **Dimensions** | 28×28 RGB |
| **Classes** | Malignant (1) vs Benign (0) |
| **Imbalance** | ~9:1 (Benign:Malignant) |

### Original Classes → Binary Mapping

| Original Class | Binary Label |
|----------------|--------------|
| Melanocytic nevi (nv) | Malignant (1) |
| Melanoma (mel) | Malignant (1) |
| Dermatofibroma (df) | Malignant (1) |
| Benign keratosis (bkl) | Benign (0) |
| Basal cell carcinoma (bcc) | Benign (0) |
| Actinic keratoses (akiec) | Benign (0) |
| Vascular lesions (vasc) | Benign (0) |

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

### Environment Variables (Set in Notebook)

```python
# ============================================================================
# CONFIGURATION - Modify these settings as needed
# ============================================================================
USE_COLAB = True      # Set to False for local execution
USE_WANDB = True      # Set to False to disable experiment tracking
USE_DRIVE = True      # Set to False to save models locally (Colab only)
SEED = 42             # Random seed for reproducibility
```

### Weights & Biases (Optional)

W&B provides experiment tracking and visualization. To enable:

1. Create free account at [wandb.ai](https://wandb.ai)
2. Get your API key from [wandb.ai/settings](https://wandb.ai/settings)
3. Set `USE_WANDB = True` in the notebook
4. When prompted, paste your API key

To disable tracking: Set `USE_WANDB = False`

### GPU Requirements

| Environment | GPU Setup |
|-------------|-----------|
| **Colab** | Runtime → Change runtime type → GPU (T4 recommended) |
| **Local** | NVIDIA GPU with CUDA support + cuDNN |

> ⚠️ Training without GPU will be significantly slower (~10x)

---

## 📁 Project Structure

```
Binary-Classification-DermaMNIST/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── Binary_Classification_DermaMNIST.ipynb # Main notebook
├── models/                                # Saved model checkpoints
│   └── .gitkeep
└── results/                               # Experiment results
    └── .gitkeep
```

---

## 🔬 Methodology

### Three-Phase Experimental Design

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Architecture Selection                                    │
│  ├── Custom CNN (baseline)                                          │
│  ├── ResNet50 ─────────────────────┐                                │
│  ├── VGG16                         │                                │
│  └── EfficientNetB0                │                                │
│                                    ▼                                │
│  PHASE 2: Hyperparameter Tuning [Winner: ResNet50]                  │
│  ├── Freeze10 (conservative)                                        │
│  ├── Freeze20 (balanced) ──────────┐                                │
│  ├── HighDropout (regularization)  │                                │
│  └── LowLR (stability)             │                                │
│                                    ▼                                │
│  PHASE 3: Final Evaluation    [Winner: Freeze20]                    │
│  ├── Threshold calibration on validation set                        │
│  ├── Test set evaluation (held out until now!)                      │
│  └── Clinical interpretation                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Recall as primary metric** | Minimize missed cancers (clinical priority) |
| **Threshold calibration** | Boost recall while maintaining precision ≥40% |
| **Class weighting** | Handle 9:1 class imbalance |
| **Test set isolation** | No peeking until final evaluation |

---

## 📈 Results

### Final Model Performance (Test Set)

| Metric | Value |
|--------|-------|
| **Recall (Sensitivity)** | 85% |
| **Precision** | 33% |
| **F1 Score** | 47% |
| **AUC** | 0.91 |

### Confusion Matrix

```
                  Predicted
                 Benign  Malignant
Actual Benign      756      145
       Malignant    12       70
```

### Clinical Interpretation

- **Total malignant cases**: 82
- **Correctly detected**: 70 (85%)
- **Missed (False Negatives)**: 12 (15%)

---

## 🛠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: medmnist` | Run `pip install medmnist` |
| `CUDA out of memory` | Reduce `batch_size` to 16 |
| `wandb: permission denied` | Set `USE_WANDB = False` or login with `wandb login` |
| `Drive mount failed` | Set `USE_DRIVE = False` for local saving |
| Very slow training | Ensure GPU is enabled (check with `tf.config.list_physical_devices('GPU')`) |

### Getting Help

1. Check existing [Issues](https://github.com/gez2code/Binary-Classification-DermaMNIST/issues)
2. Open a new issue with:
   - Error message
   - Environment (Colab/Local, OS, Python version)
   - Steps to reproduce

---

## 📚 References

- **Dataset**: Yang, J., et al. (2023). MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. [arXiv:2110.14795](https://arxiv.org/abs/2110.14795)
- **ResNet**: He, K., et al. (2016). Deep Residual Learning for Image Recognition. [CVPR 2016](https://arxiv.org/abs/1512.03385)
- **Transfer Learning**: Tan, C., et al. (2018). A Survey on Deep Transfer Learning. [arXiv:1808.01974](https://arxiv.org/abs/1808.01974)

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
