# Thesis Code Repository
This repository contains the codebase for a comparative study of additive (L2-sum) and multiplicative (L2-mul) regularization techniques in convolutional neural networks. It supports running systematic experiments on the CIFAR-10 cats-vs-deer binary classification task, measuring training dynamics, generalization, and robustness under various noise conditions. The code also includes tools for margin analysis, weight-rank evaluation, and 3D loss-landscape visualization.

---

## Repository Structure
```
├── BinaryClassificationAWN/
│   ├── experiment_utils_mc.py
│   ├── main.py
│   └── model.py
├── BinaryClassificationSKS/
│   ├── HarderNet_Sum_Scaled.py
│   ├── HarderNet_Mul_Scaled.py
│   └── HarderNet_Noreg_Scaled.py
├── LICENSE
├── README.md
└── requirements.txt
```

- **BinaryClassificationAWN**: Implements the training and evaluation pipeline with weight normalization (PaperWN) and L2 regularization (sum, mul, none).
- **BinaryClassificationSKS**: Generates 3D loss-landscape visualizations under different regularization schemes.
- **LICENSE**: MIT License terms.
- **README.md**: This file.
- **requirements.txt**: Python dependencies.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/radubauzh/multiplicative-regularization-for-nn.git
   cd multiplicative-regularization-for-nn
   ```
2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Hyperparameter Sweeps (BinaryClassificationAWN)
Run `main.py` to train CNNs across combinations of:

- L2-sum penalty (`--l2_sum_lambda`)
- L2-mul penalty (`--l2_mul_lambda`)
- Weight normalization modes (`--wn`)
- Depth/features normalization flags
- Batch normalization and bias toggles
- Learning rate, batch size, optimizer, seed, mislabel percentage

**Example**:
```bash
python BinaryClassificationAWN/main.py \
  --lr 0.01 0.001 \
  --batchsize 32 64 \
  --n_epochs 100 \
  --l2_sum_lambda 0.0 0.1 \
  --l2_mul_lambda 0.0 0.1 \
  --wn None default \
  --depth_normalization True False \
  --features_normalization None f_out \
  --batch_norm True False \
  --bias True False \
  --opt_name sgd adam \
  --seed 1 42 \
  --mislabel_percentage 0.0 0.1
```
Results (loss, accuracy, margins, weight-rank metrics) are saved under `Results/`.

### 2. Loss-Landscape Visualization (BinaryClassificationSKS)
Use the `HarderNet_*.py` scripts to generate 3D loss-landscape `.vtp` outputs:
```bash
python BinaryClassificationSKS/HarderNet_Sum_Scaled.py <lambda> <scale> <seed> <mislabel>
python BinaryClassificationSKS/HarderNet_Mul_Scaled.py <lambda> <scale> <seed> <mislabel>
python BinaryClassificationSKS/HarderNet_Noreg_Scaled.py <scale> <seed> <mislabel>
```
- `<lambda>`: regularization strength
- `<scale>`: initialization scaling
- `<seed>`: random seed
- `<mislabel>`: misclassification rate

Outputs are placed in `Results/Visualization/` by default.

---

## Key Concepts

- **L2-Sum Regularization**: Traditional penalty summing squared weights.
- **L2-Mul Regularization**: Novel multiplicative penalty using product of per-layer norms.
- **Weight Normalization**: Controlled via PaperWN wrappers to fix weight scales.
- **Margin Analysis**: Computes classification margins on train/val/test sets.
- **Weight-Rank Metrics**: Estimates effective rank of weight matrices via singular-value analysis.

---

## Requirements

```text
numpy
torch
torchvision
scikit-learn
pandas
matplotlib
losscape
vtk  # for loss-landscape output
``` 

Install all via:
```bash
pip install -r requirements.txt
```

---

## License
This work is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact
For questions, please contact **Raduba** at <raduba@mit.edu>.
