# SkyFinder Temperature Prediction via Deep Imbalanced Regression

> **Take-home test submission for the Health Intelligence Lab, UCLA (Prof. Yuzhe Yang)**
> Adapting the DIR framework (Yang et al., ICML 2021) to outdoor temperature prediction from sky images.

---

## What This Is

Standard regression models fail on rare labels — they see thousands of mild-temperature sky photos and almost none from extreme cold or heat. This repo adapts **Deep Imbalanced Regression (DIR)** to fix that, using the [SkyFinder dataset](https://cs.valdosta.edu/~rpmihail/skyfinder/) as a testbed.

**Task:** Given a sky image → predict outdoor temperature (°C)
**Problem:** Temperature distribution is heavily skewed (30:1 imbalance ratio)
**Solution:** Apply LDS and FDS from the DIR paper and measure what actually changes

---

## Key Results

| Method | Test MAE | RMSE | Low-shot MAE |
|---|---|---|---|
| Baseline | 3.279°C | 4.390°C | 5.713°C |
| **+ LDS** | **3.079°C** | **4.081°C** | **4.481°C** |
| + FDS | 3.326°C | 4.415°C | 5.653°C |
| + LDS + FDS (DIR) | 3.413°C | 4.468°C | 4.930°C |

**LDS reduced low-shot MAE by 21.5%** — the largest gain exactly where imbalance hurts most.

---

## Most Interesting Finding

FDS with default settings caused **128% MAE degradation** (3.28 → 7.47°C).

Root cause: FDS estimates per-bucket statistics in a 512-dimensional feature space. With only 1,680 training images over a 77°C range, the average bucket contains ~22 samples — far too few to estimate reliable covariance matrices. Early smoothing corrupts features instead of improving them.

**Fix:** Delaying FDS activation to epoch 10 (start_smooth=10, momentum=0.99) stabilized training and recovered performance to 3.326°C. The instability spike is visible in the training curves.

> **Practical prescription:** Always check average per-bucket sample count before enabling FDS. LDS is robust to sparsity and should be the default first choice.

---

## Results

<p align="center">
  <img src="results/comparison_bars.png" width="700"/>
</p>

<p align="center">
  <img src="results/training_curves.png" width="700"/>
</p>

<p align="center">
  <img src="results/error_by_temp.png" width="700"/>
</p>

<p align="center">
  <img src="results/scatter_baseline.png" width="340"/>
  <img src="results/scatter_lds.png" width="340"/>
</p>

---

## Repo Structure

```
├── skyfinder_dir/
│   ├── prepare_data.py     # Download SkyFinder images + build CSV
│   ├── datasets.py         # PyTorch Dataset with LDS weighting
│   ├── model.py            # ResNet-18 regression head
│   ├── fds.py              # Feature Distribution Smoothing module
│   ├── loss.py             # Weighted L1/MSE/Huber loss functions
│   ├── train.py            # Full training loop with LDS/FDS support
│   └── analyze.py          # Generate comparison plots and tables
├── results/                # All plots and summary_table.csv
├── report/                 # Full written report (PDF)
└── run_all.bat             # Run all 4 experiments sequentially
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install torch torchvision scipy scikit-learn matplotlib tqdm requests Pillow pandas

# 2. Download SkyFinder data
cd skyfinder_dir
python prepare_data.py

# 3. Run all 4 experiments
run_all.bat

# 4. Generate plots and comparison table
python analyze.py
```

Or run individually:
```bash
python train.py --experiment baseline
python train.py --experiment lds --lds --reweight sqrt_inv
python train.py --experiment fds --fds --start_update 5 --start_smooth 10 --fds_mmt 0.99
python train.py --experiment dir --lds --reweight sqrt_inv --fds --start_update 5 --start_smooth 10 --fds_mmt 0.99
```

---

## Setup Details

- **Model:** ResNet-18 pretrained on ImageNet, regression head (512 → 1)
- **Dataset:** 2,400 SkyFinder images, 10 cameras, −27°C to +50°C
- **Split:** 1,680 train / 360 val / 360 test
- **Hardware:** CPU (no GPU required)
- **Training time:** ~20 mins per experiment on modern CPU

---

## Reference

```bibtex
@inproceedings{yang2021delving,
  title={Delving into Deep Imbalanced Regression},
  author={Yang, Yuzhe and Zha, Kaiwen and Chen, Ying-Cong and Wang, Hao and Katabi, Dina},
  booktitle={ICML},
  year={2021}
}
```

---

## Author

**Manish Dhatrak**
B.Tech Electronics & Computer Engineering | Sanjivani College of Engineering
Research Intern, NTU Singapore
[Portfolio](https://manishportfolio-green.vercel.app/) · [LinkedIn](https://www.linkedin.com/in/manish-dhatrak-b759171aa/) · [Email](mailto:manishdhatrak1121@gmail.com)

---

*Submitted as take-home test for the Health Intelligence Lab internship, UCLA.*
