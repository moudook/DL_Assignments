# CS601T — Assignment 1: Perceptron for Classification and Regression

**Group 11** | Semester 2, 2026/2027

| Roll No | Name |
|---------|------|
| CS24BT055 | Himanshu Nagwanshi |
| CS24BT061 | Sidharth B |
| CS24BT060 | Anshuman Rahul |

---

## Project Structure

```
Assignment1/
├── README.md                          ← this file
├── report_final.pdf                   ← full report (10 pages, with dataset analysis + citations)
├── data_loader.py                     ← loads all 4 datasets from data/
│
├── models/
│   ├── perceptron.py                  ← single-layer perceptron (sigmoid / tanh / linear)
│   └── one_vs_one.py                  ← one-vs-one multi-class wrapper
│
├── optimizers/
│   └── gradient_descent.py            ← batch gradient descent optimiser
│
├── classification/
│   ├── run_ls.py                      ← Dataset 1: linearly separable (sigmoid + tanh)
│   └── run_nls.py                     ← Dataset 2: nonlinearly separable (sigmoid + tanh)
│
├── regression/
│   ├── run_univariate.py              ← Dataset 3: 1-D regression
│   └── run_bivariate.py               ← Dataset 4: 2-D regression
│
├── data/
│   └── Group11/
│       ├── Classification/
│       │   ├── LS_Group11/            ← Class1.txt, Class2.txt, Class3.txt
│       │   └── NLS_Group11.txt
│       └── Regression/
│           ├── UnivariateData/11.csv
│           └── BivariateData/11.csv
│
└── outputs/                           ← auto-generated plots (created on first run)
    ├── ls/
    ├── nls/
    ├── regression/univariate/
    └── datasets/                      ← raw dataset visualisations
```

## Setup


### 1. Ensure the `data/` folder exists
The `data/Group11/` directory must be present with the raw datasets. **Do not rename, move, or edit the data files.**

### 2. Install dependencies
```bash
pip install numpy matplotlib
```

The `shared/` module (located at `DL_Assignments/shared/`) provides `train_test_split`, `get_rmse`, `get_percent_rmse`, and the plotting utilities. It is imported automatically via `sys.path`.

## Running

Run each script from the `Assignment1/` directory:

```bash
# Classification
python classification/run_ls.py       # Linearly separable
python classification/run_nls.py      # Nonlinearly separable

# Regression
python regression/run_univariate.py   # 1-D
python regression/run_bivariate.py    # 2-D
```

Each script trains the model, evaluates on the test set, prints metrics (accuracy / RMSE / %RMSE), and saves plots to `outputs/`.

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Learning rate | 0.01 | Empirically chosen |
| Epochs | 100 | Assignment spec |
| Train/Test split | 70% / 30% | `shared/utils/data.py` (stratified) |
| Weight init | N(0, 0.01²) | LeCun et al. 1998 |
| Random seed | 42 | Reproducibility |
| Optimiser | Batch gradient descent | `optimizers/gradient_descent.py` |

## Results Summary

| Task | Metric | Value |
|------|--------|-------|
| LS — Sigmoid | Accuracy | 1.000 |
| LS — Tanh | Accuracy | 1.000 |
| NLS — Sigmoid | Accuracy | 0.266 |
| NLS — Tanh | Accuracy | 0.280 |
| Univariate | Test RMSE | 1.2661 (53.52%) |
| Bivariate | Test RMSE | 3.3338 (83.63%) |

## Report

See `report_final.pdf` for the full analysis including:
- Complete raw dataset characterisation (ranges, distributions, correlations)
- Mechanism-level explanation of why each result occurs
- Code snippets with design-rationale commentary
- Academic citations for all design decisions
- Dataset visualisations (scatter plots for all 4 raw datasets)

## References

1. LeCun, Bottou, Orr, Müller (1998). *Efficient BackProp*. Neural Networks: Tricks of the Trade.
2. Hastie & Tibshirani (1998). *Classification by Pairwise Coupling*. Annals of Statistics 26(2).
3. Goodfellow, Bengio, Courville (2016). *Deep Learning*. MIT Press.
4. Haykin (2009). *Neural Networks and Learning Machines*. 3rd ed., Pearson.
