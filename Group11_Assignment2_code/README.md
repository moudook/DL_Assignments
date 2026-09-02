# Group11 Assignment 2 — CS601T Deep Learning

FCNN from scratch (NumPy only) — classification tasks.

## Quick start

```bash
# Linearly separable (3 classes, 1 hidden layer)
python classification\run_ls.py

# Non-linearly separable (3 classes, 2 hidden layers)
python classification\run_nls.py
```

Override hyperparameters via CLI flags:

```bash
python classification\run_ls.py --hidden_sizes 4 8 16 --lr 0.3 --epochs 1500
```

## Folder layout

```
Group11_Assignment2_code/
├── data/                  # Dataset files (copied from Assignment 1)
├── shared/                # loaders, metrics, plotting utilities
├── models/                # FCNN class
├── optimizers/            # Pure online SGD trainer
├── classification/        # LS and NLS runners
│   ├── run_ls.py
│   ├── run_nls.py
│   └── _common.py         # Shared helpers (metrics JSON, printing)
├── outputs/               # All generated plots + metrics JSONs
│   └── classification/
│       ├── ls/            # Per-arch (h2..h32) + best/
│       └── nls/           # Per-arch (h2..h32) + best/
└── README.md
```

## What each runner produces

For every hidden size `h` in the sweep:
- `outputs/classification/<dataset>/h{h}/error_curve.png`
- `outputs/classification/<dataset>/h{h}/cm_val.png`
- `outputs/classification/<dataset>/h{h}/metrics_val.json`

For the best architecture (lowest validation MSE):
- `outputs/classification/<dataset>/best/decision_regions.png`
- `outputs/classification/<dataset>/best/cm_test.png`
- `outputs/classification/<dataset>/best/metrics_test.json`
- `outputs/classification/<dataset>/best/surface_hidden_l{1,2}_n*.png`
- `outputs/classification/<dataset>/best/surface_output_l3_n*.png`

## Notes

- Activations: sigmoid on every layer (including output). Targets are one-hot.
- Optimizer: pure online SGD — one sample per weight update.
- Best architecture is selected by lowest validation MSE.
- Data split: 60% train / 20% val / 20% test, stratified per class, seed 42.
- No TensorFlow / PyTorch / sklearn used.