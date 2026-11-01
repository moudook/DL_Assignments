# How to run (Assignment 5)

Works the same on Windows and Linux. Run every command from this folder (`Group11_Assignment5_code/`). Everything runs on CPU.

## 1. Setup

Use Python 3.12, then install the dependencies:

```
pip install -r requirements.txt
```

You need two dataset folders, each containing `train/ val/ test/`:

| Dataset | Classes |
|---|---|
| MNIST | 0, 4, 5, 6, 7 |
| Caltech | bonsai, buddha, car_side, scorpion, starfish |

Either put them at `data/mnist` and `data/caltech`, or point to them with environment variables:

```
# Linux
export A5_MNIST_DIR=/path/to/mnist
export A5_CALTECH_DIR=/path/to/caltech

# Windows (PowerShell)
$env:A5_MNIST_DIR = "C:\path\to\mnist"
$env:A5_CALTECH_DIR = "C:\path\to\caltech"
```

## 2. Run

```
python run.py smoke          # correctness checks: expect 19/19 passed
python run.py all --quick    # ~1 min pipeline test: expect "manifest complete: 47 artefacts"
python run.py all            # full run, ~5-10 min
```

The first run downloads the VGG19 ImageNet weights (~550 MB).

Results go to `out/`:

- `out/summary.json`: the headline numbers
- `out/{A,B1,B2}/results.json`: the full results for each part
- `out/{A,B1,B2}/*.png`: all figures

You can also run one part at a time with `python run.py partA`, `partB1` or `partB2`. Part B2 needs Part B1's results.

## 3. Report figures

```
python export_figures.py
```

This copies the 35 figures that `report.tex` uses into `../report/figures/`. Then upload the whole `report/` folder to Overleaf.

## Expected numbers

The run is deterministic, so you should get exactly these numbers. If yours differ, the hardcoded numbers in `report.tex` need updating.

| Part | Learning rate | Accuracy |
|---|---|---|
| A3 | 0.03 | test 99.42% |
| B1 Arch-1 | 0.0003 | val 74% |
| B1 Arch-2 | 0.0003 | val 78%, test 68% (best architecture) |
| B1 Arch-3 | 0.0003 | val 74% |
| B2 VGG19 | 0.03 | 100% on train, val and test |

The method and design choices are explained in `README.md`.
