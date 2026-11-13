# Group 11 — Assignment 6: RNN and LSTM sequence classification

Classifies two sequence datasets with an RNN and an LSTM (PyTorch), tunes the number of hidden layers
at each time step, the number of recurrent (RNN/LSTM) layers and the neurons per layer, and trains each
network until the average training error changes by less than 1e-4 between successive epochs.

## Files

| File | Contents |
|---|---|
| `run.py` | entry point (all commands below) |
| `config.py` | every setting, the hyperparameter grid, dataset descriptions, class mapping |
| `data.py` | file readers, normalisation, train/val split, batching, dataset statistics |
| `models.py` | `SeqClassifier`: hidden layers at each time step, stacked RNN/LSTM layers, prediction from the last time step |
| `train.py` | training until convergence, evaluation, confusion matrix, checkpoints |
| `experiments.py` | hyperparameter sweep, selection of the best model, tables, figures |
| `plots.py` | all figures |
| `smoke_test.py` | correctness checks |

## Setup (Linux and Windows)

Python 3.10 or newer.

```
python -m venv .venv
# Linux:   source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For a GPU on Windows, install the CUDA build of torch from https://pytorch.org instead of the plain `torch` line.

## Data layout

The code expects a `data` folder **next to** this code folder:

```
Assignment6/
  Group11_Assignment6_code/   (this folder)
  data/
    hw/Mapping_Handwritten.txt
    hw/a/train/*.txt   hw/a/dev/*.txt   ... one folder per class
    cv/Mapping_CV.txt
    cv/ba/Train/*.mfcc cv/ba/Test/*.mfcc ... one folder per class
```

Extracting the class zips with Python works the same on every OS (avoids Windows "Extract All" creating an extra folder level):

```
python -m zipfile -e path/to/a.zip ../data/hw
python -m zipfile -e path/to/ba.zip ../data/cv
```

Copy `Mapping_Handwritten.txt` into `data/hw` and `Mapping_CV.txt` into `data/cv`.
Another location can be given with `--data <folder>`. Outputs go to `../out` (or `--out <folder>`).

## Commands

```
python run.py smoke                              # correctness checks, run first
python run.py inspect                            # dataset statistics, length histograms, handwriting gallery
python run.py pilot --threads 1                  # epoch timing + one network per cell trained to convergence
python run.py sweep --threads 1                  # every hyperparameter set, both datasets, RNN and LSTM
python run.py select                             # summary tables and best model per (dataset, cell)
python run.py figures                            # all report figures
```

Useful options: `--dataset hw|cv`, `--cell rnn|lstm`, `--configs 0 1 2` (indices into `config.GRID`),
`--seeds 11 12 13`, `--device cpu|cuda`, `--force` (re-run finished configs),
`--cv-norm zscore_train|none`, `--hw-norm minmax_axis|minmax_joint|none`, `--lr 1e-4`.

Each entry of `config.GRID` is `(dense_sizes, hidden_sizes)`: the neurons of each hidden layer applied at
every time step before the recurrent layers (`[]` = none), and the neurons of each recurrent layer.

Other variants that were run:

```
python run.py sweep --threads 1 --dataset cv --cv-norm none      # CV with raw MFCCs (no standardisation)
python run.py sweep --threads 1 --seeds 11 12 13                 # extra seeds (seed 11 is the default)
python run.py select --seeds 11 12 13                            # tables over the extra seeds (separate files)
```

`select` and `figures` only use the seeds in `config.SEEDS` (default `[11]`). `select --seeds 11 12 13`
writes `summary_seeds11-12-13.csv/.tex` and `best_seeds11-12-13.json` next to the seed-11 files and never
overwrites them; do not run `figures` with several seeds.

`--threads 1` is fastest on CPU for these small networks. Several sweeps can run at the same time,
e.g. one terminal per (dataset, cell):

```
python run.py sweep --threads 1 --dataset hw --cell rnn
python run.py sweep --threads 1 --dataset hw --cell lstm
python run.py sweep --threads 1 --dataset cv --cell rnn
python run.py sweep --threads 1 --dataset cv --cell lstm
```

Finished configs are skipped, so an interrupted sweep can simply be restarted. Results from two machines
can be combined by copying their `out/<dataset>_<norm>_lr<lr>/<cell>/<run>` folders into one `out` folder
before running `select` and `figures`. The device used is recorded in each `metrics.json`; GPU and CPU
runs of the same seed are not bit-identical, so do not mix them in one table without saying so.

## Outputs

```
out/hw_minmax_axis_lr0.001/            (folder name includes the normalisation scheme and learning rate)
  split.json                           which train files were held out for validation
  data_stats.json                      class counts and sequence lengths
  rnn/h64-64_s11/model.pt              trained network + preprocessing statistics
                                       (a network with hidden layers at each time step: rnn/d32_h64_s11/)
  rnn/h64-64_s11/history.json          average training error per epoch
  rnn/h64-64_s11/metrics.json          train/val/test accuracy, confusion matrices, settings used
  rnn/summary.csv, rnn/summary.tex     table of every hyperparameter set (best marked)
  best_summary.csv                     best RNN vs best LSTM
  figures/                             PNG + PDF of every figure
out/cv_zscore_train_lr0.001/           same for the CV dataset (standardised MFCCs)
out/cv_none_lr0.001/                   same for the CV dataset (raw MFCCs)
out/pilot/                             pilot timing and learning-rate runs (lr 1e-3, 3e-4, 1e-4)
out/logs/                              console output of the sweeps
```
