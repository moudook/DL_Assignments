# Group 11 - Assignment 5 (CNNs)

Part A: CNN building blocks on the MNIST subset (digits 0, 4, 5, 6, 7).
Part B: three CNNs from scratch and VGG19 transfer learning on the Caltech-101
subset (bonsai, buddha, car_side, scorpion, starfish).

## Setup

Python 3.12; same versions on Linux and Windows:

```
pip install -r requirements.txt
```

Everything runs on CPU (about 10 minutes end to end on an 8-core laptop). The
first VGG19 run downloads the ImageNet weights (~550 MB) to `~/.keras/models`.

Data is located automatically by its class folders. If it lives elsewhere, set
`A5_MNIST_DIR` / `A5_CALTECH_DIR` to the folder containing `train/ val/ test/`.

## Commands

```
python run.py smoke            correctness checks (19 + 8 TensorFlow checks)
python run.py all --quick      whole pipeline on tiny subsets (~1 min, numbers meaningless)
python run.py all              the real run -> out/
python run.py partA | partB1 | partB2   one part
python run.py check            manifest check, summary.json, contact sheets
python export_figures.py       copy the figures report.tex uses into ../report/figures/
```

Each part runs in its own process; TensorFlow (Part B Q2 only) and PyTorch
never share one.

## Outputs

```
out/summary.json          headline numbers of every part
out/A/results.json        A1-A3: filter values, dimension checks, lr grid, metrics, traced neurons
out/B1/results.json       B1a-d: architecture size tables, lr grids, metrics, best arch, neurons
out/B2/results.json       B2a-d: VGG19 metrics, comparison with B1b, neurons, Grad-CAM probabilities
out/*/*.png               every figure (file names start with the question, e.g. A3c_, B2d_)
out/contact_sheet_*.pdf   all figures of a part on one page, for review
```

## Methodology (fixed before any training)

- **Inputs.** MNIST: pixels / 255. Caltech: RGB (greyscale replicated), resized to
  224x224 (aspect not preserved), per-channel mean/std from the train split. VGG19
  gets the same resized pixels with `vgg19.preprocess_input`.
- **Models.** Exactly as specified; ReLU after every conv; conv kernels Kaiming normal,
  FC layers Xavier normal, biases 0. Softmax output, cross-entropy loss.
- **Optimiser.** Mini-batch SGD, momentum 0.9, no weight decay, no schedule. Batch 32
  (MNIST), 16 (every Caltech model).
- **Stopping.** Early stopping on validation loss, patience 15, best epoch's weights
  restored. Same rule for every model.
- **Learning rate.** Every model runs the same grid {0.001, 0.003, 0.01, 0.03, 0.1}
  and keeps the rate with the lowest early-stopped validation loss; one extra step is
  added only if the winner is at a grid edge. Fairness means an identical procedure,
  not an identical value. Every run is reported.
- **Selections.** Images: first train file of each class (B1c/B2c: first bonsai image).
  Best architecture (B1b): highest val accuracy, then lowest val loss, then fewer
  parameters. Test data is used only after all choices are made.
- **VGG19.** Only the final `predictions` layer is replaced (Dense(5, softmax)) and
  trained; everything else frozen. It is trained on cached `fc2` features (identical
  to training through the frozen network, checked), using the same protocol as B1.
- **"Last conv layer"** means its output after ReLU, before pooling. Neurons are
  traced back through every conv/pool window to the input box they can see.
- **A3b.** The sheet asks for 10 layer-1 filters but layer 1 has 6, so all 6 are shown.

## Files

```
run.py            entry point
config.py         paths, seeds, hyperparameters, fixed selections
data.py           framework-free loading and caching (shared by torch and TF code)
conv_manual.py    A1 explicit convolution + output-size formula
layers.py         A2 ConvReLUPool class and all torch models
receptive.py      neuron -> input box tracing
protocol.py       early stopping + lr grid (shared by torch and Keras training)
train.py          torch training loop
evaluate.py       accuracy, confusion matrix, per-class recall
plots.py          figures + automatic layout check on every figure
results.py        JSON output
part_a.py, part_b1.py, part_b2_vgg.py   the three parts
smoke_test.py, smoke_test_tf.py         correctness checks
```
