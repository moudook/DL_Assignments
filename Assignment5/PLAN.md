# Assignment 5 — Implementation Plan (Group 11)

Deadline: **Saturday 31 Oct 2026, 22:00** (the sheet says "Sunday"; 31 Oct 2026 is a Saturday — plan for the earlier reading).
Sources: `CS601T_Assignment5_Details_12October2026.pdf` (the sheet) and `Class22-26-ConvolutionalNeuralNetworks-06-12Oct2026.pdf` (the notes). Where the sheet is silent we follow the notes; where both are silent we pick one rule, apply it everywhere, and state it in the report.

---

## 0. Facts measured before planning

| Item | Value |
|---|---|
| MNIST (Group 11) | `Assignment4/Data/Group_11`, classes 0 4 5 6 7, 2277 / 759 / 759 per class (train/val/test), JPEG 28×28 |
| Caltech (Group 11) | `Assignment5/Group_11`, classes bonsai buddha car_side scorpion starfish, 50 / 10 / 20 per class → 250 / 50 / 100, sizes 124–300 px |
| Greyscale images | **car_side 80/80**, buddha 4/80, starfish 3/80, others 0 → colour alone identifies car_side (see §6.4) |
| venv | `/home/anshuman139/Deep-Learning/.venv`, Python 3.12.3, torch 2.14 (+cu130 build, no GPU on this laptop), matplotlib 3.11, numpy 2.5, sklearn 1.9, PIL 12.3. **No TensorFlow/Keras, no pytest.** |
| CPU timing, batch-1 SGD | Part A net: 0.68 ms/step → **7.8 s/epoch** with `torch.set_num_threads(1)` (72 s/epoch with 8 threads — 10× slower). Part B Arch-1: 0.5 s/epoch. |

Consequence of the timing: **Part A and Part B-Q1 run comfortably on CPU on either machine**; batch-1 training gets no benefit from a GPU (kernel-launch bound). The GPU only matters for VGG19 feature extraction, and native-Windows TensorFlow has had **no GPU support since TF 2.10** (GPU needs WSL2). So the plan is designed so the VGG part is CPU-feasible too (§5.1 feature caching).

---

## 1. Environment (step 0)

1. Install into the venv: `tensorflow` (CPU build is fine; brings Keras 3) and `pytest` is **not** needed — smoke tests are a plain script so nobody has to install anything extra.
   - Verify `import tensorflow, torch` both work **and** numpy is not downgraded/broken by TF's pin. If TF's numpy pin conflicts with numpy 2.5, create a second venv for Part B-Q2 only (the code isolates TF in its own process anyway, §2).
2. `requirements.txt` with exact versions from this venv; Windows teammate installs the same versions (torch CPU wheel is enough; CUDA wheel optional).
3. First VGG19 run downloads ImageNet weights (~550 MB with top) to `~/.keras/models` — do it once on each machine before the timed run.

---

## 2. Code layout and OS-agnostic rules

```
Group11_Assignment5_code/
  run.py              single entry point: smoke | partA | partB1 | partB2 | all | check  (+ --quick, --outdir)
  config.py           seeds, hyperparameters, paths, image-selection rules — one source of truth
  data.py             framework-free loaders (PIL + numpy); caches .npz; used by BOTH torch and TF code
  conv_manual.py      A1: explicit-loop convolution + output-size formula
  layers.py           A2 ConvReLUPool class; Part A CNN; Caltech Arch-1/2/3 (torch)
  receptive.py        generic receptive-field tracer (layer list of (k, s, p)) → input box per neuron
  protocol.py         early stopping + lr-grid rule, framework-free (shared by torch and Keras training)
  train.py            torch mini-batch SGD loop using protocol.py, best-weight restore
  evaluate.py         accuracy, confusion matrices, per-class metrics
  plots.py            every figure goes through here: style, save, lint
  results.py          JSON output (numpy/NaN-safe)
  part_a.py           A1, A2, A3a-c          -> out/A/results.json
  part_b1.py          B1a-d                  -> out/B1/results.json
  part_b2_vgg.py      B2a-d — the ONLY module that imports TensorFlow -> out/B2/results.json
  smoke_test.py       correctness checks (§7), no pytest dependency
  smoke_test_tf.py    TensorFlow checks, launched by smoke_test.py in its own process
  README.md
```

OS-agnostic rules (each one is a bug we have either hit or would hit on Windows):

- **Paths:** `pathlib.Path` everywhere, resolved relative to `Path(__file__).parent`; never string-concatenate with `/` or `\`. Data dirs found by a resolver (as in A4's `resolve_data_dir`) with a `--data-*` override.
- **No Makefile / shell scripts** — the teammate has no `make`. `run.py` subcommands replace them.
- **`if __name__ == "__main__":`** guard on every entry point (Windows spawns, not forks).
- **`DataLoader(num_workers=0)`** — A4 hit Windows error 1455 (shared file mapping) with workers > 0. The datasets are tiny; everything is preloaded into memory as tensors anyway.
- **Text I/O:** `open(..., encoding="utf-8")` for every file; log/print ASCII-only (Windows console cp1252 crashes on `→`, `×`, `✓`).
- **TF and torch never share a process.** `run.py all` launches each part with `subprocess.run([sys.executable, ...])`. Avoids Windows DLL clashes and keeps Part A/B1 runnable without TF installed.
- **matplotlib backend `Agg`**, set before `pyplot` import (no display on servers, identical output on both OSes).
- **Threads:** `torch.set_num_threads(1)` was 10× faster for batch-1 steps; re-measure at batch 32/16 and set per part. Device default = CPU for torch parts; `--device cuda` available but not needed.
- **Determinism:** seed `random`, `numpy`, `torch`, `tf` from `config.SEED`; shuffling uses an explicit `torch.Generator`. Final numbers come from **one machine** (record OS, versions, device in `run_config.json`); CPU runs are the most reproducible across machines.
- **Short output paths** (Windows 260-char limit): `out/A3/a_cm_test.png`, not deep descriptive trees.
- **Filenames:** lowercase, no spaces, unique ignoring case (Windows is case-insensitive).
- **Checkpoints/caches** are saved so figures can be regenerated without retraining (`run.py partA --plots-only`).

---

## 3. Global methodology (fairness rules — fixed before any training)

| Decision | Rule | Why |
|---|---|---|
| Input scaling, MNIST | pixels / 255 → [0, 1], no mean subtraction | Same as A3/A4; keeps feature maps interpretable. Same for A1, A2, A3. |
| Input, Caltech (B1) | RGB, greyscale replicated to 3 channels, resize to 224×224 (bilinear, antialias, aspect **not** preserved — sheet says "resize"), /255, then per-channel mean/std from **train only** | Applied identically to all three architectures. |
| Input, VGG19 (B2) | the **same** resized uint8 224×224 array from `data.py`, then `keras.applications.vgg19.preprocess_input` exactly once | Same pixels as B1, so the B2a-vs-B1b comparison differs only in the model. |
| Conv activation | ReLU after every conv (sheet A2 + notes "ReLU on output of convolution") | Part B silent → notes. |
| Init | Conv: Kaiming normal, fan_in, ReLU gain (std = √(2/fan_in)). FC: Xavier normal. Biases 0. | Sheet for Part A; notes ("each kernel initialised as per Kaiming") + Part A for Part B. |
| Output | 5-way softmax; loss = cross-entropy. Implemented as logits + `CrossEntropyLoss` (softmax inside), probabilities via softmax for reporting. | Mathematically identical; state it. |
| Optimizer | **mini-batch SGD, momentum 0.9**, no weight decay, no lr schedule (early stopping ends training) | Neither sheet nor notes fix it. The notes' one-image backprop walkthrough explains the mechanics; a mini-batch averages those per-image gradients — same algorithm. Momentum SGD is the recipe of the AlexNet/VGG papers the notes' case studies cite. |
| Batch size | **32 for MNIST, 16 for every Caltech model** (B1 archs and VGG19 head) | Standard CNN range. Caltech has 250 images: 32 would give 8 updates/epoch, 16 gives ~16, keeping 15 epochs of patience meaningful. The two datasets are never compared, so different sizes there are harmless; within Part B all models are identical. |
| Learning rate | **same grid for every model** {0.001, 0.003, 0.01, 0.03, 0.1}; each model (A3, Arch-1/2/3, VGG19 head) keeps the rate with the **lowest early-stopped val loss**. Pre-set edge rule: if the winner is at a grid end, add one step that way (0.0003 or 0.3), once, for that model only; nothing else is added after seeing results. Criterion never switched to val accuracy afterwards. | Fairness = identical tuning procedure and budget, not identical values: the best rate depends on the model, and a shared rate favours whichever model it suits (worst for the VGG head, a linear layer on fixed features vs a CNN trained from scratch). Second opinion obtained and agrees. |
| Grid reporting | report **every** run: model × lr → best val loss, val acc and train acc at the restored epoch, best epoch, stop epoch, instability flag (loss spikes / NaN / no learning) | Makes the selection auditable; shows whether the Arch ranking also holds at a common lr. Does not remove selection bias — say so. |
| Validation reuse (report caveat) | the 50 Caltech val images pick the lr, stop each run, and pick the architecture → val numbers are optimistic. Test set is used once, after all choices (B1b, B2a), so test numbers are not. | Stated, not hidden. |
| Stopping | **early stopping on validation loss**: evaluate mean val cross-entropy after every epoch; stop when it has not improved (strictly lower than the best so far) for **15 consecutive epochs**. Safety cap on epochs (logged; if hit, say so in the report). Same rule and patience for every model, incl. the VGG19 head. | Neither the sheet nor the notes specify a criterion; this is the standard choice for CNNs. Training loss is not used: an over-parameterised CNN drives it to ~0 (memorisation), which says nothing about unseen data. Val **loss**, not accuracy: continuous, whereas Caltech val accuracy moves in 2 % steps. Patience 15 outlasts SGD noise (worst excursion seen in the pilot: 4 epochs). |
| Weights evaluated | **best-val-loss epoch's weights restored** before every evaluation | Part of early stopping. Caveat for the report: the 50 Caltech val images then both stop each model and pick the architecture, so the winner's val accuracy is slightly optimistic. The **test set stays untouched** until B1b/B2a and is used once, so reported test numbers are unbiased. |
| Shared initial weights | A3's conv layers start from **exactly the A2 weights** (same seed, asserted bitwise). Arch-1/2/3 share **identical initial conv1/conv2 weights** and **identical shuffle order**. | A3b compares trained vs untrained filters at the same index → must be the same filters. B1 compares architectures → remove init/order luck from the identical layers (A3 sheet convention "same initial random values"). |
| Augmentation / dropout / BN | none | Not in the sheet; any of them changes the specified architectures. |
| Reported train accuracy | recomputed in eval mode after training (not the running epoch average) | |
| Seeds | **one seed** (`config.SEEDS = [11]`). The code loops over `config.SEEDS`, so adding seeds needs no other change: every trained model (A3, Arch-1/2/3, VGG19 head) then runs per seed, accuracies become mean ± std with a per-seed table, and selection uses the mean. A seed sets initial weights and shuffle order (VGG head: head init and order; cached features are fixed). | Single run, as is usual for this course. Report caveat: with 50 val images the B1b choice can come down to one image (2 %). |
| Designated seed | `SEEDS[0]`, fixed in advance, **never the best seed**. It supplies all figures (filters, feature maps, trace-back, guided backprop, Grad-CAM) and the confusion matrices (labelled "seed 11"), and its A2 init is what A3b compares against. | No cherry-picking; figures stay one coherent model. |
| Architecture selection tie-break (B1b) | val accuracy (mean over seeds if several) → val loss → fewer parameters; rule written in config before training | 50 val images: one image = 2 %, ties are likely. |
| "Last conv layer" for trace-back | conv output **after ReLU, before pooling** | Notes: the traced neuron sits on the conv layer (slide 112); VGG neuron positions come from the 14×14 map (slide 141). |
| Neuron numbering | (feature map, row, col), **1-based in figures/report**, 0-based in code, stated in captions | Notes' format: "(7, 2) pixel in the 180th feature map". |
| Image selection | rule-based and fixed in `config.py` before training: first file (sorted name) of each class in the train split. B1c image = first file of the first **colour** class (bonsai) so the colour maps/guided backprop are meaningful. Filenames written to `selection.json` and the report. | No cherry-picking after seeing results; every downstream question reuses the same images. |

---

## 4. Part A — MNIST

Expected sizes (asserted in code, tabulated in the report): 28×28 → conv5 **6@24×24** → pool **6@12×12** → conv5 **16@8×8** → pool **16@4×4** → 256 → 64 → 5. Parameters: 156 + 2,416 + 16,448 + 325 = **19,345**.

### A1 — manual convolution
- 5 images (selection rule), one 5×5 filter, Kaiming (fan_in 25, std √(2/25)), seeded, **same filter for all 5 images**.
- `conv_manual.py`: explicit double loop over output positions, stride 1, padding 0, cross-correlation (no flip — the notes' image formula I(i+a, j+b)·K(a, b), same as every framework). No activation (sheet asks only for the convolution).
- Size check: formula ⌊(W − F + 2P)/S⌋ + 1 = 24 printed next to the measured shape.
- Outputs:
  - `A1_filter.png` — 5×5 filter as image **with the numeric values printed in each cell** (like slides 107–110), diverging colormap centred at 0.
  - `A1_feature_maps.png` — 5 rows (one per class): image | feature map (diverging, symmetric colour limits, colorbar).
  - `A1_filter_values.csv` and the table of expected vs obtained dimensions (also printed to log).

### A2 — `ConvReLUPool` class, two stacked layers (untrained)
- Class with explicit weight/bias parameters, Kaiming init, forward = conv → ReLU → maxpool(2, 2). Two instances: (1→6, 5×5) and (6→16, 5×5).
- Pass the 5 A1 images; **print** each layer's output shape and the formula for conv and pool stages (24, 12, 8, 4); assert equal.
- Choose **3 filter indices per layer**, fixed in `config.py` (recorded — A3b reuses them).
- Outputs:
  - `A2_filters_l1.png` — 3 conv1 filters (5×5, values annotated).
  - `A2_filters_l2.png` — 3 conv2 filters; each is 6×5×5 → drawn as **6 per-channel 5×5 slices** (the notes draw VGG's 3×3×3 filters per channel), shared symmetric colour scale.
  - `A2_maps_l1.png`, `A2_maps_l2.png` — for each of the 5 images and each chosen filter: post-ReLU conv map (24×24 / 8×8) and pooled map (12×12 / 4×4), labelled with class and filter index.
  - Dimension table.

### A3 — full CNN, trained
- Built **from the same `ConvReLUPool` class** and same seed → initial conv weights identical to A2 (asserted). FC layers Xavier.
- Train: momentum SGD, batch 32, lr grid + early stopping on val loss (§3). Log per-epoch train/val loss + accuracy; mark the restored best epoch on the curves.
- **A3a:** `A3_curve.png` (loss vs epoch, accuracy vs epoch); train/val/test accuracy table; **3 confusion matrices** (`A3_cm_train/val/test.png`), counts + row-%.
- **A3b:** the sheet says "10 filters from the first conv layer", but conv1 has 6. Plan (state in report): **all 6 conv1 filters** and **10 conv2 filters, including the 3 indices from A2**. Figures:
  - trained filters (conv1 5×5; conv2 per-channel slices),
  - their feature maps for the same 5 images,
  - **before/after panel** for the A2 indices (A2 init vs trained) — this is the comparison the "same position/location index" wording is asking for.
- **A3c:** trace-back exactly as the notes (slides 111–115): neuron on conv2 output (8×8, post-ReLU, pre-pool) → its 5×5 window on pool1 → 2×2 pool windows on conv1 → 5×5 conv1 windows on the input. For neuron (i, j): rows/cols **2i … 2i+13** → **14×14 box**, whole pooling windows (not just the argmax path).
  - Neuron choice: **top-5 distinct feature maps** by peak activation, each at its argmax position (avoids five near-identical boxes from one map at adjacent positions; matches how slides 116–123 count neurons). Report (map, row, col) and activation.
  - Figures: `A3c_boxes.png` — 5×5 grid (row = image, column = neuron, one box per panel, title = map/row/col/activation); `A3c_overlay.png` — each image with all 5 boxes in distinct colours + legend. Image upsampled with nearest-neighbour so the 28×28 pixels and box edges line up exactly (box drawn on pixel boundaries: `Rectangle((c0-0.5, r0-0.5), 14, 14)`).

---

## 5. Part B — Caltech-101 (5 classes, 224×224×3)

### B1 — three custom CNNs (torch)

Expected sizes (floor in the formula — 224 does not divide evenly for 11×11/s4, and 3×3/s2 pools on 54 and 22 give 25.5 and 9.5; state the floor convention since the notes' formula is printed without it):

| | Arch-1 | Arch-2 | Arch-3 |
|---|---|---|---|
| conv1 8@11×11 s4 | 8@54×54 | = | = |
| pool 3/2 | 8@26×26 | = | = |
| conv2 16@5×5 | 16@22×22 | = | = |
| pool 3/2 | 16@10×10 | = | = |
| conv3 32@3×3 | — | 32@8×8 → pool 32@3×3 | 32@8×8 |
| conv4 64@3×3 | — | — | 64@6×6 → pool 64@2×2 |
| flatten → 128 → 5 | 1600 | 288 | 256 |
| params | 211,701 | 48,405 | 62,805 |
| RF of last-conv neuron | 51×51 (jump 8) | 99×99 (jump 16) | 131×131 (jump 16) |

All three trained under §3 rules (same lr, same init of shared layers, same order, same stopping rule).

- **B1a:** for each arch: training curves, train + val accuracy, **train and val confusion matrices** (6 total). **No test numbers here.** Architecture table above generated from forward hooks (shapes + parameter counts) and asserted against the formula, so the report table cannot drift from the code.
- **B1b:** best by val accuracy (tie-break rule §3) → test accuracy + test confusion matrix. Note the 2 %-per-image granularity of val.
- **B1c:** one training image (selection rule). All **8 conv1 maps**; for each later conv layer, **8 maps selected by a stated rule** (top-8 by mean activation). Post-ReLU, pre-pool. One figure per layer, 2×4 grid, per-map colour scale + note, original image shown alongside. Observations: edges/colour in conv1 → coarser, sparser, more localised later.
- **B1d:** one training image per class (5). For each: the single most activated neuron in the best arch's last conv layer (post-ReLU, pre-pool) → receptive-field box via `receptive.py` (clipped to the image) → figure: image with box + the cropped patch beside it, label (map, row, col).

### B2 — VGG19 pretrained on ImageNet (TensorFlow Keras)

#### 5.1 Model and training
- `keras.applications.VGG19(weights="imagenet", include_top=True)`; replace **only** the final 1000-way `predictions` layer with `Dense(5, softmax)` (Xavier/Glorot init — Keras default); freeze every other layer. This is the strict reading and what the notes say for a small dataset ("retrain only the softmax layer"). Trainable params: 4096·5 + 5 = **20,485** (asserted).
- Speed: because everything below `predictions` is frozen, deterministic, and the Keras VGG19 top has **no dropout** (asserted), we compute the 4096-d `fc2` outputs **once** for all 400 images and train the 5-way softmax on those. This is mathematically the same training as running the frozen network every step, and makes CPU-only Windows/Linux fine. The trained weights are then loaded into the full Keras model, and a smoke test asserts the full model and cached-feature head give identical predictions.
- Same protocol as B1 (momentum SGD, batch 16, same lr grid and selection rule, same stopping rule). In the report the B2a-vs-B1b comparison is framed as **fixed ImageNet-pretrained features vs features learned from scratch on 250 images**, not as a pure architecture comparison.

#### 5.2 Questions
- **B2a:** train/val/test accuracy, **3 confusion matrices**; comparison table vs B1b on the **same 100 test images** (accuracy, per-class recall, trainable / total params).
- **B2b:** same 5 images as B1d. Last conv layer = `block5_conv4` (14×14×512, ReLU built in, pre-pool). Argmax neuron → RF box with the same `receptive.py` (VGG: 3×3/s1/p1 convs, 2×2/s2 pools). Theoretical RF = **252 px > 224** → box clipped to the image covers (almost) all of it; the report must say so. Same figure format as B1d for side-by-side comparison.
- **B2c:** guided backpropagation on the **B1c image**, 5 neurons in `block5_conv4` with **non-zero activation** (notes, slide 139): top-5 distinct maps at their argmax. Implementation per notes: keep the target neuron, zero all others in that layer, backprop to the input; every ReLU's backward passes gradient only where forward input > 0 **and** upstream gradient > 0 (`tf.custom_gradient`, swapped into each conv layer's `activation`). Uses ImageNet conv weights — identical to the fine-tuned model since only the head was trained (asserted). Report: image, the 5 positions as "(row, col) in the k-th feature map", the 5 gradient images (standard normalisation: zero-mean, unit-std, ×0.1, +0.5, clip [0, 1]), and observations.
- **B2d:** Grad-CAM following keras.io: same 5 images as B1d, `block5_conv4`, the **fine-tuned 5-class model** (a heat map per *our* classes), class score = **pre-softmax logit** (keras.io sets the last layer's activation to None — done on a **clone**, the trained model is not mutated). 5 images × 5 classes = **25 heat maps**: `B2d_gradcam.png` as a 5×5 grid (rows = image/true class, cols = target class, diagonal = true class outlined, title = p(target)). Per-map normalisation as keras.io, but the raw max is printed in each panel so magnitude is not hidden; an all-zero map (ReLU kills it when there is no positive evidence) is shown blank and labelled, not divided by zero.

---

## 6. Plotting standards ("clear output, no formatting issues")

1. One `plots.py`; every figure is created by a helper and saved by `save_fig()`. Global rcParams set once: DejaVu Sans, font sizes 9–11, `constrained_layout`, dpi 200 PNG **and** PDF (vector for the report).
2. Colormaps by meaning, consistent everywhere: input images `gray` (MNIST) / RGB (Caltech); **signed** quantities (filters, A1 un-activated maps, gradients) `RdBu_r` with symmetric limits centred at 0; **non-negative** (ReLU maps) `viridis` from 0; Grad-CAM `jet` overlay α = 0.4 (keras.io convention). Every heat-type panel has a colorbar or a stated shared scale.
3. Pixel-art images use `interpolation="nearest"`; ticks off on image grids; every panel titled (class name, filter/map index 1-based).
4. Confusion matrices: fixed class order everywhere (alphabetical = ImageFolder order), labels on both axes ("True"/"Predicted"), counts + row-%, text colour switches with cell darkness, x labels rotated 30° (car_side / starfish fit), accuracy in the title.
5. Box overlays: drawn on pixel edges, linewidth scaled to image size, distinct high-contrast colours + legend.
6. **Figure lint in `save_fig()`** (fails the smoke test, not just a warning): matplotlib warnings promoted to errors (missing glyphs, layout failures); after drawing, every text artist's bounding box must lie inside the canvas; titles, axis labels and tick labels must not overlap each other; figure is non-blank; size within limits.
7. `contact_sheet.pdf`: every figure thumbnailed with its filename — one-page visual review before it goes into the report.

### 6.4 Greyscale car_side caveat
Required: say in B1/B2 observations that car_side is the only greyscale class, so its accuracy can come from colour alone. Optional extra **after** all required work: retrain the best arch on greyscale-converted copies of all images, compare car_side recall. Shown as a clearly separate ablation, never mixed into the required tables.

---

## 7. Smoke tests — `python run.py smoke` (must pass on Linux AND Windows before any full run)

**Data**
1. Counts: MNIST 2277/759/759 per class; Caltech 50/10/20 per class; class order identical across splits and between the torch and TF loaders.
2. Caltech arrays all `uint8 (224, 224, 3)`; greyscale sources have three equal channels.
3. **No leakage:** no identical file content (hash) shared between train/val/test, for both datasets.
4. Selected images exist, are in the train split, one per class; `selection.json` written.

**Convolution and sizes**
5. `conv_manual` output == `F.conv2d` (atol 1e-6) on random inputs and on the 5 real images; shape == formula (24×24).
6. Output-size formula == actual tensor shape at **every** layer of Part A, Arch-1/2/3 and VGG19 (incl. floor cases 54, 26, 22, 10, 8, 3, 6, 2).
7. Parameter counts == 19,345 / 211,701 / 48,405 / 62,805; VGG trainable == 20,485.

**Initialisation**
8. Kaiming: empirical std of conv weights ≈ √(2/fan_in) (within sampling tolerance, pooled over a large draw); Xavier variance ≈ 2/(fan_in+fan_out); biases zero.
9. A3 initial conv weights == A2 weights (bitwise). Arch-1/2/3 initial conv1/conv2 identical; first-epoch shuffle order identical.

**Training mechanics**
10. Initial loss ≈ ln 5 = 1.609 (softmax/CE wired correctly).
11. **Overfit test:** 20 training images → 100 % train accuracy within a small epoch budget for Part A, each Arch, and the VGG head (proves backprop reaches every layer; also checks every parameter's gradient is non-zero after one step).
12. Early stopping on synthetic val-loss traces: stops exactly `patience` epochs after the minimum, restores that epoch's weights (checked by comparing parameters), never stops while val loss is still improving, and honours the epoch cap.

**Trace-back**
13. Notes' toy example (slide 124): 4×4 input, 2×2 filter, stride 2 → h4 ↦ pixels {11, 12, 15, 16}.
14. **Gradient-support check:** with ReLUs removed and max-pool replaced by avg-pool of the same size/stride, the non-zero support of ∂neuron/∂input equals the computed box exactly, for random neurons in Part A, Arch-1/2/3 (and VGG, cropped). Part A box size == 14×14.

**VGG / visualisation**
15. Head swap: output shape 5; all non-head weights == ImageNet weights after training; no Dropout layer in the model; `preprocess_input` applied exactly once (input statistics check).
16. Cached-feature head predictions == full Keras model predictions.
17. Guided ReLU unit test: backward zeroes (a) positions with negative forward input and (b) negative upstream gradients; guided gradient has shape 224×224×3, is non-zero, differs from the plain gradient.
18. Grad-CAM: map 14×14, non-negative; model predictions unchanged after running it (no mutation); zero-map case handled.

**Pipeline**
19. `run.py all --quick` (tiny subsets, 1–2 epochs) produces **every artefact in the manifest** (§9); the manifest check lists anything missing.
20. Every figure passes the lint (§6.6).

---

## 8. Submission

Zip is built by hand (no script). Checklist:
1. Tool: Linux `zip -r Group11_Assignment5_code.zip Group11_Assignment5_code -x "*/out/*" "*/__pycache__/*"`; Windows Explorer "Send to > Compressed (zipped) folder" or 7-Zip. **Not** Windows PowerShell 5.1 `Compress-Archive` — it stores backslash paths, which is how A4's zip extracted on Linux as files literally named `Group11_Assignment4_code\data.py`.
2. Contents: top folder `Group11_Assignment5_code/`, only `.py` files + `README.md` + `requirements.txt`; no `out/` (data caches, checkpoints, figures), no `__pycache__/`.
3. Verify: `unzip -l Group11_Assignment5_code.zip` (or open it in 7-Zip) shows forward-slash paths under that one folder.
4. Report: `Group11_Assignment5_report.pdf`, all observations included.

---

## 9. Expected-results manifest (every item the sheet asks for)

| Q | Artefacts |
|---|---|
| A1 | 5 chosen images; filter values; 5 feature maps; dimension check (expected vs obtained) |
| A2 | printed feature maps/shapes after each layer; dimension checks (conv + pool, both layers); 3 conv1 filters + 3 conv2 filters as images; their feature maps for the 5 images |
| A3a | train/val/test accuracy; 3 confusion matrices; training curve |
| A3b | 6 conv1 + 10 conv2 trained filters (incl. A2 indices); feature maps for the 5 images; before/after comparison; note on the 10-vs-6 contradiction |
| A3c | 5 images × 5 neurons, (map, row, col) + activation, 14×14 bounding boxes |
| B1a | 3 architecture descriptions (feature-map size per layer, params); train + val accuracy; 6 confusion matrices; curves |
| B1b | chosen arch + reason; test accuracy; test confusion matrix |
| B1c | 1 image; 8 conv1 maps; 8 maps per later conv layer; observations |
| B1d | 5 images; 1 neuron each; patch box + crop |
| B2a | train/val/test accuracy; 3 confusion matrices; comparison with B1b |
| B2b | same 5 images; 1 neuron each in block5_conv4; patch box (RF > image noted) |
| B2c | B1c image; 5 neuron positions; 5 guided-backprop images; observations |
| B2d | 25 Grad-CAM heat maps (5 images × 5 classes) |
| Report | observations for every question; all stated assumptions (§3, A3b, floor convention, cross-correlation, VGG head reading, neuron selection, image selection, greyscale caveat) |

---

## 10. Order of work and time budget

| Day | Work |
|---|---|
| Sun 25 | Env (TF install + check), `config/data/plots/receptive/conv_manual`, smoke tests 1–8, 13–14 |
| Mon 26 | Part A end-to-end with `--quick`; full Part A run incl. lr grid |
| Tue 27 | Part B1 (`--quick`, then full — minutes on CPU); smoke 9–12 |
| Wed 28 | Part B2 (feature cache, head, B2b–d); smoke 14–17 |
| Thu 29 | `run.py smoke` + `run.py all --quick` on **Windows**; final full run on one machine; contact-sheet review |
| Fri 30 | Report writing, observations |
| Sat 31 | Submission build + self-test, upload well before 22:00 |

## 11. Decisions to confirm (defaults above are used unless changed)
1. A3b: 6 conv1 + 10 conv2 (default) — or ask the instructor.
2. VGG head: strict (replace `predictions` only) — default, per the notes.
3. Neuron selection: top-5 distinct feature maps (default) vs top-5 raw units.
4. ~~Optimizer~~ decided: momentum SGD, batch 32 / 16, per-model lr from a shared grid (§3).
