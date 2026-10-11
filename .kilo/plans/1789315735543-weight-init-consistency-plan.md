# Plan: Assignment 3 — Controlled Values + GPU Training
## Source of Truth: Assignment 2 Manual Review

Assignment 2 implements every tunable quantity explicitly:
- `models/fcnn.py:29-37` — weight init: `np.random.default_rng(seed)`, `scale = np.sqrt(1.0 / n_in)`, bias = 0
- `shared/data.py:84-146` — split ratios `(0.6, 0.2, 0.2)`, `stratify=True`, explicit `seed=42`
- `optimizers/sgd.py:12-32` — explicit `lr`, `epochs`, `seed`, `tol`; `rng = np.random.default_rng(self.seed)` for deterministic order
- `classification/_common.py:38-57` — `RunSpec` dataclass with explicit fields
- `shared/plotting.py` — explicit DPI, figure sizes, colors, resolutions

Assignment 3 must follow the same pattern: **no hidden defaults, no magic numbers, no implicit library behavior**.

---

## Requirements

| # | Requirement |
|---|-------------|
| d | assignment_statement.txt point d: same initial weights for each architecture across all optimizers |
| p1 | Professor: check/raise error **before** `optimizer.step()`, not after weights are corrupted |
| p2 | Professor: use **one consistent library/method** everywhere — no mixing libraries with different default init ranges |
| p3 | Professor: understand the library — explicitly own constants/formulas, don't blindly use defaults |
| p4 | Professor: two motives — (1) understand learning rates + backprop algorithms, (2) understand libraries |
| p5 | Professor scolded: Assignment 2 node-output plots must be **3-D particle/scatter plots** (actual data points only), not 3-D mesh surfaces. One graph per node and split, with all classes shown in that graph. |
| p6 | Professor/User: training must run on GPU, not CPU |
| p7 | Assignment point a: use cross-entropy loss — current `nn.CrossEntropyLoss()` is correct; no change needed |
| u1 | User: exact same weights/range/formula for all permutations, no hard-coding |
| u2 | User: save checkpoints so progress is not lost if the run crashes or is interrupted |
| u3 | User: save all model outputs/checkpoints and every type of plot structurally, with redundancy allowed, so nothing needs to be retrained and outputs are easy to identify by kind/architecture/optimizer/split |

---

## Precise Interpretation

- **Assignment 3, point d**: for each fixed architecture, all 7 optimizers must start from bit-identical initial weight tensors. `experiment.py` must save that architecture's `state_dict` once and reload it before every optimizer. There are exactly **6 architectures** total, split across 3 Colab notebooks.
- **Across different architectures**: layer shapes differ, so a single full weight tensor cannot literally be shared. Use one explicit initializer, one seed policy, and the same formulas/ranges for every architecture; do not let library defaults introduce unexplained variation. There are exactly 6 architectures in this assignment.
- **Assignment 2 graph requirement**: Classification task 4 asks for each node's output with `x1`, `x2`, and `z` for train/validation/test. The required particle graph is a 3-D scatter plot of actual data points only—one graph per node and split, all classes in that graph, with no mesh surface.
- **Assignment 3 graph requirement**: average training error versus epochs, with optimizer curves superimposed. This is a separate 2-D plot and `evaluate.py` already implements it.
- **Comprehensive output/plot saving**: save every artifact structurally: model checkpoints, histories, CSV tables, confusion matrices, and every useful plot type. Redundant plotting is allowed. Directory layout must make it easy to identify plot kind, architecture, optimizer, and data split at a glance.
- **Device policy**: training must run on GPU when CUDA is available. All models, data batches, and optimizers must live on the same device.

---

## Current Problems

| # | Location | Problem |
|---|----------|---------|
| 1 | `models.py:46` | `_make_model()` calls `torch.manual_seed(INIT_SEED)` before every model → architectures with the same shape get identical weights; the global initialization stream is reset instead of used once |
| 2 | `models.py:28` | `nn.Linear` silently uses PyTorch default Kaiming-uniform — hidden, library-dependent, not explicitly owned |
| 3 | `dataset.py:44-46` | `DataLoader(shuffle=True)` uses an unseeded/default generator; each optimizer can see a different sample order, confounding optimizer comparison |
| 4 | `train.py:38-48` | `optimizer.step()` runs before any error check — weights corrupted if loss/grad is NaN/Inf |
| 5 | `train.py` | No NaN/Inf detection anywhere |
| 6 | `Group11_Assignment2_code/shared/plotting.py:179-205` | Assignment 2 currently draws a meshgrid surface (`plot_surface`) and overlays data points; professor wants the particle/scatter plot without the surface |
| 7 | `experiment.py:13` | Device selection exists, but the current training loop and data pipeline should be audited to ensure all tensors/optimizers are consistently placed on the selected device and that GPU training is actually used when available |

---

## Fixes

### 1. `models.py` — Explicit init + single global seed
- Remove `torch.manual_seed(INIT_SEED)` from `_make_model()`.
- Add `_init_weights(m)`: if `nn.Linear`, apply `nn.init.xavier_uniform_(m.weight)` + `nn.init.zeros_(m.bias)`.
- In `FCNN.__init__`, call `self.apply(_init_weights)` after building layers.
- In `get_models()`, call `torch.manual_seed(INIT_SEED)` **once** before the dict.
- Log init method.
- Top-level comment: document init method, formula, seed, rationale.

**Why Xavier?** Assignment uses Tanh. Xavier is textbook match for Tanh/sigmoid. Also TF/Keras default → most cross-library consistent. Explicitly chosen, not blindly using PyTorch default → satisfies "understand the library."

### 2. `train.py` — Error-before-update guard
- Add `_check_numerical_stability(model, loss, epoch, optimizer_name='')`:
  - `if not torch.isfinite(loss):` print diagnostics, raise `RuntimeError`.
  - After `loss.backward()`, scan all `p.grad`: `if not torch.isfinite(p.grad).all():` print diagnostics, raise `RuntimeError`.
- Call **after** `loss.backward()` and **before** `optimizer.step()`.

### 3. `dataset.py` — Deterministic, device-aware dataloaders
- Add `seed` and `device` parameters to `get_dataloaders`.
- For training DataLoader, use `shuffle=True` with a deterministic generator: `generator=torch.Generator().manual_seed(seed)` so every optimizer sees the same sample order.
- Use `pin_memory=True` when `device.type == 'cuda'` for faster host-to-device transfer.
- Keep val/test loaders non-shuffled, full-batch.

### 4. `experiment.py` — Controlled values, GPU, graceful failure, checkpoints, structured outputs
- Define all hyperparameters as module-level constants at the top of the file, matching Assignment 2's `RunSpec` style:
  - `LR = 0.001`, `MOMENTUM = 0.9`, `BETA1 = 0.9`, `BETA2 = 0.999`, `EPSILON = 1e-8`, `ALPHA = 0.99`, `TOL = 1e-4`, `INIT_SEED = 42`, `DATA_SEED = 42`, `MAX_EPOCHS = 10000`
- Select device once: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` and pass it everywhere.
- Build dataloaders with `seed=DATA_SEED` and `device=device`.
- Move model to `device` **before** constructing the optimizer so optimizer state lives on the correct device.
- Wrap `train_model()` in `try/except RuntimeError`. On catch: log `(arch_name, opt_name, error_msg)` to `errors.csv` in results dir. Continue to next pair; don't crash entire run.
- **Checkpointing:** after each completed (architecture, optimizer) run, save `torch.save({'model_state_dict': model.state_dict(), 'optimizer_name': opt_name, 'arch_name': arch_name, 'epoch': len(history['train_loss']), 'history': history}, checkpoint_path)`. On startup, scan results dir for existing checkpoints and skip/resume completed runs so progress is preserved if the script is interrupted.
- **Structured output directory layout:**
  - `results/`
    - `checkpoints/{arch_name}/{opt_name}_epoch{epoch}.pt`
    - `histories/{arch_name}/{opt_name}.csv`
    - `plots/`
      - `epoch_loss/{arch_name}/{opt_name}.png` (superimposed optimizer curves)
      - `confusion_matrix/{arch_name}/{opt_name}_{split}.png`
      - `surface_3d/{arch_name}/{opt_name}/{split}/layer{layer_idx}_node{node_idx}.png` (if/when implemented)
      - `scatter_3d/{arch_name}/{opt_name}/{split}/layer{layer_idx}_node{node_idx}.png` (if/when implemented)
    - `convergence_results.csv`
    - `errors.csv`
    - `best_model.pt`
- **Naming convention:** every file name encodes `arch_name`, `opt_name`, `epoch`, `split`, and plot kind so the source is identifiable without reading the file.
- **Redundant plotting:** generate every useful diagnostic plot even if not explicitly required by the assignment, as long as it is saved structurally and does not require retraining.

### 5. `evaluate.py` — No change
- `plot_training_curves` already produces 2D superimposed line plots. Correct.
- Report must call them "epoch-loss curves" — never "surface graphs."

### 6. Constants
Explicitly define all hyperparameters in `experiment.py` so they are visible and understood:

| Constant | Value | Used In |
|----------|-------|---------|
| `LR` | 0.001 | All optimizers |
| `MOMENTUM` | 0.9 | SGD_Momentum, NAG |
| `BETA1` | 0.9 | Adam |
| `BETA2` | 0.999 | Adam |
| `EPSILON` | 1e-8 | RMSProp, Adam, numerical checks |
| `ALPHA` | 0.99 | RMSProp |
| `TOL` | 1e-4 | Stopping |
| `INIT_SEED` | 42 | Weight init |
| `DATA_SEED` | 42 | DataLoader shuffle order |
| `MAX_EPOCHS` | 10000 | Safety cap |

---

## Validation

1. Two different architectures have different weight tensors.
2. Same architecture + two optimizers → bit-identical initial weights.
3. All DataLoader shuffles are deterministic and identical across optimizer runs for the same architecture.
4. Smoke test: train one (arch, optimizer) pair → no NaN/Inf, error guard silent.
5. Inject NaN → `RuntimeError` raised **before** `optimizer.step()`, weights uncorrupted.
6. When CUDA is available, `next(model.parameters()).is_cuda` is True and training runs on GPU.
7. Loss function is `nn.CrossEntropyLoss()` with raw logits, matching assignment point a.
8. Exactly 6 architectures are present in `get_models()`, and all 3 Colab notebooks reflect this split.
9. Report names init method/formula, explains Xavier choice, labels graphs as "epoch-loss curves."

---

## Decisions Finalized

| Decision | Choice |
|----------|--------|
| Init method | `nn.init.xavier_uniform_` (Glorot Uniform) — explicit, cross-library consistent, standard for Tanh |
| Seed | One `torch.manual_seed(42)` at start of `get_models()` |
| Error policy | Raise `RuntimeError` before `optimizer.step()`; catch in `experiment.py` |
| Graphs | 2D epoch-loss curves (current code correct) |
| Library | PyTorch only; explicit init replaces hidden default |
| Loss | `nn.CrossEntropyLoss()` with raw logits, matching assignment point a |
| Checkpoints | Save `state_dict` after every run so progress is preserved across interruptions |
| Device | `cuda` if available, otherwise `cpu`; all tensors/optimizers on same device; `pin_memory=True` for GPU dataloaders |
