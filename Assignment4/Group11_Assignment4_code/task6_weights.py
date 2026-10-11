"""
task6_weights.py — Assignment-4 Task-6: weight visualisation.

A4 Task-6 requirements:
  a. For the BEST compressed representation from the 1-hidden autoencoder, plot
     the inputs that maximally activate each neuron of the hidden representation
     ("plot of weights from the input layer to the compressed layer").
  b. Same for BOTH denoising autoencoders (20% and 40%).
  c. Compare (a) and (b).

Two readings of "maximally activate ... (plot of weights ...)": the inputs that
maximise each unit, and the encoder weight vectors themselves. We plot BOTH rows
per unit, because they answer different questions and can disagree - a unit
firing on a motif shared across many images has a diffuse weight vector but a
consistent best input. Showing only one would guess at intent; showing both makes
the comparison in (c) substantive.

Selection: "best compressed representation in one hidden layer autoencoder" =
Task-3's winning bottleneck (selection.json["3"]). The denoising AEs use the same
bottleneck, since Task-5 adopted Task-3's.

Writes results/task6.json.
"""

import json
import os

import numpy as np
import torch

import plots
from plots import ensure_dir
from evaluate import maximally_activating
from models import build_autoencoder, build_denoising_autoencoder
from run_tracker import RunTracker, atomic_write_text
from train import train_autoencoder, MAX_EPOCHS

MAX_UNITS_SHOWN = 32


def _jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.numel() > 1 \
            else obj.detach().cpu().item()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _state_matches(state_dict, bottleneck):
    """
    Does a saved 1-hidden AE state_dict have the right bottleneck width?

    The encoder's first Linear is (bottleneck, 784), so its output size is an
    exact check. Verifying before load_state_dict converts a crash into a
    harmless fallback: Task-6 runs last, so an exception there discards hours of
    completed upstream work.
    """
    try:
        w = state_dict["encoder.0.weight"]
        return tuple(w.shape) == (bottleneck, 784)
    except (KeyError, TypeError, AttributeError):
        return False


def resolve_selection(outdir):
    """Read Task-3's winning bottleneck (and Task-5's noise levels)."""
    path = os.path.join(outdir, "selection.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Task-6 depends on Task-3's best representation.")
    with open(path, encoding="utf-8") as fh:
        sel = json.load(fh)
    t3 = sel.get("3") or {}
    if t3.get("best_bottleneck") is None:
        raise KeyError("selection.json has no Task-3 best_bottleneck.")
    return int(t3["best_bottleneck"]), sel


def run_task6(data, outdir="results", max_epochs=MAX_EPOCHS, device=None,
              ae_state=None, denoise_state=None, bottleneck=None, noise_levels=(0.2, 0.4)):
    """
    Produce the Task-6 weight-visualisation figures for the plain AE and both
    denoising AEs, plus the comparison.

    ae_state / denoise_state: optional pretrained state dicts (from Task-2 /
    Task-5) so Task-6 does not retrain. When absent, models are trained here.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = plots.task_dirs(outdir, 6)
    V = {
        "plain_ae":     ensure_dir(os.path.join(dirs["weights"], "01_plain_AE")),
        "denoise20":    ensure_dir(os.path.join(dirs["weights"], "02_denoising_AE_20pct")),
        "denoise40":    ensure_dir(os.path.join(dirs["weights"], "03_denoising_AE_40pct")),
    }
    cmp_dir = ensure_dir(os.path.join(dirs["comparison"]))

    if bottleneck is None:
        bottleneck, sel = resolve_selection(outdir)
        print(f"  bottleneck from Task-3 selection: k={bottleneck}")
    else:
        _, sel = resolve_selection(outdir)

    print(f"Task-6: weight visualisation, k={bottleneck}")

    X_train = data["X_train"]
    variants = {}

    if ae_state and _state_matches(ae_state, bottleneck):
        ae = build_autoencoder("1hidden", bottleneck).to(device)
        ae.load_state_dict(ae_state)
        print("  (a) reusing supplied 1-hidden AE state")
    else:
        if ae_state:
            print(f"  (a) supplied state does not match k={bottleneck}; "
                  f"training locally instead")
        tracker = RunTracker(f"task6_ae_k{bottleneck}", outdir,
                             total_epochs=max_epochs)
        ae = build_autoencoder("1hidden", bottleneck).to(device)
        train_autoencoder(ae, data, f"ae_k{bottleneck}", tracker=tracker,
                          max_epochs=max_epochs, device=device)
    variants["plain_ae"] = (ae, f"1-hidden AE (no noise), k={bottleneck}")

    for noise in noise_levels:
        tag = f"denoise{int(noise * 100)}"
        key = f"noise{int(noise * 100)}"
        state = (denoise_state or {}).get(key)
        if state and _state_matches(state, bottleneck):
            m = build_denoising_autoencoder(bottleneck, noise=noise).to(device)
            m.load_state_dict(state)
            print(f"  (b) reusing supplied {tag} state")
        else:
            if state:
                print(f"  (b) supplied {tag} state does not match "
                      f"k={bottleneck}; training locally instead")
            gen = torch.Generator(device=device).manual_seed(42)
            tracker = RunTracker(f"task6_dae_{tag}_k{bottleneck}", outdir,
                                 total_epochs=max_epochs)
            m = build_denoising_autoencoder(bottleneck, noise=noise).to(device)
            train_autoencoder(m, data, f"dae_{tag}", tracker=tracker,
                              max_epochs=max_epochs, device=device,
                              noise_level=noise, noise_generator=gen)
        variants[tag] = (m, f"Denoising AE {int(noise * 100)}% noise, k={bottleneck}")

    results = {
        "task": 6,
        "name": "Weight visualisation",
        "bottleneck": bottleneck,
        "bottleneck_source": "Task-3 best representation (A4 Task-6a)",
        "units_shown": min(bottleneck, MAX_UNITS_SHOWN),
        "units_total": bottleneck,
        "selection_rule": ("argmax |activation|: the bottleneck is linear "
                           "(A4 mandate) so activations are signed, and "
                           "selecting by plain argmax would return the "
                           "least-negative input for a negative-preferring "
                           "unit - i.e. its weakest response, not its "
                           "strongest."),
        "variants": {},
    }

    for key, (model, title) in variants.items():
        print(f"\n  {title}")
        ma = maximally_activating(model, X_train)
        n_show = results["units_shown"]

        p_pairs = plots.maxact_grid(
            ma["inputs"][:n_show], ma["weights"][:n_show],
            f"{title} — max-activating input and encoder weight",
            os.path.join(V[key], f"task6_{key}_maxact_and_weight.png"),
            unit_labels=[f"u{i}" for i in range(n_show)])

        p_inputs = plots.maxact_grid(
            ma["inputs"][:n_show], None,
            f"{title} — max-activating inputs only",
            os.path.join(V[key], f"task6_{key}_maxact_inputs.png"))

        p_weights = plots.weight_image_grid(
            ma["weights"], f"{title} — encoder weights, all {len(ma['weights'])} units",
            os.path.join(V[key], f"task6_{key}_weight_grid.png"))

        plots.activation_histogram(
            ma["acts"], f"{title} — distribution of max activation per unit",
            os.path.join(V[key], f"task6_{key}_activation_hist.png"),
            xlabel="max activation on its best input")
        plots.activation_histogram(
            np.asarray(ma["weights"]).ravel(),
            f"{title} — distribution of encoder weight values",
            os.path.join(V[key], f"task6_{key}_weight_hist.png"),
            bins=100, xlabel="weight value")

        acts = ma["acts"]
        results["variants"][key] = {
            "title": title,
            "k": ma["k"],
            "weights": np.asarray(ma["weights"]).tolist(),
            "activations": acts,
            "activation_abs_max": [abs(a) for a in acts],
            "n_negative_units": sum(1 for a in acts if a < 0),
            "figures": {"pairs": p_pairs, "inputs": p_inputs,
                        "weights_grid": p_weights},
            "weights_recoverable_from": "trained autoencoder checkpoint",
        }
        n_neg = results["variants"][key]["n_negative_units"]
        print(f"    units shown: {n_show}/{ma['k']}   "
              f"negative-responding units: {n_neg}")

    keys = list(results["variants"])
    if len(keys) >= 2:
        plot_keys = keys[:3]
        n_show = results["units_shown"]
        fig, ax = plt_unit_comparison(results["variants"], plot_keys, n_show)
        p_cmp = save_comparison(fig, ax, os.path.join(
            cmp_dir, "task6c_comparison_activation_by_unit.png"))
        results["comparison_figure"] = p_cmp

        results["bottleneck_grid_figure"] = plt_bottleneck_grid(
            results["variants"], plot_keys, n_show,
            os.path.join(cmp_dir, "task6b_bottleneck_weight_grid_all.png"))
        results["bottleneck_sample_figure"] = plt_bottleneck_sample(
            results["variants"], plot_keys, n_show,
            os.path.join(cmp_dir, "task6b_bottleneck_weight_grid.png"))
        results["bottleneck_profile_figure"] = plt_bottleneck_profile(
            results["variants"], plot_keys,
            os.path.join(cmp_dir, "task6c_bottleneck_weight_profile.png"))
        print(f"  bottleneck comparison: {results['bottleneck_sample_figure']}")
        print(f"                        {results['bottleneck_profile_figure']}")

        summary = {}
        for k in keys:
            a = np.abs(results["variants"][k]["activations"])
            summary[k] = {
                "mean_abs_activation": float(a.mean()),
                "max_abs_activation": float(a.max()),
                "median_abs_activation": float(np.median(a)),
            }
        results["comparison_summary"] = summary
        print("\n  comparison (mean |activation| per unit):")
        for k, v in summary.items():
            print(f"    {k:<12} mean={v['mean_abs_activation']:.3f}  "
                  f"max={v['max_abs_activation']:.3f}")

        plots.comparison_bars(
            {k: summary[k]["mean_abs_activation"] for k in keys},
            "Task-6c: mean |activation| per bottleneck unit, plain vs denoising",
            os.path.join(cmp_dir, "task6c_mean_activation_by_variant.png"),
            ylabel="mean |activation|",
            labels=[results["variants"][k]["title"] for k in keys])
        plots.comparison_bars(
            {k: summary[k]["max_abs_activation"] for k in keys},
            "Task-6c: max |activation| per bottleneck unit, plain vs denoising",
            os.path.join(cmp_dir, "task6c_max_activation_by_variant.png"),
            ylabel="max |activation|",
            labels=[results["variants"][k]["title"] for k in keys])
        plots.comparison_bars(
            {k: results["variants"][k]["n_negative_units"] for k in keys},
            "Task-6c: units whose strongest response is negative",
            os.path.join(cmp_dir, "task6c_negative_units_by_variant.png"),
            ylabel="number of units",
            labels=[results["variants"][k]["title"] for k in keys])

    atomic_write_text(json.dumps(_jsonable(results), indent=2, default=float),
                      os.path.join(outdir, "task6.json"))
    print(f"\nwrote {os.path.join(outdir, 'task6.json')}")
    return results


def plt_bottleneck_grid(variants, keys, n_show, path):
    """
    Side-by-side internal-neuron weight maps: one column per variant, one row per
    bottleneck unit, so the same unit can be compared across training regimes
    directly.
    """
    import matplotlib.pyplot as plt
    n = min(n_show, len(variants[keys[0]]["weights"]))
    cols = len(keys)
    fig, axes = plt.subplots(n, cols, figsize=(2.0 * cols, 2.0 * n), squeeze=False)
    allw = np.concatenate([np.asarray(variants[k]["weights"][:n]).ravel() for k in keys])
    vmax = float(np.abs(allw).max())

    for c, k in enumerate(keys):
        W = np.asarray(variants[k]["weights"][:n])
        for r in range(n):
            ax = axes[r][c]
            ax.imshow(W[r].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.3)
            if r == 0:
                ax.set_title(f"{variants[k]['title']}\nk={variants[k]['k']}",
                             fontsize=8, pad=6)
            if c == 0:
                ax.set_ylabel(f"u{r}", fontsize=7, rotation=0, labelpad=14,
                              va="center")

    fig.suptitle("Internal-neuron weights at the bottleneck layer: "
                 "same units, three training regimes\n"
                 "red = positive weight, blue = negative, common scale",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return plots.save_atomic(fig, path)


def plt_bottleneck_sample(variants, keys, n_show, path, n_units=8):
    """
    Readable subset of the bottleneck grid: the first n_units internals, laid out
    one row per unit with all variants side by side. This is the figure a reader
    can actually inspect; the full 32-unit version is archived alongside it.
    """
    import matplotlib.pyplot as plt
    n = min(n_units, n_show, len(variants[keys[0]]["weights"]))
    cols = len(keys)
    fig, axes = plt.subplots(n, cols, figsize=(1.7 * cols, 1.7 * n), squeeze=False)
    allw = np.concatenate([np.asarray(variants[k]["weights"][:n]).ravel() for k in keys])
    vmax = float(np.abs(allw).max())

    for c, k in enumerate(keys):
        W = np.asarray(variants[k]["weights"][:n])
        for r in range(n):
            ax = axes[r][c]
            ax.imshow(W[r].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                short = variants[k]["title"].split(",")[0]
                ax.set_title(f"{short}\nk={variants[k]['k']}", fontsize=7.5, pad=5)
            if c == 0:
                ax.set_ylabel(f"u{r}", fontsize=7.5, rotation=0, labelpad=12,
                              va="center")

    fig.suptitle("Internal-neuron weights at the bottleneck layer, units u0-u%d:\n"
                 "vanilla autoencoder against both denoising autoencoders "
                 "(common colour scale, red positive / blue negative)" % (n - 1),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return plots.save_atomic(fig, path)


def plt_bottleneck_profile(variants, keys, path):
    """
    Quantitative companion to the grid: per-unit L2 weight norm, which says how
    much signal each unit actually carries, alongside the cosine similarity between
    each denoising encoder and the plain one.
    """
    import matplotlib.pyplot as plt
    n = min(len(variants[keys[0]]["weights"]),
            min(len(variants[k]["weights"]) for k in keys))
    base = np.asarray(variants[keys[0]]["weights"][:n])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.0))

    for k in keys:
        W = np.asarray(variants[k]["weights"][:n])
        norms = np.linalg.norm(W, axis=1)
        ax1.plot(np.arange(n), norms, marker="o", ms=4, lw=1.4,
                 label=variants[k]["title"])
        if k != keys[0]:
            cs = [float(np.dot(base[i], W[i]) /
                        (np.linalg.norm(base[i]) * np.linalg.norm(W[i]) + 1e-12))
                  for i in range(n)]
            ax2.plot(np.arange(n), cs, marker="s", ms=4, lw=1.4,
                     label=variants[k]["title"])

    ax1.set_xlabel("bottleneck unit")
    ax1.set_ylabel("L2 norm of that unit's weight vector")
    ax1.set_title("Weight magnitude per internal neuron")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)

    ax2.set_xlabel("bottleneck unit")
    ax2.set_ylabel("cosine similarity with the plain AE's unit")
    ax2.set_title("Directional change caused by denoising training")
    ax2.set_ylim(-1, 1)
    ax2.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)

    fig.suptitle("Quantifying what denoising does to the internal neurons", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return plots.save_atomic(fig, path)


def plt_unit_comparison(variants, keys, n_show):
    """Grouped bars of |activation| per unit across the three variants."""
    import matplotlib.pyplot as plt
    n = min(n_show, len(variants[keys[0]]["activations"]))
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(n)
    for i, k in enumerate(keys):
        a = np.abs(variants[k]["activations"][:n])
        ax.bar(x + i * width - 0.4 + width / 2, a, width,
               label=variants[k]["title"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"u{j}" for j in range(n)], fontsize=6, rotation=90)
    ax.set_xlabel("bottleneck unit")
    ax.set_ylabel("|activation| on its max-activating input")
    ax.set_title("Task-6(c): how strongly each unit responds, plain vs denoising AE")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def save_comparison(fig, ax, path):
    import plots
    return plots.save_atomic(fig, path)
