"""
run.py - the single entry point (works the same on Linux and Windows).

    python run.py smoke                    correctness checks (torch + TF)
    python run.py partA  [--quick]         Part A (MNIST)
    python run.py partB1 [--quick]         Part B Q1 (three CNNs)
    python run.py partB2 [--quick]         Part B Q2 (VGG19; needs partB1 for B2a)
    python run.py all    [--quick]         all three parts, then check
    python run.py check                    manifest + summary.json + contact sheets

--quick: tiny subsets and 2 epochs; verifies the pipeline, numbers meaningless.
Default output: out/ (out_quick/ with --quick); override with --outdir.
Each part runs in its own process: TensorFlow and torch never share one.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import config

PARTS = {"partA": "part_a.py", "partB1": "part_b1.py", "partB2": "part_b2_vgg.py"}

# Every artefact the assignment needs.
MANIFEST = {
    "A": ["results.json", "A1_filter.png", "A1_feature_maps.png",
          "A2_filters_l1.png", "A2_filters_l2.png", "A2_maps_l1.png", "A2_maps_l2.png",
          "A3a_cm_train.png", "A3a_cm_val.png", "A3a_cm_test.png", "A3a_curves.png",
          "A3a_lr_grid.png", "A3b_filters_l1.png", "A3b_filters_l2.png",
          "A3b_before_after_l1.png", "A3b_before_after_l2.png", "A3b_maps_l1.png",
          "A3b_maps_l2.png", "A3c_patches_grid.png", "A3c_patches_overlay.png"],
    "B1": ["results.json", "B1b_best_cm_test.png", "B1c_conv1_maps.png",
           "B1c_conv2_maps.png", "B1d_patches.png"]
          + ["B1a_arch%d_%s.png" % (n, k) for n in (1, 2, 3)
             for k in ("cm_train", "cm_val", "curves", "lr_grid")],
    "B2": ["results.json", "B2a_cm_train.png", "B2a_cm_val.png", "B2a_cm_test.png",
           "B2a_curves.png", "B2a_lr_grid.png", "B2a_comparison.png", "B2b_patches.png",
           "B2c_guided_backprop.png", "B2d_gradcam.png"],
}


def run_part(name, outdir, quick):
    cmd = [sys.executable, str(config.CODE_DIR / PARTS[name]), "--outdir", str(outdir)]
    if quick:
        cmd.append("--quick")
    print("=== %s: %s" % (name, " ".join(cmd[1:])), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(config.CODE_DIR))
    print("=== %s finished in %.0fs (exit %d)" % (name, time.perf_counter() - t0, r.returncode),
          flush=True)
    return r.returncode


def summary(outdir):
    """Headline numbers of every part in one JSON."""
    from results import load_json, save_json
    out = {}
    pa = outdir / "A" / "results.json"
    if pa.exists():
        a = load_json(pa)
        out["A3"] = {"lr": a["A3"]["lr_grid"]["winner_lr"],
                     **{s: a["A3"]["metrics"][s]["accuracy"] for s in config.SPLITS}}
    pb = outdir / "B1" / "results.json"
    if pb.exists():
        b = load_json(pb)
        out["B1"] = {k: {"name": v["name"], "params": v["params"],
                         "lr": v["lr_grid"]["winner_lr"],
                         "train": v["metrics"]["train"]["accuracy"],
                         "val": v["metrics"]["val"]["accuracy"],
                         "val_loss": v["metrics"]["val"]["loss"]}
                     for k, v in b["architectures"].items()}
        out["B1b"] = {"best": b["B1b"]["name"], "test": b["B1b"]["test"]["accuracy"]}
    pv = outdir / "B2" / "results.json"
    if pv.exists():
        v = load_json(pv)
        out["B2a"] = {"lr": v["lr_grid"]["winner_lr"],
                      **{s: v["B2a"]["metrics"][s]["accuracy"] for s in config.SPLITS}}
    return save_json(out, outdir / "summary.json"), out


def check(outdir):
    import plots
    missing = [str(Path(p) / f) for p, files in MANIFEST.items() for f in files
               if not (outdir / p / f).exists()]
    for p in MANIFEST:
        pngs = sorted((outdir / p).glob("*.png"))
        if pngs:
            plots.contact_sheet(pngs, outdir / ("contact_sheet_%s.pdf" % p))
    path, s = summary(outdir)
    print("summary: %s" % path)
    for k, v in s.items():
        print("  %s: %s" % (k, v))
    if missing:
        print("MISSING %d artefacts:\n  %s" % (len(missing), "\n  ".join(missing)))
        return 1
    print("manifest complete: %d artefacts" % sum(len(f) for f in MANIFEST.values()))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["smoke", "all", "check"] + list(PARTS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = Path(args.outdir) if args.outdir else (
        config.CODE_DIR / ("out_quick" if args.quick else "out"))

    if args.command == "smoke":
        return subprocess.run([sys.executable, str(config.CODE_DIR / "smoke_test.py")],
                              cwd=str(config.CODE_DIR)).returncode
    if args.command == "check":
        return check(outdir)
    names = list(PARTS) if args.command == "all" else [args.command]
    for name in names:
        if run_part(name, outdir, args.quick) != 0:
            return 1
    return check(outdir) if args.command == "all" else 0


if __name__ == "__main__":
    sys.exit(main())
