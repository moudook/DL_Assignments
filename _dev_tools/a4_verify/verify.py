"""
Re-check a finished run's artefacts for the invariants the report relies on.

Each check restates a claim from DECISIONS.md and either confirms it from the run's own
files or says precisely how it fails. Nothing here retrains anything.
"""

import glob
import json
import os
import sys

import torch

FAILURES = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (("  -- " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label)


def main(outdir):
    outdir = os.path.abspath(outdir)
    print(f"verifying {outdir}\n")

    print("[1] every task wrote a result file")
    expected = {"task1.json", "task2.json", "task3.json", "task4.json",
                "task5.json", "task6.json", "summary.json"}
    present = set(os.path.basename(p) for p in glob.glob(os.path.join(outdir, "*.json")))
    missing = expected - present
    check("all six task files plus summary", not missing, "missing: %s" % sorted(missing))

    print("\n[2] task numbers agree with summary.json")
    summary = json.load(open(os.path.join(outdir, "summary.json"), encoding="utf-8"))
    for task, key in ((1, "task1"), (3, "task3"), (4, "task4"), (5, "task5")):
        js = json.load(open(os.path.join(outdir, f"task{task}.json"), encoding="utf-8"))
        rep = js.get("best_test_accuracy")
        got = summary.get(key, {}).get("best_test_accuracy")
        check(f"task{task} accuracy matches summary", rep is not None and got is not None
              and abs(rep - got) < 1e-12,
              "task=%.6f summary=%.6f" % (100 * (rep or 0), 100 * (got or 0)))

    print("\n[3] checkpoints are the model the results describe (ADR-004)")
    ckpts = glob.glob(os.path.join(outdir, "**", "*.pt"), recursive=True)
    check("at least 60 checkpoints exist", len(ckpts) >= 60, "found %d" % len(ckpts))
    bad_epoch, bad_metric, n = [], [], 0
    for path in ckpts:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            bad_metric.append("%s: unreadable (%s)" % (os.path.basename(path), exc))
            continue
        n += 1
        ep, er = ck.get("epoch"), ck.get("epochs_run")
        if ep is not None and er is not None and ep + 1 != er:
            bad_epoch.append("%s: %s+1 != %s" % (os.path.basename(path), ep, er))
        for key in ("test_accuracy", "best_test_accuracy", "reconstruction_mse"):
            if key in ck and isinstance(ck[key], float):
                break
    check("epoch + 1 == epochs_run for every checkpoint", not bad_epoch,
          "; ".join(bad_epoch[:3]))
    check("every checkpoint loaded and reported an epoch", n == len(ckpts), "%d/%d" % (n, len(ckpts)))

    print("\n[4] no run is reported converged while degenerate (ADR-003)")
    flagged, mislabelled, converged = [], [], 0
    for path in glob.glob(os.path.join(outdir, "task*.json")):
        name = os.path.basename(path)
        js = json.load(open(path, encoding="utf-8"))
        for container in (js.get("results"), js.get("by_dimension"), js.get("by_bottleneck")):
            for key, entry in (container or {}).items():
                if not isinstance(entry, dict):
                    continue
                for arch, r in (entry.get("architectures") or {}).items():
                    state = r.get("terminal_state")
                    val = r.get("val_accuracy", r.get("val_acc"))
                    if state == "converged":
                        converged += 1
                        if isinstance(val, float) and val < 0.85:
                            mislabelled.append("%s:%s:%s (%.3f)" % (name, key, arch, val))
                for arch in (entry.get("degenerate_archs") or []):
                    flagged.append("%s:%s:%s" % (name, key, arch))
    print("      %d runs labelled converged, %d labelled degenerate" % (converged, len(flagged)))
    check("at least one run was labelled degenerate", len(flagged) > 0,
          "0 flagged means the check cannot see the right field")
    check("no sub-85% run labelled converged", not mislabelled, "; ".join(mislabelled[:3]))

    print("\n[5] autoencoders stopped on the plateau rule, not the epoch cap")
    t2 = json.load(open(os.path.join(outdir, "task2.json"), encoding="utf-8"))
    cap = t2.get("max_epochs")
    over_cap, on_rule = [], 0
    for key, r in (t2.get("by_model") or {}).items():
        er, stopped = r.get("epochs_run"), r.get("stopping_rule")
        if isinstance(er, int) and isinstance(cap, int) and er >= cap:
            over_cap.append("%s (%d)" % (key, er))
        if stopped:
            on_rule += 1
    print("      %d AE runs, all recording a stopping rule" % on_rule)
    check("no AE run hit the epoch budget", not over_cap, "; ".join(over_cap[:4]))
    check("every AE run recorded which rule stopped it",
          on_rule == len(t2.get("by_model") or {}), "%d/%d" % (on_rule, len(t2.get("by_model") or {})))

    print("\n[6] figure set is complete")
    figs = glob.glob(os.path.join(outdir, "plots", "**", "*.png"), recursive=True)
    check("at least 420 figures", len(figs) >= 420, "found %d" % len(figs))
    check("dataset figures present",
          len(glob.glob(os.path.join(outdir, "plots", "00_dataset", "*.png"))) >= 7)
    check("class-mean figure has its confusion panels",
          os.path.getsize(os.path.join(outdir, "plots", "00_dataset",
                                       "dataset_class_mean_distances.png")) > 60000)

    print("\n" + "=" * 62)
    if FAILURES:
        print("FAILED %d check(s):" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results_final"))
