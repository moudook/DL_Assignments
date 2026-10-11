# Assignment-4 verification tools

Development tooling, deliberately kept **outside** the submission folder
(`Assignment4/Group11_Assignment4_code/`) so the submitted code contains only the
pipeline itself.

Run any of them with the code directory as the working directory:

```bash
cd Assignment4/Group11_Assignment4_code
python ../../_dev_tools/a4_verify/verify.py results_final
python ../../_dev_tools/a4_verify/verify_report_claims.py
python ../../_dev_tools/a4_verify/verify_from_checkpoints.py
python ../../_dev_tools/a4_verify/verify_ae_from_checkpoints.py
python ../../_dev_tools/a4_verify/verify_independent.py
```

| script | what it proves |
|---|---|
| `verify.py` | pipeline invariants: 66 checkpoints load, epoch invariant holds, degeneracy flags match the 85% floor, no run hit the epoch cap, figure set complete |
| `verify_report_claims.py` | all 58 quantitative claims written in the report are re-derived from `results_final/` |
| `verify_from_checkpoints.py` | Task-1 accuracies and confusion matrices recomputed from stored weights and the PCA fit, ignoring the JSON |
| `verify_ae_from_checkpoints.py` | Task-2/3/4 accuracies, confusion matrices and reconstruction errors recomputed from stored weights |
| `verify_independent.py` | cross-file consistency: grid maxima, summary agreement, selection-bias arithmetic, A3 baseline, report table values, degeneracy consistency |

`verify_from_checkpoints.py` and `verify_ae_from_checkpoints.py` are the authoritative
ones: they re-predict the test set from the trained parameters rather than reading back
the pipeline's own output. See ADR-020 in `Group11_Assignment4_code/DECISIONS.md`.