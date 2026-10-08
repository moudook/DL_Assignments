Dataset: MNIST subset, 5 classes, 28x28 images, flattened to 784-dim vectors
Train/val/test already separated (same as Assignment-3)
Same 5 classes as Assignment-3 (0, 4, 5, 6, 7)

Task-1: PCA dimension reduction
- Reduce dimensions to 32, 64, 128, 256
- Eigen vectors from training data
- Mean subtracted training, validation, test data projected onto eigen vectors
- Use mean vector from training set for validation and test mean subtraction
- FCNN classification model for each reduced dimension
- Present validation accuracy for different FCNN architectures
- Present test accuracy and confusion matrix for best architecture (based on validation)
- Observe best reduced dimension from test accuracies
- Compare with best results from Assignment-3

Task-2: Autoencoder reconstruction
- Build autoencoders with 1-hidden layer and 3-hidden layer architectures
- 3-hidden layer: 400 neurons in first and third layers
- Bottleneck (middle) layer: 32, 64, 128, 256 neurons
- Train with Adam optimizer
- Bottleneck layer is always linear (no activation)
- Sigmoid (logistic or tanh, use one consistently) for remaining hidden layers
- Observe average reconstruction errors for train, val, test (computed after training)
- Take one image per class from train, val, test; show reconstructed images for each architecture (with originals)

Task-3: Classification using 1-hidden autoencoder compressed representation
- Save output of middle layer (compressed) from each 1-hidden autoencoder encoder
- Obtain compressed representation for train, val, test
- Experiment for each reduced dimension representation
- FCNN classification model for each representation
- Same FCNN architectures as Task-1
- Present validation accuracy for different architectures
- Present test accuracy and confusion matrix for best architecture
- Observe best reduced dimension from test accuracies
- Compare with Assignment-3 and Task-1 results

Task-4: Classification using 2-hidden autoencoder compressed representation
- Save output of middle layer from 2-hidden autoencoder encoder
- Obtain compressed representation for train, val, test
- Experiment for each reduced representation
- FCNN classification model for each representation
- Same FCNN architectures as Task-1
- Present validation accuracy for different architectures
- Present test accuracy and confusion matrix for best architecture
- Observe best reduced dimension from test accuracies
- Compare with Assignment-3, Task-1, Task-3 results

Task-5: Denoising autoencoders
- 1-hidden layer autoencoder with 20% noise and 40% noise
- Bottleneck neurons based on best test accuracy of best reduced dimension from 1-hidden autoencoder
- Observe average reconstruction errors for train, val, test
- Take one image per class from train, val, test; show reconstructed images (with originals)
- Reduced dimension representation classification
- Same best architecture as Task-3 for this reduced representation
- Present validation and test accuracy

Task-6: Weight visualization
- Best compressed representation from 1-hidden autoencoder: plot inputs that maximally activate each hidden neuron (weights from input to compressed layer)
- Same for both denoising autoencoders
- Compare (a) and (b)

Code/Submission:
- Code in .py file
- Folder name: Group11_Assignment4_code
- Zip file: Group11_Assignment4_code.zip
- Report: Group11_Assignment4_report.pdf
- Upload code zip and report PDF to Moodle

EARLY STOPPING (MEASURED CORRECTION - differs from the naive inherited rule):
- A single-epoch test |L_t - L_{t-1}| < 1e-4 is NOT sufficient. Measured on this
  dataset: the 5L_A classifier [64,32,16,8,4] stalls with loss 1.6089 at epoch 13
  then OSCILLATES upward (1.6091, 1.6096, 1.6102, 1.6108, 1.6111). Adam's momentum
  swinging inside a local plateau produced a one-epoch |dL| of 1.2e-05, under the
  tolerance, so the naive rule declared "converged" while validation accuracy was
  exactly 20.00% (chance). The loss was flat AND RISING - the opposite of convergence.
- IMPLEMENTED: patience window. Stop only after |dL| < 1e-4 holds for STOP_PATIENCE
  = 15 CONSECUTIVE epochs. Verified against measured traces:
    healthy 3L_A: fires at epoch ~241 (loss settles to 4e-07) - CORRECT
    oscillating 5L_A plateau: never fires - CORRECT
- Consequence: 5L_A trains to 97.94% over 594 epochs instead of falsely stopping
  at chance. Without this fix it was reported as a valid 20% result.
- The convergence watcher's state (patience streak, prev/best loss) MUST be
  checkpointed. RunTracker.check() builds its OWN payload and only merges `extra`,
  so watch_state has to be passed through `extra` or it is silently discarded.
  Verified: without it, a resumed run restarts the streak at 0 and stops EARLIER
  than the criterion intends, making results depend on where a crash happened.
- DEGENERATE-RUN DETECTION: any run finishing at chance accuracy is flagged
  `degenerate` and EXCLUDED from best-architecture selection, with test accuracy
  recomputed for the re-selected survivor. Observed for real in Task-4 at k=256
  (3-hidden AE): 3L_B, 4L_A and 5L_A all collapsed to 20.00%; 3L_A was correctly
  selected instead. Report MUST state this rather than showing a 20% cell as a
  result - it is an honest negative finding about sigmoid depth, not a defect.

VERIFIED PIPELINE RESULTS (2026-10-08, full run: 60 training runs, all 4
dimensions, RTX 2050, 3.3 min wall-clock):
  Task-1 PCA      best dim 32   test 98.23%  (-0.53 pp vs A3 98.76%)
  Task-3 1-hidden best k=256    test 98.58%  (-0.18 pp)  <- closes most of the gap
  Task-4 3-hidden best k=128    test 98.10%  (-0.66 pp)
  Task-5 denoising best 20%     test 98.52%  (k=256, Task-3's architecture)
  79 figures, no atomic-write failures, no crash
- Note the pilot's timing (36 ep/s for the heaviest AE) was measured on a wider
  architecture than the final one, so it remains conservative.
- Per-class (best config per task, from test confusion matrices):
    Task-1 PCA k=32      0:98.4  4:98.3  5:98.2  6:98.3  7:98.0
    Task-3 1-hidden k=256 0:99.2  4:98.6  5:98.0  6:98.7  7:98.4
    Task-4 3-hidden k=128 0:98.6  4:98.0  5:98.0  6:97.8  7:98.2
    Task-5 20% noise      0:98.8  4:98.8  5:98.2  6:98.4  7:98.4
    Task-5 40% noise      0:99.2  4:98.6  5:97.9  6:98.0  7:98.6
  Digit 5 is consistently the weakest class; the dominant confusion is the
  5<->6 pair (7 errors each direction at Task-3's best), followed by 7<->4 and
  5<->7 at 4 each. This matches A3's reported confusion structure, so the
  reduced representations preserve A3's error profile rather than introducing
  new failure modes - a substantive point for the report's analysis.

Hidden constraints (from Assignment-3 code and context):
- Reproducible weight initialization (seed 42)
- Same initial weights for different optimizers/architectures
- Xavier Glorot Uniform weight initialization
- Full-batch evaluation for test set
- Checkpoint and resume support for long training runs
- Early stopping based on loss convergence (tol=1e-4)

Checkpointing (MANDATORY - highest priority after correctness):
- ALWAYS ON, for EVERY training run, for EVERY task. No exceptions, no opt-in flag.
  There is no "quick mode" that skips checkpointing.
- Cadence: NOT every epoch. Every CHECKPOINT_EVERY = 10 epochs (run_tracker.py), plus
  an unconditional final checkpoint at convergence or max_epochs.
  Rationale: bounds crash loss to 10 epochs of work while avoiding a write every
  single epoch. A per-epoch checkpoint on a fast epoch is pure I/O overhead; for the
  heaviest autoencoders (784 -> 400 -> k -> 400 -> 784) a checkpoint is megabytes,
  so writing one every epoch means hundreds of MB of pointless disk traffic across a
  10k-epoch run. 10 epochs is the compromise: cheap, and bounded loss.
- If a run proves slow enough that 10 epochs is still a meaningful loss of work,
  RAISE the interval for that run rather than lowering it globally. The cost model is
  (time per epoch) x (interval), not the interval itself.
- Do NOT save a checkpoint after every epoch. Cadence exists specifically to avoid
  that I/O. Only the progress line is per-epoch (cheap, append-only text).
- Same rule for the status JSON: it is a small overwrite, but writing it every epoch
  on ~40 runs still adds avoidable disk churn. Throttle it to the checkpoint cadence
  (or every few epochs) rather than every epoch.
- Contents of every checkpoint: run_id, epoch, model_state_dict, optimizer_state_dict,
  start_time (original, preserved across resume so elapsed/ETA stay truthful),
  and the full metric history.
- ATOMIC WRITES ARE MANDATORY: torch.save to "<path>.tmp" then os.replace onto the
  final path. Never write directly to the checkpoint path. A process death during a
  plain torch.save() leaves a truncated file that fails to load on resume - which is
  strictly worse than having no checkpoint, because it looks resumable and is not.
  os.replace is atomic on both Windows and POSIX. Same rule for JSON status files
  (add flush + fsync before the rename).
- MANDATORY RESUME: if a checkpoint exists for a run_id, LOAD IT and continue from
  the stored epoch. Never silently restart a run from epoch 0 when a checkpoint is
  present - that discards days of compute and produces numbers that do not match the
  log.
- On resume: restore model AND optimizer state, and restore start_time from the
  checkpoint. A resumed run must not reset its elapsed clock, or the ETA restarts at
  zero and every subsequent ETA is wrong.
- Checkpoints are also the ERROR-RECOVERY path: after a crash, inspect the last
  checkpoint's history to see how far the run got and what loss/accuracy it had
  reached. This is the primary debugging artifact for a multi-day job.
- Keep the FULL history in the checkpoint (not just the last few points) so curves
  can be replotted after a crash without re-running.
- Every checkpointed run is independently reproducible from its own checkpoint.
  Do not make run N depend on live state from run N-1 in memory.

Progress tracking (MANDATORY):
- Every run reports progress via run_tracker.RunTracker: emits a machine-readable
  "PROGRESS epoch=... wall=... loss=... rate=... eta=..." line to
  <outdir>/logs/<run_id>.log and refreshes <outdir>/status/<run_id>.json.
- monitor.py is READ-ONLY and must never touch the training process. It may be run,
  interrupted, or run from another terminal with zero effect on training.
- ETA MUST use a ROLLING window (RATE_WINDOW = 50 recent epochs), never a whole-run
  mean. Rationale: this is a laptop RTX 2050; sustained load thermally throttles, so
  later epochs are slower than early ones. A mean-rate ETA only ever drifts more
  optimistic, which is backwards. Rolling-window rate makes throttling visible as a
  lengthening ETA while it is still happening.
- monitor.py must detect and display the "snapshot says running BUT the PID is gone"
  case. That is how a crashed or externally-killed run gets noticed.
- monitor.py must surface GPU temperature and flag >= 85C as a throttling warning.

PLOTTING (MANDATORY - but SELECTIVE, not after every run):
- Core principle: a multi-day run can die at any point, so a plot produced ONLY at the
  very end is worthless if the run dies first. BUT plotting every one of ~40 runs
  wastes the very time budget we are protecting. Both are true, so plotting is
  MILESTONE-DRIVEN, not exhaustive.
- DO plot at every MILESTONE - these are the non-negotiable plot points:
  1. Task boundary: any plot that closes out a task's headline result.
  2. Best-run-so-far for a task: the curve for the architecture/dimension that
     currently leads on validation accuracy (re-rendered only when the leader changes).
  3. Anything that feeds a later task's decision (e.g. Task-3's winner, which selects
     the Task-5 bottleneck size).
  4. Final plot at the end of each task, for the selected/best configuration.
- DO NOT plot every intermediate run. A losing architecture at epoch 300 does not need
  its own figure. Its numbers go in a TABLE; the table is the compact record, the plot
  is for the runs you will actually discuss.
- Superset/superseded curves are cheap to REDRAW from checkpoint history later without
  re-running training. So: checkpoint everything, plot the milestones, regenerate the
  rest on demand from history. That is the whole strategy - it removes the
  plot-everything cost while keeping full recoverability.
- A task is DONE only when its milestone plots exist. Individual runs within it need
  only their checkpoints (see Checkpointing), which is what makes selective plotting
  safe.
- Never defer a milestone plot to a single "make all plots" pass at the end of the
  pipeline. Plot at the milestone, then move on.
- Plots are written atomically too (write .png.tmp, then os.replace). A truncated PNG
  from an interrupted save is an unreadable image with a valid-looking name.
- Required plot set (Assignment-3 used a "training error vs epochs" figure per
  architecture with all optimizers superimposed; A4 extends this):
  - Loss / reconstruction-error vs epoch for SELECTED runs, with the early-stopping
    threshold line drawn so convergence is visible, not just asserted.
  - Validation metric vs epoch (accuracy or recon error), train and val on one axes
    where meaningful, so overfitting is visible.
  - Confusion matrix for the best architecture of each task.
  - Reconstruction image grids (original vs reconstructed) for each autoencoder -
    these are required per architecture by the assignment, so they are NOT optional.
  - Maximally-activating-input grids for Task-6 (all three variants - this is a
    comparison figure and is explicitly required).
  - Comparison bar charts across bottleneck sizes (32/64/128/256) per task - these
    answer "which dimension is best" and are the highest-value figures in the report.
    Bar charts are cheap (built from a results table, not from training), so always
    make these.
- Every plot gets a descriptive filename encoding what it shows, written under
  <outdir>/plots/<task>/. Never overwrite a previous step's figure with a bare
  name like "loss.png" - a days-long run whose figures overwrite each other is
  unrecoverable.
- Plots must be regenerable from saved checkpoints/history alone. If a plot cannot be
  rebuilt without re-running training, the pipeline is not resumable.

Long-run budget:
- Keep Assignment-3's stopping tolerance (1e-4) and learning rate (0.001) for
  comparability with the A3 baseline, but apply an explicit per-run MAX_EPOCHS cap.
- Every long run MUST print an epoch rate early (first ~10 epochs) so total cost is
  known before committing to it, not discovered hours in.
- Measure the real epoch rate on this hardware before fixing the cap. Laptop
  throttling means an extrapolated ETA from a short pilot underestimates total time.
- Adam optimizer for autoencoder training (specified in assignment)
- Bottleneck layer always linear (no activation)
- Sigmoid OR tanh consistently (not mixed)
- Reconstruction error computed post-training, not during
- One image per class from train/val/test for reconstruction visuals

Report constraints:
- Tooling: LaTeX, matching Assignment-3's report (pdfTeX source). NOT Word.
- Proper mathematics: numbered display equations (assign each its own \label{eq:...})
  - PCA objective: covariance eigendecomposition, variance retained by top-k components
  - Encoder/decoder mappings for 1-hidden and 3-hidden autoencoders
  - Reconstruction error definition
  - Denoising autoencoder objective
  - Classification cross-entropy (carried over from Assignment-3)
- Notation table early in the report (Assignment-3 used §1.2); one symbol = one meaning,
  reused consistently throughout. Never redefine a symbol mid-report.
- Every figure/table is referenced inline at the exact point it is needed, immediately
  after the sentence that calls for it. Use \ref{fig:...}/\ref{tab:...} cross-references,
  not hardcoded numbers.
- Figures appear visually inline, at the point of discussion, in float order.
- An end-of-report figure gallery archives EVERY image in one place, so "where is the
  image for X?" has a single answer. Gallery is additive, not a replacement for inline
  placement.
- Deep inline analysis is mandatory: any figure/table discussed in body text must be
  visually analysed in that same paragraph, with substantive inference drawn from it.
  Referencing without analysing does not satisfy this.
- The analysis must be genuinely thought through BEFORE the inference is written into
  the report. Do not pattern-match on the numbers and generate plausible-sounding
  commentary; open the image, look at it, and reason from what is actually there.
- Per-element report format, applied to each figure / figure-group / table / element:
    reference  ->  the element (or group)  ->  caption  ->  inference
  Assignment-3's per-figure "Observations (from Figure N):" bullet blocks are the
  precedent; extend them from bullets to substantive prose inference for A4.

Assignment-3 baseline to compare against (from Group11_Assignment3_report.pdf,
the submission of record):
- Best model: A_128_64_32 (784 -> 128 -> 64 -> 32 -> 5) with NAG
- Validation accuracy 98.84%, Test accuracy 98.76%, 18 epochs to converge
- NOTE: Assignment-3/Assignment3/results/convergence_results.csv disagrees
  (best = SGD_Momentum @ 98.68% val, only 3L_A rows present). The PDF report is
  the submission of record; cite the PDF. Do not cite that CSV.

Known errors in the Assignment-3 report (do NOT carry these into A4):
- It states "PyTorch default Kaiming uniform" but models.py uses xavier_uniform_
- It states a 5000 epoch cap but experiment.py sets MAX_EPOCHS = 10000
- Its architecture list has 4 entries (A/B/C/D). The CURRENT models.py defines 6 keys
  (3L_A, 3L_B, 4L_A, 4L_B, 5L_A, 5L_B), and its own docstring and startup print
  both say "5" while returning 6. So models.py has three disagreeing counts:
  report = 4, docstring/print = 5, actual = 6. Settle which set A4 uses and state
  the choice in the report.
- VERIFIED 2026-10-08: models.py was modified at 12:20 (uncommitted; git shows all
  8 A3 files as modified). The architecture list changed during this session, so
  re-verify against the live file rather than trusting any earlier reading of it.
  Current shapes:
    3L_A [128,64,32]      3L_B [256,64,16]
    4L_A [256,128,32,16]  4L_B [128,64,32,16]
    5L_A [256,128,64,32,16]  5L_B [512,256,128,64,32]
  Note 3L_B is [256,64,16] here, NOT [32,32,16] as an earlier revision had.

USE OF ASSIGNMENT-3 CODE (MANDATORY - inherit selectively, never wholesale):
- Reusing A3's constants, dataset conventions and structural approach is EXPLICITLY
  ALLOWED. A4 is a follow-on assignment and is expected to build on A3.
- What is FORBIDDEN is blind copying - dropping A3 files in and editing around them.
  Rationale: A3 has known defects, and inheriting them silently would propagate the
  same mistakes into A4's code and report. Every reused idea must be re-examined,
  not assumed correct.
- A3 PROPERTIES THAT MAY BE INHERITED:
    - Dataset location, split sizes, class set {0,4,5,6,7}
    - Constant VALUES: seed 42, Xavier Glorot uniform init, lr, betas, eps,
      tolerance 1e-4
    - Structural conventions: ImageFolder over class folders, Grayscale -> ToTensor
      -> flatten to 784, FCNN with raw logits + CrossEntropyLoss, tanh-style funnel
      architectures, full-batch val/test evaluation
    - The comparison baseline: 98.84% val / 98.76% test
- A3 KNOWN DEFECTS THAT MUST NOT BE INHERITED (verify each is absent in A4):
    1. Report/code mismatch on weight init: A3's report says "Kaiming uniform"
       but its code uses xavier_uniform_. A4 must describe what its code does.
    2. Report/code mismatch on the epoch cap: report says 5000, code sets 10000.
       A4 must state one number and make the code match the prose.
    3. Undocumented full-batch failure at lr=0.001: A3's Batch GD sits at ~20%
       (chance) while satisfying the 1e-4 stopping rule in 2 epochs, and the A4
       pilot reproduced this at full batch / lr=0.001. A4 must not ship that
       combination, and must not let the stopping rule certify a chance-level model
       as "converged".
    4. Architecture-count drift: A3's report lists 4, its docstring says 5, its code
       returned 6 (and changed mid-session). A4's counts must agree everywhere.
    5. results/ directory is a partial leftover - it disagrees with the submitted
       report. Do not treat any A3 results folder as authoritative.
- A4 FEATURES WITH NO A3 COUNTERPART (must be built, not adapted):
    PCA decomposition, autoencoders (1-hidden and 3-hidden with 400-unit outer
    layers), denoising autoencoders, reconstruction-error reporting, compressed-
    representation extraction, maximally-activating-input visualisation, atomic
    checkpointing, rolling-window ETA, status snapshots, milestone plotting.
- A4's reporting is FAR richer than A3's. A3 produced essentially basic loss-vs-
    epoch plots and confusion matrices. A4 additionally requires per-task
    reconstruction grids, comparison bar charts across bottleneck sizes, denoising
    comparisons, weight-visualisation grids, and a figure gallery. Do not reuse A3's
    plotting scope or structure as-is; A4's figure set is substantially larger and
    the milestone-based plotting policy applies instead of A3's
    "plot at the end" approach.
- Anything reused must be re-derived and re-verified for A4 rather than trusted
  because A3 got it right. Where A4 deliberately differs (sigmoid not tanh,
  classifier lr 0.01, atomic checkpoints, new figure set), say so in the report.

GPU / hardware constraints:
- VERIFIED ENVIRONMENT (do not re-derive, do not assume different):
  GPU: NVIDIA GeForce RTX 2050 (laptop), 4 GB VRAM, compute capability 8.6
  torch 2.6.0+cu124, torchvision 0.21.0+cu124, Python 3.13.14
  sklearn 1.9.1, pandas 3.0.5, numpy 2.5.1, matplotlib 3.11.1,
  seaborn 0.13.2, scipy 1.18.0
  cuda.is_available() == True, device_count == 1
- Device selection is dynamic, never hardcoded:
  device = cuda if torch.cuda.is_available() else cpu, with a working CPU
  fallback path. Print the resolved device, GPU name, CUDA version, and torch
  version at startup, as Assignment-3's experiment.py does.
- Use the GPU for all training and all feature extraction (encoder bottleneck
  outputs, PCA projection, weight-visualization activations).
- fp32 remains the default compute dtype. AMP fp16 is OPT-IN, not automatic.
  Rationale specific to this card: the RTX 2050 is a low-power laptop GPU with
  modest fp16 throughput, and Assignment-3 already observed pathological
  behaviour under fp16 + batch_size=1 (Adam loss spikes to ~6.0 on the wide
  architecture). A4 uses larger batches, so the risk is lower, but fp16 must
  never be silently enabled.
- Where AMP IS enabled: gate it behind a module-level ENABLE_AMP flag, and use
  the GradScaler + isfinite() guard pattern already proven in Assignment-3's
  train.py:49-88 (scaler.step() must skip non-finite batches, not hard-fail).
- NEVER run these in fp16 / under autocast - they must be bit-reproducible fp32:
  - PCA: the covariance matrix and its eigendecomposition
  - mean computation and mean subtraction
  - reconstruction error reporting (the reported numbers)
  - the weight-visualization maximally-activating-input search (argmax ordering)
- VRAM discipline: 4 GB is small. Budget before allocating. Reference figures:
  train set as fp32 = 11,385 x 784 x 4 B ~= 34 MB; val/test ~= 11 MB each;
  whole dataset ~= 56 MB. All three splits fit in VRAM simultaneously with
  enormous headroom, so preload once (as dataset_2.preload_tensors does) rather
  than re-reading JPEGs per epoch.
- Do NOT keep multiple full copies of the dataset alive (e.g. a CPU copy plus a
  GPU copy plus a per-architecture duplicate). Load once, share across all
  architectures and all bottleneck sizes.
- DataLoader: num_workers=0 is REQUIRED on Windows. This is not a performance
  oversight - Assignment-3 recorded Windows shared-memory failures
  ("Couldn't open shared file mapping ... error code 1455") and MemoryError when
  num_workers>0. pin_memory=True is fine and should be enabled when on CUDA.
- Reproducibility on GPU requires more than a seed:
  torch.manual_seed(42) AND, if bitwise determinism is claimed in the report,
  torch.use_deterministic_algorithms(True) plus
  torch.backends.cudnn.deterministic = True / benchmark = False.
  Be precise in the report about which reproducibility level is actually claimed:
  seeding guarantees reproducible init and reproducible shuffling; it does NOT
  guarantee bitwise-identical reductions on GPU. Do not overstate this.
- torch.compile is OPT-IN (ENABLE_COMPILE flag, default off for A4). On Windows
  + sm86 the first call per architecture costs ~20-40 s of JIT. A4 builds many
  models (4 PCA dims x 4 archs, plus 8 autoencoders, plus classifiers), so it can
  amortise - but it must degrade gracefully: wrap in try/except and fall back to
  the eager model on any failure, exactly as Assignment-3's experiment_2.py
  does. Never let a compile failure kill a run.
- Full-batch evaluation for train/val/test is unchanged by GPU use, and is still
  required. Do not switch to minibatched evaluation to "use the GPU better" - it
  would make A4 numbers incomparable with A3.
- When reporting: state the GPU model and the fp32/AMP choice in the report's
  experimental setup section. A grader reproducing this needs to know it was
  run on a 4 GB laptop GPU, not a datacenter card.
- No distributed training, no multi-GPU, no DataParallel. Single device only.