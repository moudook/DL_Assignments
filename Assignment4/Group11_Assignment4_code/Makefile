PYTHON ?= python

.PHONY: help quick train verify verify-claims verify-weights verify-all report check clean

help:
	@echo "quick         - smoke test the whole pipeline (~1 min, numbers meaningless)"
	@echo "train         - the real run (~22 min), writes results_final/"
	@echo "verify        - pipeline invariants: checkpoints, degeneracy, figure set"
	@echo "verify-claims - every number quoted in the report, re-derived from the run"
	@echo "verify-weights - re-predict the test set from stored weights, ignoring the JSON"
	@echo "verify-all    - all of the above"
	@echo "report        - compile report/report.tex twice (cross-refs resolve on pass 2)"
	@echo "check         - compile-check and import-check every module"
	@echo "clean         - remove transient run outputs"

quick:
	$(PYTHON) run_all.py --quick --outdir build/quick

train:
	$(PYTHON) run_all.py --outdir results_final

verify:
	$(PYTHON) scripts/verify.py results_final

verify-claims:
	$(PYTHON) scripts/verify_report_claims.py

verify-weights:
	$(PYTHON) scripts/verify_from_checkpoints.py
	$(PYTHON) scripts/verify_ae_from_checkpoints.py

verify-all: verify verify-claims verify-weights
	$(PYTHON) scripts/verify_independent.py

report:
	cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex

check:
	$(PYTHON) -m compileall -q . && $(PYTHON) -c \
	  "import sys, pathlib, importlib; sys.path.insert(0, '.'); \
	   [importlib.import_module(p.stem) for p in sorted(pathlib.Path('.').glob('*.py'))]; \
	   print('import ok')"

clean:
	rm -rf build
