PYTHON ?= python
export PYTHONPATH := src
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1

.PHONY: install-dev compile test claims drift gate benchmark agent-memory gpu-benchmark holdout autonomy-dry-run build release clean

install-dev:
	$(PYTHON) -m pip install -e '.[dev]'

compile:
	$(PYTHON) -m compileall -q src scripts tests

test:
	$(PYTHON) -m pytest -q

claims:
	$(PYTHON) scripts/autonomy/validate_claims.py

drift:
	$(PYTHON) scripts/autonomy/detect_drift.py

gate: compile test claims drift

benchmark:
	$(PYTHON) scripts/benchmark_fabric.py --threads 1 --output results/fabric_benchmark.json

agent-memory:
	$(PYTHON) scripts/evaluation/agent_memory_benchmark.py --output results/agent_memory_benchmark.json

gpu-benchmark:
	$(PYTHON) scripts/evaluation/benchmark_gpu.py --output results/gpu_benchmark.json

holdout:
	$(PYTHON) scripts/autonomy/holdout_evaluator.py --source src --seed 17072026 --cases 3 --output results/holdout_selftest.json

autonomy-dry-run:
	$(PYTHON) scripts/autonomy/run_codex_loop.py --mode dry-run --max-hypotheses 2 --output results/autonomy_dry_run.json

build:
	$(PYTHON) -m build

release: gate build
	$(PYTHON) scripts/release/build_release.py

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache
	find src scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +
