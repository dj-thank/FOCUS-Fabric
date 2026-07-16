# Reproducibility

## Reference environment

The committed evidence was generated on Linux with Python 3.13.5, PyTorch 2.10.0+cpu, one CPU thread, no CUDA, and no Triton installation. Exact timing is machine-specific; accuracy artifacts use deterministic seeds.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## Full local gate

```bash
make gate
```

This compiles Python sources, runs unit tests, verifies claim artifact hashes/paths, and scans documentation/evidence drift.

## Rebuild evidence

```bash
make benchmark
make agent-memory
make gpu-benchmark
```

`make benchmark` overwrites `results/fabric_benchmark.json`, CSV, and PNG. Runtime fields will differ. Accuracy fields should be close or identical for the reference software stack. After intentionally accepting new evidence, refresh claim hashes:

```bash
python scripts/autonomy/validate_claims.py --refresh-digests
python scripts/autonomy/validate_claims.py
```

Do not refresh hashes merely to make a failing gate pass. Review metric changes and update `docs/CLAIMS.md` first.

## Randomized holdout

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/autonomy/holdout_evaluator.py \
  --source src --seed 17072026 --cases 3 \
  --output results/holdout_selftest.json
```

The release also retains seeds `314159` and `271828`; `results/randomized_holdout_suite.json` aggregates all three raw runs without replacing them.

Autonomous execute mode chooses a new seed after the code change and compares root/candidate source on identical cases. The preregistered version-2 policy rejects schema/seed/case mismatches, any paired case above its regression ceiling, and candidates that have no measurable effect in the randomized cases.

## Archived checkpoint integrity

The public GitHub source does not redistribute the historical `model.safetensors` files because their original training-data and redistribution provenance is incomplete. Their expected byte sizes and SHA-256 digests are recorded in `checkpoints/README.md`. In a controlled environment where an authorized local copy is available, place it at the documented path and run:

```bash
python -c "from focus_native.io import load_checkpoint; print(load_checkpoint('checkpoints/focus-native-small')[2])"
```

Checkpoint-specific integration tests skip cleanly when those optional local artifacts are absent.

## External pretrained-model Q/K/V traces

Install optional Hugging Face dependencies and force a PyTorch SDPA forward. The collector records Q/K/V at the actual SDPA boundary, after architecture-specific positional transforms. Architectures that bypass this boundary fail explicitly.

```bash
pip install -e '.[hf]'
python scripts/evaluation/collect_hf_sdpa_traces.py \
  --model /path/to/version-pinned-model \
  --text-file /path/to/prompt.txt \
  --device cuda \
  --output results/external/model-traces.safetensors
```

Record the model revision, weight hashes, tokenizer revision, prompt hash, hardware, and Transformers/PyTorch versions before using the trace in a public comparison.

## Official datasets

Datasets are not downloaded automatically. Use local, version-pinned files and record their hashes. Example:

```bash
python scripts/evaluation/run_official_benchmarks.py   --suite ruler   --data /data/ruler.jsonl   --backend-command '/path/to/backend --json-stdio'   --output results/ruler_predictions.json
```

Then invoke the suite's official scorer. Keep generic diagnostics separate from official scores.

## GPU

```bash
python scripts/evaluation/benchmark_gpu.py --output results/gpu_benchmark.json
```

Without CUDA/Triton it emits a valid `not_executed` record. With GPU it checks fused/reference numerical error and event latency. Physical HBM must be collected separately with Nsight Compute; the script never labels estimated tensor traffic as a hardware counter.

## Determinism caveats

K-means and SVD/eigendecomposition can vary across BLAS/GPU stacks. Query-aware restarts reduce initialization sensitivity but do not make all low-level linear algebra bitwise deterministic. Compare tolerances and aggregate evidence rather than blindly requiring byte-identical JSON across platforms.
