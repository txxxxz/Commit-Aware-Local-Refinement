# Commit-Aware-Local-Refinement

Clean research code for commit-aware local refinement in discrete diffusion code generation. The repository keeps the historical CLI stable while moving the large evaluation implementation behind smaller eval modules.

## Layout

- `src/decoding/`: risk-aware PF decoder, calibration, local beam, sampler adapters, and risk utilities.
- `src/eval/run_eval.py`: stable compatibility CLI for `python -m eval.run_eval`.
- `src/eval/_runtime.py`: evaluation runtime implementation shared by the compatibility layer.
- `src/eval/io_utils.py`, `postprocess.py`, `visible_tests.py`, `dream_official.py`, `branch_observe.py`, `metrics.py`, `cli.py`: focused eval helper modules.
- `scripts/`: aggregation utilities for experiment summaries.
- `artifacts/final-log/`: curated final-log `summary.json` files only.

## Install

```bash
pip install -r requirements.txt
```

For LLaDA/Dream backends, install optional model dependencies:

```bash
pip install -r requirements.txt -r requirements-llada.txt
```

## Quick Run

```bash
PYTHONPATH=src python -m eval.run_eval --backend placeholder --decoder baseline --max_samples 1
```

Dream official baseline example:

```bash
PYTHONPATH=src python -m eval.run_eval \
  --backend dream \
  --dream_model_path Dream-org/Dream-Coder-v0-Instruct-7B \
  --dream_device cuda \
  --decoder baseline \
  --dataset humaneval \
  --dream_diffusion_steps 768 \
  --dream_max_new_tokens 768
```

## Results

Each run writes `<result_root>/<timestamp>/json/summary.json` and related analysis files. Raw sample details are optional and can be disabled with `--no_output_rawdata`.

The archived final-log content in this repository follows the same policy requested for publication: only `summary.json` files are versioned; raw trace logs and rawdata directories are excluded.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/test_run_eval_outputs.py -q
```

The full test suite may require optional dependencies such as `torch`.
