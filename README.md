# Reduce-then-Optimize for the Capacitated Vehicle Routing Problem (CVRP / CVRPTW)

This repository contains the code to train a GNN-based arc predictor and run a reduce-then-optimize pipeline for CVRP (and the time-windowed variant CVRPTW). The decoder backend is either Gurobi (exact) or HGS via PyVRP.

## Local Setup

A conda env named `cvrp` is assumed (any env with PyTorch, PyTorch Geometric, PyVRP, Gurobi, Hydra, and W&B installed works).

To execute any script, run from the project root and export the repo root onto `PYTHONPATH`:
```bash
PYTHONPATH=$PWD python scripts/<example_script.py>
```

W&B is opt-in via the `wandb_use` config flag — set `wandb_use=false` if you don't have an API key.

Note: If Gurobi is run with a commercial or academic license, ensure your license file is discoverable (default: `$HOME/gurobi.lic`).

## Step 1: Data Generation

The active routine in `02_generate_samples.py` re-solves an existing CVRP `.pkl.gz` sample folder with a shorter HGS budget and writes the new solutions to an output directory (auto-created with `exist_ok=True`).

Example:
```bash
# Re-solve a slice of existing CVRP samples with a shorter HGS budget
PYTHONPATH=$PWD python scripts/01_data/02_generate_samples.py \
  --config-path "$PWD/configs/training" --config-name config \
  samples_input_dir=data/samples_Munich_100 \
  samples_output_dir=data/samples_generated \
  regul_lambda=1 \
  num_samples=0 \
  max_iterations_FW=5
```

I/O paths come from `configs/training/config.yaml` (`samples_input_dir`, `samples_output_dir`). The slice is controlled by `num_samples` (start) and `max_iterations_FW` (end); `regul_lambda` is the HGS time budget in seconds.

To switch routines (e.g. `generate_cvrptw_restricted`, `generate_directed_solution_TW`, `generate_undirected_solution`, `generate_slice_bigger_instances`), edit the active call inside `main()` in `scripts/01_data/02_generate_samples.py`.

---
**NOTE**
Use different seeds when generating training and benchmarking instances to prevent leakage.

---

## Step 2: Training and Model Selection

Example:
```bash
PYTHONPATH=$PWD python scripts/02_training_and_evaluation/01_train_sol_edge_predictor.py \
  --config-path "$PWD/configs/training" --config-name config \
  data_path=data/samples_Munich_100 \
  validation_data_path=data/samples_Munich_100_test_cluster \
  model=gcnn \
  model.num_conv_layers=6 \
  model.num_dense_layers=2 \
  model.hidden_layer_dim=20 \
  out_dir=trained_models \
  seed=0 \
  cross_validate=true \
  num_samples=20 \
  train_batch_size=10 \
  directed=false \
  is_time_windows=false \
  project_name=my_cvrp_run \
  wandb_use=false
```

Key flags introduced for this pipeline:
- `directed` — `True`: asymmetric GNN message passing; `False`: symmetric (use for undirected CVRP)
- `is_time_windows` — `True` for CVRPTW, `False` for CVRP
- `wandb_use` — gate W&B (`wandb.login()` only runs if `True`)
- `project_name` — W&B project name (when `wandb_use=true`)

`cross_validate=true` uses file-based K-fold (5 folds) over `.pt` feature files under `train_pt/` or the configured features directory; `cross_validate=false` does a single train/val split.

## Step 3: Benchmarking

The reduce-then-optimize pipeline uses a trained checkpoint to reduce the arc set per instance, then solves the reduced problem with either an exact decoder or HGS.

Example (ML reduce + HGS decoder):
```bash
MODEL=trained_models/model_gcnn_features_graph_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_conv_hidden_layer_dim_20_num_conv_layers_6_num_dense_layers_2/application/best_checkpoint.pth.tar

PYTHONPATH=$PWD python scripts/03_benchmarking/01_run_benchmarking_experiments.py \
  --config-path "$PWD/configs/benchmarking" --config-name config \
  decoder=hgs \
  method=ml-reduction \
  method.model_path=$MODEL \
  method.model_name=my_gnn \
  instance_dir=data/samples_Munich_100 \
  solution_dir=benchmarking/output \
  start=0 end=49 \
  HGS_runtime=5 heu_time=5 \
  'method.size_threshold=[0.99,0.5,0.3,0.1,0.05]'
```

Decoder groups available: `exact`, `hgs`, `lp`, `ts`, `ea` (see `configs/benchmarking/decoder/`).

Decoder/reduction settings now live in `configs/benchmarking/config.yaml` (`pyvrp_version`, `heu_time`, `threshold_type`, `top_k_eval`, `size_threshold`, `probability_threshold`, `instance_dir`, `solution_dir`), independent of the training config.

Example (exact decoder):
```bash
PYTHONPATH=$PWD python scripts/03_benchmarking/01_run_benchmarking_experiments.py \
  --config-path "$PWD/configs/benchmarking" --config-name config \
  decoder=exact \
  method=exact \
  instance_dir=data/samples_Munich_100 \
  solution_dir=benchmarking/output_exact \
  start=0 end=49 \
  num_threads=1 method.grb_timeout=60
```

## Step 4: Analyses and Visualizations

`scripts/04_analyses/visualization.py` is a library of plotting routines. Because the directory name starts with a digit, import it via `importlib` rather than `import`:

```bash
PYTHONPATH=$PWD python - <<'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("viz", "scripts/04_analyses/visualization.py")
viz = importlib.util.module_from_spec(spec); spec.loader.exec_module(viz)

# Optimality gap per instance, one subplot per method (threshold subdir)
viz.generate_benchmark_plot(
    "benchmarking/output/ml-reduction/size/hgs-heu_time_100/my_gnn"
)

# Runtime vs missing-arc curve (no bars)
viz.generate_runtime_missing_arc_plot(
    "benchmarking/output/ml-reduction/size/hgs-heu_time_100/my_gnn"
)

# Same curve with bar overlay variant
viz.generate_runtime_missing_arc_plot_with_bars(
    "benchmarking/output/ml-reduction/size/hgs-heu_time_100/my_gnn"
)
EOF
```

The `cost_dict` (per-instance baseline) is now built on the fly from each result pickle's `exact_objective_value`; no external `cost_dict.pkl` aggregator is required.

## Additional Experiments and Analyses

### Hyperparameter Screening
Override default training values via CLI to sweep hyperparameters (`model.num_conv_layers`, `model.hidden_layer_dim`, `learning_rate`, `train_batch_size`, ...). For overfitting assessment, disable LR decay and early stopping (`lr_decay=false early_stopping=10000 max_num_epochs=200`).

### PyVRP backend switch
Set `pyvrp_version=old` (default, uses `pyvrp` package) or `pyvrp_version=new` (uses `PyVRP.pyvrp`). The new backend is loaded lazily inside the heuristics dispatcher so the module imports cleanly when only one backend is installed.

## Add Problem Variants
To extend to new problem variants, adjust the following:

**Step 1: Instances and Samples**
* Add class to describe instance in `core/utils/cvrp.py`.
* Implement MIP formulation as Gurobi model in `core/cvrp_solvers/ip_grb.py`.
* Implement instance generation in `scripts/01_data/02_generate_samples.py`.
* Extend instance loader in `core/data_processing/data_utils.py`.

**Step 2: Features, ML Model, and Training**
* Adjust features (`core/utils/ml_utils.py`) and (if necessary) ML models (`core/ml_models/gnn.py`, `core/ml_models/cvrp_sol_predictor.py`) to represent the new problem variant.
* Extend the training wrapper in `core/utils/ml_utils.py` if new metrics/branches are required.
* Adjust the training script in `scripts/02_training_and_evaluation/01_train_sol_edge_predictor.py`.

**Step 3: Benchmarking**
* Adjust `scripts/03_benchmarking/01_run_benchmarking_experiments.py` to include solvers for the new variant.
* Extend the reduce-then-optimize wrapper in `core/ml_models/wrapper.py` (decoder dispatch, completion heuristic, `pyvrp_version` propagation).
* Add benchmarking config groups under `configs/benchmarking/method/` and `configs/benchmarking/decoder/` as needed.

## Pipeline Smoke Test (≈1 minute)

End-to-end sanity check on a small Munich slice. Useful after pulling changes.

```bash
cd /path/to/cvrp_reduce_then_optimize
export PYTHONPATH=$PWD

# 1) Generate (~10 s)
python scripts/01_data/02_generate_samples.py \
  --config-path "$PWD/configs/training" --config-name config \
  samples_input_dir=data/samples_Munich_100 \
  samples_output_dir=data/samples_smoke_out \
  regul_lambda=1 num_samples=0 max_iterations_FW=5

# 2) Train (~20 s, early stops)
python scripts/02_training_and_evaluation/01_train_sol_edge_predictor.py \
  --config-path "$PWD/configs/training" --config-name config \
  data_path=data/samples_Munich_100 \
  validation_data_path=data/samples_Munich_100_test_cluster \
  model=gcnn model.num_conv_layers=6 model.num_dense_layers=2 model.hidden_layer_dim=20 \
  out_dir=trained_models_smoke seed=0 cross_validate=false \
  num_samples=20 train_batch_size=10 \
  directed=false is_time_windows=false project_name=smoke_test wandb_use=false

# 3) Benchmark (~15 s)
MODEL=trained_models_smoke/model_gcnn_features_graph_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_conv_hidden_layer_dim_20_num_conv_layers_6_num_dense_layers_2/application/best_checkpoint.pth.tar

python scripts/03_benchmarking/01_run_benchmarking_experiments.py \
  --config-path "$PWD/configs/benchmarking" --config-name config \
  decoder=hgs method=ml-reduction \
  method.model_path=$MODEL \
  method.model_name=smoke_test_model \
  instance_dir=data/samples_Munich_100 \
  solution_dir=benchmarking_smoke/output \
  start=0 end=4 HGS_runtime=2 heu_time=2

# 4) Visualize
python - <<'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("viz", "scripts/04_analyses/visualization.py")
viz = importlib.util.module_from_spec(spec); spec.loader.exec_module(viz)
viz.generate_benchmark_plot(
    "benchmarking_smoke/output/ml-reduction/size/hgs-heu_time_100/smoke_test_model"
)
viz.generate_runtime_missing_arc_plot(
    "benchmarking_smoke/output/ml-reduction/size/hgs-heu_time_100/smoke_test_model"
)
viz.generate_runtime_missing_arc_plot_with_bars(
    "benchmarking_smoke/output/ml-reduction/size/hgs-heu_time_100/smoke_test_model"
)
EOF
```
