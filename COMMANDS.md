# diss3d Commands

Run commands from the repo root:

```bash
cd diss3d
```

This file only lists the commands used in the final dissertation workflow.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install modal
python3 -m modal setup
```

## Build The Canonical Scaled Datasets

Build the `8x` and `16x` scene-scale variants from the `1x` master source:

```bash
python3 scripts/build_scene_scale_dataset.py \
  --source-dir data/objects/shark/real_views_master_5ring \
  --output-dir data/objects/shark/real_views_scale8x_master_5ring \
  --scale 8 \
  --clean

python3 scripts/build_scene_scale_dataset.py \
  --source-dir data/objects/shark/real_views_master_5ring \
  --output-dir data/objects/shark/real_views_scale16x_master_5ring \
  --scale 16 \
  --clean
```

## Build The Study Datasets

Orbit coverage:

```bash
python3 scripts/build_study_view_datasets.py \
  --plan configs/shark_orbit_coverage_16x_datasets.json
```

View count:

```bash
python3 scripts/build_study_view_datasets.py \
  --plan configs/shark_view_count_16x_datasets.json
```

Resolution:

```bash
python3 scripts/build_study_view_datasets.py \
  --plan configs/shark_resolution_16x_datasets.json
```

## Build The Experiment Configs

Scale:

```bash
python3 scripts/build_experiment_matrix.py \
  --matrix configs/study_scale_pilot_matrix.json
```

Orbit coverage:

```bash
python3 scripts/build_experiment_matrix.py \
  --matrix configs/study_orbit_coverage_matrix.json
```

View count:

```bash
python3 scripts/build_experiment_matrix.py \
  --matrix configs/study_view_count_16x_matrix.json
```

Resolution:

```bash
python3 scripts/build_experiment_matrix.py \
  --matrix configs/study_resolution_16x_matrix.json
```

## Run One Experiment

```bash
python3 scripts/run_real_experiment.py \
  --config configs/generated/scale_pilot/shark_gs_scale16x_150v_5orbit_768.json
```

Other examples:

```bash
python3 scripts/run_real_experiment.py \
  --config configs/generated/orbit_coverage/shark_gs_orbits5_views30_768_16x.json

python3 scripts/run_real_experiment.py \
  --config configs/generated/view_count/shark_gs_views100_5orbit_768_16x.json

python3 scripts/run_real_experiment.py \
  --config configs/generated/resolution/shark_gs_res512_150v_5orbit_16x.json
```

## Compare GauStudio Images Against Reference Views

```bash
python3 scripts/compare_render_pairs.py \
  --experiment_id shark_gs_scale16x_150v_5orbit_768 \
  --reference_dir data/objects/shark/real_views_scale16x_master_5ring \
  --predicted_dir results/experiments/shark_gs_scale16x_150v_5orbit_768/mesh/gaustudio_mesh/images \
  --analysis_record results/experiments/shark_gs_scale16x_150v_5orbit_768/evaluation/analysis_record.json \
  --disable_lpips
```

## Aggregate All Experiment Records

```bash
python3 scripts/aggregate_experiment_metrics.py
```

Main outputs:

- `results/analysis/study_summary.csv`
- `results/analysis/experiment_metrics.csv`
- `results/analysis/reconstruction_study.db`
- `results/analysis/dissertation_graphs.ipynb`

## Useful Help Commands

```bash
python3 scripts/build_scene_scale_dataset.py --help
python3 scripts/build_study_view_datasets.py --help
python3 scripts/build_experiment_matrix.py --help
python3 scripts/run_real_experiment.py --help
python3 scripts/compare_render_pairs.py --help
python3 scripts/aggregate_experiment_metrics.py --help
```
