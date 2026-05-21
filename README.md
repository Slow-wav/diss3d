# diss3d

This repo is the working codebase for a dissertation on **information loss in a `3D -> 2D -> 3D` reconstruction loop**.

The pipeline is intentionally narrow:

**Blender-rendered multi-view images -> Gaussian Splatting -> GauStudio mesh extraction -> quantitative evaluation**

The project is not trying to be a general 3D reconstruction framework. It is a study workspace built around one object family, one reconstruction backend, and a small set of controlled experiment axes.

## Research Question

The main question is:

> How much information is lost when a known 3D object is projected into 2D views and reconstructed back into 3D under different observation conditions?

The final experiment axes are:

- `scale`
- `orbit coverage`
- `view count`
- `image resolution`

## What This Repo Actually Does

At a practical level, the repo supports four recurring tasks:

1. Build controlled view datasets from a canonical Blender-rendered source set.
2. Run one real-view Gaussian Splatting + GauStudio experiment from a concrete config.
3. Evaluate geometry, scale drift, runtime, and image fidelity.
4. Aggregate results into CSV and SQLite outputs for analysis and plotting.

## Final Workflow

The final dissertation workflow is:

1. Start from a canonical real-view dataset for `shark`.
2. Derive study datasets for scale, orbit coverage, view count, and resolution.
3. Expand those study definitions into concrete experiment configs.
4. Run experiments through Gaussian Splatting and GauStudio.
5. Evaluate geometry against the ground-truth mesh.
6. Compare GauStudio image outputs against the reference views for `PSNR` and `SSIM`.
7. Aggregate all experiment summaries into flat tables and a SQLite database.

## Repo Layout

```text
.
├── configs/
│   ├── baseline_realviews_3dgs_gaustudio_whitebg.json
│   ├── shark_orbit_coverage_16x_datasets.json
│   ├── shark_resolution_16x_datasets.json
│   ├── shark_view_count_16x_datasets.json
│   ├── study_orbit_coverage_matrix.json
│   ├── study_resolution_16x_matrix.json
│   ├── study_scale_pilot_matrix.json
│   └── study_view_count_16x_matrix.json
├── data/
│   └── objects/
│       └── shark/
│           ├── ground_truth/
│           ├── input/
│           ├── real_views_master_5ring/
│           ├── real_views_scale8x_master_5ring/
│           ├── real_views_scale16x_master_5ring/
│           └── study_views/
├── modal/
│   ├── gaussian_splatting_app.py
│   └── gaustudio_mesh_app.py
├── results/
│   ├── analysis/
│   └── experiments/
├── scripts/
└── src/
```

## Core Scripts

These are the scripts that matter for the final dissertation workflow:

- `scripts/build_scene_scale_dataset.py`
- `scripts/build_study_view_datasets.py`
- `scripts/build_experiment_matrix.py`
- `scripts/run_real_experiment.py`
- `scripts/compare_render_pairs.py`
- `scripts/aggregate_experiment_metrics.py`

The lower-level wrappers still exist because they are useful for debugging and reruns, but the repo is meant to be read through the study-building and experiment-running scripts above.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install modal
python3 -m modal setup
```

## Main Outputs

The main analysis outputs are:

- `results/analysis/study_summary.csv`
- `results/analysis/experiment_metrics.csv`
- `results/analysis/reconstruction_study.db`
- `results/analysis/dissertation_graphs.ipynb`

The SQLite database contains:

- `experiment_summaries`
- `render_runs`
- `render_frame_metrics`

## Notes

- `LPIPS` is optional and only available if its dependency stack installs cleanly.
- The image-based comparisons in the final workflow use the GauStudio image outputs because they preserve the reconstruction appearance more faithfully than plain mesh re-renders.
- CloudCompare and MeshLab were used as supporting qualitative inspection tools, not as the primary automated evaluation pipeline.
