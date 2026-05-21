# Methodology Note

This repo ended up narrower than it first started, and that is deliberate.

The final version of the project is built around one practical research loop:

1. Start with a known 3D object rendered from Blender.
2. Treat the rendered views as the controlled 2D observation set.
3. Reconstruct with Gaussian Splatting.
4. Extract a mesh with GauStudio.
5. Measure what was preserved and what was lost.

The code is organised around the four experiment axes used in the dissertation:

- `scale`
- `orbit coverage`
- `view count`
- `image resolution`

The design is intentionally study-first rather than framework-first.

That is why the repo contains:

- study dataset builders
- matrix-to-config expansion
- one main experiment runner
- one evaluation and aggregation path

and not a large number of interchangeable reconstruction backends or generation branches.

The most important outputs are not the intermediate manifests, but the analysis records and the final aggregated tables:

- `results/analysis/study_summary.csv`
- `results/analysis/experiment_metrics.csv`
- `results/analysis/reconstruction_study.db`

Image-based metrics use the GauStudio image outputs rather than plain mesh re-renders wherever possible, because those images preserve the appearance of the reconstruction more directly than a simplified surface render.

This is a research codebase rather than a reusable software package. The structure is meant to make the final dissertation experiments easy to rerun, inspect, and explain.
