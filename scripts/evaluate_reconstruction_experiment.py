#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import (
    analyze_transforms_json,
    build_default_research_questions,
    expected_artifact_paths,
    list_image_files,
    load_real_experiment_config,
    resolve_experiment_root,
    resolve_study_parameters,
    stage_evaluation_paths,
    write_json,
)
from utils.evaluation_metrics import (
    build_efficiency_metrics,
    compute_geometry_metrics,
    compute_render_metrics,
    find_first_existing,
    read_optional_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the main geometry, scale, coverage, and runtime outputs for one dissertation experiment."
    )
    parser.add_argument("--config", required=True, help="Path to a real experiment config JSON.")
    parser.add_argument(
        "--surface_sample_count",
        type=int,
        default=10000,
        help="How many surface samples to use when computing geometry metrics.",
    )
    parser.add_argument(
        "--completeness_threshold_ratio",
        type=float,
        default=0.01,
        help="Distance threshold ratio, relative to scene bounding-box diagonal, used for completeness/accuracy.",
    )
    return parser.parse_args()


def count_pngs(directory: Path) -> int | None:
    if not directory.exists():
        return None
    return len([path for path in list_image_files(directory, recursive=True) if path.suffix.lower() == ".png"])


def build_metric_groups(config: dict[str, Any], artifacts: dict[str, Path]) -> dict[str, Any]:
    metrics = config["evaluation"].get("metrics", {})
    mesh_path = artifacts.get("mesh")
    mesh_ready = mesh_path is not None and mesh_path.exists() and artifacts["ground_truth_mesh"].exists()
    return {
        "novel_view_render": {
            "metrics": metrics.get("novel_view_render", ["psnr", "ssim", "lpips"]),
            "status": "planned",
            "notes": "Requires rendered evaluation images from Gaussian Splatting and reference views matched by filename.",
        },
        "geometry_and_mesh": {
            "metrics": metrics.get(
                "geometry_and_mesh",
                ["chamfer_distance", "rmse", "hausdorff_distance", "mesh_completeness", "mesh_accuracy", "f1_score", "normal_consistency"],
            ),
            "status": "ready" if mesh_ready else "blocked",
            "notes": "Requires a predicted mesh and a ground-truth reference mesh.",
        },
        "scale_and_alignment": {
            "metrics": metrics.get(
                "scale_and_alignment",
                ["scale_factor_error", "bbox_diagonal_ratio", "center_offset", "dimensional_accuracy"],
            ),
            "status": "ready" if mesh_ready else "blocked",
            "notes": "Tracks size drift, extent distortion, and centroid offset relative to ground truth.",
        },
        "efficiency": {
            "metrics": metrics.get(
                "efficiency",
                ["input_preparation_runtime", "reconstruction_runtime", "meshing_runtime", "total_runtime"],
            ),
            "status": "partial",
            "notes": "Use runtime together with quality thresholds when analysing convergence and stability.",
        },
        "qualitative": {
            "metrics": metrics.get(
                "qualitative",
                ["collapse", "fragmentation", "blobs", "surface_artifacts", "scale_instability"],
            ),
            "status": "ready",
            "notes": "Use this checklist during CloudCompare or MeshLab inspection of meshes and render outputs.",
        },
    }


def build_input_stage_metrics(config: dict[str, Any], artifacts: dict[str, Path]) -> dict[str, Any]:
    measured_coverage = analyze_transforms_json(artifacts["transforms_json"])
    view_count = measured_coverage["actual_frame_count"] or count_pngs(artifacts["dataset_dir"])
    discovered_png_count = count_pngs(artifacts["dataset_dir"])
    input_manifest = read_optional_json(find_first_existing([
        artifacts.get("generation_modal_manifest"),
        artifacts.get("generation_manifest"),
    ]))
    input_runtime = build_efficiency_metrics(input_manifest, None, None)["input_preparation_runtime"]
    return {
        "stage": "input_observation",
        "input_mode": config["input_condition"]["mode"],
        "dataset_dir": str(artifacts["dataset_dir"]),
        "view_count": view_count,
        "discovered_png_count_recursive": discovered_png_count,
        "measured_coverage": measured_coverage,
        "study_parameters": resolve_study_parameters(config),
        "transforms_json_exists": artifacts["transforms_json"].exists(),
        "metric_plan": {
            "reference_alignment": ["psnr", "ssim", "lpips"],
            "cross_view_consistency": ["adjacent_view_consistency", "qualitative_drift_notes"],
            "coverage": [
                "actual_frame_count",
                "actual_orbit_count",
                "distinct_elevation_bands",
                "azimuth_coverage_span_deg",
                "per_orbit_frame_counts",
            ],
        },
        "readiness": "partial" if input_manifest is not None else "baseline_only",
        "notes": "This stage records the controlled 2D observation setting before Gaussian Splatting reconstruction.",
        "manifest": input_manifest,
        "runtime_seconds": input_runtime,
    }


def build_pose_stage_metrics(config: dict[str, Any], artifacts: dict[str, Path]) -> dict[str, Any]:
    measured_coverage = analyze_transforms_json(artifacts["transforms_json"])
    return {
        "stage": "pose_assignment",
        "pose_regime": "known_blender_orbit",
        "transforms_json": str(artifacts["transforms_json"]),
        "transforms_json_exists": artifacts["transforms_json"].exists(),
        "measured_coverage": measured_coverage,
        "metric_plan": {
            "pose_source": ["known", "native_assigned", "future_pose_estimation_check"],
            "consistency": [
                "view_count_with_transforms",
                "distinct_elevation_bands",
                "azimuth_coverage_span_deg",
                "future_colmap_alignment_error",
            ],
        },
        "readiness": "ready" if artifacts["transforms_json"].exists() else "blocked",
        "notes": "Tracks the camera-pose assumptions used by the reconstruction pipeline.",
    }


def build_reconstruction_stage_metrics(config: dict[str, Any], artifacts: dict[str, Path]) -> dict[str, Any]:
    reconstruction_manifest_path = find_first_existing([
        artifacts.get("reconstruction_modal_manifest"),
        artifacts.get("reconstruction_manifest"),
    ])
    reconstruction_manifest = read_optional_json(reconstruction_manifest_path)
    backend = config["reconstruction"]["backend"]
    reconstruction_runtime = build_efficiency_metrics(None, reconstruction_manifest, None)["reconstruction_runtime"]
    return {
        "stage": "reconstruction",
        "backend": backend,
        "manifest_path": str(reconstruction_manifest_path) if reconstruction_manifest_path is not None else None,
        "manifest_exists": reconstruction_manifest_path.exists() if reconstruction_manifest_path is not None else False,
        "metric_plan": {
            "render_quality": ["psnr", "ssim", "lpips"],
            "efficiency": ["reconstruction_runtime", "iteration_count", "completion_status"],
            "failure_modes": ["floaters", "fragmentation", "underfitting", "pose_sensitivity_notes"],
        },
        "readiness": "partial" if reconstruction_manifest is not None else "blocked",
        "notes": "This stage records Gaussian Splatting training status, runtime, and reconstruction-side failure modes.",
        "manifest": reconstruction_manifest,
        "runtime_seconds": reconstruction_runtime,
    }


def build_mesh_stage_metrics(
    config: dict[str, Any],
    artifacts: dict[str, Path],
    geometry_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    mesh_path = artifacts.get("mesh")
    mesh_exists = mesh_path.exists() if mesh_path is not None else False
    meshing_manifest_path = find_first_existing([
        artifacts.get("meshing_modal_manifest"),
        artifacts.get("meshing_manifest"),
    ])
    meshing_manifest = read_optional_json(meshing_manifest_path)
    meshing_runtime = build_efficiency_metrics(None, None, meshing_manifest)["meshing_runtime"]
    return {
        "stage": "meshing",
        "backend": config["meshing"]["backend"],
        "mesh_path": str(mesh_path) if mesh_path is not None else None,
        "mesh_exists": mesh_exists,
        "ground_truth_mesh": str(artifacts["ground_truth_mesh"]),
        "ground_truth_mesh_exists": artifacts["ground_truth_mesh"].exists(),
        "manifest_path": str(meshing_manifest_path) if meshing_manifest_path is not None else None,
        "manifest": meshing_manifest,
        "runtime_seconds": meshing_runtime,
        "metric_plan": {
            "geometry": ["chamfer_distance", "mesh_completeness", "normal_consistency"],
            "topology": ["vertex_count", "face_count", "connected_components"],
            "qualitative": ["holes", "oversmoothing", "spurious_surfaces", "surface_noise"],
        },
        "readiness": "ready" if mesh_exists and artifacts["ground_truth_mesh"].exists() else "blocked",
        "notes": "This stage exposes geometric loss, scale drift, and alignment drift after mesh extraction.",
        "geometry_metrics": geometry_metrics,
    }


def compute_quantitative_metrics(
    config: dict[str, Any],
    artifacts: dict[str, Path],
    surface_sample_count: int,
    completeness_threshold_ratio: float,
) -> dict[str, Any]:
    mesh_path = artifacts.get("mesh")
    geometry_metrics = None
    if mesh_path is not None and mesh_path.exists() and artifacts["ground_truth_mesh"].exists():
        geometry_metrics = compute_geometry_metrics(
            predicted_mesh_path=mesh_path,
            ground_truth_mesh_path=artifacts["ground_truth_mesh"],
            sample_count=surface_sample_count,
            completeness_threshold_ratio=completeness_threshold_ratio,
        )

    render_metrics = {
        "available": False,
        "reference_dir": None,
        "predicted_dir": None,
        "matched_image_count": 0,
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "notes": "No render metric directories configured yet.",
    }
    render_config = config["evaluation"].get("novel_view_render_dirs")
    if isinstance(render_config, dict):
        reference_dir_value = render_config.get("reference_dir")
        predicted_dir_value = render_config.get("predicted_dir")
        if reference_dir_value and predicted_dir_value:
            reference_dir = Path(reference_dir_value)
            predicted_dir = Path(predicted_dir_value)
            if not reference_dir.is_absolute():
                reference_dir = (PROJECT_ROOT / reference_dir).resolve()
            if not predicted_dir.is_absolute():
                predicted_dir = (PROJECT_ROOT / predicted_dir).resolve()
            render_metrics = compute_render_metrics(reference_dir, predicted_dir)

    input_manifest = read_optional_json(find_first_existing([
        artifacts.get("generation_modal_manifest"),
        artifacts.get("generation_manifest"),
    ]))
    reconstruction_manifest = read_optional_json(find_first_existing([
        artifacts.get("reconstruction_modal_manifest"),
        artifacts.get("reconstruction_manifest"),
    ]))
    meshing_manifest = read_optional_json(find_first_existing([
        artifacts.get("meshing_modal_manifest"),
        artifacts.get("meshing_manifest"),
    ]))
    efficiency_metrics = build_efficiency_metrics(input_manifest, reconstruction_manifest, meshing_manifest)

    return {
        "experiment_id": config["experiment_id"],
        "object_id": config["object_id"],
        "input_condition": config["input_condition"]["mode"],
        "reconstruction_backend": config["reconstruction"]["backend"],
        "meshing_backend": config["meshing"]["backend"],
        "study_parameters": resolve_study_parameters(config),
        "geometry_and_mesh": geometry_metrics,
        "novel_view_render": render_metrics,
        "efficiency": efficiency_metrics,
    }


def build_analysis_record(
    config: dict[str, Any],
    artifacts: dict[str, Path],
    stage_payloads: dict[str, dict[str, Any]],
    quantitative_metrics: dict[str, Any],
) -> dict[str, Any]:
    geometry = quantitative_metrics.get("geometry_and_mesh") or {}
    render_metrics = quantitative_metrics.get("novel_view_render") or {}
    efficiency = quantitative_metrics.get("efficiency") or {}
    mesh_path = artifacts.get("mesh")
    study_parameters = resolve_study_parameters(config)
    measured_coverage = stage_payloads["input_stage"].get("measured_coverage") or {}
    return {
        "experiment_id": config["experiment_id"],
        "object_id": config["object_id"],
        "input_condition": config["input_condition"]["mode"],
        "reconstruction_backend": config["reconstruction"]["backend"],
        "meshing_backend": config["meshing"]["backend"],
        "reconstruction_model": study_parameters.get("reconstruction_model"),
        "study_view_count": study_parameters.get("view_count"),
        "study_number_of_orbits": study_parameters.get("number_of_orbits"),
        "study_elevation_coverage": study_parameters.get("elevation_coverage"),
        "study_image_resolution": study_parameters.get("image_resolution"),
        "study_scale": study_parameters.get("scale"),
        "control_experiment_id": config["evaluation"].get("control_experiment_id"),
        "dataset_dir": str(artifacts["dataset_dir"]),
        "view_count": stage_payloads["input_stage"].get("view_count"),
        "actual_view_count": measured_coverage.get("actual_frame_count"),
        "actual_orbit_count": measured_coverage.get("actual_orbit_count"),
        "measured_elevation_bands_deg": measured_coverage.get("elevation_bands_deg"),
        "measured_min_elevation_deg": measured_coverage.get("min_elevation_deg"),
        "measured_max_elevation_deg": measured_coverage.get("max_elevation_deg"),
        "measured_azimuth_coverage_span_deg": measured_coverage.get("azimuth_coverage_span_deg"),
        "measured_per_orbit_frame_counts": measured_coverage.get("per_orbit_frame_counts"),
        "transforms_json_exists": stage_payloads["pose_stage"].get("transforms_json_exists"),
        "mesh_path": str(mesh_path) if mesh_path is not None else None,
        "mesh_exists": mesh_path.exists() if mesh_path is not None else False,
        "mesh_source_backend": config["meshing"]["backend"],
        "mesh_has_geometry": geometry.get("mesh_has_geometry"),
        "ground_truth_mesh": str(artifacts["ground_truth_mesh"]),
        "ground_truth_mesh_exists": artifacts["ground_truth_mesh"].exists(),
        "chamfer_distance": geometry.get("chamfer_distance"),
        "rmse": geometry.get("rmse"),
        "hausdorff_distance": geometry.get("hausdorff_distance"),
        "mesh_completeness": geometry.get("mesh_completeness"),
        "mesh_accuracy": geometry.get("mesh_accuracy"),
        "f1_score": geometry.get("f1_score"),
        "normal_consistency": geometry.get("normal_consistency"),
        "scale_factor_error": geometry.get("scale_factor_error"),
        "bbox_diagonal_ratio": geometry.get("bbox_diagonal_ratio"),
        "center_offset": geometry.get("center_offset"),
        "extent_ratio_x": ((geometry.get("dimensional_accuracy") or {}).get("axis_ratio") or {}).get("x"),
        "extent_ratio_y": ((geometry.get("dimensional_accuracy") or {}).get("axis_ratio") or {}).get("y"),
        "extent_ratio_z": ((geometry.get("dimensional_accuracy") or {}).get("axis_ratio") or {}).get("z"),
        "vertex_count": (geometry.get("predicted_topology") or {}).get("vertex_count"),
        "face_count": (geometry.get("predicted_topology") or {}).get("face_count"),
        "connected_components": (geometry.get("predicted_topology") or {}).get("connected_components"),
        "bounding_box_diagonal": (geometry.get("predicted_topology") or {}).get("bounding_box_diagonal"),
        "render_metrics_available": render_metrics.get("available"),
        "matched_render_count": render_metrics.get("matched_image_count"),
        "psnr": render_metrics.get("psnr"),
        "ssim": render_metrics.get("ssim"),
        "lpips": render_metrics.get("lpips"),
        "input_preparation_runtime": efficiency.get("input_preparation_runtime"),
        "reconstruction_runtime": efficiency.get("reconstruction_runtime"),
        "meshing_runtime": efficiency.get("meshing_runtime"),
        "total_runtime": efficiency.get("total_runtime"),
        "input_stage_readiness": stage_payloads["input_stage"]["readiness"],
        "pose_stage_readiness": stage_payloads["pose_stage"]["readiness"],
        "reconstruction_stage_readiness": stage_payloads["reconstruction_stage"]["readiness"],
        "mesh_stage_readiness": stage_payloads["mesh_stage"]["readiness"],
        "analysis_version": 3,
    }


def build_experiment_summary(
    config: dict[str, Any],
    artifacts: dict[str, Path | None],
    stage_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    measured_coverage = stage_payloads["input_stage"].get("measured_coverage") or {}
    return {
        "experiment_id": config["experiment_id"],
        "object_id": config["object_id"],
        "control_experiment_id": config["evaluation"].get("control_experiment_id"),
        "comparison_axes": {
            "input_condition": config["input_condition"]["mode"],
            "reconstruction_backend": config["reconstruction"]["backend"],
            "meshing_backend": config["meshing"]["backend"],
        },
        "study_parameters": resolve_study_parameters(config),
        "measured_dataset_coverage": measured_coverage,
        "stage_readiness": {
            stage_name: payload["readiness"] for stage_name, payload in stage_payloads.items()
        },
        "artifact_status": {
            name: {"path": str(path) if path is not None else None, "exists": path.exists() if path is not None else False}
            for name, path in artifacts.items()
            if name != "experiment_root"
        },
        "research_questions": config["evaluation"].get("research_questions", build_default_research_questions()),
        "summary_notes": [
            "Use controlled one-factor-at-a-time comparisons to isolate information loss in the 3D-to-2D-to-3D pipeline.",
            "Report geometry, scale, render fidelity, and runtime together rather than treating any single metric as sufficient.",
            "Use CloudCompare or MeshLab for manual inspection of failure modes such as collapse, fragmentation, and scale drift.",
        ],
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_real_experiment_config(config_path)
    experiment_root = resolve_experiment_root(PROJECT_ROOT, config["experiment_id"])
    artifacts = expected_artifact_paths(PROJECT_ROOT, config)

    evaluation_paths = stage_evaluation_paths(PROJECT_ROOT, config)
    evaluation_paths["evaluation_root"].mkdir(parents=True, exist_ok=True)

    quantitative_metrics = compute_quantitative_metrics(
        config=config,
        artifacts=artifacts,
        surface_sample_count=args.surface_sample_count,
        completeness_threshold_ratio=args.completeness_threshold_ratio,
    )

    input_stage_payload = build_input_stage_metrics(config, artifacts)
    pose_stage_payload = build_pose_stage_metrics(config, artifacts)
    reconstruction_stage_payload = build_reconstruction_stage_metrics(config, artifacts)
    mesh_stage_payload = build_mesh_stage_metrics(
        config,
        artifacts,
        quantitative_metrics.get("geometry_and_mesh"),
    )
    stage_payloads = {
        "input_stage": input_stage_payload,
        "pose_stage": pose_stage_payload,
        "reconstruction_stage": reconstruction_stage_payload,
        "mesh_stage": mesh_stage_payload,
    }

    write_json(evaluation_paths["input_stage"], input_stage_payload)
    write_json(evaluation_paths["pose_stage"], pose_stage_payload)
    write_json(evaluation_paths["reconstruction_stage"], reconstruction_stage_payload)
    write_json(evaluation_paths["mesh_stage"], mesh_stage_payload)

    experiment_summary = build_experiment_summary(config, artifacts, stage_payloads)
    write_json(evaluation_paths["experiment_summary"], experiment_summary)
    write_json(evaluation_paths["quantitative_metrics"], quantitative_metrics)

    analysis_record = build_analysis_record(config, artifacts, stage_payloads, quantitative_metrics)
    write_json(evaluation_paths["analysis_record"], analysis_record)

    protocol_payload = {
        "experiment_id": config["experiment_id"],
        "object_id": config["object_id"],
        "control_experiment_id": config["evaluation"].get("control_experiment_id"),
        "config_path": str(config_path),
        "dataset_summary": {
            "dataset_dir": str(artifacts["dataset_dir"]),
            "view_count": input_stage_payload.get("view_count"),
            "discovered_png_count_recursive": input_stage_payload.get("discovered_png_count_recursive"),
            "transforms_json": str(artifacts["transforms_json"]),
            "measured_coverage": input_stage_payload.get("measured_coverage"),
        },
        "study_parameters": resolve_study_parameters(config),
        "research_questions": config["evaluation"].get("research_questions", build_default_research_questions()),
        "comparison_axes": {
            "input_condition": config["input_condition"]["mode"],
            "reconstruction_backend": config["reconstruction"]["backend"],
            "meshing_backend": config["meshing"]["backend"],
        },
        "artifact_status": experiment_summary["artifact_status"],
        "metric_groups": build_metric_groups(config, artifacts),
        "stage_metric_files": {
            "input_stage": str(evaluation_paths["input_stage"]),
            "pose_stage": str(evaluation_paths["pose_stage"]),
            "reconstruction_stage": str(evaluation_paths["reconstruction_stage"]),
            "mesh_stage": str(evaluation_paths["mesh_stage"]),
            "experiment_summary": str(evaluation_paths["experiment_summary"]),
            "quantitative_metrics": str(evaluation_paths["quantitative_metrics"]),
            "analysis_record": str(evaluation_paths["analysis_record"]),
        },
        "quantitative_metrics": quantitative_metrics,
        "qualitative_checks": config["evaluation"].get(
            "qualitative_checks",
            [
                "Inspect multi-view consistency across generated views.",
                "Inspect missing or hallucinated geometry in the final mesh.",
                "Inspect texture drift or floaters in rendered outputs.",
                "Inspect smoothing or oversurface artifacts introduced by meshing.",
            ],
        ),
        "manifests": {
            "runner": read_optional_json(artifacts["runner_manifest"]),
            "input_preparation": read_optional_json(find_first_existing([
                artifacts.get("generation_modal_manifest"),
                artifacts.get("generation_manifest"),
            ])),
            "reconstruction": read_optional_json(find_first_existing([
                artifacts.get("reconstruction_modal_manifest"),
                artifacts.get("reconstruction_manifest"),
            ])),
            "meshing": read_optional_json(find_first_existing([
                artifacts.get("meshing_modal_manifest"),
                artifacts.get("meshing_manifest"),
            ])),
        },
        "next_steps": [
            "Render held-out evaluation views from the trained Gaussian Splatting model for PSNR, SSIM, and LPIPS when render pairs are available.",
            "Use CloudCompare or MeshLab to validate failure cases and cross-check scale or alignment issues reported numerically.",
            "Use analysis_record.json as the normalized source for view-count, orbit-count, scale, and runtime plots.",
        ],
    }

    output_path = experiment_root / "evaluation" / "evaluation_protocol.json"
    write_json(output_path, protocol_payload)
    print(json.dumps({"experiment_id": config["experiment_id"], "evaluation_protocol": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
