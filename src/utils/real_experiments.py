from pathlib import Path
from typing import Any

from .io import load_json


def load_real_experiment_config(config_path: Path) -> dict[str, Any]:
    payload = load_json(config_path)
    required = ["experiment_id", "object_id", "input_condition", "reconstruction", "meshing", "evaluation"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Experiment config is missing required key: {key}")
    payload.setdefault("study_parameters", {})
    return payload


def normalize_input_mode(config: dict[str, Any]) -> str:
    mode = config["input_condition"]["mode"]
    if mode != "real_views":
        raise ValueError("The active workflow only supports input_condition.mode='real_views'.")
    return mode


def resolve_view_source(config: dict[str, Any]) -> str:
    normalize_input_mode(config)
    return "real_views"


def resolve_experiment_root(project_root: Path, experiment_id: str) -> Path:
    return (project_root / "results" / "experiments" / experiment_id).resolve()


def resolve_ground_truth_mesh(project_root: Path, config: dict[str, Any]) -> Path:
    explicit = config["evaluation"].get("ground_truth_mesh")
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else (project_root / path).resolve()
    return (project_root / "data" / "objects" / config["object_id"] / "ground_truth" / "mesh.obj").resolve()


def build_default_research_questions() -> list[dict[str, str]]:
    return [
        {
            "id": "RQ1",
            "question": "How much geometric and visual information is lost in 3D-to-2D-to-3D reconstruction under controlled observation settings?",
        },
        {
            "id": "RQ2",
            "question": "How do view count, elevation coverage, image resolution, and scale affect reconstruction fidelity in Gaussian Splatting?",
        },
        {
            "id": "RQ3",
            "question": "Which controlled factors most strongly affect geometric loss, scale distortion, runtime, and failure cases?",
        },
    ]


def resolve_real_view_dataset_dir(project_root: Path, config: dict[str, Any], object_root: Path) -> Path:
    dataset_dir_value = (
        config.get("reconstruction", {})
        .get("dataset_prep", {})
        .get("dataset_dir")
        or config.get("input_condition", {})
        .get("real_view_transforms", {})
        .get("images_dir")
    )
    if dataset_dir_value:
        path = Path(dataset_dir_value)
        return path if path.is_absolute() else (project_root / path).resolve()
    return (object_root / "real_views_master_5ring").resolve()


def resolve_study_parameters(config: dict[str, Any]) -> dict[str, Any]:
    payload = dict(config.get("study_parameters", {}))
    real_view_args = config.get("input_condition", {}).get("real_view_transforms", {})
    width = real_view_args.get("width")
    height = real_view_args.get("height")

    payload.setdefault("reconstruction_model", config.get("reconstruction", {}).get("backend"))
    payload.setdefault("view_count", real_view_args.get("num_views"))
    payload.setdefault("number_of_orbits", real_view_args.get("number_of_orbits", 1))
    payload.setdefault("elevation_coverage", "single_orbit" if payload.get("number_of_orbits", 1) == 1 else "multi_orbit")
    payload.setdefault("image_width", width)
    payload.setdefault("image_height", height)
    if width is not None and height is not None:
        payload.setdefault("image_resolution", f"{width}x{height}")
    payload.setdefault("scale", 1.0)
    return payload


def stage_evaluation_paths(project_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    experiment_root = resolve_experiment_root(project_root, config["experiment_id"])
    evaluation_root = experiment_root / "evaluation"
    return {
        "evaluation_root": evaluation_root,
        "input_stage": evaluation_root / "input_stage_metrics.json",
        "pose_stage": evaluation_root / "pose_stage_metrics.json",
        "reconstruction_stage": evaluation_root / "reconstruction_stage_metrics.json",
        "mesh_stage": evaluation_root / "mesh_stage_metrics.json",
        "experiment_summary": evaluation_root / "experiment_summary.json",
        "evaluation_protocol": evaluation_root / "evaluation_protocol.json",
        "quantitative_metrics": evaluation_root / "quantitative_metrics.json",
        "analysis_record": evaluation_root / "analysis_record.json",
    }


def expected_artifact_paths(project_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    experiment_root = resolve_experiment_root(project_root, config["experiment_id"])
    object_root = (project_root / "data" / "objects" / config["object_id"]).resolve()
    normalize_input_mode(config)
    reconstruction_backend = config["reconstruction"]["backend"]
    meshing_backend = config["meshing"]["backend"]
    view_source = resolve_view_source(config)
    dataset_dir = resolve_real_view_dataset_dir(project_root, config, object_root)

    artifacts: dict[str, Path] = {
        "experiment_root": experiment_root,
        "dataset_dir": dataset_dir,
        "transforms_json": dataset_dir / "transforms.json",
        "ground_truth_mesh": resolve_ground_truth_mesh(project_root, config),
        "runner_manifest": experiment_root / "experiment_manifest.json",
    }
    artifacts.update(stage_evaluation_paths(project_root, config))
    artifacts["generation_modal_manifest"] = None

    if reconstruction_backend != "gaussian_splatting":
        raise ValueError("The active workflow only supports reconstruction.backend='gaussian_splatting'.")

    artifacts["prepared_dataset_dir"] = experiment_root / "intermediates" / "gaussian_splatting_dataset"
    artifacts["prepared_train_transforms"] = artifacts["prepared_dataset_dir"] / "transforms_train.json"
    artifacts["prepared_test_transforms"] = artifacts["prepared_dataset_dir"] / "transforms_test.json"
    artifacts["reconstruction_manifest"] = (
        project_root / "results" / "gaussian_splatting" / f"{config['object_id']}_{view_source}_manifest.json"
    ).resolve()
    artifacts["reconstruction_modal_manifest"] = (
        project_root / "results" / "gaussian_splatting" / f"{config['object_id']}_{view_source}_modal_manifest.json"
    ).resolve()
    artifacts["gaussian_model_dir"] = experiment_root / "reconstruction" / "gaussian_splatting_model"
    if meshing_backend == "gaustudio":
        artifacts["gaustudio_output_dir"] = experiment_root / "mesh" / "gaustudio_mesh"
        artifacts["gaustudio_source_path"] = artifacts["gaussian_model_dir"] / "cameras.json"
        artifacts["mesh"] = artifacts["gaustudio_output_dir"] / "fused_mesh.ply"
        artifacts["meshing_manifest"] = artifacts["gaustudio_output_dir"] / "gaustudio_mesh_wrapper_manifest.json"
        artifacts["meshing_modal_manifest"] = artifacts["gaustudio_output_dir"] / "gaustudio_mesh_manifest.json"
    else:
        raise ValueError("The active workflow only supports meshing.backend='gaustudio'.")

    return artifacts
