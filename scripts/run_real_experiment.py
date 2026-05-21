#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import (
    build_default_research_questions,
    expected_artifact_paths,
    load_real_experiment_config,
    normalize_input_mode,
    resolve_experiment_root,
    resolve_study_parameters,
    resolve_view_source,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one full dissertation experiment from a concrete config: dataset prep, reconstruction, meshing, and evaluation."
    )
    parser.add_argument("--config", required=True, help="Path to a real experiment config JSON.")
    parser.add_argument("--dry_run", action="store_true", help="Pass dry-run mode through to all downstream wrappers where supported.")
    parser.add_argument(
        "--keep_existing",
        action="store_true",
        help="Do not delete the existing experiment directory before rerunning. By default reruns start from a clean experiment root.",
    )
    return parser.parse_args()


def add_option(command: list[str], flag: str, value: Any | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def add_bool_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def build_stage_commands(
    config: dict[str, Any],
    artifacts: dict[str, Path],
    dry_run: bool,
) -> list[dict[str, Any]]:
    object_id = config["object_id"]
    view_source = resolve_view_source(config)
    python = sys.executable
    # Build the pipeline as data first so it can be inspected before execution.
    stage_commands: list[dict[str, Any]] = []

    real_args = config["input_condition"].get("real_view_transforms", {})
    if real_args.get("use_existing_transforms"):
        stage_commands.append(
            {
                "stage": "reuse_real_view_transforms",
                "command": None,
                "skipped": True,
                "reason": "Using an existing real-view transforms.json instead of regenerating an orbit file.",
            }
        )
    else:
        command = [python, str((PROJECT_ROOT / "scripts" / "write_real_view_transforms.py").resolve()), "--object_id", object_id]
        for flag in ["images_dir", "output", "num_views", "width", "height", "focal_length_mm", "sensor_width_mm", "radius", "elevation_ratio", "number_of_orbits"]:
            add_option(command, f"--{flag}", real_args.get(flag))
        stage_commands.append({"stage": "prepare_real_view_transforms", "command": command})

    reconstruction = config["reconstruction"]
    meshing = config["meshing"]
    prep_args = reconstruction.get("dataset_prep", {})
    prepare_command = [
        python,
        str((PROJECT_ROOT / "scripts" / "prepare_gaussian_splatting_dataset.py").resolve()),
        "--object_id",
        object_id,
        "--view_source",
        view_source,
        "--output_dir",
        str(artifacts["prepared_dataset_dir"]),
    ]
    add_bool_flag(prepare_command, "--eval", bool(prep_args.get("eval", False)))
    add_option(prepare_command, "--test_holdout", prep_args.get("test_holdout"))
    add_bool_flag(prepare_command, "--copy_images", bool(prep_args.get("copy_images", False)))
    add_option(prepare_command, "--dataset_dir", prep_args.get("dataset_dir"))
    stage_commands.append({"stage": "prepare_gaussian_splatting_dataset", "command": prepare_command})

    gs_args = reconstruction.get("gaussian_splatting", {})
    gs_command = [
        python,
        str((PROJECT_ROOT / "scripts" / "run_gaussian_splatting.py").resolve()),
        "--object_id",
        object_id,
        "--view_source",
        view_source,
        "--dataset_dir",
        str(artifacts["prepared_dataset_dir"]),
        "--model_path",
        str(artifacts["gaussian_model_dir"]),
    ]
    add_option(gs_command, "--iterations", gs_args.get("iterations"))
    add_bool_flag(gs_command, "--eval", bool(gs_args.get("eval", False)))
    add_bool_flag(gs_command, "--quiet", bool(gs_args.get("quiet", False)))
    add_bool_flag(gs_command, "--white_background", bool(gs_args.get("white_background", False)))
    if dry_run:
        gs_command.append("--dry_run")
    stage_commands.append({"stage": "run_gaussian_splatting", "command": gs_command})

    if meshing["backend"] == "gaustudio":
        gaustudio_args = meshing.get("gaustudio", {})
        gaustudio_command = [
            python,
            str((PROJECT_ROOT / "scripts" / "run_gaustudio_mesh.py").resolve()),
            "--object_id",
            object_id,
            "--view_source",
            view_source,
            "--model_dir",
            str(artifacts["gaussian_model_dir"]),
            "--output_dir",
            str(artifacts["gaustudio_output_dir"]),
        ]
        add_option(gaustudio_command, "--source_path", gaustudio_args.get("source_path"))
        for flag in ["load_iteration", "resolution", "sh_degree"]:
            add_option(gaustudio_command, f"--{flag}", gaustudio_args.get(flag))
        add_bool_flag(gaustudio_command, "--white_background", bool(gaustudio_args.get("white_background", False)))
        add_bool_flag(gaustudio_command, "--clean", bool(gaustudio_args.get("clean", False)))
        if dry_run:
            gaustudio_command.append("--dry_run")
        stage_commands.append({"stage": "run_gaustudio_mesh", "command": gaustudio_command})

    evaluation_command = [
        python,
        str((PROJECT_ROOT / "scripts" / "evaluate_reconstruction_experiment.py").resolve()),
        "--config",
        str(config["__config_path"]),
    ]
    stage_commands.append({"stage": "evaluate_experiment", "command": evaluation_command})
    return stage_commands


def validate_active_workflow(config: dict[str, Any]) -> None:
    if normalize_input_mode(config) != "real_views":
        raise ValueError("The active dissertation workflow is now scoped to real_views experiments only.")
    if config["reconstruction"]["backend"] != "gaussian_splatting":
        raise ValueError("The active dissertation workflow is now scoped to gaussian_splatting only.")
    if config["meshing"]["backend"] != "gaustudio":
        raise ValueError("The active dissertation workflow is now scoped to meshing.backend='gaustudio' only.")


def reset_experiment_root(experiment_root: Path) -> None:
    if experiment_root.exists():
        shutil.rmtree(experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_real_experiment_config(config_path)
    validate_active_workflow(config)
    config["__config_path"] = str(config_path)
    config["evaluation"].setdefault("research_questions", build_default_research_questions())
    experiment_root = resolve_experiment_root(PROJECT_ROOT, config["experiment_id"])
    if not args.dry_run and not args.keep_existing:
        reset_experiment_root(experiment_root)
    else:
        experiment_root.mkdir(parents=True, exist_ok=True)
    artifacts = expected_artifact_paths(PROJECT_ROOT, config)
    study_parameters = resolve_study_parameters(config)

    stage_commands = build_stage_commands(config, artifacts, args.dry_run)
    manifest_path = experiment_root / "experiment_manifest.json"
    write_json(
        manifest_path,
        {
            "experiment_id": config["experiment_id"],
            "object_id": config["object_id"],
            "config_path": str(config_path),
            "dry_run": args.dry_run,
            "study_parameters": study_parameters,
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "stages": [
                {
                "stage": item["stage"],
                "command": item["command"],
                "skipped": item.get("skipped", False),
                "reason": item.get("reason"),
                }
                for item in stage_commands
            ],
            "fresh_run": not args.keep_existing,
        },
    )

    for item in stage_commands:
        if item.get("skipped"):
            continue
        subprocess.run(item["command"], check=True)

    print(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "experiment_root": str(experiment_root),
                "manifest": str(manifest_path),
                "dry_run": args.dry_run,
                "stage_count": len(stage_commands),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
