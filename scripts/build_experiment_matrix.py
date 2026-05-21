#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand one handwritten study matrix into the concrete experiment configs used for a dissertation sweep."
    )
    parser.add_argument("--matrix", required=True, help="Path to the experiment-matrix specification JSON.")
    return parser.parse_args()


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _set_nested_value(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = payload
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            nested = {}
            target[part] = nested
        target = nested
    target[parts[-1]] = value


def _apply_updates(config: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for dotted_key, value in updates.items():
        _set_nested_value(updated, dotted_key, value)
    return updated


def _build_variant_config(
    base_config: dict[str, Any],
    variant_name: str,
    axis_name: str,
    baseline_experiment_id: str,
    variant_spec: dict[str, Any],
) -> dict[str, Any]:
    config = _apply_updates(base_config, variant_spec.get("updates", {}))
    experiment_id = str(variant_spec.get("experiment_id") or f"{baseline_experiment_id}_{variant_name}")
    config["experiment_id"] = experiment_id

    study_parameters = dict(config.get("study_parameters", {}))
    study_parameters["matrix_axis"] = axis_name
    study_parameters["matrix_variant"] = variant_name
    study_parameters.update(variant_spec.get("study_parameters", {}))
    config["study_parameters"] = study_parameters

    evaluation = dict(config.get("evaluation", {}))
    evaluation["control_experiment_id"] = str(variant_spec.get("control_experiment_id") or baseline_experiment_id)
    config["evaluation"] = evaluation
    return config


def main() -> None:
    args = parse_args()
    matrix_path = _resolve_path(args.matrix)
    matrix_spec = load_json(matrix_path)
    base_config_path = _resolve_path(matrix_spec["base_config"])
    output_dir = _resolve_path(matrix_spec["output_dir"])
    base_config = load_json(base_config_path)
    baseline_spec = matrix_spec.get("baseline", {})
    baseline_experiment_id = str(baseline_spec.get("experiment_id") or base_config["experiment_id"])

    generated: list[dict[str, Any]] = []

    baseline_config = _build_variant_config(
        base_config=base_config,
        variant_name=str(baseline_spec.get("name", "baseline")),
        axis_name="baseline",
        baseline_experiment_id=baseline_experiment_id,
        variant_spec=baseline_spec,
    )
    baseline_config["experiment_id"] = baseline_experiment_id
    baseline_path = output_dir / f"{baseline_experiment_id}.json"
    write_json(baseline_path, baseline_config)
    generated.append(
        {
            "axis": "baseline",
            "variant_name": baseline_spec.get("name", "baseline"),
            "experiment_id": baseline_experiment_id,
            "config_path": str(baseline_path),
        }
    )

    for axis in matrix_spec.get("axes", []):
        axis_name = str(axis["axis"])
        for variant in axis.get("variants", []):
            variant_name = str(variant["name"])
            config = _build_variant_config(
                base_config=baseline_config,
                variant_name=variant_name,
                axis_name=axis_name,
                baseline_experiment_id=baseline_experiment_id,
                variant_spec=variant,
            )
            config_path = output_dir / f"{config['experiment_id']}.json"
            write_json(config_path, config)
            generated.append(
                {
                    "axis": axis_name,
                    "variant_name": variant_name,
                    "experiment_id": config["experiment_id"],
                    "config_path": str(config_path),
                }
            )

    manifest_path = output_dir / "matrix_manifest.json"
    write_json(
        manifest_path,
        {
            "matrix_id": matrix_spec.get("matrix_id"),
            "matrix_source": str(matrix_path),
            "base_config": str(base_config_path),
            "output_dir": str(output_dir),
            "generated_config_count": len(generated),
            "generated_configs": generated,
        },
    )

    print(
        json.dumps(
            {
                "matrix_id": matrix_spec.get("matrix_id"),
                "generated_config_count": len(generated),
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
