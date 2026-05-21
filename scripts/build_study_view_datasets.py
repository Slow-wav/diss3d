#!/usr/bin/env python3
import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the real-view study datasets used in the dissertation by subsetting and resizing one canonical source set."
    )
    parser.add_argument("--plan", required=True, help="Path to the study dataset plan JSON.")
    return parser.parse_args()


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _extract_frame_z(frame: dict[str, Any]) -> float:
    return round(float(frame["transform_matrix"][2][3]), 6)


def _select_evenly_spaced_frames(frames: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("Requested frame count must be non-negative.")
    if count > len(frames):
        raise ValueError(f"Requested {count} frames from a ring that only contains {len(frames)} frames.")
    if count == 0:
        return []
    if count == len(frames):
        return list(frames)
    return [frames[math.floor(index * len(frames) / count)] for index in range(count)]


def _scale_intrinsics(source_payload: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    source_width = int(source_payload["w"])
    source_height = int(source_payload["h"])
    scale_x = width / source_width
    scale_y = height / source_height
    fl_x = float(source_payload["fl_x"]) * scale_x
    fl_y = float(source_payload["fl_y"]) * scale_y
    camera_angle_x = 2.0 * math.atan((0.5 * width) / fl_x)
    camera_angle_y = 2.0 * math.atan((0.5 * height) / fl_y)
    return {
        "w": int(width),
        "h": int(height),
        "fl_x": round(fl_x, 6),
        "fl_y": round(fl_y, 6),
        "cx": round(float(source_payload["cx"]) * scale_x, 6),
        "cy": round(float(source_payload["cy"]) * scale_y, 6),
        "camera_angle_x": round(camera_angle_x, 6),
        "camera_angle_y": round(camera_angle_y, 6),
    }


def _copy_or_resize_image(source_image: Path, destination_image: Path, width: int, height: int) -> None:
    if destination_image.exists():
        destination_image.unlink()
    if width == 0 or height == 0:
        raise ValueError("Resize target must be greater than zero.")

    with Image.open(source_image) as image:
        source_width, source_height = image.size
        if source_width == width and source_height == height:
            shutil.copy2(source_image, destination_image)
            return
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        resized.save(destination_image)


def _load_source(source_name: str, source_spec: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = _resolve_path(source_spec["dataset_dir"])
    transforms_path = dataset_dir / "transforms.json"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Source dataset directory does not exist: {dataset_dir}")
    if not transforms_path.exists():
        raise FileNotFoundError(f"Source dataset is missing transforms.json: {transforms_path}")

    payload = load_json(transforms_path)
    frames = list(payload["frames"])
    unique_z_bands = sorted({_extract_frame_z(frame) for frame in frames})
    ring_names = source_spec.get("ring_names") or [f"ring_{index + 1}" for index in range(len(unique_z_bands))]
    if len(ring_names) != len(unique_z_bands):
        raise ValueError(
            f"Source '{source_name}' defines {len(ring_names)} ring names but has {len(unique_z_bands)} z bands."
        )

    ring_name_by_z = {z_value: ring_name for z_value, ring_name in zip(unique_z_bands, ring_names)}
    ring_frames: dict[str, list[dict[str, Any]]] = {ring_name: [] for ring_name in ring_names}
    indexed_frames: list[dict[str, Any]] = []
    for source_index, frame in enumerate(frames):
        ring_name = ring_name_by_z[_extract_frame_z(frame)]
        enriched = {
            "source_index": source_index,
            "source_frame": int(frame.get("frame", source_index + 1)),
            "source_file_path": str(frame["file_path"]),
            "ring_name": ring_name,
            "frame": frame,
        }
        ring_frames[ring_name].append(enriched)
        indexed_frames.append(enriched)

    return {
        "name": source_name,
        "dataset_dir": dataset_dir,
        "transforms_path": transforms_path,
        "payload": payload,
        "ring_names": ring_names,
        "ring_frames": ring_frames,
        "indexed_frames": indexed_frames,
    }


def _build_variant_frame_selection(
    source: dict[str, Any],
    variant_name: str,
    selection_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    counts_per_ring = selection_spec.get("counts_per_ring")
    if not isinstance(counts_per_ring, dict):
        raise ValueError(f"Variant '{variant_name}' must define selection.counts_per_ring.")

    selected: list[dict[str, Any]] = []
    for ring_name in source["ring_names"]:
        ring_frames = source["ring_frames"][ring_name]
        requested_count = int(counts_per_ring.get(ring_name, 0))
        selected.extend(_select_evenly_spaced_frames(ring_frames, requested_count))

    if not selected:
        raise ValueError(f"Variant '{variant_name}' selected zero frames.")

    selected.sort(key=lambda frame: frame["source_index"])
    return selected


def _write_variant_dataset(
    output_dir: Path,
    source: dict[str, Any],
    source_payload: dict[str, Any],
    selected_frames: list[dict[str, Any]],
    width: int,
    height: int,
    variant_spec: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = _scale_intrinsics(source_payload, width, height)

    output_frames: list[dict[str, Any]] = []
    selected_frame_manifest: list[dict[str, Any]] = []
    ring_counts: dict[str, int] = {}

    for view_index, selected in enumerate(selected_frames, start=1):
        source_file_path = Path(selected["source_file_path"])
        source_image = (source["dataset_dir"] / source_file_path).resolve()
        destination_name = f"{view_index:04d}.png"
        destination_image = output_dir / destination_name
        _copy_or_resize_image(source_image, destination_image, width, height)

        frame_payload = dict(selected["frame"])
        frame_payload["file_path"] = destination_name
        frame_payload["view_index"] = view_index - 1
        frame_payload["source_frame"] = selected["source_frame"]
        frame_payload["source_file_path"] = source_file_path.as_posix()
        frame_payload["ring_name"] = selected["ring_name"]
        output_frames.append(frame_payload)

        ring_counts[selected["ring_name"]] = ring_counts.get(selected["ring_name"], 0) + 1
        selected_frame_manifest.append(
            {
                "output_file": destination_name,
                "source_file": source_file_path.as_posix(),
                "source_frame": selected["source_frame"],
                "source_index": selected["source_index"],
                "ring_name": selected["ring_name"],
            }
        )

    transforms_payload = {
        **intrinsics,
        "frames": output_frames,
        "number_of_orbits": int(variant_spec.get("study_parameters", {}).get("number_of_orbits", len(ring_counts))),
        "views_per_orbit": dict(ring_counts),
    }

    transforms_path = output_dir / "transforms.json"
    manifest_path = output_dir / "study_dataset_manifest.json"
    write_json(transforms_path, transforms_payload)
    write_json(
        manifest_path,
        {
            "variant_name": variant_spec["name"],
            "source_dataset": source["name"],
            "source_dataset_dir": str(source["dataset_dir"]),
            "source_transforms": str(source["transforms_path"]),
            "output_dir": str(output_dir),
            "transforms_json": str(transforms_path),
            "selection": variant_spec["selection"],
            "resize": {"width": width, "height": height},
            "frame_count": len(output_frames),
            "ring_counts": ring_counts,
            "study_parameters": variant_spec.get("study_parameters", {}),
            "selected_frames": selected_frame_manifest,
        },
    )

    return {
        "variant_name": variant_spec["name"],
        "dataset_dir": str(output_dir),
        "transforms_json": str(transforms_path),
        "manifest": str(manifest_path),
        "frame_count": len(output_frames),
        "ring_counts": ring_counts,
        "study_parameters": variant_spec.get("study_parameters", {}),
    }


def main() -> None:
    args = parse_args()
    plan_path = _resolve_path(args.plan)
    plan = load_json(plan_path)
    output_root = _resolve_path(plan["output_root"])

    sources = {
        source_name: _load_source(source_name, source_spec)
        for source_name, source_spec in plan.get("sources", {}).items()
    }
    if not sources:
        raise ValueError("Study dataset plan does not define any sources.")

    generated_variants: list[dict[str, Any]] = []
    for variant_spec in plan.get("variants", []):
        source_name = variant_spec["source"]
        if source_name not in sources:
            raise ValueError(f"Variant '{variant_spec['name']}' references unknown source '{source_name}'.")
        source = sources[source_name]
        resize_spec = variant_spec.get("resize", {})
        width = int(resize_spec.get("width", source["payload"]["w"]))
        height = int(resize_spec.get("height", source["payload"]["h"]))
        selected_frames = _build_variant_frame_selection(source, variant_spec["name"], variant_spec["selection"])
        output_dir = output_root / variant_spec["name"]
        generated_variants.append(
            _write_variant_dataset(
                output_dir=output_dir,
                source=source,
                source_payload=source["payload"],
                selected_frames=selected_frames,
                width=width,
                height=height,
                variant_spec=variant_spec,
            )
        )

    manifest_path = output_root / "study_views_manifest.json"
    write_json(
        manifest_path,
        {
            "study_id": plan.get("study_id"),
            "plan_path": str(plan_path),
            "output_root": str(output_root),
            "source_count": len(sources),
            "variant_count": len(generated_variants),
            "variants": generated_variants,
        },
    )

    print(
        json.dumps(
            {
                "study_id": plan.get("study_id"),
                "output_root": str(output_root),
                "variant_count": len(generated_variants),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
