#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import ensure_dir, list_image_files, load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a scene-scale dataset by reusing one image set and scaling only camera translations in transforms.json."
    )
    parser.add_argument("--source-dir", required=True, help="Dataset directory that already contains the source PNGs and transforms.json.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the scaled dataset.")
    parser.add_argument("--scale", required=True, type=float, help="Scene scale factor applied to camera translations.")
    parser.add_argument("--clean", action="store_true", help="Remove existing files in the output directory first.")
    return parser.parse_args()


def _clean_directory(directory: Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _scaled_transforms_payload(source_payload: dict, scale: float) -> dict:
    scaled = dict(source_payload)
    scaled_frames: list[dict] = []
    for frame in source_payload.get("frames", []):
        scaled_frame = dict(frame)
        matrix = [[float(value) for value in row] for row in frame["transform_matrix"]]
        for row_index in range(3):
            matrix[row_index][3] *= scale
        scaled_frame["transform_matrix"] = matrix
        scaled_frames.append(scaled_frame)
    scaled["frames"] = scaled_frames
    return scaled


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_transforms_path = source_dir / "transforms.json"

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source dataset directory: {source_dir}")
    if not source_transforms_path.exists():
        raise FileNotFoundError(f"Missing source transforms.json: {source_transforms_path}")

    if args.clean:
        _clean_directory(output_dir)
    ensure_dir(output_dir)

    image_paths = list_image_files(source_dir)
    for image_path in image_paths:
        shutil.copy2(image_path, output_dir / image_path.name)

    source_payload = load_json(source_transforms_path)
    scaled_payload = _scaled_transforms_payload(source_payload, args.scale)
    write_json(output_dir / "transforms.json", scaled_payload)

    manifest = {
        "dataset_type": "scene_scale_variant",
        "source_dataset_dir": str(source_dir),
        "output_dir": str(output_dir),
        "scene_scale_factor": float(args.scale),
        "frame_count": len(image_paths),
        "image_copy_mode": "copy2",
        "images_reused_from_source": True,
        "transform_translation_scale_only": True,
        "source_transforms": str(source_transforms_path),
        "output_transforms": str(output_dir / "transforms.json"),
    }
    write_json(output_dir / "scene_scale_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
