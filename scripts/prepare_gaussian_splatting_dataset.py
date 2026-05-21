#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

VIEW_SOURCE_DIRS = {"real_views": "real_views_master_5ring"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one real-view dataset folder into the dataset layout expected by the Gaussian Splatting trainer."
    )
    parser.add_argument("--object_id", required=True, help="Object folder name under data/objects/, for example 'shark'.")
    parser.add_argument(
        "--view_source",
        choices=["real_views"],
        default="real_views",
        help="Which object-local real-view folder to convert.",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Optional override source dataset directory. Defaults to the folder implied by --view_source.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. Defaults to results/gaussian_splatting_datasets/<object>_<view_source>/.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Create a simple train/test split. Without this flag, all frames go into transforms_train.json and test is empty.",
    )
    parser.add_argument(
        "--test_holdout",
        type=int,
        default=8,
        help="If --eval is used, every Nth frame goes to test. Default matches LLFF-style holdout 8.",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images into the output dataset directory. Otherwise the transforms point back to the original images in-place.",
    )
    return parser.parse_args()


def resolve_dataset_dir(object_root: Path, view_source: str, dataset_dir_arg: str | None) -> Path:
    if dataset_dir_arg is not None:
        dataset_dir = Path(dataset_dir_arg)
        if dataset_dir.is_absolute():
            return dataset_dir.resolve()
        return (PROJECT_ROOT / dataset_dir).resolve()
    return (object_root / VIEW_SOURCE_DIRS[view_source]).resolve()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).resolve()
    return (PROJECT_ROOT / "results" / "gaussian_splatting_datasets" / f"{args.object_id}_{args.view_source}").resolve()


def strip_extension(file_path: str) -> str:
    path = Path(file_path)
    return (path.parent / path.stem).as_posix()


def normalize_frame_for_blender(frame: dict) -> dict:
    return {
        "file_path": strip_extension(frame["file_path"]),
        "transform_matrix": frame["transform_matrix"],
    }


def split_frames(frames: list[dict], eval_enabled: bool, test_holdout: int) -> tuple[list[dict], list[dict]]:
    if not eval_enabled:
        return frames, []

    train_frames: list[dict] = []
    test_frames: list[dict] = []
    for index, frame in enumerate(frames):
        if test_holdout > 0 and index % test_holdout == 0:
            test_frames.append(frame)
        else:
            train_frames.append(frame)
    return train_frames, test_frames


def copy_frame_images(frames: list[dict], source_dir: Path, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_frames: list[dict] = []
    for frame in frames:
        source_image = source_dir / f"{frame['file_path']}.png"
        destination = output_dir / f"{Path(frame['file_path']).name}.png"
        destination.write_bytes(source_image.read_bytes())
        copied_frames.append(
            {
                "file_path": destination.stem,
                "transform_matrix": frame["transform_matrix"],
            }
        )
    return copied_frames


def build_payload(camera_angle_x: float, frames: list[dict]) -> dict:
    return {
        "camera_angle_x": camera_angle_x,
        "frames": frames,
    }


def main() -> None:
    args = parse_args()
    object_root = (PROJECT_ROOT / "data" / "objects" / args.object_id).resolve()
    dataset_dir = resolve_dataset_dir(object_root, args.view_source, args.dataset_dir)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms_path = dataset_dir / "transforms.json"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms.json in source dataset: {transforms_path}")

    source_payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    normalized_frames = [normalize_frame_for_blender(frame) for frame in source_payload["frames"]]
    train_frames, test_frames = split_frames(normalized_frames, args.eval, args.test_holdout)

    if args.copy_images:
        images_out = output_dir / "images"
        train_frames = copy_frame_images(train_frames, dataset_dir, images_out)
        test_frames = copy_frame_images(test_frames, dataset_dir, images_out)
    else:
        train_frames = [
            {
                "file_path": Path(__import__("os").path.relpath((dataset_dir / f"{frame['file_path']}.png").with_suffix(""), output_dir)).as_posix(),
                "transform_matrix": frame["transform_matrix"],
            }
            for frame in train_frames
        ]
        test_frames = [
            {
                "file_path": Path(__import__("os").path.relpath((dataset_dir / f"{frame['file_path']}.png").with_suffix(""), output_dir)).as_posix(),
                "transform_matrix": frame["transform_matrix"],
            }
            for frame in test_frames
        ]

    train_payload = build_payload(source_payload["camera_angle_x"], train_frames)
    test_payload = build_payload(source_payload["camera_angle_x"], test_frames)

    transforms_train = output_dir / "transforms_train.json"
    transforms_test = output_dir / "transforms_test.json"
    manifest_path = output_dir / "gaussian_splatting_dataset_manifest.json"

    transforms_train.write_text(json.dumps(train_payload, indent=2) + "\n", encoding="utf-8")
    transforms_test.write_text(json.dumps(test_payload, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "object_id": args.object_id,
                "view_source": args.view_source,
                "source_dataset_dir": str(dataset_dir),
                "source_transforms_json": str(transforms_path),
                "output_dir": str(output_dir),
                "transforms_train_json": str(transforms_train),
                "transforms_test_json": str(transforms_test),
                "eval": args.eval,
                "test_holdout": args.test_holdout,
                "copy_images": args.copy_images,
                "train_frame_count": len(train_frames),
                "test_frame_count": len(test_frames),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "object_id": args.object_id,
                "view_source": args.view_source,
                "source_dataset_dir": str(dataset_dir),
                "output_dir": str(output_dir),
                "transforms_train_json": str(transforms_train),
                "transforms_test_json": str(transforms_test),
                "manifest": str(manifest_path),
                "train_frame_count": len(train_frames),
                "test_frame_count": len(test_frames),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
