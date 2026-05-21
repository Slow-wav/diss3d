import io
import json
import shlex
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if not SRC_ROOT.exists():
    PROJECT_ROOT = Path("/root")
    SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import load_object_metadata, resolve_object_metadata_path, resolve_object_root


APP_NAME = "diss3d-gaussian-splatting"
GAUSSIAN_SPLATTING_REPO_URL = "https://github.com/graphdeco-inria/gaussian-splatting.git"
GAUSSIAN_SPLATTING_REPO_COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"

VIEW_SOURCE_DIRS = {"real_views": "real_views"}

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "build-essential",
        "clang",
        "git",
        "ninja-build",
        "libgl1",
        "libglib2.0-0",
    )
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install setuptools==69.5.1 wheel packaging",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2",
        "python -m pip install 'numpy<2' plyfile tqdm opencv-python joblib tensorboard",
        f"git clone --recursive {GAUSSIAN_SPLATTING_REPO_URL} /root/gaussian-splatting",
        "cd /root/gaussian-splatting && git checkout "
        f"{GAUSSIAN_SPLATTING_REPO_COMMIT} && git submodule update --init --recursive",
        "python -c \"from pathlib import Path; "
        "path = Path('/root/gaussian-splatting/scene/dataset_readers.py'); "
        "text = path.read_text(encoding='utf-8'); "
        "old = '            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), \\\"RGB\\\")\\n'; "
        "new = '            image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), \\\"RGB\\\")\\n'; "
        "assert old in text, 'Expected dataset_readers.py line was not found for patching'; "
        "path.write_text(text.replace(old, new), encoding='utf-8')\"",
        "cd /root/gaussian-splatting && MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6 python -m pip install --no-build-isolation submodules/diff-gaussian-rasterization",
        "cd /root/gaussian-splatting && MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6 python -m pip install --no-build-isolation submodules/simple-knn",
        "cd /root/gaussian-splatting && MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6 python -m pip install --no-build-isolation submodules/fused-ssim",
    )
    .add_local_dir(SRC_ROOT, remote_path="/root/src", copy=True)
)


def _resolve_prepared_dataset_dir(object_id: str, view_source: str, dataset_dir: str | None) -> Path:
    if dataset_dir:
        return Path(dataset_dir).resolve()
    return (PROJECT_ROOT / "results" / "gaussian_splatting_datasets" / f"{object_id}_{view_source}").resolve()


def _resolve_model_path(object_id: str, view_source: str, model_path: str | None) -> Path:
    if model_path:
        return Path(model_path).resolve()
    return (PROJECT_ROOT / "results" / "gaussian_splatting" / f"{object_id}_{view_source}").resolve()


def _candidate_image_paths(dataset_dir: Path, frame_file_path: str) -> list[Path]:
    base_path = (dataset_dir / frame_file_path).resolve()
    suffixes = [".png", ".jpg", ".jpeg", ".webp"]
    candidates = [base_path.with_suffix(suffix) for suffix in suffixes]
    if base_path.suffix:
        candidates.insert(0, base_path)
    return candidates


def _resolve_frame_image_path(dataset_dir: Path, frame_file_path: str) -> Path:
    for candidate in _candidate_image_paths(dataset_dir, frame_file_path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve an image file for frame path {frame_file_path!r} from dataset {dataset_dir}"
    )


def _collect_gaussian_dataset_bundle(dataset_dir: Path) -> tuple[dict[str, bytes], dict[str, int]]:
    bundle: dict[str, bytes] = {}
    stats = {
        "bundle_file_count": 0,
        "referenced_image_count": 0,
        "train_frame_count": 0,
        "test_frame_count": 0,
    }

    transforms_train = dataset_dir / "transforms_train.json"
    transforms_test = dataset_dir / "transforms_test.json"
    if not transforms_train.exists() or not transforms_test.exists():
        raise FileNotFoundError(
            f"Prepared dataset must contain transforms_train.json and transforms_test.json: {dataset_dir}"
        )

    metadata_files = [
        transforms_train,
        transforms_test,
        dataset_dir / "gaussian_splatting_dataset_manifest.json",
    ]
    for path in metadata_files:
        if path.exists():
            bundle[str(path.relative_to(PROJECT_ROOT).as_posix())] = path.read_bytes()

    seen_images: set[str] = set()
    for split_name, transforms_path in (("train", transforms_train), ("test", transforms_test)):
        payload = json.loads(transforms_path.read_text(encoding="utf-8"))
        frames = payload.get("frames", [])
        stats[f"{split_name}_frame_count"] = len(frames)
        for frame in frames:
            image_path = _resolve_frame_image_path(dataset_dir, frame["file_path"])
            relative_key = str(image_path.relative_to(PROJECT_ROOT).as_posix())
            if relative_key not in seen_images:
                bundle[relative_key] = image_path.read_bytes()
                seen_images.add(relative_key)

    stats["bundle_file_count"] = len(bundle)
    stats["referenced_image_count"] = len(seen_images)
    return bundle, stats


def _extract_latest_iteration_from_model_dir(model_dir: Path) -> int | None:
    point_cloud_root = model_dir / "point_cloud"
    candidates = sorted(point_cloud_root.glob("iteration_*"))
    if not candidates:
        return None
    return max(int(path.name.split("_")[-1]) for path in candidates)


def _build_model_archive(model_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(model_dir.rglob("*")):
            if not path.is_file():
                continue
            tar.add(path, arcname=str(path.relative_to(model_dir)))
    return buffer.getvalue()


def _extract_model_archive(archive_bytes: bytes, model_path: Path) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        tar.extractall(path=model_path)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    startup_timeout=60 * 60,
)
def run_gaussian_splatting_remote(
    dataset_bundle: dict[str, bytes],
    dataset_dir_relative: str,
    iterations: int = 30000,
    eval_enabled: bool = False,
    quiet: bool = False,
    data_device: str = "cuda",
    white_background: bool = False,
) -> dict:
    import os
    import subprocess

    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="gaussian_splatting_") as temp_dir:
        workspace_root = Path(temp_dir) / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)

        for relative_path, file_bytes in dataset_bundle.items():
            destination = workspace_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file_bytes)

        dataset_dir = workspace_root / dataset_dir_relative
        model_dir = workspace_root / "outputs" / "gaussian_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "/root/gaussian-splatting/train.py",
            "-s",
            str(dataset_dir),
            "-m",
            str(model_dir),
            "--iterations",
            str(iterations),
            "--test_iterations",
            "-1",
            "--save_iterations",
            str(iterations),
            "--data_device",
            data_device,
            "--disable_viewer",
        ]
        if eval_enabled:
            command.append("--eval")
        if quiet:
            command.append("--quiet")
        if white_background:
            command.append("--white_background")

        env = os.environ.copy()
        env["MAX_JOBS"] = "4"
        env["TORCH_CUDA_ARCH_LIST"] = "8.6"

        try:
            completed = subprocess.run(
                command,
                cwd="/root/gaussian-splatting",
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Gaussian Splatting training failed. "
                f"stdout_tail={exc.stdout[-3000:] if exc.stdout else ''!r} "
                f"stderr_tail={exc.stderr[-3000:] if exc.stderr else ''!r}"
            ) from exc

        finished_at = datetime.now(timezone.utc)
        runtime_seconds = round(time.perf_counter() - started_perf, 6)
        latest_iteration = _extract_latest_iteration_from_model_dir(model_dir)
        point_cloud_path = (
            model_dir / "point_cloud" / f"iteration_{latest_iteration}" / "point_cloud.ply"
            if latest_iteration is not None
            else None
        )

        return {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "runtime_seconds": runtime_seconds,
            "modal_gpu": "A10G",
            "repo_commit": GAUSSIAN_SPLATTING_REPO_COMMIT,
            "iterations": iterations,
            "eval": eval_enabled,
            "quiet": quiet,
            "data_device": data_device,
            "white_background": white_background,
            "latest_iteration": latest_iteration,
            "point_cloud_exists": point_cloud_path.exists() if point_cloud_path is not None else False,
            "model_archive": _build_model_archive(model_dir),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }


@app.local_entrypoint()
def main(
    object_id: str = "shark",
    view_source: str = "real_views",
    dataset_dir: str = "",
    model_path: str = "",
    iterations: int = 30000,
    eval: bool = False,
    quiet: bool = False,
    data_device: str = "cuda",
    white_background: bool = False,
    dry_run: bool = False,
) -> None:
    if view_source not in VIEW_SOURCE_DIRS:
        raise ValueError(f"Unsupported view_source: {view_source}")

    object_root = resolve_object_root(PROJECT_ROOT, object_id)
    metadata_path = resolve_object_metadata_path(PROJECT_ROOT, object_id)
    metadata = load_object_metadata(metadata_path)
    dataset_dir_path = _resolve_prepared_dataset_dir(object_id, view_source, dataset_dir or None)
    model_path_path = _resolve_model_path(object_id, view_source, model_path or None)

    transforms_train = dataset_dir_path / "transforms_train.json"
    transforms_test = dataset_dir_path / "transforms_test.json"
    if not dataset_dir_path.exists():
        raise FileNotFoundError(f"Prepared Gaussian Splatting dataset does not exist: {dataset_dir_path}")
    if not transforms_train.exists() or not transforms_test.exists():
        raise FileNotFoundError(
            f"Prepared dataset is missing transforms_train.json or transforms_test.json: {dataset_dir_path}"
        )

    dataset_bundle, bundle_stats = _collect_gaussian_dataset_bundle(dataset_dir_path)
    dataset_dir_relative = str(dataset_dir_path.relative_to(PROJECT_ROOT).as_posix())
    manifest_path = PROJECT_ROOT / "results" / "gaussian_splatting" / f"{object_id}_{view_source}_modal_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_command = [
        sys.executable,
        "/root/gaussian-splatting/train.py",
        "-s",
        f"/tmp/<workspace>/{dataset_dir_relative}",
        "-m",
        "/tmp/<workspace>/outputs/gaussian_model",
        "--iterations",
        str(iterations),
        "--test_iterations",
        "-1",
        "--save_iterations",
        str(iterations),
        "--data_device",
        data_device,
        "--disable_viewer",
    ]
    if eval:
        resolved_command.append("--eval")
    if quiet:
        resolved_command.append("--quiet")
    if white_background:
        resolved_command.append("--white_background")

    if dry_run:
        print(
            json.dumps(
                {
                    "object_id": object_id,
                    "view_source": view_source,
                    "dataset_dir": str(dataset_dir_path),
                    "transforms_train_json": str(transforms_train),
                    "transforms_test_json": str(transforms_test),
                    "model_path": str(model_path_path),
                    "manifest": str(manifest_path),
                    "modal_app": APP_NAME,
                    "modal_gpu": "A10G",
                    "repo_commit": GAUSSIAN_SPLATTING_REPO_COMMIT,
                    "iterations": iterations,
                    "eval": eval,
                    "quiet": quiet,
                    "data_device": data_device,
                    "white_background": white_background,
                    "bundle_file_count": bundle_stats["bundle_file_count"],
                    "referenced_image_count": bundle_stats["referenced_image_count"],
                    "train_frame_count": bundle_stats["train_frame_count"],
                    "test_frame_count": bundle_stats["test_frame_count"],
                    "object_metadata_radius": metadata.get("radius"),
                    "object_metadata_elevation_ratio": metadata.get("elevation_ratio"),
                    "dry_run": True,
                    "resolved_command": " ".join(shlex.quote(part) for part in resolved_command),
                },
                indent=2,
            )
        )
        return

    remote_result = run_gaussian_splatting_remote.remote(
        dataset_bundle=dataset_bundle,
        dataset_dir_relative=dataset_dir_relative,
        iterations=iterations,
        eval_enabled=eval,
        quiet=quiet,
        data_device=data_device,
        white_background=white_background,
    )

    _extract_model_archive(remote_result["model_archive"], model_path_path)
    manifest_path.write_text(
        json.dumps(
            {
                "object_id": object_id,
                "view_source": view_source,
                "dataset_dir": str(dataset_dir_path),
                "transforms_train_json": str(transforms_train),
                "transforms_test_json": str(transforms_test),
                "model_path": str(model_path_path),
                "modal_app": APP_NAME,
                "modal_gpu": remote_result["modal_gpu"],
                "repo_commit": remote_result["repo_commit"],
                "iterations": remote_result["iterations"],
                "eval": remote_result["eval"],
                "quiet": remote_result["quiet"],
                "data_device": remote_result["data_device"],
                "white_background": remote_result["white_background"],
                "latest_iteration": remote_result["latest_iteration"],
                "point_cloud_exists": remote_result["point_cloud_exists"],
                "started_at_utc": remote_result["started_at_utc"],
                "finished_at_utc": remote_result["finished_at_utc"],
                "runtime_seconds": remote_result["runtime_seconds"],
                "bundle_file_count": bundle_stats["bundle_file_count"],
                "referenced_image_count": bundle_stats["referenced_image_count"],
                "train_frame_count": bundle_stats["train_frame_count"],
                "test_frame_count": bundle_stats["test_frame_count"],
                "stdout_tail": remote_result["stdout_tail"],
                "stderr_tail": remote_result["stderr_tail"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "object_id": object_id,
                "view_source": view_source,
                "dataset_dir": str(dataset_dir_path),
                "model_path": str(model_path_path),
                "manifest": str(manifest_path),
                "runtime_seconds": remote_result["runtime_seconds"],
                "modal_gpu": remote_result["modal_gpu"],
                "white_background": remote_result["white_background"],
                "latest_iteration": remote_result["latest_iteration"],
                "point_cloud_exists": remote_result["point_cloud_exists"],
            },
            indent=2,
        )
    )
