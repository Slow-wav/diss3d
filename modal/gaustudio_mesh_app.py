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


APP_NAME = "diss3d-gaustudio-mesh"
GAUSTUDIO_REPO_URL = "https://github.com/GAP-LAB-CUHK-SZ/gaustudio.git"
GAUSTUDIO_REPO_COMMIT = "132d749d863f11a6f158aa10c617ea4afc5b7268"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "build-essential",
        "clang",
        "cmake",
        "git",
        "ninja-build",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install setuptools==69.5.1 wheel packaging",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2",
        "python -m pip install 'numpy<2' plyfile tqdm opencv-python-headless trimesh omegaconf einops kiui scipy click open3d vdbfusion rembg",
        f"git clone --recursive {GAUSTUDIO_REPO_URL} /root/gaustudio",
        "cd /root/gaustudio && git checkout "
        f"{GAUSTUDIO_REPO_COMMIT} && git submodule update --init --recursive",
        "cd /root/gaustudio/submodules/gaustudio-diff-gaussian-rasterization && "
        "MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6 python setup.py install",
        "cd /root/gaustudio && python setup.py develop",
    )
)


def _resolve_model_dir(object_id: str, view_source: str, model_dir: str | None) -> Path:
    if model_dir:
        return Path(model_dir).resolve()
    return (PROJECT_ROOT / "results" / "gaussian_splatting" / f"{object_id}_{view_source}").resolve()


def _resolve_output_dir(model_dir: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    return (model_dir.parent / "gaustudio_mesh").resolve()


def _resolve_source_path(model_dir: Path, source_path: str | None) -> Path:
    if source_path:
        return Path(source_path).resolve()
    return (model_dir / "cameras.json").resolve()


def _build_model_archive(model_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(model_dir.rglob("*")):
            if not path.is_file():
                continue
            tar.add(path, arcname=str(path.relative_to(model_dir)))
    return buffer.getvalue()


def _extract_archive(archive_bytes: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        tar.extractall(path=destination)


def _build_output_archive(output_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            tar.add(path, arcname=str(path.relative_to(output_dir)))
    return buffer.getvalue()


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    startup_timeout=60 * 60,
)
def run_gaustudio_mesh_remote(
    model_archive: bytes,
    source_path_bytes: bytes | None = None,
    source_path_name: str | None = None,
    load_iteration: int = -1,
    resolution: int = 2,
    sh_degree: int = 0,
    white_background: bool = False,
    clean: bool = False,
) -> dict:
    import os
    import subprocess

    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="gaustudio_mesh_") as temp_dir:
        temp_root = Path(temp_dir)
        model_dir = temp_root / "gaussian_splatting_model"
        output_dir = temp_root / "gaustudio_output"
        _extract_archive(model_archive, model_dir)

        source_path = None
        if source_path_bytes is not None:
            source_path = temp_root / (source_path_name or "source_path.json")
            source_path.write_bytes(source_path_bytes)

        command = [
            sys.executable,
            "/root/gaustudio/gaustudio/scripts/extract_mesh.py",
            "--gpu",
            "0",
            "--model",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--load_iteration",
            str(load_iteration),
            "--resolution",
            str(resolution),
            "--sh",
            str(sh_degree),
        ]
        if source_path is not None:
            command.extend(["--source_path", str(source_path)])
        if white_background:
            command.append("--white_background")
        if clean:
            command.append("--clean")

        env = os.environ.copy()
        env["MAX_JOBS"] = "4"
        env["TORCH_CUDA_ARCH_LIST"] = "8.6"

        try:
            completed = subprocess.run(
                command,
                cwd="/root/gaustudio",
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "GauStudio mesh extraction failed. "
                f"stdout_tail={exc.stdout[-3000:] if exc.stdout else ''!r} "
                f"stderr_tail={exc.stderr[-3000:] if exc.stderr else ''!r}"
            ) from exc

        mesh_path = output_dir / "fused_mesh.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"GauStudio finished but did not create fused_mesh.ply in {output_dir}")

        finished_at = datetime.now(timezone.utc)
        runtime_seconds = round(time.perf_counter() - started_perf, 6)

        return {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "runtime_seconds": runtime_seconds,
            "modal_gpu": "A10G",
            "repo_commit": GAUSTUDIO_REPO_COMMIT,
            "load_iteration": load_iteration,
            "resolution": resolution,
            "sh_degree": sh_degree,
            "white_background": white_background,
            "clean": clean,
            "mesh_bytes": mesh_path.read_bytes(),
            "output_archive": _build_output_archive(output_dir),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }


@app.local_entrypoint()
def main(
    object_id: str = "shark",
    view_source: str = "real_views",
    model_dir: str = "",
    output_dir: str = "",
    source_path: str = "",
    load_iteration: int = -1,
    resolution: int = 2,
    sh_degree: int = 0,
    white_background: bool = False,
    clean: bool = False,
    dry_run: bool = False,
) -> None:
    model_dir_path = _resolve_model_dir(object_id, view_source, model_dir or None)
    output_dir_path = _resolve_output_dir(model_dir_path, output_dir or None)
    mesh_path = output_dir_path / "fused_mesh.ply"
    manifest_path = output_dir_path / "gaustudio_mesh_manifest.json"

    source_path_resolved = _resolve_source_path(model_dir_path, source_path or None)

    resolved_command = [
        sys.executable,
        "/root/gaustudio/gaustudio/scripts/extract_mesh.py",
        "--gpu",
        "0",
        "--model",
        "/tmp/<uploaded>/gaussian_splatting_model",
        "--output-dir",
        "/tmp/<outputs>/gaustudio_output",
        "--load_iteration",
        str(load_iteration),
        "--resolution",
        str(resolution),
        "--sh",
        str(sh_degree),
    ]
    resolved_command.extend(["--source_path", f"/tmp/<uploaded>/{source_path_resolved.name}"])
    if white_background:
        resolved_command.append("--white_background")
    if clean:
        resolved_command.append("--clean")

    if dry_run:
        print(
            json.dumps(
                {
                    "object_id": object_id,
                    "view_source": view_source,
                    "model_dir": str(model_dir_path),
                    "output_dir": str(output_dir_path),
                    "source_path": str(source_path_resolved),
                    "mesh": str(mesh_path),
                    "manifest": str(manifest_path),
                    "modal_app": APP_NAME,
                    "repo_commit": GAUSTUDIO_REPO_COMMIT,
                    "load_iteration": load_iteration,
                    "resolution": resolution,
                    "sh_degree": sh_degree,
                    "white_background": white_background,
                    "clean": clean,
                    "dry_run": True,
                    "resolved_command": " ".join(shlex.quote(part) for part in resolved_command),
                },
                indent=2,
            )
        )
        return

    if not model_dir_path.exists():
        raise FileNotFoundError(f"GauStudio input model directory does not exist: {model_dir_path}")
    if not source_path_resolved.exists():
        raise FileNotFoundError(f"GauStudio source_path does not exist: {source_path_resolved}")

    source_bytes = source_path_resolved.read_bytes()
    remote_result = run_gaustudio_mesh_remote.remote(
        model_archive=_build_model_archive(model_dir_path),
        source_path_bytes=source_bytes,
        source_path_name=source_path_resolved.name,
        load_iteration=load_iteration,
        resolution=resolution,
        sh_degree=sh_degree,
        white_background=white_background,
        clean=clean,
    )

    output_dir_path.mkdir(parents=True, exist_ok=True)
    _extract_archive(remote_result["output_archive"], output_dir_path)
    if not mesh_path.exists():
        mesh_path.write_bytes(remote_result["mesh_bytes"])

    manifest_path.write_text(
        json.dumps(
            {
                "object_id": object_id,
                "view_source": view_source,
                "model_dir": str(model_dir_path),
                "output_dir": str(output_dir_path),
                "mesh": str(mesh_path),
                "source_path": str(source_path_resolved),
                "modal_app": APP_NAME,
                "repo_commit": remote_result["repo_commit"],
                "load_iteration": remote_result["load_iteration"],
                "resolution": remote_result["resolution"],
                "sh_degree": remote_result["sh_degree"],
                "white_background": remote_result["white_background"],
                "clean": remote_result["clean"],
                "started_at_utc": remote_result["started_at_utc"],
                "finished_at_utc": remote_result["finished_at_utc"],
                "runtime_seconds": remote_result["runtime_seconds"],
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
                "model_dir": str(model_dir_path),
                "output_dir": str(output_dir_path),
                "mesh": str(mesh_path),
                "manifest": str(manifest_path),
                "runtime_seconds": remote_result["runtime_seconds"],
            },
            indent=2,
        )
    )
