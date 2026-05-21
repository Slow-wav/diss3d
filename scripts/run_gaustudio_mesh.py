#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODAL_APP_REF = f"{(PROJECT_ROOT / 'modal' / 'gaustudio_mesh_app.py').resolve()}::main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GauStudio mesh-extraction step for one trained reconstruction.")
    parser.add_argument("--object_id", required=True, help="Object folder name under data/objects/, for example 'shark'.")
    parser.add_argument("--view_source", choices=["real_views"], default="real_views", help="Which Gaussian Splatting real-view branch to mesh.")
    parser.add_argument("--model_dir", default=None, help="Optional direct path to a Gaussian Splatting model directory.")
    parser.add_argument("--output_dir", default=None, help="Optional output directory for GauStudio artifacts.")
    parser.add_argument("--source_path", default=None, help="Optional direct path to cameras.json or source dataset override.")
    parser.add_argument("--load_iteration", type=int, default=-1, help="Gaussian iteration to load; -1 uses the latest iteration.")
    parser.add_argument("--resolution", type=int, default=2, help="Downscale factor used by GauStudio during extraction.")
    parser.add_argument("--sh_degree", type=int, default=0, help="SH degree override passed to GauStudio.")
    parser.add_argument("--white_background", action="store_true", help="Pass GauStudio's --white_background flag.")
    parser.add_argument("--clean", action="store_true", help="Run GauStudio's connected-component cleanup pass.")
    parser.add_argument("--dry_run", action="store_true", help="Print the resolved Modal command without executing it.")
    return parser.parse_args()


def add_option(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def build_modal_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "modal",
        "run",
        MODAL_APP_REF,
        "--object-id",
        args.object_id,
        "--view-source",
        args.view_source,
    ]
    add_option(command, "--model-dir", Path(args.model_dir).resolve() if args.model_dir else None)
    add_option(command, "--output-dir", Path(args.output_dir).resolve() if args.output_dir else None)
    add_option(command, "--source-path", Path(args.source_path).resolve() if args.source_path else None)
    add_option(command, "--load-iteration", args.load_iteration)
    add_option(command, "--resolution", args.resolution)
    add_option(command, "--sh-degree", args.sh_degree)
    if args.white_background:
        command.append("--white-background")
    if args.clean:
        command.append("--clean")
    if args.dry_run:
        command.append("--dry-run")
    return command


def manifest_path(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif args.model_dir:
        output_dir = Path(args.model_dir).resolve().parent / "gaustudio_mesh"
    else:
        output_dir = (PROJECT_ROOT / "results" / "gaussian_splatting" / f"{args.object_id}_{args.view_source}" / "gaustudio_mesh").resolve()
    return output_dir / "gaustudio_mesh_wrapper_manifest.json"


def write_wrapper_manifest(args: argparse.Namespace, command: list[str]) -> Path:
    path = manifest_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "object_id": args.object_id,
                "view_source": args.view_source,
                "model_dir": str(Path(args.model_dir).resolve()) if args.model_dir else None,
                "output_dir": str(Path(args.output_dir).resolve()) if args.output_dir else None,
                "source_path": str(Path(args.source_path).resolve()) if args.source_path else None,
                "load_iteration": args.load_iteration,
                "resolution": args.resolution,
                "sh_degree": args.sh_degree,
                "white_background": args.white_background,
                "clean": args.clean,
                "execution_backend": "modal-wrapper",
                "modal_app": MODAL_APP_REF,
                "resolved_command": subprocess.list2cmdline(command) if sys.platform.startswith("win") else " ".join(command),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    command = build_modal_command(args)
    path = write_wrapper_manifest(args, command)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "object_id": args.object_id,
                    "view_source": args.view_source,
                    "manifest": str(path),
                    "resolved_command": subprocess.list2cmdline(command) if sys.platform.startswith("win") else " ".join(command),
                    "dry_run": True,
                },
                indent=2,
            )
        )
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
