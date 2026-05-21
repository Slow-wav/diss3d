#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODAL_APP_REF = f"{(PROJECT_ROOT / 'modal' / 'gaussian_splatting_app.py').resolve()}::main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Gaussian Splatting training step for one prepared dissertation dataset."
    )
    parser.add_argument("--object_id", required=True, help="Object folder name under data/objects/, for example 'shark'.")
    parser.add_argument("--view_source", choices=["real_views"], default="real_views", help="Which object-local real-view folder should be prepared and trained.")
    parser.add_argument("--dataset_dir", default=None, help="Optional override prepared Gaussian Splatting dataset directory.")
    parser.add_argument("--iterations", type=int, default=30000, help="Training iterations.")
    parser.add_argument("--model_path", default=None, help="Output model path.")
    parser.add_argument("--eval", action="store_true", help="Enable train/test split aware training.")
    parser.add_argument("--quiet", action="store_true", help="Pass through Gaussian Splatting's --quiet flag.")
    parser.add_argument("--white_background", action="store_true", help="Composite RGBA inputs onto white inside the remote trainer.")
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
    add_option(command, "--dataset-dir", Path(args.dataset_dir).resolve() if args.dataset_dir else None)
    add_option(command, "--model-path", Path(args.model_path).resolve() if args.model_path else None)
    add_option(command, "--iterations", args.iterations)
    if args.eval:
        command.append("--eval")
    if args.quiet:
        command.append("--quiet")
    if args.white_background:
        command.append("--white-background")
    if args.dry_run:
        command.append("--dry-run")
    return command


def manifest_path(args: argparse.Namespace) -> Path:
    return (PROJECT_ROOT / "results" / "gaussian_splatting" / f"{args.object_id}_{args.view_source}_manifest.json").resolve()


def write_wrapper_manifest(args: argparse.Namespace, command: list[str]) -> Path:
    path = manifest_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "object_id": args.object_id,
                "view_source": args.view_source,
                "dataset_dir": str(Path(args.dataset_dir).resolve()) if args.dataset_dir else None,
                "model_path": str(Path(args.model_path).resolve()) if args.model_path else None,
                "iterations": args.iterations,
                "eval": args.eval,
                "quiet": args.quiet,
                "white_background": args.white_background,
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
