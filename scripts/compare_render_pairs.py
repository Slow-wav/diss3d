#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import ensure_dir, load_json, replace_render_run, upsert_experiment_summary, write_json
from utils.evaluation_metrics import compare_render_directories, write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reconstruction images against matching reference views, save side-by-side outputs, and store the summary in SQLite."
    )
    parser.add_argument("--reference_dir", required=True, help="Directory containing the reference images.")
    parser.add_argument("--predicted_dir", required=True, help="Directory containing rendered reconstruction images.")
    parser.add_argument("--experiment_id", default=None, help="Experiment identifier used for output naming and database rows.")
    parser.add_argument("--run_name", default="render_eval", help="Logical name for this render-comparison run.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory where summary files and frame comparisons will be written. Defaults under results/analysis/render_comparisons/.",
    )
    parser.add_argument(
        "--database",
        default=str((PROJECT_ROOT / "results" / "analysis" / "reconstruction_study.db").resolve()),
        help="SQLite database path for summary and per-frame rows.",
    )
    parser.add_argument(
        "--analysis_record",
        default=None,
        help="Optional analysis_record.json to merge with the new render metrics before writing the summary database row.",
    )
    parser.add_argument(
        "--disable_lpips",
        action="store_true",
        help="Skip LPIPS even if torch/lpips is installed locally.",
    )
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace, experiment_id: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    return (PROJECT_ROOT / "results" / "analysis" / "render_comparisons" / experiment_id / args.run_name).resolve()


def _resolve_experiment_id(args: argparse.Namespace) -> str:
    if args.experiment_id:
        return args.experiment_id
    if args.analysis_record:
        return str(load_json(Path(args.analysis_record).resolve()).get("experiment_id") or "unnamed_experiment")
    return f"{Path(args.predicted_dir).resolve().name}_vs_{Path(args.reference_dir).resolve().name}"


def main() -> None:
    args = parse_args()
    experiment_id = _resolve_experiment_id(args)
    reference_dir = Path(args.reference_dir).resolve()
    predicted_dir = Path(args.predicted_dir).resolve()
    output_dir = _resolve_output_dir(args, experiment_id)
    comparison_dir = output_dir / "comparison_frames"
    database_path = Path(args.database).resolve()

    ensure_dir(output_dir)
    result = compare_render_directories(
        reference_dir=reference_dir,
        predicted_dir=predicted_dir,
        comparison_dir=comparison_dir,
        include_per_frame=True,
        compute_lpips=not args.disable_lpips,
    )

    summary = dict(result["summary"])
    frame_rows = list(result["frames"])
    summary["experiment_id"] = experiment_id
    summary["run_name"] = args.run_name

    summary_json_path = output_dir / "render_metrics_summary.json"
    frame_jsonl_path = output_dir / "render_frame_metrics.jsonl"
    frame_csv_path = output_dir / "render_frame_metrics.csv"

    write_json(summary_json_path, summary)
    write_jsonl(frame_jsonl_path, frame_rows)
    write_csv(frame_csv_path, frame_rows)

    replace_render_run(
        database_path,
        {
            "experiment_id": experiment_id,
            "run_name": args.run_name,
            "reference_dir": str(reference_dir),
            "predicted_dir": str(predicted_dir),
            "comparison_dir": str(comparison_dir),
            "matched_image_count": summary.get("matched_image_count"),
            "psnr": summary.get("psnr"),
            "ssim": summary.get("ssim"),
            "lpips": summary.get("lpips"),
            "summary_json_path": str(summary_json_path),
        },
        frame_rows,
    )

    if args.analysis_record:
        merged_summary: dict[str, Any] = load_json(Path(args.analysis_record).resolve())
        merged_summary["psnr"] = summary.get("psnr")
        merged_summary["ssim"] = summary.get("ssim")
        merged_summary["lpips"] = summary.get("lpips")
        merged_summary["matched_render_count"] = summary.get("matched_image_count")
        merged_summary["analysis_record_path"] = str(Path(args.analysis_record).resolve())
        upsert_experiment_summary(database_path, merged_summary)

    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "run_name": args.run_name,
                "summary_json": str(summary_json_path),
                "frame_jsonl": str(frame_jsonl_path),
                "frame_csv": str(frame_csv_path),
                "comparison_dir": str(comparison_dir),
                "database": str(database_path),
                "matched_image_count": summary.get("matched_image_count"),
                "psnr": summary.get("psnr"),
                "ssim": summary.get("ssim"),
                "lpips": summary.get("lpips"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
