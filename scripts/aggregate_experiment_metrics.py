#!/usr/bin/env python3
import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import load_json, upsert_experiment_summary, write_json
from utils.evaluation_metrics import flatten_record, write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate finished dissertation experiments into the final CSV, JSONL, and SQLite analysis outputs."
    )
    parser.add_argument(
        "--experiments_root",
        default=str((PROJECT_ROOT / "results" / "experiments").resolve()),
        help="Root directory containing experiment folders.",
    )
    parser.add_argument(
        "--output_dir",
        default=str((PROJECT_ROOT / "results" / "analysis").resolve()),
        help="Directory where aggregated analysis files will be written.",
    )
    parser.add_argument(
        "--database",
        default=str((PROJECT_ROOT / "results" / "analysis" / "reconstruction_study.db").resolve()),
        help="SQLite database path for experiment summaries and render-comparison records.",
    )
    return parser.parse_args()


def load_database_summaries(database_path: Path) -> dict[str, dict[str, object]]:
    if not database_path.exists():
        return {}
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT * FROM experiment_summaries").fetchall()
    finally:
        connection.close()
    return {
        str(row["experiment_id"]): {key: row[key] for key in row.keys()}
        for row in rows
        if row["experiment_id"] is not None
    }


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    database_path = Path(args.database).resolve()
    database_summaries = load_database_summaries(database_path)

    analysis_records: list[dict[str, object]] = []
    for record_path in sorted(experiments_root.glob("*/evaluation/analysis_record.json")):
        payload = load_json(record_path)
        payload["analysis_record_path"] = str(record_path)
        experiment_id = str(payload.get("experiment_id") or "")
        summary_row = database_summaries.get(experiment_id)
        if summary_row is not None:
            for field in [
                "psnr",
                "ssim",
                "lpips",
                "matched_render_count",
                "reconstruction_runtime",
                "meshing_runtime",
                "total_runtime",
            ]:
                if summary_row.get(field) is not None:
                    payload[field] = summary_row[field]
        analysis_records.append(payload)
        upsert_experiment_summary(database_path, payload)

    flattened_rows = [flatten_record(record) for record in analysis_records]
    summary_rows = [
        {
            "experiment_id": record.get("experiment_id"),
            "object_id": record.get("object_id"),
            "view_count": record.get("study_view_count"),
            "number_of_orbits": record.get("study_number_of_orbits"),
            "elevation_coverage": record.get("study_elevation_coverage"),
            "image_resolution": record.get("study_image_resolution"),
            "scale": record.get("study_scale"),
            "scale_factor_found": record.get("bbox_diagonal_ratio"),
            "scale_factor_error": record.get("scale_factor_error"),
            "rmse_mm": record.get("rmse"),
            "f1_score": record.get("f1_score"),
            "chamfer_distance": record.get("chamfer_distance"),
            "psnr": record.get("psnr"),
            "ssim": record.get("ssim"),
            "lpips_visual_loss": record.get("lpips"),
            "matched_render_count": record.get("matched_render_count"),
            "reconstruction_runtime": record.get("reconstruction_runtime"),
            "meshing_runtime": record.get("meshing_runtime"),
            "total_runtime": record.get("total_runtime"),
        }
        for record in analysis_records
    ]
    summary = {
        "experiments_root": str(experiments_root),
        "output_dir": str(output_dir),
        "database": str(database_path),
        "record_count": len(analysis_records),
        "source_files": [record["analysis_record_path"] for record in analysis_records],
    }

    write_json(output_dir / "aggregation_summary.json", summary)
    write_json(output_dir / "experiment_metrics.json", {"records": analysis_records})
    write_jsonl(output_dir / "experiment_metrics.jsonl", analysis_records)
    write_csv(output_dir / "experiment_metrics.csv", flattened_rows)
    write_csv(output_dir / "study_summary.csv", summary_rows)

    print(
        json.dumps(
            {
                "record_count": len(analysis_records),
                "summary": str((output_dir / "aggregation_summary.json").resolve()),
                "json": str((output_dir / "experiment_metrics.json").resolve()),
                "jsonl": str((output_dir / "experiment_metrics.jsonl").resolve()),
                "csv": str((output_dir / "experiment_metrics.csv").resolve()),
                "table_csv": str((output_dir / "study_summary.csv").resolve()),
                "database": str(database_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
