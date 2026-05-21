import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .io import ensure_dir


EXPERIMENT_SUMMARY_COLUMNS = [
    "experiment_id",
    "object_id",
    "reconstruction_model",
    "study_view_count",
    "study_number_of_orbits",
    "study_elevation_coverage",
    "study_image_resolution",
    "study_scale",
    "chamfer_distance",
    "rmse",
    "hausdorff_distance",
    "mesh_completeness",
    "mesh_accuracy",
    "f1_score",
    "scale_factor_error",
    "bbox_diagonal_ratio",
    "center_offset",
    "psnr",
    "ssim",
    "lpips",
    "matched_render_count",
    "reconstruction_runtime",
    "meshing_runtime",
    "total_runtime",
    "analysis_record_path",
    "updated_at_utc",
]


def _connect_results_database(path: Path) -> sqlite3.Connection:
    ensure_dir(path.parent)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_summaries (
            experiment_id TEXT PRIMARY KEY,
            object_id TEXT,
            reconstruction_model TEXT,
            study_view_count INTEGER,
            study_number_of_orbits INTEGER,
            study_elevation_coverage TEXT,
            study_image_resolution TEXT,
            study_scale REAL,
            chamfer_distance REAL,
            rmse REAL,
            hausdorff_distance REAL,
            mesh_completeness REAL,
            mesh_accuracy REAL,
            f1_score REAL,
            scale_factor_error REAL,
            bbox_diagonal_ratio REAL,
            center_offset REAL,
            psnr REAL,
            ssim REAL,
            lpips REAL,
            matched_render_count INTEGER,
            reconstruction_runtime REAL,
            meshing_runtime REAL,
            total_runtime REAL,
            analysis_record_path TEXT,
            updated_at_utc TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS render_runs (
            experiment_id TEXT,
            run_name TEXT,
            reference_dir TEXT,
            predicted_dir TEXT,
            comparison_dir TEXT,
            matched_image_count INTEGER,
            psnr REAL,
            ssim REAL,
            lpips REAL,
            summary_json_path TEXT,
            created_at_utc TEXT,
            PRIMARY KEY (experiment_id, run_name)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS render_frame_metrics (
            experiment_id TEXT,
            run_name TEXT,
            frame_name TEXT,
            reference_path TEXT,
            predicted_path TEXT,
            comparison_image_path TEXT,
            width INTEGER,
            height INTEGER,
            mse REAL,
            mae REAL,
            psnr REAL,
            ssim REAL,
            lpips REAL,
            PRIMARY KEY (experiment_id, run_name, frame_name)
        )
        """
    )
    connection.commit()
    return connection


def upsert_experiment_summary(database_path: Path, row: dict[str, Any]) -> None:
    connection = _connect_results_database(database_path)
    payload = {column: row.get(column) for column in EXPERIMENT_SUMMARY_COLUMNS}
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    assignments = ", ".join(f"{column} = excluded.{column}" for column in EXPERIMENT_SUMMARY_COLUMNS[1:])
    placeholders = ", ".join("?" for _ in EXPERIMENT_SUMMARY_COLUMNS)
    with connection:
        connection.execute(
            f"""
            INSERT INTO experiment_summaries ({", ".join(EXPERIMENT_SUMMARY_COLUMNS)})
            VALUES ({placeholders})
            ON CONFLICT(experiment_id) DO UPDATE SET {assignments}
            """,
            [payload[column] for column in EXPERIMENT_SUMMARY_COLUMNS],
        )
    connection.close()


def replace_render_run(
    database_path: Path,
    run_summary: dict[str, Any],
    frame_rows: Sequence[dict[str, Any]],
) -> None:
    connection = _connect_results_database(database_path)
    run_name = str(run_summary["run_name"])
    experiment_id = str(run_summary["experiment_id"])
    with connection:
        connection.execute(
            """
            INSERT INTO render_runs (
                experiment_id,
                run_name,
                reference_dir,
                predicted_dir,
                comparison_dir,
                matched_image_count,
                psnr,
                ssim,
                lpips,
                summary_json_path,
                created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id, run_name) DO UPDATE SET
                reference_dir = excluded.reference_dir,
                predicted_dir = excluded.predicted_dir,
                comparison_dir = excluded.comparison_dir,
                matched_image_count = excluded.matched_image_count,
                psnr = excluded.psnr,
                ssim = excluded.ssim,
                lpips = excluded.lpips,
                summary_json_path = excluded.summary_json_path,
                created_at_utc = excluded.created_at_utc
            """,
            [
                experiment_id,
                run_name,
                run_summary.get("reference_dir"),
                run_summary.get("predicted_dir"),
                run_summary.get("comparison_dir"),
                run_summary.get("matched_image_count"),
                run_summary.get("psnr"),
                run_summary.get("ssim"),
                run_summary.get("lpips"),
                run_summary.get("summary_json_path"),
                datetime.now(timezone.utc).isoformat(),
            ],
        )
        connection.execute(
            "DELETE FROM render_frame_metrics WHERE experiment_id = ? AND run_name = ?",
            [experiment_id, run_name],
        )
        connection.executemany(
            """
            INSERT INTO render_frame_metrics (
                experiment_id,
                run_name,
                frame_name,
                reference_path,
                predicted_path,
                comparison_image_path,
                width,
                height,
                mse,
                mae,
                psnr,
                ssim,
                lpips
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                [
                    experiment_id,
                    run_name,
                    row.get("frame_name"),
                    row.get("reference_path"),
                    row.get("predicted_path"),
                    row.get("comparison_image_path"),
                    row.get("width"),
                    row.get("height"),
                    row.get("mse"),
                    row.get("mae"),
                    row.get("psnr"),
                    row.get("ssim"),
                    row.get("lpips"),
                ]
                for row in frame_rows
            ],
        )
    connection.close()
