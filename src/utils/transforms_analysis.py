import math
from pathlib import Path
from typing import Any

from .io import load_json


def _normalize_angle_deg(angle_deg: float) -> float:
    normalized = angle_deg % 360.0
    return normalized if normalized >= 0.0 else normalized + 360.0


def _band_label(elevation_deg: float) -> str:
    return f"{round(elevation_deg, 3):.3f}".rstrip("0").rstrip(".")


def _circular_span_deg(azimuths_deg: list[float]) -> float:
    if not azimuths_deg:
        return 0.0
    if len(azimuths_deg) == 1:
        return 0.0

    ordered = sorted(_normalize_angle_deg(value) for value in azimuths_deg)
    gaps = [ordered[index + 1] - ordered[index] for index in range(len(ordered) - 1)]
    gaps.append((ordered[0] + 360.0) - ordered[-1])
    return 360.0 - max(gaps)


def analyze_transforms_json(transforms_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "transforms_json": str(transforms_path),
        "exists": transforms_path.exists(),
        "actual_frame_count": 0,
        "actual_orbit_count": 0,
        "elevation_bands_deg": [],
        "min_elevation_deg": None,
        "max_elevation_deg": None,
        "azimuth_min_deg": None,
        "azimuth_max_deg": None,
        "azimuth_coverage_span_deg": None,
        "per_orbit_frame_counts": {},
        "duplicate_frame_file_count": 0,
        "unique_frame_file_count": 0,
    }
    if not transforms_path.exists():
        return summary

    payload = load_json(transforms_path)
    frames = list(payload.get("frames", []))
    summary["actual_frame_count"] = len(frames)
    if not frames:
        return summary

    elevation_bands: dict[str, float] = {}
    per_orbit_frame_counts: dict[str, int] = {}
    azimuths_deg: list[float] = []
    frame_paths: list[str] = []
    raw_elevations_deg: list[float] = []

    for frame in frames:
        matrix = frame.get("transform_matrix") or []
        if len(matrix) < 3 or any(len(row) < 4 for row in matrix[:3]):
            continue

        x = float(matrix[0][3])
        y = float(matrix[1][3])
        z = float(matrix[2][3])
        horizontal_radius = math.hypot(x, y)
        raw_elevation_deg = math.degrees(math.atan2(z, horizontal_radius))
        raw_elevations_deg.append(raw_elevation_deg)
        band = _band_label(raw_elevation_deg)
        elevation_bands[band] = round(raw_elevation_deg, 3)
        per_orbit_frame_counts[band] = per_orbit_frame_counts.get(band, 0) + 1
        azimuths_deg.append(_normalize_angle_deg(math.degrees(math.atan2(x, y))))
        frame_paths.append(str(frame.get("file_path", "")))

    ordered_bands = [elevation_bands[key] for key in sorted(elevation_bands, key=lambda item: elevation_bands[item])]
    summary["actual_orbit_count"] = len(ordered_bands)
    summary["elevation_bands_deg"] = ordered_bands
    summary["min_elevation_deg"] = round(min(raw_elevations_deg), 6) if raw_elevations_deg else None
    summary["max_elevation_deg"] = round(max(raw_elevations_deg), 6) if raw_elevations_deg else None
    summary["azimuth_min_deg"] = round(min(azimuths_deg), 6) if azimuths_deg else None
    summary["azimuth_max_deg"] = round(max(azimuths_deg), 6) if azimuths_deg else None
    summary["azimuth_coverage_span_deg"] = round(_circular_span_deg(azimuths_deg), 6) if azimuths_deg else None
    summary["per_orbit_frame_counts"] = {
        band: per_orbit_frame_counts[band] for band in sorted(per_orbit_frame_counts, key=lambda item: elevation_bands[item])
    }
    summary["unique_frame_file_count"] = len(set(frame_paths))
    summary["duplicate_frame_file_count"] = len(frame_paths) - len(set(frame_paths))
    return summary
