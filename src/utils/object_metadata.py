import json
from pathlib import Path
from typing import Any


def resolve_object_root(project_root: Path, object_id: str) -> Path:
    return (project_root / "data" / "objects" / object_id).resolve()


def load_object_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Object metadata file does not exist: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_object_metadata_path(project_root: Path, object_id: str) -> Path:
    return resolve_object_root(project_root, object_id) / "object_metadata.json"


def resolve_object_input_image(object_root: Path) -> Path:
    input_dir = object_root / "input"
    candidates = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not candidates:
        raise FileNotFoundError(f"No input image found in {input_dir}")
    return candidates[0]


def try_resolve_object_input_image(object_root: Path) -> Path | None:
    try:
        return resolve_object_input_image(object_root)
    except FileNotFoundError:
        return None
