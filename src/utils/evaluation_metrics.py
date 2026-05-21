import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.spatial import cKDTree

from .io import ensure_dir, list_image_files, load_json


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray


def read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return load_json(path)


def find_first_existing(paths: Iterable[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def load_mesh(path: Path) -> MeshData:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _load_obj_mesh(path)
    if suffix == ".ply":
        return _load_ply_mesh(path)
    raise ValueError(f"Unsupported mesh format: {path}")


def _load_obj_mesh(path: Path) -> MeshData:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            parts = line.split()[1:]
            indices = []
            for part in parts:
                token = part.split("/")[0]
                if token:
                    indices.append(int(token) - 1)
            faces.extend(_triangulate_face(indices))
    return MeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32) if faces else np.empty((0, 3), dtype=np.int32),
    )


def _load_ply_mesh(path: Path) -> MeshData:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    vertices = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64, copy=False)

    if "face" not in ply:
        return MeshData(vertices=vertices, faces=np.empty((0, 3), dtype=np.int32))

    faces: list[list[int]] = []
    for face in ply["face"].data["vertex_indices"]:
        faces.extend(_triangulate_face([int(idx) for idx in face]))
    return MeshData(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int32) if faces else np.empty((0, 3), dtype=np.int32),
    )


def _triangulate_face(indices: Sequence[int]) -> list[list[int]]:
    if len(indices) < 3:
        return []
    if len(indices) == 3:
        return [list(indices)]
    return [[indices[0], indices[i], indices[i + 1]] for i in range(1, len(indices) - 1)]


def mesh_has_geometry(mesh: MeshData) -> bool:
    return mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0


def mesh_topology_metrics(mesh: MeshData) -> dict[str, Any]:
    vertex_count = int(mesh.vertices.shape[0])
    face_count = int(mesh.faces.shape[0])
    if vertex_count == 0:
        return {
            "vertex_count": 0,
            "face_count": face_count,
            "connected_components": 0,
            "bounding_box_diagonal": 0.0,
        }

    min_corner = mesh.vertices.min(axis=0)
    max_corner = mesh.vertices.max(axis=0)
    bounding_box_diagonal = float(np.linalg.norm(max_corner - min_corner))
    connected_components = _count_face_components(mesh.faces)
    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "connected_components": connected_components,
        "bounding_box_diagonal": round(bounding_box_diagonal, 6),
    }


def _bounding_box_extents(mesh: MeshData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mesh.vertices.shape[0] == 0:
        zeros = np.zeros(3, dtype=np.float64)
        return zeros, zeros, zeros
    min_corner = mesh.vertices.min(axis=0)
    max_corner = mesh.vertices.max(axis=0)
    return min_corner, max_corner, max_corner - min_corner


def _vertex_centroid(mesh: MeshData) -> np.ndarray:
    if mesh.vertices.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)
    return mesh.vertices.mean(axis=0)


def _count_face_components(faces: np.ndarray) -> int:
    if faces.shape[0] == 0:
        return 0

    parent = list(range(faces.shape[0]))
    rank = [0] * faces.shape[0]
    vertex_owner: dict[int, int] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for face_index, face in enumerate(faces):
        for vertex_index in face:
            owner = vertex_owner.get(int(vertex_index))
            if owner is None:
                vertex_owner[int(vertex_index)] = face_index
            else:
                union(face_index, owner)

    roots = {find(index) for index in range(faces.shape[0])}
    return len(roots)


def sample_mesh_surface(mesh: MeshData, sample_count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if not mesh_has_geometry(mesh) or sample_count <= 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    triangles = mesh.vertices[mesh.faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    face_normals = np.cross(edge_a, edge_b)
    areas = np.linalg.norm(face_normals, axis=1) * 0.5
    valid = areas > 1e-12
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    triangles = triangles[valid]
    face_normals = face_normals[valid]
    areas = areas[valid]
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)

    probabilities = areas / areas.sum()
    rng = np.random.default_rng(seed)
    chosen_faces = rng.choice(len(triangles), size=sample_count, p=probabilities)

    r1 = np.sqrt(rng.random(sample_count))
    r2 = rng.random(sample_count)
    a = triangles[chosen_faces, 0]
    b = triangles[chosen_faces, 1]
    c = triangles[chosen_faces, 2]
    points = ((1.0 - r1)[:, None] * a) + ((r1 * (1.0 - r2))[:, None] * b) + ((r1 * r2)[:, None] * c)
    normals = face_normals[chosen_faces]
    return points.astype(np.float64, copy=False), normals.astype(np.float64, copy=False)


def compute_geometry_metrics(
    predicted_mesh_path: Path,
    ground_truth_mesh_path: Path,
    sample_count: int = 10000,
    completeness_threshold_ratio: float = 0.01,
) -> dict[str, Any]:
    predicted_mesh = load_mesh(predicted_mesh_path)
    ground_truth_mesh = load_mesh(ground_truth_mesh_path)
    predicted_topology = mesh_topology_metrics(predicted_mesh)
    ground_truth_topology = mesh_topology_metrics(ground_truth_mesh)

    if not mesh_has_geometry(predicted_mesh) or not mesh_has_geometry(ground_truth_mesh):
        return {
            "mesh_format": predicted_mesh_path.suffix.lower(),
            "mesh_has_geometry": mesh_has_geometry(predicted_mesh),
            "ground_truth_has_geometry": mesh_has_geometry(ground_truth_mesh),
            "sample_count": sample_count,
            "predicted_topology": predicted_topology,
            "ground_truth_topology": ground_truth_topology,
            "chamfer_distance": None,
            "mesh_completeness": None,
            "mesh_accuracy": None,
            "normal_consistency": None,
        }

    predicted_points, predicted_normals = sample_mesh_surface(predicted_mesh, sample_count, seed=0)
    ground_truth_points, ground_truth_normals = sample_mesh_surface(ground_truth_mesh, sample_count, seed=1)

    if predicted_points.shape[0] == 0 or ground_truth_points.shape[0] == 0:
        return {
            "mesh_format": predicted_mesh_path.suffix.lower(),
            "mesh_has_geometry": False,
            "ground_truth_has_geometry": False,
            "sample_count": sample_count,
            "predicted_topology": predicted_topology,
            "ground_truth_topology": ground_truth_topology,
            "chamfer_distance": None,
            "mesh_completeness": None,
            "mesh_accuracy": None,
            "normal_consistency": None,
        }

    gt_tree = cKDTree(ground_truth_points)
    pred_tree = cKDTree(predicted_points)

    pred_to_gt_distances, pred_to_gt_indices = gt_tree.query(predicted_points, k=1)
    gt_to_pred_distances, gt_to_pred_indices = pred_tree.query(ground_truth_points, k=1)

    chamfer_distance = float(pred_to_gt_distances.mean() + gt_to_pred_distances.mean())
    rmse = float(np.sqrt(np.mean(np.concatenate([pred_to_gt_distances**2, gt_to_pred_distances**2]))))
    hausdorff_distance = float(max(pred_to_gt_distances.max(initial=0.0), gt_to_pred_distances.max(initial=0.0)))

    diagonal = max(
        predicted_topology["bounding_box_diagonal"],
        ground_truth_topology["bounding_box_diagonal"],
        1e-6,
    )
    threshold = diagonal * completeness_threshold_ratio
    mesh_accuracy = float(np.mean(pred_to_gt_distances <= threshold))
    mesh_completeness = float(np.mean(gt_to_pred_distances <= threshold))
    f1_score = (
        float((2.0 * mesh_accuracy * mesh_completeness) / (mesh_accuracy + mesh_completeness))
        if (mesh_accuracy + mesh_completeness) > 1e-12
        else 0.0
    )

    pred_normal_matches = ground_truth_normals[pred_to_gt_indices]
    gt_normal_matches = predicted_normals[gt_to_pred_indices]
    pred_alignment = np.abs(np.sum(predicted_normals * pred_normal_matches, axis=1))
    gt_alignment = np.abs(np.sum(ground_truth_normals * gt_normal_matches, axis=1))
    normal_consistency = float((pred_alignment.mean() + gt_alignment.mean()) / 2.0)

    predicted_centroid = _vertex_centroid(predicted_mesh)
    ground_truth_centroid = _vertex_centroid(ground_truth_mesh)
    center_offset = float(np.linalg.norm(predicted_centroid - ground_truth_centroid))

    _, _, predicted_extents = _bounding_box_extents(predicted_mesh)
    _, _, ground_truth_extents = _bounding_box_extents(ground_truth_mesh)
    safe_ground_truth_extents = np.where(np.abs(ground_truth_extents) <= 1e-12, 1.0, ground_truth_extents)
    extent_ratio = predicted_extents / safe_ground_truth_extents

    ground_truth_diagonal = max(ground_truth_topology["bounding_box_diagonal"], 1e-6)
    bbox_diagonal_ratio = float(predicted_topology["bounding_box_diagonal"] / ground_truth_diagonal)
    scale_factor_error = float(abs(1.0 - bbox_diagonal_ratio))

    return {
        "mesh_format": predicted_mesh_path.suffix.lower(),
        "mesh_has_geometry": True,
        "ground_truth_has_geometry": True,
        "sample_count": sample_count,
        "completeness_threshold": round(threshold, 6),
        "predicted_topology": predicted_topology,
        "ground_truth_topology": ground_truth_topology,
        "chamfer_distance": round(chamfer_distance, 6),
        "rmse": round(rmse, 6),
        "hausdorff_distance": round(hausdorff_distance, 6),
        "mesh_completeness": round(mesh_completeness, 6),
        "mesh_accuracy": round(mesh_accuracy, 6),
        "f1_score": round(f1_score, 6),
        "normal_consistency": round(normal_consistency, 6),
        "bbox_diagonal_ratio": round(bbox_diagonal_ratio, 6),
        "scale_factor_error": round(scale_factor_error, 6),
        "center_offset": round(center_offset, 6),
        "dimensional_accuracy": {
            "predicted_extents": [round(float(value), 6) for value in predicted_extents.tolist()],
            "ground_truth_extents": [round(float(value), 6) for value in ground_truth_extents.tolist()],
            "axis_ratio": {
                "x": round(float(extent_ratio[0]), 6),
                "y": round(float(extent_ratio[1]), 6),
                "z": round(float(extent_ratio[2]), 6),
            },
            "axis_absolute_error": {
                "x": round(float(abs(predicted_extents[0] - ground_truth_extents[0])), 6),
                "y": round(float(abs(predicted_extents[1] - ground_truth_extents[1])), 6),
                "z": round(float(abs(predicted_extents[2] - ground_truth_extents[2])), 6),
            },
        },
        "centroids": {
            "predicted": [round(float(value), 6) for value in predicted_centroid.tolist()],
            "ground_truth": [round(float(value), 6) for value in ground_truth_centroid.tolist()],
        },
    }


def _relative_image_map(directory: Path) -> dict[str, Path]:
    return {path.relative_to(directory).as_posix(): path for path in list_image_files(directory, recursive=True)}


def match_image_pairs(reference_dir: Path, predicted_dir: Path) -> list[tuple[Path, Path]]:
    reference_images = _relative_image_map(reference_dir)
    predicted_images = _relative_image_map(predicted_dir)
    common = sorted(reference_images.keys() & predicted_images.keys())
    if common:
        return [(reference_images[name], predicted_images[name]) for name in common]

    reference_by_name = {path.name: path for path in reference_images.values()}
    predicted_by_name = {path.name: path for path in predicted_images.values()}
    if len(reference_by_name) != len(reference_images) or len(predicted_by_name) != len(predicted_images):
        return []

    common_names = sorted(reference_by_name.keys() & predicted_by_name.keys())
    return [(reference_by_name[name], predicted_by_name[name]) for name in common_names]


def compute_render_metrics(reference_dir: Path, predicted_dir: Path) -> dict[str, Any]:
    return compare_render_directories(reference_dir, predicted_dir)["summary"]


def compare_render_directories(
    reference_dir: Path,
    predicted_dir: Path,
    comparison_dir: Path | None = None,
    include_per_frame: bool = True,
    compute_lpips: bool = True,
) -> dict[str, Any]:
    pairs = match_image_pairs(reference_dir, predicted_dir)
    if not pairs:
        return {
            "summary": {
                "available": False,
                "reference_dir": str(reference_dir),
                "predicted_dir": str(predicted_dir),
                "comparison_dir": str(comparison_dir) if comparison_dir is not None else None,
                "matched_image_count": 0,
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "notes": "No matched image filenames found for render-based evaluation.",
            },
            "frames": [],
        }

    if comparison_dir is not None:
        ensure_dir(comparison_dir)

    lpips_model, lpips_note = _try_create_lpips_model() if compute_lpips else (None, "LPIPS disabled for this run.")
    frame_rows: list[dict[str, Any]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    for reference_path, predicted_path in pairs:
        reference = _load_rgb_image(reference_path)
        predicted = _load_rgb_image(predicted_path, target_size=(reference.shape[1], reference.shape[0]))
        psnr_value = _compute_psnr(reference, predicted)
        ssim_value = _compute_ssim(reference, predicted)
        mse_value = float(np.mean((reference - predicted) ** 2))
        mae_value = float(np.mean(np.abs(reference - predicted)))
        lpips_value = _compute_lpips(reference, predicted, lpips_model)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        if lpips_value is not None:
            lpips_values.append(lpips_value)

        comparison_image_path = None
        if comparison_dir is not None:
            comparison_image_path = comparison_dir / reference_path.name
            _write_render_comparison_image(
                reference,
                predicted,
                comparison_image_path,
                title=reference_path.name,
                psnr=psnr_value,
                ssim=ssim_value,
                lpips=lpips_value,
            )

        if include_per_frame:
            frame_rows.append(
                {
                    "frame_name": reference_path.name,
                    "reference_path": str(reference_path),
                    "predicted_path": str(predicted_path),
                    "comparison_image_path": str(comparison_image_path) if comparison_image_path is not None else None,
                    "width": int(reference.shape[1]),
                    "height": int(reference.shape[0]),
                    "mse": round(mse_value, 6),
                    "mae": round(mae_value, 6),
                    "psnr": round(psnr_value, 6),
                    "ssim": round(ssim_value, 6),
                    "lpips": round(lpips_value, 6) if lpips_value is not None else None,
                }
            )

    summary = {
        "available": True,
        "reference_dir": str(reference_dir),
        "predicted_dir": str(predicted_dir),
        "comparison_dir": str(comparison_dir) if comparison_dir is not None else None,
        "matched_image_count": len(pairs),
        "psnr": round(float(np.mean(psnr_values)), 6),
        "ssim": round(float(np.mean(ssim_values)), 6),
        "lpips": round(float(np.mean(lpips_values)), 6) if lpips_values else None,
        "notes": lpips_note,
    }
    return {"summary": summary, "frames": frame_rows}


def _load_rgb_image(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if target_size is not None and image.size != target_size:
        image = image.resize(target_size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float64) / 255.0


def _compute_psnr(reference: np.ndarray, predicted: np.ndarray) -> float:
    mse = float(np.mean((reference - predicted) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def _compute_ssim(reference: np.ndarray, predicted: np.ndarray) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    scores: list[float] = []
    for channel in range(reference.shape[2]):
        ref = reference[:, :, channel]
        pred = predicted[:, :, channel]
        mu_x = float(ref.mean())
        mu_y = float(pred.mean())
        sigma_x = float(ref.var())
        sigma_y = float(pred.var())
        sigma_xy = float(((ref - mu_x) * (pred - mu_y)).mean())
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        scores.append(numerator / denominator if denominator > 0 else 0.0)
    return float(np.mean(scores))


def _try_create_lpips_model() -> tuple[Any | None, str]:
    try:
        import lpips
        import torch
    except ImportError:
        return None, "LPIPS is unavailable locally because torch/lpips is not installed."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model, f"LPIPS computed with alex on {device}."


def _compute_lpips(reference: np.ndarray, predicted: np.ndarray, model: Any | None) -> float | None:
    if model is None:
        return None
    import torch

    def to_tensor(image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return ((tensor * 2.0) - 1.0).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        value = model(to_tensor(reference), to_tensor(predicted))
    return float(value.item())


def _write_render_comparison_image(
    reference: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title: str,
    psnr: float,
    ssim: float,
    lpips: float | None,
) -> None:
    diff = np.abs(reference - predicted).mean(axis=2)
    ref_image = Image.fromarray(np.clip(reference * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    pred_image = Image.fromarray(np.clip(predicted * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    diff_gray = np.clip(diff * 255.0 * 4.0, 0, 255).astype(np.uint8)
    diff_image = Image.fromarray(diff_gray, mode="L").convert("RGB")

    width, height = ref_image.size
    header_height = 34
    canvas = Image.new("RGB", (width * 3, height + header_height), color=(24, 24, 24))
    canvas.paste(ref_image, (0, header_height))
    canvas.paste(pred_image, (width, header_height))
    canvas.paste(diff_image, (width * 2, header_height))

    draw = ImageDraw.Draw(canvas)
    lpips_text = "n/a" if lpips is None else f"{lpips:.4f}"
    draw.text(
        (10, 9),
        f"{title} | PSNR {psnr:.3f} | SSIM {ssim:.4f} | LPIPS {lpips_text}",
        fill=(235, 235, 235),
    )
    draw.text((10, height + header_height - 18), "Reference", fill=(255, 255, 255))
    draw.text((width + 10, height + header_height - 18), "Rendered", fill=(255, 255, 255))
    draw.text((width * 2 + 10, height + header_height - 18), "Absolute Diff", fill=(255, 255, 255))
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def runtime_seconds_from_manifest(manifest: dict[str, Any] | None) -> float | None:
    if not manifest:
        return None
    if isinstance(manifest.get("runtime_seconds"), (float, int)):
        return float(manifest["runtime_seconds"])

    for key, value in manifest.items():
        if isinstance(value, dict):
            nested = runtime_seconds_from_manifest(value)
            if nested is not None:
                return nested
    return None


def build_efficiency_metrics(
    input_manifest: dict[str, Any] | None,
    reconstruction_manifest: dict[str, Any] | None,
    meshing_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    input_runtime = runtime_seconds_from_manifest(input_manifest)
    reconstruction_runtime = runtime_seconds_from_manifest(reconstruction_manifest)
    meshing_runtime = runtime_seconds_from_manifest(meshing_manifest)
    runtimes = [value for value in [input_runtime, reconstruction_runtime, meshing_runtime] if value is not None]
    total_runtime = round(sum(runtimes), 6) if runtimes else None
    return {
        "input_preparation_runtime": input_runtime,
        "reconstruction_runtime": reconstruction_runtime,
        "meshing_runtime": meshing_runtime,
        "total_runtime": total_runtime,
    }


def flatten_record(payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                walk(next_prefix, nested)
        elif isinstance(value, list):
            flattened[prefix] = json.dumps(value, sort_keys=True)
        else:
            flattened[prefix] = value

    walk("", payload)
    return flattened


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
