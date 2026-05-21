#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Matrix
except ImportError as exc:
    raise SystemExit(
        "This script must be run from Blender, for example:\n"
        "blender --background --python scripts/render_evaluation_mesh.py -- "
        "--mesh /abs/path/to/fused_mesh.ply "
        "--transforms /abs/path/to/transforms.json "
        "--output_dir /abs/path/to/predicted_renders"
    ) from exc


def blender_cli_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a reconstructed mesh from the exact cameras stored in a NeRF-style transforms.json."
    )
    parser.add_argument("--mesh", required=True, help="Path to the reconstructed mesh to render.")
    parser.add_argument("--transforms", required=True, help="Path to the reference transforms.json.")
    parser.add_argument("--output_dir", required=True, help="Directory where rendered PNGs will be written.")
    parser.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE"], default="CYCLES", help="Blender render engine.")
    parser.add_argument("--samples", type=int, default=64, help="Render samples for Cycles.")
    parser.add_argument("--camera_scale", type=float, default=1.0, help="Optional multiplier applied only to camera translations.")
    parser.add_argument("--mesh_scale", type=float, default=1.0, help="Optional uniform scale applied to the imported mesh.")
    parser.add_argument("--white_background", action="store_true", help="Render against a white background.")
    parser.add_argument("--transparent_background", action="store_true", help="Render with film transparency enabled.")
    parser.add_argument("--camera_name", default="EvaluationCamera", help="Name for the temporary evaluation camera.")
    parser.add_argument("--manifest_output", default=None, help="Optional JSON manifest describing the render run.")
    return parser.parse_args(blender_cli_args())


def load_transforms(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block_collection in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.materials):
        for block in list(block_collection):
            if block.users == 0:
                block_collection.remove(block)


def import_mesh(mesh_path: Path) -> bpy.types.Object:
    suffix = mesh_path.suffix.lower()
    before = set(bpy.data.objects.keys())

    if suffix == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(mesh_path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(mesh_path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(mesh_path))
        else:
            bpy.ops.import_scene.obj(filepath=str(mesh_path))
    elif suffix == ".stl":
        bpy.ops.import_mesh.stl(filepath=str(mesh_path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")

    imported_names = [name for name in bpy.data.objects.keys() if name not in before]
    imported_objects = [bpy.data.objects[name] for name in imported_names]
    mesh_objects = [obj for obj in imported_objects if obj.type == "MESH"]
    if not mesh_objects:
        raise ValueError(f"No mesh objects were imported from {mesh_path}")

    active = mesh_objects[0]
    bpy.context.view_layer.objects.active = active
    for obj in mesh_objects:
        obj.select_set(True)
    return active


def create_camera(name: str) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new(name=name)
    camera_object = bpy.data.objects.new(name, camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object
    return camera_object


def configure_world(white_background: bool, transparent_background: bool) -> None:
    scene = bpy.context.scene
    scene.render.film_transparent = transparent_background
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0) if white_background else (0.0, 0.0, 0.0, 1.0)


def configure_render_settings(payload: dict[str, object], engine: str, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if scene.render.film_transparent else "RGB"
    scene.render.resolution_x = int(payload["w"])
    scene.render.resolution_y = int(payload["h"])
    scene.render.resolution_percentage = 100
    if engine == "CYCLES":
        scene.cycles.samples = samples


def apply_intrinsics(camera_object: bpy.types.Object, payload: dict[str, object]) -> None:
    camera = camera_object.data
    width = float(payload["w"])
    height = float(payload["h"])
    fl_x = float(payload["fl_x"])
    fl_y = float(payload["fl_y"])

    camera.type = "PERSP"
    camera.sensor_fit = "HORIZONTAL"
    camera.sensor_width = 36.0
    camera.sensor_height = 36.0 * (height / width if width > 0 else 1.0)
    camera.lens = fl_x * camera.sensor_width / width if width > 0 else 50.0

    if width > 0 and height > 0 and abs(fl_x - fl_y) > 1e-6:
        camera.sensor_fit = "AUTO"


def scaled_matrix(matrix_rows: list[list[float]], camera_scale: float) -> Matrix:
    matrix = Matrix(matrix_rows)
    if abs(camera_scale - 1.0) <= 1e-12:
        return matrix
    for row_index in range(3):
        matrix[row_index][3] *= camera_scale
    return matrix


def output_path_for_frame(output_dir: Path, frame_entry: dict[str, object], index: int) -> Path:
    file_path = frame_entry.get("file_path")
    if isinstance(file_path, str) and file_path:
        return output_dir / Path(file_path)
    return output_dir / f"{index + 1:04d}.png"


def render_frames(
    camera_object: bpy.types.Object,
    frames: list[dict[str, object]],
    output_dir: Path,
    camera_scale: float,
) -> list[str]:
    scene = bpy.context.scene
    rendered_files: list[str] = []
    for index, frame_entry in enumerate(frames):
        transform_rows = frame_entry.get("transform_matrix")
        if not isinstance(transform_rows, list) or len(transform_rows) != 4:
            raise ValueError(f"Frame {index} is missing a valid transform_matrix")

        camera_object.matrix_world = scaled_matrix(transform_rows, camera_scale)
        output_path = output_path_for_frame(output_dir, frame_entry, index)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        rendered_files.append(str(output_path))
    return rendered_files


def write_manifest(
    manifest_output: Path,
    mesh_path: Path,
    transforms_path: Path,
    output_dir: Path,
    rendered_files: list[str],
    args: argparse.Namespace,
) -> None:
    payload = {
        "mesh": str(mesh_path),
        "transforms": str(transforms_path),
        "output_dir": str(output_dir),
        "render_count": len(rendered_files),
        "camera_scale": args.camera_scale,
        "mesh_scale": args.mesh_scale,
        "engine": args.engine,
        "samples": args.samples,
        "white_background": args.white_background,
        "transparent_background": args.transparent_background,
        "rendered_files_preview": rendered_files[:5],
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    mesh_path = Path(args.mesh).resolve()
    transforms_path = Path(args.transforms).resolve()
    output_dir = Path(args.output_dir).resolve()
    payload = load_transforms(transforms_path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in transforms file: {transforms_path}")

    clear_scene()
    imported_mesh = import_mesh(mesh_path)
    imported_mesh.scale = (args.mesh_scale, args.mesh_scale, args.mesh_scale)
    bpy.context.view_layer.update()

    camera_object = create_camera(args.camera_name)
    configure_world(args.white_background, args.transparent_background)
    configure_render_settings(payload, args.engine, args.samples)
    apply_intrinsics(camera_object, payload)
    rendered_files = render_frames(camera_object, frames, output_dir, args.camera_scale)

    if args.manifest_output:
        write_manifest(Path(args.manifest_output).resolve(), mesh_path, transforms_path, output_dir, rendered_files, args)

    print(
        json.dumps(
            {
                "mesh": str(mesh_path),
                "transforms": str(transforms_path),
                "output_dir": str(output_dir),
                "render_count": len(rendered_files),
                "camera_scale": args.camera_scale,
                "mesh_scale": args.mesh_scale,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
