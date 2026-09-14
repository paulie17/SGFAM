import blenderproc as bproc

import argparse
import copy
import os
import shutil

import bpy # type: ignore
import numpy as np
import cv2
import yaml
from PIL import Image


CATEGORY_EXTENSIONS = (".ply", ".glb", ".obj")
FLAT_EXTENSIONS = (".ply", ".glb")
CATEGORY_SKIP_KEYWORDS = ("fixed", "simplified", "manifold")
FLAT_SKIP_KEYWORDS = ("simplified",)
DEFAULT_Z_MAX = 50.0
DEFAULT_LIGHT_LAYOUT = "distributed"


def resolve_path(path_value, base_dir):
    if path_value is None or os.path.isabs(path_value):
        return path_value

    return os.path.normpath(os.path.join(base_dir, path_value))


def resolve_config_paths(config):
    config_dir = config["_config_dir"]

    for key in ("model_dir", "obj_pose", "cam_pose", "output_dir"):
        if config.get(key):
            config[key] = resolve_path(config[key], config_dir)

    return config


def normalize_mesh_extension(extension):
    if extension is None:
        return None

    normalized_extension = extension.strip().lower()
    if not normalized_extension:
        return None

    if not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"

    return normalized_extension


def get_allowed_extensions(config, categorized):
    preferred_extension = normalize_mesh_extension(config.get("preferred_mesh_extension"))
    if preferred_extension is not None:
        return (preferred_extension,)

    return CATEGORY_EXTENSIONS if categorized else FLAT_EXTENSIONS


def get_skip_keywords(config, categorized):
    configured_keywords = config.get("skip_mesh_keywords")
    if configured_keywords is not None:
        return tuple(keyword.lower() for keyword in configured_keywords)

    return CATEGORY_SKIP_KEYWORDS if categorized else FLAT_SKIP_KEYWORDS


def get_z_max(config):
    return config.get("z_max", DEFAULT_Z_MAX)


def get_light_layout(config):
    light_layout = config.get("light_layout", DEFAULT_LIGHT_LAYOUT)
    normalized_layout = light_layout.strip().lower()

    if normalized_layout not in {"distributed", "two-points"}:
        raise ValueError(
            "light_layout must be either 'distributed' or 'two-points'"
        )

    return normalized_layout


def load_materials_from_blend(blend_file_path):
    materials = {}
    objs = bproc.loader.load_blend(blend_file_path)

    for obj in objs:
        obj_materials = obj.get_materials()
        if not obj_materials:
            continue

        name = obj.get_name()
        if name in materials:
            raise ValueError(f"Material with name {name} already exists.")

        materials[name] = obj_materials[0]

    if objs:
        bproc.object.delete_multiple(objs)

    if not materials:
        raise ValueError(f"No materials found in blend file: {blend_file_path}")

    print(f"Successfully loaded {len(materials)} materials from {blend_file_path}")
    return materials


def load_material_from_config(material_config, config_dir):
    if not material_config:
        return None

    blend_file_path = material_config.get("blend_file")
    if not blend_file_path:
        return None

    if not os.path.isabs(blend_file_path):
        blend_file_path = os.path.normpath(os.path.join(config_dir, blend_file_path))

    if not os.path.exists(blend_file_path):
        raise FileNotFoundError(f"Material blend file not found: {blend_file_path}")

    materials = load_materials_from_blend(blend_file_path)
    material_name = material_config.get("name")

    if material_name is not None:
        if material_name not in materials:
            available_materials = ", ".join(sorted(materials))
            raise ValueError(
                f"Material '{material_name}' not found in {blend_file_path}. "
                f"Available materials: {available_materials}"
            )
        return materials[material_name]

    if len(materials) == 1:
        return next(iter(materials.values()))

    available_materials = ", ".join(sorted(materials))
    raise ValueError(
        "Multiple materials found in the blend file. Set material.name in the config. "
        f"Available materials: {available_materials}"
    )


def depth_to_pointcloud(depth_map, K, scale=1.0, z_max=1.3):
    """
    Convert a depth map to a point cloud using camera intrinsics.

    Parameters:
    - depth_map: (H, W) numpy array, depth values.
    - K: (3, 3) numpy array, intrinsic matrix.
    - scale: Scaling factor for depth values if needed.
    - z_max: Maximum depth value; points with Z > z_max will be set to NaN.

    Returns:
    - points: (N, 3) numpy array, 3D points in the camera frame.
    """
    h, w = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Normalize pixel coordinates
    x = (u - cx) / fx
    y = (v - cy) / fy

    # Get depth values and scale them
    Z = depth_map * scale
    X = x * Z
    Y = y * Z

    # Apply truncation condition
    mask = Z > z_max
    X[mask], Y[mask], Z[mask] = np.nan, np.nan, np.nan

    # Stack into (N, 3) point cloud format
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    return points


def load_original_poses(config):
    original_poses = np.load(config["cam_pose"])

    if config["poses"] == "upper":
        original_poses = original_poses[original_poses[:, 2, 3] >= 0]

    return original_poses


def configure_camera(config):
    K_matrix = np.reshape(config["cam"]["K"], (3, 3))
    bproc.camera.set_intrinsics_from_K_matrix(
        K_matrix,
        config["cam"]["width"],
        config["cam"]["height"],
    )
    return K_matrix


def is_valid_mesh_file(filename, config, categorized):
    extensions = get_allowed_extensions(config, categorized)
    skip_keywords = get_skip_keywords(config, categorized)

    return filename.endswith(extensions) and not any(keyword in filename.lower() for keyword in skip_keywords)


def matches_model_filters(filename, model_index, model_name):
    if model_index is not None:
        current_index = int(filename.split("_")[1].split(".")[0])
        if current_index != model_index:
            print(current_index)
            return False

    if model_name is not None and model_name not in filename:
        print(f"Skipping {filename} as it does not match the model name {model_name}")
        return False

    return True


def iter_mesh_entries(config):
    mesh_dir = config["model_dir"]
    model_index = config["model_index"]
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    all_items = sorted(os.listdir(mesh_dir))
    categories = [
        item for item in all_items
        if os.path.isdir(os.path.join(mesh_dir, item)) and "backup" not in item
    ]

    if categories:
        print(f"Found {len(categories)} categories: {categories}")
        models_category = config["models_category"]
        if models_category is not None:
            if models_category not in categories:
                raise ValueError(
                    f"Category '{models_category}' not found in {mesh_dir}. "
                    f"Available categories: {categories}"
                )
            categories = [models_category]

        for category in categories:
            category_path = os.path.join(mesh_dir, category)
            for filename in sorted(os.listdir(category_path)):
                print("Processing file:", filename)
                if not is_valid_mesh_file(filename, config, categorized=True):
                    continue
                if not matches_model_filters(filename, model_index, model_name):
                    continue

                yield {
                    "filename": filename,
                    "obj_path": os.path.join(category_path, filename),
                    "category_id": f"{category}_{filename[:-4]}",
                    "output_dir": os.path.join(
                        config["output_dir"],
                        dataset_name,
                        category,
                        f"obj_{filename[:-4]}",
                    ),
                }
        return

    for filename in sorted(os.listdir(mesh_dir)):
        if not is_valid_mesh_file(filename, config, categorized=False):
            continue
        if not matches_model_filters(filename, model_index, model_name):
            continue

        yield {
            "filename": filename,
            "obj_path": os.path.join(mesh_dir, filename),
            "category_id": filename[:-4],
            "output_dir": os.path.join(
                config["output_dir"],
                dataset_name,
                f"obj_{filename[:-4]}",
            ),
        }


def load_mesh_object(obj_path, filename):
    loaded_objects = bproc.loader.load_obj(obj_path)
    if not loaded_objects:
        raise ValueError(f"No objects loaded from mesh: {obj_path}")

    if not filename.endswith(".glb"):
        return loaded_objects[0]

    for obj in loaded_objects:
        if obj.get_bound_box_volume() == 0.0:
            continue
        for material in obj.get_materials():
            material.set_principled_shader_value("Normal", [1, 1, 1])
        return obj

    raise ValueError(f"No non-empty object found in GLB file: {obj_path}")


def prepare_object(obj, category_id):
    obj.set_origin(mode="CENTER_OF_MASS")
    obj_center_position = obj.get_location()
    obj.set_cp("category_id", category_id)
    obj.set_shading_mode("auto")
    obj_dim = np.sqrt(
        obj.blender_obj.dimensions.x ** 2
        + obj.blender_obj.dimensions.y ** 2
        + obj.blender_obj.dimensions.z ** 2
    )
    return obj_center_position, obj_dim


def maybe_assign_material(obj, filename, material_config, config_dir):
    if not filename.endswith(".ply"):
        return

    configured_material = load_material_from_config(material_config, config_dir)
    if configured_material is not None:
        obj.replace_materials(configured_material)


def add_lights(obj_dim, obj_center_position, light_layout):
    if light_layout == "distributed":
        cube_vertices = [
            [obj_dim, obj_dim, obj_dim],
            [obj_dim, obj_dim, -obj_dim],
            [obj_dim, -obj_dim, obj_dim],
            [obj_dim, -obj_dim, -obj_dim],
            [-obj_dim, obj_dim, obj_dim],
            [-obj_dim, obj_dim, -obj_dim],
            [-obj_dim, -obj_dim, obj_dim],
            [-obj_dim, -obj_dim, -obj_dim],
        ]
        light_positions = [2 * np.array(vertex) + np.array(obj_center_position) for vertex in cube_vertices]
        energy = 200 * obj_dim
        radius = 0.1 * obj_dim
    elif light_layout == "two-points":
        light_positions = [
            [obj_dim, obj_dim, obj_dim],
            [-obj_dim, -obj_dim, -obj_dim],
        ]
        energy = 100 * obj_dim
        radius = obj_dim
    else:
        raise ValueError(f"Unsupported light layout: {light_layout}")

    for position in light_positions:
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(position)
        light.set_energy(energy)
        light.set_radius(radius)


def add_camera_poses(original_poses, factor, distance_scaling, obj_dim, obj_center_position):
    poses = copy.deepcopy(original_poses)
    current_poses = np.zeros_like(poses)

    for index, pose in enumerate(poses):
        pose[:3, 3] *= factor * distance_scaling * obj_dim
        pose[:3, 3] += obj_center_position
        camera_pose = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
            pose,
            ["X", "-Y", "-Z"],
        )
        bproc.camera.add_camera_pose(camera_pose)
        current_poses[index, :, :] = np.linalg.inv(pose)

    return current_poses


def configure_renderer():
    bproc.renderer.set_light_bounces(
        diffuse_bounces=60,
        glossy_bounces=60,
        max_bounces=60,
        transmission_bounces=60,
        transparent_max_bounces=60,
        volume_bounces=60,
    )

    bproc.python.renderer.RendererUtility.render_init()
    bproc.renderer.set_max_amount_of_samples(50)
    bproc.renderer.set_noise_threshold(0.001)
    bproc.renderer.set_denoiser("INTEL")
    bproc.renderer.toggle_light_tree(enable=True)
    bproc.renderer.enable_segmentation_output(map_by=["instance"])
    bproc.renderer.enable_normals_output()


def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
        return

    print(f"Directory '{output_dir}' already exists.")


def copy_source_mesh_to_output(mesh_path, output_dir):
    mesh_filename = os.path.basename(mesh_path)
    destination_path = os.path.join(output_dir, mesh_filename)
    shutil.copy2(mesh_path, destination_path)
    print(f"Copied source mesh to '{destination_path}'.")


def save_render_outputs(data, current_poses, output_dir, config, K_matrix, z_max):
    width = config["cam"]["width"]
    height = config["cam"]["height"]
    black_img = Image.new("RGB", (width, height))

    for idx_frame in range(current_poses.shape[0]):
        rgb = Image.fromarray(np.uint8(data["colors"][idx_frame]))
        mask = Image.fromarray(np.uint8(data["instance_segmaps"][idx_frame]) * 255)
        mask.save(os.path.join(output_dir, f"mask_{idx_frame:06d}.png"))

        normals = Image.fromarray(np.uint8(data["normals"][idx_frame] * 255))
        normals = Image.composite(normals, black_img, mask)
        normals.save(os.path.join(output_dir, f"normals_{idx_frame:06d}.png"))

        depth = data["depth"][idx_frame]
        depth_map_uint16 = (depth * 1000).astype(np.uint16)
        cv2.imwrite(
            os.path.join(output_dir, f"depth_{idx_frame:06d}.png"),
            depth_map_uint16,
        )

        points = depth_to_pointcloud(depth, K_matrix, z_max=z_max)
        np.save(os.path.join(output_dir, f"pcd_{idx_frame:06d}.npy"), arr=points)

        img = Image.composite(rgb, black_img, mask)
        img.save(os.path.join(output_dir, f"{idx_frame:06d}.png"))

    np.save(os.path.join(output_dir, "obj_poses.npy"), current_poses)


def render_mesh_entry(entry, config, original_poses):
    filename = entry["filename"]
    light_layout = get_light_layout(config)
    print(f"Loading object {filename}")

    obj = load_mesh_object(entry["obj_path"], filename)
    obj_center_position, obj_dim = prepare_object(obj, entry["category_id"])
    print(
        f"Object {filename} has dimensions: {obj.blender_obj.dimensions}, "
        f"diagonal length: {obj_dim}"
    )

    maybe_assign_material(obj, filename, config.get("material"), config["_config_dir"])
    K_matrix = configure_camera(config)
    add_lights(obj_dim, obj_center_position, light_layout)
    current_poses = add_camera_poses(
        original_poses,
        config["mesh_scale"],
        config["camera_distance_scaling"],
        obj_dim,
        obj_center_position,
    )
    configure_renderer()
    data = bproc.renderer.render()

    ensure_output_dir(entry["output_dir"])
    copy_source_mesh_to_output(entry["obj_path"], entry["output_dir"])
    save_render_outputs(
        data,
        current_poses,
        entry["output_dir"],
        config,
        K_matrix,
        get_z_max(config),
    )

    bproc.utility.reset_keyframes()
    bproc.clean_up(clean_up_camera=True)

def render(config):
    bproc.init()
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    original_poses = load_original_poses(config)

    for entry in iter_mesh_entries(config):
        render_mesh_entry(entry, config, original_poses)


def resolve_config_path(config_path, script_dir):
    if os.path.isabs(config_path):
        return config_path

    candidate_paths = [
        os.path.abspath(config_path),
        os.path.join(script_dir, config_path),
        os.path.join(os.path.dirname(script_dir), config_path),
    ]

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    return os.path.abspath(candidate_paths[0])


def load_config(config_path):
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    config["_config_dir"] = os.path.dirname(os.path.abspath(config_path))
    return resolve_config_paths(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", default="mesh_templates_cfg.yaml", help="Path to config file")
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)

    config_path = resolve_config_path(args.config_path, dirname)
    config = load_config(config_path)

    render(config)
