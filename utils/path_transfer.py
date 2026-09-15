import numpy as np
import torch
import open3d as o3d
import vedo
import matplotlib.cm as cm
from scipy.interpolate import splprep, splev
from utils.mesh_loader import ObjectData

def viridis_gradient(n: int, include_alpha: bool = False, scale: float = 1.0) -> np.ndarray:
    """
    Generates a sequence of RGB (or RGBA) colors from the Viridis colormap.
    """
    if n <= 0:
        return np.array([])
    viridis_cmap = cm.get_cmap('viridis')
    norm_values = np.linspace(0, 1, n)
    rgba_colors = viridis_cmap(norm_values)
    if include_alpha:
        return rgba_colors * scale
    return rgba_colors[:, :3] * scale


def interpolate_curve(curve: np.ndarray, num_points: int = 500, s: float = 0.0005, w: list = None) -> np.ndarray:
    """
    Interpolates a 3D curve using B-splines.
    """
    tck, u = splprep(curve.T, s=s, w=w)
    u_new = np.linspace(0, 1, num_points)
    interpolated_curve = np.array(splev(u_new, tck)).T
    return interpolated_curve


def load_trajectory(filepath: str, scale_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a trajectory from a text file (x y z roll pitch yaw [weight]).
    """
    points = []
    weights = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                x, y, z = map(float, parts[:3])
                points.append([x * scale_factor, y * scale_factor, z * scale_factor])
                weights.append(float(parts[6]))
            elif len(parts) >= 3:
                x, y, z = map(float, parts[:3])
                points.append([x * scale_factor, y * scale_factor, z * scale_factor])
                weights.append(1.0)
    return np.array(points), np.array(weights)


def transfer_path_from_indices(
    source_indices: list,
    p2p_12: np.ndarray,
    obj1: ObjectData,
    obj2: ObjectData,
    num_points: int = 500
) -> tuple:
    """
    Traces geodesic paths on source & target meshes via vedo, interpolates curves,
    and projects points onto exact mesh surfaces using PyTorch Volumetric SDF.
    """
    vedo_mesh_1 = vedo.Mesh([obj1.vertices, obj1.faces])
    vedo_mesh_2 = vedo.Mesh([obj2.vertices, obj2.faces])

    # Trace source path
    current_source_path = []
    weights_1 = []
    for i, idx in enumerate(source_indices[:-1]):
        weights_1.append(1.0)
        path = vedo_mesh_1.geodesic(idx, source_indices[i + 1])
        segment = path.pointdata["VertexIDs"].tolist()
        current_source_path.extend(segment[:-1])
        weights_1.extend([0.3] * (len(segment) - 2))
    current_source_path.append(source_indices[-1])
    weights_1.append(1.0)

    # Trace target path
    target_indices = []
    seen = set()
    for idx in p2p_12[source_indices]:
        if idx not in seen:
            target_indices.append(idx)
            seen.add(idx)

    current_target_path = []
    weights_2 = []
    for i, idx in enumerate(target_indices[:-1]):
        weights_2.append(1.0)
        path = vedo_mesh_2.geodesic(idx, target_indices[i + 1])
        segment = path.pointdata["VertexIDs"].tolist()
        current_target_path.extend(segment[:-1])
        weights_2.extend([0.3] * (len(segment) - 2))
    current_target_path.append(target_indices[-1])
    weights_2.append(1.0)

    # Clean consecutive duplicates
    curve1 = obj1.vertices[current_source_path]
    dup_mask_1 = np.insert(np.any(np.diff(curve1, axis=0), axis=1), 0, True)
    curve1 = curve1[dup_mask_1]
    weights_1 = np.array(weights_1)[dup_mask_1].tolist()

    curve2 = obj2.vertices[current_target_path]
    dup_mask_2 = np.insert(np.any(np.diff(curve2, axis=0), axis=1), 0, True)
    curve2 = curve2[dup_mask_2]
    weights_2 = np.array(weights_2)[dup_mask_2].tolist()

    # Interpolate B-Splines
    curve1_int = interpolate_curve(curve1, num_points=num_points, s=0.0005, w=weights_1)
    curve2_int = interpolate_curve(curve2, num_points=num_points, s=0.0005, w=weights_2)

    # SDF Projection
    sdf_vals_1, sdf_grads_1 = obj1.sdf_evaluator(torch.tensor(curve1_int, dtype=torch.float32))
    proj_curve1 = (torch.tensor(curve1_int, dtype=torch.float32) - sdf_vals_1.unsqueeze(1) * sdf_grads_1).numpy()

    sdf_vals_2, sdf_grads_2 = obj2.sdf_evaluator(torch.tensor(curve2_int, dtype=torch.float32))
    proj_curve2 = (torch.tensor(curve2_int, dtype=torch.float32) - sdf_vals_2.unsqueeze(1) * sdf_grads_2).numpy()

    return proj_curve1, proj_curve2, weights_1, weights_2


def transfer_trajectory_file(
    traj_filepath: str,
    p2p_12: np.ndarray,
    obj1: ObjectData,
    obj2: ObjectData,
    scale_factor: float = 1.0,
    num_points: int = 500
) -> tuple:
    """
    Loads an input trajectory text file, maps surface indices to the target mesh,
    computes geodesic target path, interpolates, and projects onto target SDF surface.
    """
    curve1_int, weights1_loaded = load_trajectory(traj_filepath, scale_factor=scale_factor)

    nns_1 = o3d.core.nns.NearestNeighborSearch(obj1.vertices)
    nns_1.knn_index()
    nearest_indices_1, _ = nns_1.knn_search(curve1_int, knn=1)
    nearest_indices_1 = nearest_indices_1.numpy()[:, 0]

    target_indices = p2p_12[nearest_indices_1]
    unique_target_indices = []
    unique_target_weights = []

    if len(target_indices) > 0:
        unique_target_indices.append(target_indices[0])
        unique_target_weights.append(weights1_loaded[0])
        for i in range(1, len(target_indices)):
            if target_indices[i] != target_indices[i - 1]:
                unique_target_indices.append(target_indices[i])
                unique_target_weights.append(weights1_loaded[i])

    vedo_mesh_2 = vedo.Mesh([obj2.vertices, obj2.faces])
    current_target_path = []
    weights_2 = []

    for i, idx in enumerate(unique_target_indices[:-1]):
        weights_2.append(unique_target_weights[i])
        path = vedo_mesh_2.geodesic(idx, unique_target_indices[i + 1])
        segment = path.pointdata["VertexIDs"].tolist()
        current_target_path.extend(segment[:-1])
        weights_2.extend([0.3] * (len(segment) - 2))

    if unique_target_indices:
        current_target_path.append(unique_target_indices[-1])
        weights_2.append(unique_target_weights[-1])

    curve2 = obj2.vertices[current_target_path]
    dup_mask = np.insert(np.any(np.diff(curve2, axis=0), axis=1), 0, True)
    curve2 = curve2[dup_mask]
    weights_2 = np.array(weights_2)[dup_mask].tolist()

    curve2_int = interpolate_curve(curve2, num_points=num_points, s=0.0005, w=weights_2)
    sdf_vals_2, sdf_grads_2 = obj2.sdf_evaluator(torch.tensor(curve2_int, dtype=torch.float32))
    proj_curve2 = (torch.tensor(curve2_int, dtype=torch.float32) - sdf_vals_2.unsqueeze(1) * sdf_grads_2).numpy()

    return curve1_int, proj_curve2, weights1_loaded.tolist(), weights_2
