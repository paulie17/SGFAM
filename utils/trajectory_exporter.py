import os
import time
import numpy as np
import torch
import open3d as o3d
import roma
from scipy.spatial.transform import Rotation as R
import pytorch_volumetric as pv

def compute_path_lrfs(path_points: np.ndarray, sdf_evaluator: pv.MeshSDF) -> list[np.ndarray]:
    """
    Computes Local Reference Frames (tangent, normal, binormal axes) along a 3D path
    using surface SDF gradients and Procrustes rotation alignment (roma.special_procrustes).
    """
    path_torch = torch.tensor(path_points, dtype=torch.float32)
    sdf_vals, sdf_grads = sdf_evaluator(path_torch)
    sdf_grads = sdf_grads / (sdf_grads.norm(dim=-1, keepdim=True) + 1e-8)

    lrfs_list = []
    num_points = len(path_points)

    for i, point in enumerate(path_points):
        z_axis = -sdf_grads[i].numpy()

        if i < num_points - 1:
            x_axis = path_points[i + 1] - path_points[i]
        else:
            x_axis = path_points[i] - path_points[i - 1]

        if np.linalg.norm(x_axis) > 1e-6:
            x_axis = x_axis / np.linalg.norm(x_axis)
        else:
            if i > 0:
                prev_x = path_points[i] - path_points[i - 1]
                x_axis = prev_x / np.linalg.norm(prev_x) if np.linalg.norm(prev_x) > 1e-6 else np.array([1.0, 0.0, 0.0])
            else:
                x_axis = np.array([1.0, 0.0, 0.0])

        y_axis = np.cross(z_axis, x_axis)
        if np.linalg.norm(y_axis) > 1e-6:
            y_axis = y_axis / np.linalg.norm(y_axis)
        else:
            if np.abs(z_axis[0]) < 0.9:
                y_axis = np.cross(z_axis, np.array([1.0, 0.0, 0.0]))
            else:
                y_axis = np.cross(z_axis, np.array([0.0, 1.0, 0.0]))
            y_axis = y_axis / np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)

        orientation = np.vstack([x_axis, y_axis, z_axis]).T
        orientation = roma.special_procrustes(torch.tensor(orientation, dtype=torch.float32)).numpy()
        lrfs_list.append(orientation)

    return lrfs_list


def export_trajectory(
    filepath: str,
    path_points: np.ndarray,
    lrfs_list: list[np.ndarray],
    weights: list = None,
    scale: str = 'm'
) -> None:
    """
    Saves positions (m or mm), Euler angles (ZYX), and optional weights to a text file.
    Format: x y z roll pitch yaw [weight]
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    scale_factor = 1000.0 if scale == 'mm' else 1.0

    with open(filepath, 'w') as f:
        for i, (position, orientation) in enumerate(zip(path_points, lrfs_list)):
            r = R.from_matrix(orientation)
            euler_angles = r.as_euler('ZYX')
            scaled_pos = position * scale_factor

            if weights is not None and len(weights) > 0:
                weight_idx = i / max(1, len(path_points) - 1) * (len(weights) - 1)
                w1 = weights[int(weight_idx)]
                w2 = weights[min(int(weight_idx) + 1, len(weights) - 1)]
                w = w1 + (w2 - w1) * (weight_idx - int(weight_idx))
                f.write(f"{scaled_pos[0]:.6f} {scaled_pos[1]:.6f} {scaled_pos[2]:.6f} {euler_angles[0]:.6f} {euler_angles[1]:.6f} {euler_angles[2]:.6f} {w:.4f}\n")
            else:
                f.write(f"{scaled_pos[0]:.6f} {scaled_pos[1]:.6f} {scaled_pos[2]:.6f} {euler_angles[0]:.6f} {euler_angles[1]:.6f} {euler_angles[2]:.6f}\n")

    print(f"Saved trajectory to: {filepath} (scale: {scale})")


def animate_trajectory_frames(mesh: o3d.geometry.TriangleMesh, path_points: np.ndarray, lrfs_list: list[np.ndarray], frame_size: float = 0.05) -> None:
    """
    Visualizes coordinate frames moving along the target 3D path using Open3D.
    """
    tool_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    vis = o3d.visualization.Visualizer()
    vis.create_window('SGFAM Trajectory Animation')

    curve_pcd = o3d.geometry.PointCloud()
    curve_pcd.points = o3d.utility.Vector3dVector(path_points)

    vis.add_geometry(mesh)
    vis.add_geometry(curve_pcd)
    vis.add_geometry(tool_frame)

    current_rotation = np.eye(3)

    for i, orientation in enumerate(lrfs_list):
        tool_frame.rotate(current_rotation.T)
        tool_frame.rotate(orientation)

        translation = path_points[i] - tool_frame.get_center()
        tool_frame.translate(translation)
        current_rotation = orientation

        vis.update_geometry(tool_frame)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)

    vis.destroy_window()
