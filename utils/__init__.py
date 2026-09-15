"""
SGFAM Utility Package
Contains modular helpers for mesh loading, functional map optimization, path transfer, and trajectory export.
"""

from utils.mesh_loader import load_object_data, create_vertex_colors, ObjectData
from utils.functional_map import reduce_descriptors, solve_functional_map, compute_spectral_colors
from utils.path_transfer import (
    viridis_gradient,
    interpolate_curve,
    transfer_path_from_indices,
    transfer_trajectory_file,
    load_trajectory
)
from utils.trajectory_exporter import compute_path_lrfs, export_trajectory, animate_trajectory_frames

__all__ = [
    "load_object_data",
    "create_vertex_colors",
    "ObjectData",
    "reduce_descriptors",
    "solve_functional_map",
    "compute_spectral_colors",
    "viridis_gradient",
    "interpolate_curve",
    "transfer_path_from_indices",
    "transfer_trajectory_file",
    "load_trajectory",
    "compute_path_lrfs",
    "export_trajectory",
    "animate_trajectory_frames"
]
