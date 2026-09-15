#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import os

def pick_points(pcd) -> list[int]:
    """
    Interactive Open3D window to pick start and end points on a point cloud / mesh.
    """
    import open3d as o3d

    print("\n[Interactive Point Selection]")
    print("1) Pick start point and end point using [Shift + Left Mouse Click].")
    print("2) Press 'Q' when finished to close the window.\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window('Pick Start and End Points')
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()


def main(args):
    import numpy as np
    import open3d as o3d

    from utils.mesh_loader import load_object_data
    from utils.functional_map import reduce_descriptors, solve_functional_map, compute_spectral_colors
    from utils.path_transfer import (
        viridis_gradient,
        transfer_path_from_indices,
        transfer_trajectory_file,
        load_trajectory
    )
    from utils.trajectory_exporter import compute_path_lrfs, export_trajectory, animate_trajectory_frames

    if args.output_directory != "":
        os.makedirs(args.output_directory, exist_ok=True)

    # 1. Load Object 1 & Object 2 Descriptors and Meshes
    print("\n--- [Step 1/5] Loading Object Data & Meshes ---")
    obj1 = load_object_data(
        args.input_1,
        only_sem=args.only_sem,
        only_geom=args.only_geom,
        simplify=args.simplify,
        load_colors=args.load_colors
    )

    obj2 = load_object_data(
        args.input_2,
        only_sem=args.only_sem,
        only_geom=args.only_geom,
        simplify=args.simplify,
        load_colors=args.load_colors
    )

    # Visualize voxel point clouds side by side
    pcd_1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj1.voxel_points))
    pcd_2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj2.voxel_points))
    pcd_2.translate(np.array([obj1.diagonal_length * 1.1, 0., 0.]))
    print("Displaying voxel point clouds side-by-side...")
    o3d.visualization.draw_geometries([pcd_1, pcd_2])
    pcd_2.translate(np.array([-obj1.diagonal_length * 1.1, 0., 0.]))

    # 2. Descriptor Reduction & Functional Map Solver
    print("\n--- [Step 2/5] Functional Map Optimization ---")
    c1 = obj1.descriptors
    c2 = obj2.descriptors

    if args.reduce_dim and (c1.shape[1] > args.target_dim or c2.shape[1] > args.target_dim):
        c1_red, c2_red = reduce_descriptors(c1, c2, target_dim=args.target_dim, pca_type=args.pca_type)
    else:
        c1_red, c2_red = c1, c2

    opt_time, emb1, emb2, p2p_21, p2p_12 = solve_functional_map(
        obj1.vertices, obj1.faces,
        obj2.vertices, obj2.faces,
        c1_red, c2_red,
        n_ev=args.n_ev
    )

    # 3. Spectral Eigen-embedding Visualization
    print("\n--- [Step 3/5] Visualizing Functional Map Spectral Embeddings ---")
    emb1_colors, emb2_colors = compute_spectral_colors(emb1, emb2)
    obj1.mesh.vertex_colors = o3d.utility.Vector3dVector(emb1_colors)
    obj2.mesh.vertex_colors = o3d.utility.Vector3dVector(emb2_colors)

    obj2.mesh.translate(np.array([obj1.diagonal_length * 1.1, 0., 0.]))
    o3d.visualization.draw_geometries([obj1.mesh, obj2.mesh])
    obj2.mesh.translate(np.array([-obj1.diagonal_length * 1.1, 0., 0.]))

    if args.save_emb_meshes and args.output_directory != "":
        m1_path = os.path.join(args.output_directory, "mesh_1_embeddings.ply")
        m2_path = os.path.join(args.output_directory, "mesh_2_embeddings.ply")
        o3d.io.write_triangle_mesh(m1_path, obj1.mesh)
        o3d.io.write_triangle_mesh(m2_path, obj2.mesh)

    # 4. Path Transfer & Trajectory Mapping
    curve1_int, curve2_int = None, None
    final_weights_1, final_weights_2 = [], []

    do_path_transfer = True
    if not args.input_traj_path:
        user_choice = input("\nDo you want to test path transfer / matching? (y/n): ").strip().lower()
        do_path_transfer = user_choice in ['y', 'yes']

    if do_path_transfer:
        print("\n--- [Step 4/5] Path Transfer & Trajectory Surface Projection ---")
        if args.input_traj_path:
            print(f"Loading input trajectory file: {args.input_traj_path}")
            curve1_int, curve2_int, final_weights_1, final_weights_2 = transfer_trajectory_file(
                args.input_traj_path, p2p_12, obj1, obj2, scale_factor=args.scale_input_traj
            )

            c1_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(curve1_int))
            c1_pcd.colors = o3d.utility.Vector3dVector(viridis_gradient(n=len(curve1_int)))

            c2_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(curve2_int))
            c2_pcd.colors = o3d.utility.Vector3dVector(viridis_gradient(n=len(curve2_int)))

            obj2.original_mesh.translate(np.array([obj1.diagonal_length * 1.1, 0., 0.]))
            c2_pcd.translate(np.array([obj1.diagonal_length * 1.1, 0., 0.]))
            o3d.visualization.draw_geometries([c1_pcd, c2_pcd, obj1.original_mesh, obj2.original_mesh])
            c2_pcd.translate(np.array([-obj1.diagonal_length * 1.1, 0., 0.]))
            obj2.original_mesh.translate(np.array([-obj1.diagonal_length * 1.1, 0., 0.]))
        else:
            # Interactive Path Selection
            continue_picking = True
            while continue_picking:
                pcd_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj1.vertices))
                picked_indices = pick_points(pcd_src)

                if len(picked_indices) < 2:
                    print("Warning: Need at least 2 picked points to trace a geodesic path.")
                    break

                proj1, proj2, w1, w2 = transfer_path_from_indices(picked_indices, p2p_12, obj1, obj2)
                curve1_int, curve2_int = proj1, proj2
                final_weights_1, final_weights_2 = w1, w2

                c1_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(proj1))
                c1_pcd.colors = o3d.utility.Vector3dVector(viridis_gradient(n=len(proj1)))

                c2_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(proj2))
                c2_pcd.colors = o3d.utility.Vector3dVector(viridis_gradient(n=len(proj2)))

                obj2.original_mesh.translate(np.array([obj1.diagonal_length * 1.1, 0., 0.]))
                c2_pcd.translate(np.array([obj1.diagonal_length * 1.1, 0., 0.]))
                o3d.visualization.draw_geometries([c1_pcd, c2_pcd, obj1.original_mesh, obj2.original_mesh])
                c2_pcd.translate(np.array([-obj1.diagonal_length * 1.1, 0., 0.]))
                obj2.original_mesh.translate(np.array([-obj1.diagonal_length * 1.1, 0., 0.]))

                user_input = input("\nDo you want to pick another path? (y/n): ").lower().strip()
                continue_picking = user_input in ['y', 'yes']

    # 5. Trajectory Export & Local Reference Frame (LRF) Computation
    print("\n--- [Step 5/5] Trajectory Export & Local Reference Frames ---")
    if curve2_int is not None and args.save_target_traj:
        lrfs_target = compute_path_lrfs(curve2_int, obj2.sdf_evaluator)
        tgt_file = os.path.join(args.output_directory, "target_trajectory.txt") if args.output_directory else "target_trajectory.txt"
        export_trajectory(tgt_file, curve2_int, lrfs_target, scale=args.scale)

        if args.animation:
            print("Running debug animation of tool coordinate frames along target path...")
            animate_trajectory_frames(obj2.original_mesh, curve2_int, lrfs_target)

    if curve1_int is not None and args.save_input_traj:
        lrfs_source = compute_path_lrfs(curve1_int, obj1.sdf_evaluator)
        src_file = os.path.join(args.output_directory, "input_trajectory.txt") if args.output_directory else "input_trajectory.txt"
        export_trajectory(src_file, curve1_int, lrfs_source, weights=final_weights_1, scale=args.scale)

    # Save point-to-point mapping metadata
    if args.output_directory != "":
        mapping_file = os.path.join(args.output_directory, "point_to_point_mapping.npz")
        np.savez_compressed(
            mapping_file,
            mapping=p2p_12,
            n_ev=args.n_ev,
            object_1_mesh=obj1.path_to_mesh,
            object_2_mesh=obj2.path_to_mesh,
            vertices_1=obj1.vertices,
            faces_1=obj1.faces,
            vertices_2=obj2.vertices,
            faces_2=obj2.faces,
            obj_1_dim=obj1.diagonal_length,
            obj_2_dim=obj2.diagonal_length,
            opt_time=opt_time
        )
        print(f"Point-to-point mapping metadata saved to: {mapping_file}")

    print("\nSGFAM Inference Pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGFAM Functional Map & Path Transfer Inference Pipeline")
    parser.add_argument("--input-1", type=str, required=True, help="Path to the first input .npz descriptor file.")
    parser.add_argument("--input-2", type=str, required=True, help="Path to the second input .npz descriptor file.")
    parser.add_argument('--n-ev', type=int, default=50, help="Number of eigenvectors for functional map.")
    parser.add_argument('--simplify', action='store_true', help="Simplify mesh with point_cloud_utils watertight decimation.")
    parser.add_argument('--reduce-dim', action='store_true', help="Reduce descriptor dimensionality.")
    parser.add_argument('--target-dim', type=int, default=128, help="Target dimensionality for reduction.")
    parser.add_argument('--only-sem', action='store_true', help="Use only semantic descriptors.")
    parser.add_argument('--only-geom', action='store_true', help="Use only geometric descriptors.")
    parser.add_argument('--load-colors', action='store_true', help="Extract vertex colors from UV texture maps.")
    parser.add_argument('--save-emb-meshes', action='store_true', help="Save PLY meshes colored with functional map embeddings.")
    parser.add_argument('--output-directory', type=str, default="", help="Output directory for saved results.")
    parser.add_argument('--save-input-traj', action='store_true', help="Save source/input trajectory text file.")
    parser.add_argument('--save-target-traj', action='store_true', help="Save target trajectory text file.")
    parser.add_argument('--input-traj-path', type=str, default=None, help="Path to input trajectory text file.")
    parser.add_argument('--scale-input-traj', type=float, default=1.0, help="Scale factor for input trajectory.")
    parser.add_argument('--animation', action='store_true', help="Animate coordinate frame visualization.")
    parser.add_argument('--scale', type=str, choices=['m', 'mm'], default='m', help="Scale for trajectory output ('m' or 'mm').")
    parser.add_argument('--pca-type', type=str, choices=['kernel', 'linear'], default='kernel', help="PCA method: 'kernel' (Kernel PCA) or 'linear'.")

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()
    main(args)
