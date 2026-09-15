import os
from dataclasses import dataclass
import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
import pytorch_volumetric as pv

@dataclass
class ObjectData:
    mesh: o3d.geometry.TriangleMesh
    original_mesh: o3d.geometry.TriangleMesh
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    descriptors: np.ndarray
    voxel_points: np.ndarray
    voxel_size: float
    diagonal_length: float
    path_to_mesh: str
    sdf_evaluator: pv.MeshSDF

def create_vertex_colors(mesh: o3d.geometry.TriangleMesh, verbose: bool = False) -> o3d.utility.Vector3dVector:
    """
    Extracts vertex colors from UV texture maps for Open3D rendering.
    """
    vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    
    if verbose:
        print("Creating vertex colors from UV texture maps, please wait...")
    
    num_triangles = len(mesh.triangles)
    progress_step = max(1, num_triangles // 100)
    
    for triangle_index in range(num_triangles):
        texture_index = mesh.triangle_material_ids[triangle_index]
        texture_image = mesh.textures[texture_index]
        texture_np = np.asarray(texture_image)
        height, width, _ = texture_np.shape
        
        for local_vertex in range(3):
            u, v = mesh.triangle_uvs[triangle_index * 3 + local_vertex]
            x = int(u * (width - 1))
            y = int(v * (height - 1))
            global_vertex_index = mesh.triangles[triangle_index][local_vertex]
            vertex_colors[global_vertex_index] = texture_np[y, x] / 255.0
        
        if verbose and (triangle_index % progress_step) == 0:
            print('#', end='', flush=True)
    
    if verbose:
        print()
    
    return o3d.utility.Vector3dVector(vertex_colors)


def load_object_data(
    npz_path: str,
    only_sem: bool = False,
    only_geom: bool = False,
    simplify: bool = False,
    load_colors: bool = False
) -> ObjectData:
    """
    Loads object descriptor file, 3D mesh, performs optional decimation/simplification,
    and maps voxel descriptors to mesh vertices via nearest-neighbor search.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Descriptor file not found: {npz_path}")

    obj_data = np.load(npz_path)
    voxel_points = obj_data["points"]

    # Determine descriptor type
    if only_sem:
        if "dino_descriptors" in obj_data:
            descriptors = obj_data["dino_descriptors"]
        elif "dift_descriptors" in obj_data:
            descriptors = obj_data["dift_descriptors"]
        else:
            descriptors = obj_data["descriptors"]
    elif only_geom:
        descriptors = obj_data["geom_descriptors"]
    else:
        descriptors = obj_data["descriptors"]

    path_to_mesh = str(obj_data["path_to_mesh"])
    voxel_size = float(obj_data["voxel_size"])

    # Compute bounding box diagonal
    pcd_voxels = o3d.geometry.PointCloud()
    pcd_voxels.points = o3d.utility.Vector3dVector(voxel_points)
    obb = pcd_voxels.get_oriented_bounding_box()
    diagonal_length = float(np.linalg.norm(obb.extent))

    # Read 3D triangle mesh
    print(f"Loading mesh from: {path_to_mesh} (bounding diagonal: {diagonal_length:.4f})")
    if not os.path.exists(path_to_mesh):
        raise FileNotFoundError(f"Mesh file not found at path: {path_to_mesh}")

    original_mesh = o3d.io.read_triangle_mesh(path_to_mesh)
    original_mesh.compute_triangle_normals()
    original_mesh.compute_vertex_normals()

    if load_colors and len(original_mesh.textures) > 0 and len(original_mesh.triangle_uvs) > 0:
        original_mesh.vertex_colors = create_vertex_colors(original_mesh, verbose=True)

    mesh = o3d.geometry.TriangleMesh(original_mesh)
    print(f"Mesh initial stats: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces.")

    # Watertight decimation / simplification in memory
    if simplify:
        print(f"Simplifying mesh with point_cloud_utils watertight decimation...")
        v_watertight, f_watertight = pcu.make_mesh_watertight(
            np.array(mesh.vertices), np.array(mesh.triangles), resolution=2000
        )
        target_num_faces = int(f_watertight.shape[0] * 0.99)
        v_dec, f_dec, _, _ = pcu.decimate_triangle_mesh(v_watertight, f_watertight, target_num_faces)

        if f_dec.shape[0] > 10000:
            target_num_faces = int(f_dec.shape[0] * 0.6)
            v_dec, f_dec, _, _ = pcu.decimate_triangle_mesh(v_dec, f_dec, target_num_faces)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v_dec)
        mesh.triangles = o3d.utility.Vector3iVector(f_dec)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
    else:
        mesh.remove_non_manifold_edges()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.triangles)
    normals = np.array(mesh.vertex_normals)

    print(f"Processed mesh stats: {len(vertices):,} vertices, {len(faces):,} faces.")

    # Map voxel descriptors to mesh vertices using nearest-neighbor search
    nns_voxels = o3d.core.nns.NearestNeighborSearch(voxel_points)
    nns_voxels.knn_index()
    indices, _ = nns_voxels.knn_search(vertices, knn=1)
    vertex_descriptors = descriptors[indices.numpy()[:, 0]]

    # Initialize PyTorch Volumetric SDF for surface projection
    source_obj = pv.MeshObjectFactory(path_to_mesh)
    sdf_evaluator = pv.MeshSDF(source_obj)

    return ObjectData(
        mesh=mesh,
        original_mesh=original_mesh,
        vertices=vertices,
        faces=faces,
        normals=normals,
        descriptors=vertex_descriptors,
        voxel_points=voxel_points,
        voxel_size=voxel_size,
        diagonal_length=diagonal_length,
        path_to_mesh=path_to_mesh,
        sdf_evaluator=sdf_evaluator
    )
