import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from torchvision import transforms
import warnings

import os
import sys

import torch
import torch.nn as nn
import packaging.version

from torch_kdtree import build_kd_tree

from collections import defaultdict
import requests
import time
import argparse

MESH_EXTENSIONS = {'.ply', '.obj', '.glb'}

# DIFT-specific imports
if packaging.version.parse(torch.__version__) >= packaging.version.parse('1.12.0'):
    torch.backends.cuda.matmul.allow_tf32 = True

try:
    from descriptors_precomp.dift_sd import SDFeaturizer4Eval
    DIFT_AVAILABLE = True
except ImportError:
    print("Warning: DIFT dependencies not available. DIFT functionality will be disabled.")
    DIFT_AVAILABLE = False

try:
    from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig
    SFAST_AVAILABLE = True
except ImportError:
    print("Warning: stable-fast not available. DIFT will run without compilation optimizations.")
    SFAST_AVAILABLE = False

DINO_AVAILABLE = True

warnings.filterwarnings(
    "ignore",
    message=r"xFormers is available.*",
    category=UserWarning,
)

# DIFT configuration
if DIFT_AVAILABLE and SFAST_AVAILABLE:
    config = CompilationConfig.Default()
    try:
        import xformers
        config.enable_xformers = True
    except ImportError:
        print('xformers not installed, skip')
    try:
        import triton
        config.enable_triton = True
    except ImportError:
        print('Triton not installed, skip')
    config.enable_cuda_graph = True
    config.enable_fused_linear_geglu = False


if DIFT_AVAILABLE:
    def quantize_unet(m):
        from diffusers.utils import USE_PEFT_BACKEND
        assert USE_PEFT_BACKEND
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                dtype=torch.qint8,
                                                inplace=True)
        return m


def transform_points_to_object_frame_torch(points_camera:torch.Tensor, 
                                           T_object_to_camera:torch.Tensor):
    """
    Transforms points from the camera frame to the object frame.

    Parameters:
    - points_camera (torch.Tensor): Nx3 tensor of points in the camera frame.
    - T_object_to_camera (torch.Tensor): 4x4 tensor from object to camera frame.

    Returns:
    - points_object (torch.Tensor): Nx3 tensor of points in the object frame.
    """
    # Compute the inverse transformation matrix
    R = T_object_to_camera[:3, :3]  # Rotation
    t = T_object_to_camera[:3, 3]  # Translation
    T_camera_to_object = torch.eye(4, device=points_camera.device)  # Initialize on the same device
    T_camera_to_object[:3, :3] = R.T
    T_camera_to_object[:3, 3] = -R.T @ t

    # Convert points to homogeneous coordinates
    num_points = points_camera.size(0)
    points_camera_h = torch.cat((points_camera, torch.ones(num_points, 1, device=points_camera.device)), dim=1)  # Nx4

    # Apply the transformation
    points_object_h = (T_camera_to_object @ points_camera_h.T).T  # Transform and transpose back

    # Return only the 3D coordinates
    return points_object_h[:, :3]


def compute_padding(image: Image.Image, 
                    target_size: tuple):
    """
    Calculates the padding applied.

    Parameters:
    - image (PIL.Image.Image): The original input image.
    - target_size (tuple): The target size as (width, height).

    Returns:
    - padding_top (int): Padding added to the top.
    - padding_bottom (int): Padding added to the bottom.
    - padding_left (int): Padding added to the left.
    - padding_right (int): Padding added to the right.
    """
    # Original image size
    W_orig, H_orig = image.size

    # Target size (W_target, H_target)
    W_target, H_target = target_size

    # Calculate total padding to be added on both dimensions
    padding_x_total = W_target - W_orig
    padding_y_total = H_target - H_orig

    # Split padding equally, and handle odd padding by adding extra to right or bottom
    padding_left = padding_x_total // 2
    padding_right = padding_x_total - padding_left  # Extra pixel goes to the right if odd

    padding_top = padding_y_total // 2
    padding_bottom = padding_y_total - padding_top  # Extra pixel goes to the bottom if odd

    return padding_top, padding_bottom, padding_left, padding_right


def extract_dino_features(args, rgb_files, path_to_views, voxel_grid, device):
    """Extract DINO features for all views"""

    input_size = 840
    
    # ViTExtractor
    model_size = args.dino_size # small - base - large - giant
    model_dict = {
            'small': 'dinov2_vits14',
            'base': 'dinov2_vitb14',
            'large': 'dinov2_vitl14',
            'giant': 'dinov2_vitg14'
        }
    
    model_type = model_dict[model_size]
    extractor = torch.hub.load('facebookresearch/dinov2', model_type)
    extractor = extractor.to(device)
    extractor.eval()

    patch_size = extractor.patch_embed.patch_size[0]
    num_patches = input_size // patch_size

    dino_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    # Determine feature dimensions
    if 's' in model_type:
        feature_dim = 384
    elif 'b' in model_type:
        feature_dim = 768
    elif 'l' in model_type:
        feature_dim = 1024
    elif 'g' in model_type:
        feature_dim = 1536
    
    imgs_list = []
    img = None
    for i in range(len(rgb_files)):
        img = Image.open(os.path.join(path_to_views, rgb_files[i]))
        rgb_resized = ImageOps.pad(img, [input_size, input_size])
        rgb_dino_input = dino_preprocess(rgb_resized).unsqueeze(0)
        imgs_list.append(rgb_dino_input)

    if img is None:
        raise ValueError(f"No RGB files found in {path_to_views}")

    img_size = np.array(img).shape[:2]
    new_size = max(img_size)
    padding_info = compute_padding(img, (new_size, new_size))

    batched_imgs = torch.cat(imgs_list, dim=0)
    with torch.no_grad():
        features = extractor.forward_features(batched_imgs.to(device))
        patch_tokens = features["x_norm_patchtokens"]

        token_count = patch_tokens.shape[1]
        inferred_patches = int(np.sqrt(token_count))
        if inferred_patches * inferred_patches != token_count:
            raise ValueError(f"Unexpected number of DINO patch tokens: {token_count}")

        if inferred_patches != num_patches:
            num_patches = inferred_patches

        batch_dino_fts = patch_tokens.permute(0, 2, 1).reshape(-1, patch_tokens.shape[-1], num_patches, num_patches)

    feature_dim = batch_dino_fts.shape[1]

    return batch_dino_fts, feature_dim, padding_info


def extract_dift_features(args, rgb_file, path_to_views):
    """Extract DIFT features for a single view"""
    if not DIFT_AVAILABLE:
        raise ImportError("DIFT dependencies not available. Please install required packages.")
    
    if not hasattr(extract_dift_features, 'dift'):
        # Initialize DIFT model once
        extract_dift_features.dift = SDFeaturizer4Eval()
        if SFAST_AVAILABLE:
            extract_dift_features.dift.pipe = compile(extract_dift_features.dift.pipe, config)
        extract_dift_features.dift.pipe.unet = quantize_unet(extract_dift_features.dift.pipe.unet)
        extract_dift_features.warmup = True
    
    rgb = Image.open(os.path.join(path_to_views, rgb_file))
    img_size = np.array(rgb).shape[:2]

    if extract_dift_features.warmup:
        prompt = args.sd_prompt
        warmup_fts = extract_dift_features.dift.forward(rgb, category=prompt, ensemble_size=2)
        extract_dift_features.warmup = False

    prompt = args.sd_prompt
    dift_fts = extract_dift_features.dift.forward(rgb, category=prompt, ensemble_size=2)
    dift_fts = dift_fts.permute(0,1,3,2)

    new_size = max(img_size)
    padding_info = compute_padding(rgb, (new_size,new_size))
    
    return dift_fts, 1280, padding_info  # DIFT has 1280 dimensions


def main(args):
    save = args.save
    visualize = args.visualize
    
    # Validate semantic descriptor choice
    if args.semantic_desc == 'dino' and not DINO_AVAILABLE:
        raise ImportError("DINO selected but dependencies not available. Please install required packages.")
    elif args.semantic_desc == 'dift' and not DIFT_AVAILABLE:
        raise ImportError("DIFT selected but dependencies not available. Please install required packages.")
    
    # Torch settings
    torch.cuda.set_device(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path_to_views = os.path.abspath(args.templates_path)
    all_files = os.listdir(path_to_views)

    mesh_files = [
        filename for filename in all_files
        if os.path.isfile(os.path.join(path_to_views, filename))
        and os.path.splitext(filename)[1].lower() in MESH_EXTENSIONS
    ]

    if len(mesh_files) != 1:
        raise ValueError(
            f"Expected exactly one mesh file ({sorted(MESH_EXTENSIONS)}) in {path_to_views}, "
            f"found {len(mesh_files)}: {mesh_files}"
        )

    mesh_filename = mesh_files[0]
    path_to_mesh = os.path.join(path_to_views, mesh_filename)
    model_name = os.path.splitext(mesh_filename)[0]
    print(f"Using mesh file: {path_to_mesh}")

    object_mesh = o3d.io.read_triangle_mesh(path_to_mesh)

    if args.rotate:
        R = object_mesh.get_rotation_matrix_from_axis_angle([np.pi / 2, 0, 0])
        object_mesh.rotate(R, center=(0, 0, 0))
    
    object_pcd = object_mesh.sample_points_uniformly(number_of_points=100000)

    obb = object_pcd.get_oriented_bounding_box()
    diagonal_length = np.sqrt(obb.extent[0]**2 + obb.extent[1]**2 + obb.extent[2]**2)
    voxel_size = diagonal_length / 85
    print("Voxel size:", voxel_size)

    object_pcd_downsampled = object_pcd.voxel_down_sample(voxel_size = voxel_size)
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(object_pcd,
                                                                voxel_size = voxel_size)
    voxels = voxel_grid.get_voxels()
    voxels_centers = np.array([voxel_grid.get_voxel_center_coordinate(voxels[i].grid_index) for i in range(len(voxels))])
    
    voxels_centers_tensor = torch.Tensor(voxels_centers).cuda()
    torch_kdtree = build_kd_tree(voxels_centers_tensor)
    
    # Separate files into different lists
    exclude_prefixes = ('mask_', 'depth_', 'normals_')
    rgb_files = sorted([
        f for f in all_files 
        if f.endswith('.png') and not any(f.startswith(prefix) for prefix in exclude_prefixes)
    ])

    pcd_files = sorted([f for f in all_files if f.startswith('pcd_') and f.endswith('.npy')])
    
    # For alignment-based aggregation, we also need normals files
    if args.aggregation == 'alignment':
        normals_files = sorted([f for f in all_files if f.startswith('normals_') and f.endswith('.png')])
    
    poses = np.load(path_to_views + "/obj_poses.npy")
    
    # Extract semantic features based on the selected model
    if args.semantic_desc == 'dino':
        print("Using DINO for semantic feature extraction...")
        batch_semantic_fts, feature_dim, padding_info = extract_dino_features(args, rgb_files, path_to_views, voxel_grid, device)
        semantic_batch_mode = True
    else:  # args.semantic_desc == 'dift'
        print("Using DIFT for semantic feature extraction...")
        feature_dim = 1280
        semantic_batch_mode = False
    
    # Prepare tensors to store descriptors based on aggregation method
    if args.aggregation == 'alignment':
        # Alignment-based: need to store per-view descriptors and angles
        view_angles = torch.zeros([len(voxel_grid.get_voxels()), len(rgb_files)]).cuda()
        point_feats = torch.zeros([len(rgb_files), len(voxel_grid.get_voxels()), feature_dim]).cuda()
    else:  # args.aggregation == 'average'
        # Simple averaging: accumulate descriptors and count
        view_count = torch.zeros([len(voxel_grid.get_voxels())]).cuda()
        point_feats = torch.zeros([len(voxel_grid.get_voxels()), feature_dim]).cuda()

    # Pre-allocate tensors outside the loop
    voxel_centers_camera = torch.empty((len(voxels_centers_tensor), 3), device=device)

    # Start iterating over all the frames
    if args.aggregation == 'alignment':
        # Alignment-based aggregation
        for idx, (rgb_file, pcd_file, pose) in enumerate(zip(rgb_files, pcd_files, poses)):    
            print(f"Integrating file {idx}.")
        
            points = np.load(os.path.join(path_to_views,pcd_file))
            points_cleaned = points[~np.isnan(points).any(axis=1)]
            query_points = torch.Tensor(points_cleaned).cuda()
        
            # Load normals for alignment calculation
            normals_file = normals_files[idx]
            normals = Image.open(os.path.join(path_to_views, normals_file))
            normals = np.array(normals)
            normals = normals.reshape(points.shape)/255
            normals = normals * 2 -1
            normals = normals[~np.isnan(points).any(axis=1)]

            pose_torch = torch.Tensor(pose).cuda()
        
            # Transform voxel centers into the camera frame
            voxel_centers_h = torch.cat((voxels_centers_tensor, torch.ones(len(voxels_centers_tensor), 1, device=device)), dim=1).T
            voxel_centers_camera = (pose_torch @ voxel_centers_h).T[:, :3]
        
            if args.debug_alignment:
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector((pose @ np.hstack([np.array(object_pcd_downsampled.points),np.ones((np.array(object_pcd_downsampled.points).shape[0],1))]).T).T[:, :3])
                pcd1.paint_uniform_color(np.array([1.,0.,0.]))
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(points_cleaned)
                pcd2.paint_uniform_color(np.array([0.,1.,0.]))
        
                frame =  o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, )
                frame2 =  o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, )
                frame2 = frame2.transform(pose)
        
                o3d.visualization.draw_geometries([pcd1, pcd2, frame, frame2])
        
            # Extract semantic features for this view
            if semantic_batch_mode:  # DINO
                semantic_fts = batch_semantic_fts[idx]
                padding_top, padding_bottom, padding_left, padding_right = padding_info
                img_size = np.array(Image.open(os.path.join(path_to_views, rgb_file))).shape[:2]
                new_size = max(img_size)
                semantic_fts = semantic_fts.unsqueeze(0)
            else:  # DIFT
                semantic_fts, _, padding_info = extract_dift_features(args, rgb_file, path_to_views)
                padding_top, padding_bottom, padding_left, padding_right = padding_info
                img_size = np.array(Image.open(os.path.join(path_to_views, rgb_file))).shape[:2]
                new_size = max(img_size)
        
            semantic_fts_upsampled = nn.Upsample(size=(new_size,new_size), mode='bilinear')(semantic_fts)
            semantic_fts_upsampled = semantic_fts_upsampled[:, :, padding_top:new_size-padding_bottom, padding_left:new_size-padding_right]
        
            semantic_fts_reshaped = semantic_fts_upsampled.view(1,semantic_fts_upsampled.shape[1],-1)
            semantic_fts_reshaped = semantic_fts_reshaped[:,:,~np.isnan(points).any(axis=1)]
        
            k = 1   
            points_object = transform_points_to_object_frame_torch(query_points, pose_torch)
            _, inds = torch_kdtree.query(points_object, nr_nns_searches=k)

            # calculate dot product of normals with z axis
            dot_product = np.dot(normals, np.array([0, 0, 1]))
            point_alignment_scores = dot_product
        
            voxel_to_points = defaultdict(list)
            for point_idx, voxel_idx in enumerate(inds):  
                voxel_to_points[voxel_idx.item()].append(point_idx)

            # Initialize a tensor for voxel alignment scores for the current view
            voxel_alignment_scores_current_view = torch.zeros(len(voxels_centers_tensor), device=device)

            for voxel_idx, point_indices in voxel_to_points.items():
                # Get the alignment scores for all points associated with this voxel
                scores_for_voxel_points = point_alignment_scores[point_indices]
                
                # Aggregate (e.g., average)
                voxel_alignment_scores_current_view[voxel_idx] = scores_for_voxel_points.mean()
        
            angles = voxel_alignment_scores_current_view

            # Store the view angles for the current view
            view_angles[:, idx] = angles
            
            # Average descriptors for each voxel
            for voxel_idx, point_indices in voxel_to_points.items():
                # Store the average descriptor in the point_feats tensor for the current view
                point_feats[idx, voxel_idx, :] = torch.stack([semantic_fts_reshaped[0, :, idx] for idx in point_indices], dim=0).mean(dim=0)
        
        # Apply softmax on view_angles to compute weights
        view_weights = torch.nn.functional.softmax(view_angles, dim=1)
        
        # Weighted aggregation of descriptors across views
        view_weights = view_weights.permute(1, 0)  # Shape becomes [num_views, num_voxels]
        
        # 3D semantic descriptors
        final_semantic_voxel_descriptors = torch.einsum('vw,vwc->wc', view_weights, point_feats)  # Weighted sum
        
    else:  # args.aggregation == 'average'
        # Simple averaging aggregation
        for idx, (rgb_file, pcd_file, pose) in enumerate(zip(rgb_files, pcd_files, poses)):    
            print(f"Integrating file {idx}.")
        
            points = np.load(os.path.join(path_to_views,pcd_file))
            points_cleaned = points[~np.isnan(points).any(axis=1)]
            query_points = torch.Tensor(points_cleaned).cuda()
        
            pose_torch = torch.Tensor(pose).cuda()
        
            # Transform voxel centers into the camera frame
            voxel_centers_h = torch.cat((voxels_centers_tensor, torch.ones(len(voxels_centers_tensor), 1, device=device)), dim=1).T
            voxel_centers_camera = (pose_torch @ voxel_centers_h).T[:, :3]
        
            if args.debug_alignment:
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector((pose @ np.hstack([np.array(object_pcd_downsampled.points),np.ones((np.array(object_pcd_downsampled.points).shape[0],1))]).T).T[:, :3])
                pcd1.paint_uniform_color(np.array([1.,0.,0.]))
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(points_cleaned)
                pcd2.paint_uniform_color(np.array([0.,1.,0.]))
        
                frame =  o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, )
                frame2 =  o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, )
                frame2 = frame2.transform(pose)
        
                o3d.visualization.draw_geometries([pcd1, pcd2, frame, frame2])
        
            # Extract semantic features for this view
            if semantic_batch_mode:  # DINO
                semantic_fts = batch_semantic_fts[idx].unsqueeze(0)
                padding_top, padding_bottom, padding_left, padding_right = padding_info
                img_size = np.array(Image.open(os.path.join(path_to_views, rgb_file))).shape[:2]
                new_size = max(img_size)
            else:  # DIFT
                semantic_fts, _, padding_info = extract_dift_features(args, rgb_file, path_to_views)
                padding_top, padding_bottom, padding_left, padding_right = padding_info
                img_size = np.array(Image.open(os.path.join(path_to_views, rgb_file))).shape[:2]
                new_size = max(img_size)
        
            semantic_fts_upsampled = nn.Upsample(size=(new_size,new_size), mode='bilinear')(semantic_fts)
            semantic_fts_upsampled = semantic_fts_upsampled[:, :, padding_top:new_size-padding_bottom, padding_left:new_size-padding_right]
        
            semantic_fts_reshaped = semantic_fts_upsampled.view(1,semantic_fts_upsampled.shape[1],-1)
            semantic_fts_reshaped = semantic_fts_reshaped[:,:,~np.isnan(points).any(axis=1)]
        
            k = 1   
            points_object = transform_points_to_object_frame_torch(query_points, pose_torch)
            _, inds = torch_kdtree.query(points_object, nr_nns_searches=k)
        
            voxel_to_points = defaultdict(list)
            for point_idx, voxel_idx in enumerate(inds):
                voxel_to_points[voxel_idx.item()].append(point_idx)

            # Simple averaging: accumulate descriptors and count for each voxel
            for voxel_idx, point_indices in voxel_to_points.items():
                point_feats[voxel_idx] += torch.stack([semantic_fts_reshaped[0, :, idx] for idx in point_indices], dim=0).mean(dim=0)
                view_count[voxel_idx] += 1

            torch.cuda.empty_cache()
            
        # Average the accumulated features by dividing by the count
        # Avoid division by zero by adding a small epsilon where count is zero
        view_count = view_count.unsqueeze(1)  # Make it broadcastable
        final_semantic_voxel_descriptors = point_feats / (view_count + 1e-8)
    
    ## Clean up semantic model
    if args.semantic_desc == 'dift' and hasattr(extract_dift_features, 'dift'):
        del extract_dift_features.dift
        torch.cuda.empty_cache()
    
    ## Remove unobserved voxels before computing geometric descriptors
    rows_with_all_zeros = np.all(final_semantic_voxel_descriptors.cpu().numpy() < 1e-15, axis=1)
    indices_of_all_zero_rows = np.where(rows_with_all_zeros)[0]

    final_semantic_voxel_descriptors_normalized = final_semantic_voxel_descriptors / (torch.norm(final_semantic_voxel_descriptors, dim=1, keepdim=True) + 1e-8)
    final_semantic_voxel_descriptors_normalized = np.delete(final_semantic_voxel_descriptors_normalized.cpu().numpy(), indices_of_all_zero_rows, axis=0)

    voxels_centers = np.delete(voxels_centers, indices_of_all_zero_rows, axis=0)
    
    ## Now retrieve geometric descriptors
    pcd_voxels = o3d.geometry.PointCloud()
    pcd_voxels.points = o3d.utility.Vector3dVector(voxels_centers)
    
    obb = pcd_voxels.get_oriented_bounding_box()
    diagonal_length = np.sqrt(obb.extent[0]**2 + obb.extent[1]**2 + obb.extent[2]**2)
    
    if args.geometric_desc == 'gedi':
        # Compute the diagonal length of the bounding box
        scales = [0.3, 0.4]
    elif args.geometric_desc == 'fcgf':
        scales = [0.02, 0.05]
    elif args.geometric_desc == 'fpfh':
        radius_normal = voxel_size * 2
        pcd_voxels.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        scales = [0.3, 0.4]

    geom_descriptors_list = []
    
    # Server URL
    if args.geometric_desc == 'gedi':
        url = "http://localhost:5000/compute_gedi_descriptors"
    elif args.geometric_desc == 'fcgf':
        url = "http://localhost:8000/compute_fcgf_features"

    only_semantic = getattr(args, f'only_{args.semantic_desc}', False)
    
    if not only_semantic:
        for s in scales:
            # Send request to the server
            if (args.geometric_desc == 'gedi') or (args.geometric_desc == 'fcgf'):
                print(f"Sending request to compute {args.geometric_desc} descriptors...")
                while True:
                    try:
                        response = requests.post(url, json={"point_cloud": voxels_centers.tolist(),
                                                            "r_lrf": s*diagonal_length,
                                                            "voxel_size": s*diagonal_length})
                        if response.status_code == 200:
                            break
                        else:
                            print(f"Request failed with status code {response.status_code}, retrying...")
                            time.sleep(5)
                    except requests.exceptions.RequestException as e:
                        print(f"Request failed with error: {e}, retrying...")
                        time.sleep(5)
                
            elif args.geometric_desc == 'fpfh':

                radius_feature = s*diagonal_length
                print("Computing FPFH features...")
                # Compute FPFH features
                pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd_voxels,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                fpfh_descriptors = np.array(pcd_fpfh.data).T
                geom_descriptors_list.append(fpfh_descriptors)
                continue
                
            if response.status_code == 200 and args.geometric_desc == 'gedi':
                # Parse the response
                data = response.json()
                gedi_descriptors = np.array(data['descriptors'])
                print("Received descriptors:", gedi_descriptors.shape)
                geom_descriptors_list.append(gedi_descriptors)
            elif response.status_code == 200 and args.geometric_desc == 'fcgf':
                data = response.json()
                fcgf_descriptors = np.array(data['features'])
                downsampled_pcd_fcgf = np.array(data['downsampled_point_cloud'])
                print("Received descriptors:", fcgf_descriptors.shape)

                # Build a KDTree for the downsampled point cloud
                kdtree = cKDTree(downsampled_pcd_fcgf)

                # Find the nearest neighbor for each voxel center
                distances, indices = kdtree.query(voxels_centers)

                # Arrange the respective FCGF features for the voxel centers
                voxel_center_features = fcgf_descriptors[indices]

                geom_descriptors_list.append(voxel_center_features)
            else:
                print("Error:", response.json())
    
        full_geom_descriptor = np.concatenate(geom_descriptors_list, axis=1)
    
        # Finalize 3D descriptors
    
        # Normalize the vectors
        full_geom_descriptor_normalized = full_geom_descriptor / np.linalg.norm(full_geom_descriptor, axis=1, keepdims=True)
        print("Computed geometric descriptors of dimension:", full_geom_descriptor_normalized.shape)
    
        final_voxel_descriptors = np.concatenate([final_semantic_voxel_descriptors_normalized,full_geom_descriptor_normalized], axis=1)
        print("Final combined descriptors dimension:", final_voxel_descriptors.shape)
    
    else:
        final_voxel_descriptors = final_semantic_voxel_descriptors_normalized
        print("Final descriptors dimension:", final_voxel_descriptors.shape)
        
    # Remove rows with NaNs if any
    rows_with_nan = np.isnan(final_voxel_descriptors).any(axis=1)
    if np.any(rows_with_nan):
        print(f"Warning: Found {np.sum(rows_with_nan)} rows with NaN values in the final descriptors. Removing these rows.")
        final_voxel_descriptors = final_voxel_descriptors[~rows_with_nan]
        voxels_centers = voxels_centers[~rows_with_nan]

    ## PCA VISUALIZATION OF DESCRIPTORS
    
    if visualize:
    
        pcd_voxels.points = o3d.utility.Vector3dVector(voxels_centers)
        
        pca = PCA(n_components=3) 
        print(final_voxel_descriptors.shape)
        data_pca = pca.fit_transform(final_voxel_descriptors)
        min_max_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        data_pca_normalized = np.array([min_max_scaler(component) for component in data_pca.T]).T
        # Use the normalized PCA components as RGB values
        colors = data_pca_normalized
    
        pcd_voxels.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd_voxels])
        
    
    if save:
        if args.output_name is not None:
            output_file_name = args.output_name.replace(args.suffix, '') + ".npz"
        else:
            output_file_name = model_name.replace(args.suffix, '') + ".npz"
    
        if only_semantic:
            np.savez(output_file_name, 
                    points=voxels_centers, 
                    descriptors = final_semantic_voxel_descriptors_normalized,
                    path_to_mesh = path_to_mesh,
                    voxel_size = voxel_size
                    )
        else:
            semantic_desc_key = f'{args.semantic_desc}_descriptors'
            np.savez(output_file_name, 
                    points = voxels_centers, 
                    geom_descriptors = full_geom_descriptor_normalized,
                    **{semantic_desc_key: final_semantic_voxel_descriptors_normalized},
                    descriptors = final_voxel_descriptors,
                    path_to_mesh = path_to_mesh,
                    voxel_size = voxel_size
                    )
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Unified FoundPath descriptor extraction pipeline with configurable semantic and geometric descriptors.')
    parser.add_argument('--templates-path', type=str, required=True,
                       help='Path to directory containing rendered templates and exactly one mesh file (.ply/.obj/.glb)')
    parser.add_argument('--semantic-desc', type=str, choices=['dino', 'dift'], default='dino', 
                       help='Semantic descriptor type: "dino" for DINOv2, "dift" for DIFT')
    parser.add_argument('--aggregation', type=str, choices=['average', 'alignment'], default='alignment', 
                       help='Aggregation method: "average" for simple averaging, "alignment" for alignment-based weighted aggregation')
    
    # DINO-specific arguments
    parser.add_argument('--dino-size', type=str, default='base', choices=['small', 'base', 'large', 'giant'],
                       help='DINO model size (only used when semantic_desc=dino)')
    parser.add_argument('--only-dino', action='store_true',
                       help='Only extract DINO descriptors without geometric features (only used when semantic_desc=dino)')
    
    # DIFT-specific arguments
    parser.add_argument('--sd-prompt', type=str, default=None,
                       help='Stable Diffusion prompt (only used when semantic_desc=dift)')
    parser.add_argument('--only-dift', action='store_true',
                       help='Only extract DIFT descriptors without geometric features (only used when semantic_desc=dift)')
    
    # Common arguments
    parser.add_argument('--geometric-desc', type=str, default='gedi', choices=['gedi', 'fcgf', 'fpfh'],
                       help='Geometric descriptor type')
    parser.add_argument('--suffix', type=str, default='_manifold.obj')
    parser.add_argument('--debug-alignment', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--output-name', default=None, type=str)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--rotate', action='store_true')

    args = parser.parse_args()

    main(args)