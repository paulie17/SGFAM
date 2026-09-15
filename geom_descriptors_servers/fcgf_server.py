from flask import Flask, request, jsonify
import torch
import numpy as np
import open3d as o3d
import argparse
import os

from warpconvnet.models.fcgf import ResUNetBN2C
from warpconvnet.geometry.types.voxels import Voxels

app = Flask(__name__)

model = None
device = None


def init_model(checkpoint_path, conv1_kernel_size=7):
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUNetBN2C(
        in_channels=1,
        out_channels=32,
        normalize_feature=True,
        conv1_kernel_size=conv1_kernel_size
    ).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading FCGF WarpConvNet checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: FCGF Checkpoint file not found at {checkpoint_path}")

    model.eval()


@app.route('/compute_fcgf_features', methods=['POST'])
def compute_fcgf_features():
    try:
        data = request.get_json()
        point_cloud = np.array(data['point_cloud'])
        voxel_size = data['voxel_size']

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        xyz_down = np.asarray(pcd_down.points)

        coords = torch.floor(torch.tensor(xyz_down, dtype=torch.float32, device=device) / voxel_size).int()
        feats = torch.ones((len(coords), 1), dtype=torch.float32, device=device)
        vox = Voxels([coords], [feats]).to(device)

        with torch.no_grad():
            descriptors = model(vox).feature_tensor

        desc_list = descriptors.cpu().numpy().tolist()
        return jsonify({
            "features": desc_list,
            "descriptors": desc_list,
            "downsampled_point_cloud": xyz_down.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCGF Feature Server (WarpConvNet Backend)')
    parser.add_argument('--checkpoint', type=str, default='fcgf-3dmatch-wcn.pth', help='Path to FCGF model checkpoint')
    parser.add_argument('--port', type=int, default=8000, help='Port for FCGF server')
    parser.add_argument('--conv1_kernel_size', type=int, default=7, help='Kernel size for conv1')
    args = parser.parse_args()

    init_model(args.checkpoint, conv1_kernel_size=args.conv1_kernel_size)
    app.run(host='0.0.0.0', port=args.port)