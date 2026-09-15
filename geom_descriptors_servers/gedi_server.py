from flask import Flask, request, jsonify
import torch
import numpy as np
import open3d as o3d
import os
import argparse
from gedi import GeDi

app = Flask(__name__)

gedi = None


def init_model(chkpt_path='data/chkpts/3dmatch/chkpt.tar'):
    global gedi
    config = {
        'dim': 32,
        'samples_per_batch': 500,
        'samples_per_patch_lrf': 4000,
        'samples_per_patch_out': 512,
        'r_lrf': 0.3,
        'fchkpt_gedi_net': chkpt_path
    }
    print(f"Initializing GeDi model from checkpoint: {chkpt_path}")
    gedi = GeDi(config=config)


@app.route('/compute_gedi_descriptors', methods=['POST'])
def compute_descriptors():
    try:
        data = request.get_json()
        point_cloud = np.array(data['point_cloud'])
        radius = float(data['r_lrf'])
        voxel_size = float(data.get('voxel_size', radius))

        pcd_tensor = torch.tensor(point_cloud).float()

        # If voxel_size is strictly less than radius and positive, downsample point cloud
        if 0 < voxel_size < radius:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            pcd_downsampled = pcd.voxel_down_sample(voxel_size)
            pts_tensor = torch.tensor(np.asarray(pcd_downsampled.points)).float()
        else:
            pts_tensor = pcd_tensor

        # Dynamically set r_lrf for this request
        gedi.r_lrf = radius

        with torch.no_grad():
            descriptors = gedi.compute(pts=pts_tensor, pcd=pcd_tensor)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify({
            "descriptors": descriptors.tolist(),
            "downsampled_point_cloud": pts_tensor.numpy().tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeDi Feature Server')
    parser.add_argument('--port', type=int, default=5000, help='Port for GeDi server')
    parser.add_argument('--chkpt', type=str, default='data/chkpts/3dmatch/chkpt.tar', help='Path to GeDi checkpoint')
    args = parser.parse_args()

    init_model(args.chkpt)
    app.run(host='0.0.0.0', port=args.port)
