import numpy as np
import torch
import open3d as o3d
from sklearn.decomposition import PCA
from utils.kpca_torch import KernelPCA
from functional_maps.functional_map import compute_surface_map

DEFAULT_FIT_PARAMS = {
    'w_descr': 1e1,
    'w_lap': 1e3,      # Isometric length preservation
    'w_dcomm': 1e2,    # Commutativity with descriptors
    'w_orient': 0,     # Chirality preservation
    'w_area': 0,
    'w_conformal': 0,
    'w_p2p': 0,
    'w_stochastic': 0,
    'w_ent': 1e1,
    'w_range01': 0,
    'w_sumto1': 1e1,
    'optinit': 'zeros',
    'maxiter': 5000,
}

def reduce_descriptors(
    c1: np.ndarray,
    c2: np.ndarray,
    target_dim: int = 128,
    pca_type: str = 'kernel'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs joint dimensionality reduction across two descriptor matrices (c1 and c2).
    Supports 'kernel' (Kernel PCA via GPU PyTorch) and 'linear' (sklearn PCA).
    """
    if c1.shape[1] <= target_dim and c2.shape[1] <= target_dim:
        return c1, c2

    print(f"Reducing descriptor dimensionality from {c1.shape[1]} to {target_dim} using {pca_type} PCA...")
    combined_descs = np.concatenate([c1, c2], axis=0)

    if pca_type == 'kernel':
        gamma = 0.5
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        combined_torch = torch.tensor(combined_descs, dtype=torch.float32).to(device)

        kernel_pca = KernelPCA(
            n_components=target_dim,
            kernel_name="rbf",
            kernel_params={'sigma2': 1.0 / gamma}
        ).to(device)

        kernel_pca.fit(combined_torch)
        reduced_torch = kernel_pca.transform(combined_torch)
        combined_reduced = reduced_torch.detach().cpu().numpy()
    else:
        pca = PCA(n_components=target_dim)
        combined_reduced = pca.fit_transform(combined_descs)

    c1_reduced = combined_reduced[:len(c1)]
    c2_reduced = combined_reduced[len(c1):]
    return c1_reduced, c2_reduced


def solve_functional_map(
    vertices_1: np.ndarray,
    faces_1: np.ndarray,
    vertices_2: np.ndarray,
    faces_2: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    n_ev: int = 50,
    fit_params: dict = None
) -> tuple:
    """
    Computes functional map optimization between two surface meshes.
    Returns (opt_time, emb1, emb2, p2p_21, p2p_12).
    """
    if fit_params is None:
        fit_params = DEFAULT_FIT_PARAMS

    print(f"Solving functional map using {n_ev} eigenvectors...")
    output = compute_surface_map(
        vertices_1, faces_1,
        vertices_2, faces_2,
        c1, c2,
        n_ev=n_ev,
        fit_params=fit_params,
        verbose=False
    )

    opt_time, emb1, emb2, p2p_21, p2p_12 = output
    print(f"Functional map optimization completed in {opt_time:.3f} seconds.")
    return opt_time, emb1, emb2, p2p_21, p2p_12


def compute_spectral_colors(emb1: np.ndarray, emb2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Maps functional map eigen-embeddings to 3D PCA RGB colors for side-by-side Open3D visualization.
    """
    combined_emb = np.concatenate([emb1, emb2], axis=0)
    pca = PCA(n_components=3)
    combined_emb_3d = pca.fit_transform(combined_emb)

    emb1_3d = combined_emb_3d[:len(emb1)]
    emb2_3d = combined_emb_3d[len(emb1):]

    emb1_colors = (emb1_3d - emb1_3d.min()) / (emb1_3d.max() - emb1_3d.min() + 1e-8)
    emb2_colors = (emb2_3d - emb2_3d.min()) / (emb2_3d.max() - emb2_3d.min() + 1e-8)

    return emb1_colors, emb2_colors
