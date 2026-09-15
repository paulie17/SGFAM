import time
from functional_maps.pyFM.mesh import TriMesh
from functional_maps.pyFM.functional import FunctionalMapping

def compute_surface_map(
    mesh1_vertices,
    mesh1_faces,
    mesh2_vertices,
    mesh2_faces,
    c1,
    c2,
    n_ev=50,
    optimizer='L-BFGS-B',
    fit_params=None,
    verbose=True
):
    """
    Computes Functional Map optimization between mesh 1 and mesh 2 using input descriptors c1 and c2.

    Returns:
        opt_time (float): Optimization time in seconds.
        emb1 (np.ndarray): Spectral embedding coordinates for mesh 1.
        emb2 (np.ndarray): Spectral embedding coordinates for mesh 2.
        p2p_21 (np.ndarray): Point-to-point mapping from mesh 2 to mesh 1.
        p2p_12 (np.ndarray): Point-to-point mapping from mesh 1 to mesh 2.
    """
    mesh1 = TriMesh(mesh1_vertices, mesh1_faces)
    mesh2 = TriMesh(mesh2_vertices, mesh2_faces)

    process_params = {
        'n_ev': (n_ev, n_ev),
        'n_descr': c1.shape[1],
        'landmarks': None,
        'descr1': c1,
        'descr2': c2,
        'subsample_step': 1,
    }

    model = FunctionalMapping(mesh1, mesh2, partial=False, optimizer=optimizer)
    model.preprocess(**process_params, verbose=verbose)

    start_s = time.time()
    model.fit(**(fit_params or {}), verbose=verbose)
    opt_time = time.time() - start_s

    _, _, emb1, emb2 = model.get_p2p(n_jobs=1)

    p2p_21 = (model.mapped_indicator * model.eta[..., None]).argmax(axis=1)
    p2p_12 = (model.mapped_indicator * model.eta[..., None]).argmax(axis=0)

    return opt_time, emb1, emb2, p2p_21, p2p_12