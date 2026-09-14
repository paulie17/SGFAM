from functional_maps.pyFM.mesh import TriMesh
from functional_maps.pyFM.functional import FunctionalMapping
import time

def compute_surface_map(mesh1_vertices,
                        mesh1_faces,
                        mesh2_vertices,
                        mesh2_faces, 
                        c1, c2, 
                        n_ev=50, 
                        optimizer='L-BFGS-B', 
                        descr_type="neural", 
                        fit_params=None,
                        verbose = True):
    '''
    Returns:
        mapping_2to1: torch.Tensor, shape (N2,) closest vertex on mesh 1 for each vertex on mesh 2
        mapping_1to2: torch.Tensor, shape (N1,) closest vertex on mesh 2 for each vertex on mesh 1
    '''
    # make sure the TriMesh is initialized correctly. Loading with o3d and pytorch3d produces different results.
    assert descr_type in ["neural", "HKS", "WKS"]

    mesh1 = TriMesh(mesh1_vertices, mesh1_faces)
    mesh2 = TriMesh(mesh2_vertices, mesh2_faces)

    if descr_type == "HKS":
        process_params = {
        'n_ev': (n_ev,n_ev),  # Number of eigenvalues on source and Target
        'landmarks': None,
        'n_descr': 16, 
        'descr_type': 'HKS',
        'subsample_step': 1,
        }
    elif descr_type == "WKS":
        process_params = {
        'n_ev': (n_ev,n_ev),  # Number of eigenvalues on source and Target
        'n_descr': 2048,
        'landmarks': None,
        'subsample_step': 1,  # In order not to use too many descriptors
        'descr_type': 'WKS',  # WKS or HKS
        }
    elif descr_type == "neural":
        process_params = {
        'n_ev': (n_ev,n_ev),  # Number of eigenvalues on source and Target
        'n_descr': c1.shape[1],
        'landmarks': None,
        'descr1': c1,
        'descr2': c2,
        'subsample_step': 1,
        }
        
    model = FunctionalMapping(mesh1, mesh2, partial=False, optimizer=optimizer)
    model.preprocess(**process_params,verbose=verbose)
    fit_params = fit_params

    start_s = time.time()
    model.fit(**fit_params, verbose=verbose)
    opt_time = time.time() - start_s

    _ , _ , emb1, emb2 = model.get_p2p(n_jobs=1) # sets model.mapped_indicator

    p2p_21 = (model.mapped_indicator * model.eta[..., None]).argmax(axis=1) # override the above
    p2p_12 = (model.mapped_indicator * model.eta[..., None]).argmax(axis=0)
    
    return opt_time, emb1, emb2, p2p_21, p2p_12, 