import tenpy
import tenpy.networks.site as tns
from tenpy.models.model import MPOModel, CouplingModel
from tenpy.networks.mps import MPS
import numpy as np
from tenpy.algorithms import dmrg
import os 
import tqdm
import argparse
import pickle

class ANNNI(CouplingModel):
    def __init__(self, model_param):
        self.lat = self.init_lattice(model_param)
        self.init_terms(model_param)
        self.name = 'ANNNI'

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', None)
        #self.logger.info("%s: set conserve to %s", self.name, conserve)
        site = tns.SpinHalfSite(conserve)
        return [site]

    def init_lattice(self, model_params):
        L = model_params.get("L", 2)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = 'periodic' if bc_MPS == 'infinite' else 'open'
        sites = self.init_sites(model_params)
        lat = tenpy.models.lattice.Lattice([L], sites, bc=bc, bc_MPS=bc_MPS)
        return lat

    def init_terms(self, model_params):
        CouplingModel.__init__(self, self.lat)
        j = model_params.get("j", 1.0)
        h = model_params.get("h", 1.0)
        k = model_params.get("k", 0.0)
        #lamb = model_params.get("lamb", 0.0)
        self.explicit_plus_hc =False
        self.add_multi_coupling(-j, [('Sigmax',0,0),('Sigmax',1,0)],plus_hc=False)
        self.add_multi_coupling(+k, [('Sigmax',0,0),('Sigmax',2,0)],plus_hc=False)
        self.add_onsite(h, 0, 'Sigmaz')
        MPOModel.__init__(self, self.lat, self.calc_H_MPO())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DMRG states of the ANNNI model')

    parser.add_argument('--side', type=int, default=51, metavar='INT', 
                        help='Discretization of h and k')

    parser.add_argument('--L', type=int, default=12, metavar='INT', 
                        help='Number of spins')

    parser.add_argument('--chi', type=int, default=12, metavar='INT', 
                        help='Maximum bond dimension')

    parser.add_argument('--path', type=str, default='./', metavar='STR', 
                        help='Path to store results')

    parser.add_argument(
        "--hide",
        action="store_true",
        help="suppress progress bars",
    )

    args = parser.parse_args()

    # Computing simulations
    L = args.L
    chi = args.chi
    path = args.path

    initial_state = ['up'] * L
    hs = np.linspace(0.0, 2.0, args.side)
    ks = np.linspace(0.0, 1.0, args.side)

    if not os.path.exists(path):
        os.makedirs(path)

    # Dictionary to store results
    mps_data = {}

    progress = tqdm.tqdm(range(args.side * args.side), disable=args.hide)
    for h in hs:
        for k in ks:
            key = (float(k), float(h))
            progress.update(1)

            model_params = {'L': L, 'J': 1.0, 'k': k, 'h': -h, 'bc_MPS': 'finite'}
            bc_MPS = model_params["bc_MPS"]
            model = ANNNI(model_params)
            psi = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc=bc_MPS)
            
            dmrg_params = {
                'mixer': False,
                'chi_list': {0: chi},
                'min_sweeps': 10,
                'max_sweeps': 4,
                'N_sweeps_check': 4,
                'combine': True,
                'norm_tol': 1,
                'max_trunc_err': 10,
            }
            eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
            E, psi = eng.run()

            # Compute and store shapes and tensors
            tensors = [psi.get_B(i)._data[0] for i in range(L)]
            shapes = [t.shape for t in tensors]

            # Save in dictionary
            mps_data[key] = (shapes, tensors)

    # Save entire dictionary as a single pickle file
    save_path = f"{path}/ANNNI_L{L}_X{chi}.pkl"
    with open(save_path, "wb") as file:
        pickle.dump(mps_data, file)

    print(f"Saved all MPS data in {save_path}")