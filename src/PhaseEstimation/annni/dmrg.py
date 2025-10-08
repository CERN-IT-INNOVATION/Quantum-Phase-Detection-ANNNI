import tenpy
import tenpy.networks.site as tns
from tenpy.algorithms import dmrg
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.mps import MPS


def DMRG_PARAMS(chi: int):
    return {
        "mixer": True,  # helps convergence if stuck in local minima
        "chi_list": {
            0: chi // 4,
            10: chi // 2,
            20: chi,  # gradually increase χ to improve precision
        },
        "min_sweeps": 20,  # more sweeps
        "max_sweeps": 100,  # allow more time to converge
        "N_sweeps_check": 10,  # check convergence every 10 sweeps
        "combine": True,  # helps with better updates
        "norm_tol": 1e-10,  # tighter tolerance
    }


class ANNNI(CouplingModel):
    def __init__(self, model_param):
        self.lat = self.init_lattice(model_param)
        self.init_terms(model_param)
        self.name = "ANNNI"

    def init_sites(self, model_params):
        conserve = model_params.get("conserve", None)
        # self.logger.info("%s: set conserve to %s", self.name, conserve)
        site = tns.SpinHalfSite(conserve)
        return [site]

    def init_lattice(self, model_params):
        L = model_params.get("L", 2)
        bc_MPS = model_params.get("bc_MPS", "finite")
        bc = "periodic" if bc_MPS == "infinite" else "open"
        sites = self.init_sites(model_params)
        lat = tenpy.models.lattice.Lattice([L], sites, bc=bc, bc_MPS=bc_MPS)
        return lat

    def init_terms(self, model_params):
        CouplingModel.__init__(self, self.lat)
        j = model_params.get("J")
        h = model_params.get("h")
        k = model_params.get("k")
        self.explicit_plus_hc = False
        self.add_multi_coupling(-j, [("Sigmax", 0, 0), ("Sigmax", 1, 0)], plus_hc=False)
        self.add_multi_coupling(
            +j * k, [("Sigmax", 0, 0), ("Sigmax", 2, 0)], plus_hc=False
        )
        self.add_onsite(j * h, 0, "Sigmaz")
        MPOModel.__init__(self, self.lat, self.calc_H_MPO())


def dmrg_annni(L, chi, k, h, initial_state=["up"]):
    model_params = {"L": L, "J": 1, "k": k, "h": -max(h, 0.1), "bc_MPS": "finite"}
    bc_MPS = model_params["bc_MPS"]
    model = ANNNI(model_params)
    psi = MPS.from_product_state(model.lat.mps_sites(), initial_state * L, bc=bc_MPS)

    eng = dmrg.TwoSiteDMRGEngine(psi, model, DMRG_PARAMS(chi))
    E, psi = eng.run()

    # Compute and store shapes and tensors
    tensors = [psi.get_B(i)._data[0] for i in range(L)]
    # shapes = [t.shape for t in tensors]

    return E, tensors
