from data_manager import ExperimentDataManager
import matplotlib.pyplot as plt

from src.qcomp_nlce.utils.stats import mc_disp_from_res_dict
from src.qcomp_nlce.utils.dispersion import exact_dispersion_1qp_1d
import numpy as np

if __name__ == "__main__":

    edm2 = ExperimentDataManager(
        "statevector_noise_add_sensitivity", dry_run=False, project="nlce"
    )
    fpath = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_12\workflow_tfim_hadam_stefan_QEXA_1000000shots_00001"
    edm = ExperimentDataManager.load(fpath)
    _d = edm.load_last_saved_data_file
    ks = _d["inputs"]["ks"]
    n_shots = _d["inputs"]["n_shots"]
    sigma_add = 1 / np.sqrt(10000)

    fig, ax = plt.subplots()

    pow_disps = []

    for noise_pow in np.linspace(0, 1.5, 20):
        noise = 10**noise_pow
        disps = mc_disp_from_res_dict(
            ks,
            _d,
            "vqe",
            n_samples=10000,
            sampler_opts={"sigma_scale": noise, "sigma_add": sigma_add},
        )
        pow_disps.append({"noise": noise, "disps": disps})
        ax.errorbar(
            ks,
            disps["statevector"]["mean"],
            disps["statevector"]["std"],
            label=rf"${noise:.3e}\times$",
        )

    ax.plot(ks, exact_dispersion_1qp_1d(1, ks), label="exact")
    ax.legend()

    edm2.save_dict(pow_disps)
    edm2.save_figure(fig)

    plt.show()
