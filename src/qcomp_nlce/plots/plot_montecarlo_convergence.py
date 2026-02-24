from data_manager import ExperimentDataManager
from qcomp_nlce.utils.stats import mc_disp_from_res_dict
import matplotlib.pyplot as plt
from fauplotstyle import style
import numpy as np
from pathlib import Path
from qcomp_nlce.utils.save import DATA_DIR

style()
dry_run = False
m_edm = ExperimentDataManager(
    "plot_montecarlo_convergence", project="nlce", dry_run=dry_run
)
fpath = r"2026_02_02\vqe_workflow_TFIM_hadam_stefan_AQT_200shots_00001"


edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
d = edm.load_last_saved_data_file
ks = d["inputs"]["ks"]

# you got a mean std
mean_stds = []
samples = []

for pow in np.linspace(1, 4, 30):
    n_mc_samples = 10**pow
    mc_disp = mc_disp_from_res_dict(ks, d, "vqe", int(n_mc_samples), to_eotm=True)
    std = mc_disp["fake_backend"]["std"]
    mean_stds.append(np.mean(std))
    samples.append(n_mc_samples)


fig, ax = plt.subplots()
ax.plot(samples, mean_stds, "x-")

ax.set_xlabel("Number of samples")
ax.set_xscale("log")
ax.set_ylabel("Mean EOTM")
m_edm.save_figure(fig, filename=f"montecarlo_convergence", add_timestamp=False)
plt.show()
