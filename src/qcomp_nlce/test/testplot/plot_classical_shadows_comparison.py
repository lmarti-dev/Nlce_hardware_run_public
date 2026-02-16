from data_manager import ExperimentDataManager
import matplotlib.pyplot as plt
import numpy as np
from fauplotstyle import style


style()

NOISY_IBM = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_16\classical_shadows_vs_the_world_00001"
IBM_NONOISE = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_16\classical_shadows_vs_the_world_IBM_GENERIC"

edm = ExperimentDataManager.load(IBM_NONOISE)
jobj = edm.load_last_saved_data_file


dry_run = False
edm_out = ExperimentDataManager(
    "plot_classical_shadows_vs_the_world", project="nlce", dry_run=dry_run
)

n_shots = jobj["list_shots"]
gs = jobj["ground_energy"]

DICT_LABELS = {
    "classical_shadows": "Shadows",
    "estimator": "Estimator",
    "statevector": "SV",
    "subspace_diag": "Subspace",
}


edm_out.save_dict(jobj)

fig, ax = plt.subplots()
for k in DICT_LABELS:
    ax.errorbar(
        n_shots,
        np.abs(np.array(jobj[k]["vals"]).flatten() - gs),
        np.array(jobj[k]["stds"]).flatten(),
        label=DICT_LABELS[k],
        fmt="o--",
        linewidth=0.75,
        capsize=1,
        markersize=4,
    )

ax.legend()
ax.set_xlabel("Number of shots")
ax.set_ylabel(r"$|\tilde{E}-E|$")
ax.set_xscale("log")
# ax.set_yscale("log")


edm_out.save_figure(
    fig,
    "classical_shadows_vs_the_world_TFIM_100j_0hl_5q",
    add_timestamp=False,
    fig_shape="nice-y",
)
plt.show()


print(jobj["bitstring_prop"])
