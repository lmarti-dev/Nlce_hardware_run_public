from data_manager import ExperimentDataManager
import matplotlib.pyplot as plt
import numpy as np
from fauplotstyle import style


style()


dry_run = False
edm_out = ExperimentDataManager(
    "plot_classical_shadows_vs_the_world", project="nlce", dry_run=dry_run
)


NOISY_IBM = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_16\classical_shadows_vs_the_world_00001"
IBM_NONOISE = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_17\classical_shadows_vs_the_world_IBM_PERFECT_00002"


edm_out.var_dump(NOISY_IBM=NOISY_IBM, IBM_NONOISE=IBM_NONOISE)

for fpath in (NOISY_IBM, IBM_NONOISE):
    if fpath == IBM_NONOISE:
        name = "noiseless"
    elif fpath == NOISY_IBM:
        name = "noisy"

    edm = ExperimentDataManager.load(fpath)
    jobj = edm.load_last_saved_data_file

    n_shots = jobj["list_shots"]
    gs = jobj["ground_energy"]

    DICT_LABELS = {
        "classical_shadows": "Shadows",
        "estimator": "Estimator",
        # "statevector": "SV",
        "subspace_diag": "Subspace",
    }

    edm_out.save_dict(jobj)

    fig, ax = plt.subplots()
    for k in DICT_LABELS:

        vals = np.array(jobj[k]["vals"]).flatten()
        stds = np.array(jobj[k]["stds"]).flatten()
        delta = np.abs(vals - gs)

        pluserr = np.abs(vals + stds - gs)
        minerr = np.abs(vals - stds - gs)

        out = np.array([minerr, delta, pluserr])
        outm = np.mean(out, axis=0)
        outs = np.std(out, axis=0)

        ax.errorbar(
            n_shots,
            outm,
            outs,
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
        f"classical_shadows_vs_the_world_TFIM_100j_0hl_5q_{name}",
        add_timestamp=False,
        fig_shape="nice-y",
    )
    plt.show()

    print(name, jobj["bitstring_prop"])
