from data_manager import ExtendedJSONDecoder

import io
import json

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_json(fpath: Path) -> list:
    with io.open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read(), cls=ExtendedJSONDecoder)
    return jobj


def get_matrix(jobj, size, which):
    return (
        jobj["results"]["vqe"][size][which]["fake_backend"]["values"],
        jobj["results"]["vqe"][size][which]["fake_backend"]["stds"],
    )


def get_gs(jobj, size):
    return (
        jobj["results"]["vqe"][size]["gsd"]["fake_backend"]["energy"],
        jobj["results"]["vqe"][size]["gsd"]["fake_backend"]["std"],
    )


if __name__ == "__main__":

    aqt_fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_09\workflow_tfim_hadam_stefan_AQT_2000shots\run_00000\data\grapefruit_00000.json"
    iqm_fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_09\workflow_tfim_hadam_stefan_IQM_2000shots\run_00000\data\pineapple_00000.json"

    aqt_jobj = load_json(aqt_fpath)
    iqm_jobj = load_json(iqm_fpath)
    textcolors = ["k", "w"]

    for jobj in (aqt_jobj, iqm_jobj):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        pn = jobj["inputs"]["provider_name"]
        nq = jobj["inputs"]["n_qubits"]

        for ind_x, size in enumerate(("dict_l", "dict_s")):
            for ind_y, which in enumerate(("eld", "exd")):
                mat, stds = get_matrix(jobj, size, which)
                astds = np.abs(stds)
                ax: plt.Axes = axes[ind_x, ind_y]
                im = ax.imshow(np.abs(stds))
                threshold = im.norm(np.max(astds)) / 2

                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        ttype = "val"
                        if ttype == "val":
                            s = f"{mat[i, j].real:.3f}\n{np.abs(mat[i, j].imag):.3f}"
                        elif ttype == "std":
                            rstd = np.abs(stds[i, j].real)
                            istd = np.abs(stds[i, j].imag)
                            s = f"{float(rstd):.2f}\n{float(istd):.2f}"
                        text = ax.text(
                            j,
                            i,
                            s,
                            ha="center",
                            va="center",
                            color=textcolors[int(im.norm(astds[i, j]) < threshold)],
                        )
                cbar = ax.figure.colorbar(
                    im,
                    ax=ax,
                )
                ax.set_title(f"{size}-{which}")
        fig.tight_layout()
        fig.suptitle(pn)

        fig.savefig(f"error_analysis_{pn}_{nq}.pdf", format="pdf")
        plt.show()
