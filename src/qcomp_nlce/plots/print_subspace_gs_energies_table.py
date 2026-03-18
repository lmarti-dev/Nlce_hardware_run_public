from data_manager import ExperimentDataManager
from qcomp_nlce.utils.workflow_utils import (
    plot_all_backends_dispersions_montecarlo,
)
import matplotlib.pyplot as plt
from fauplotstyle import style
from qcomp_nlce.utils import DATA_DIR
from pathlib import Path
import re


def etopow(a: float):
    r = f"{a:.2e}"
    r = re.sub(r"e\+0*", r"\\cdot 10^{", r)
    r = re.sub(r"e-0*", r"\\cdot 10^{-", r)
    f = r + "}"
    return f


if __name__ == "__main__":
    style()

    fpath_tfim_100j_0hl = r"vqe_tfim_100j_0hl_5q"
    fpath_tfim_80j_0hl_sweep = r"sweep_tfim_80j_0hl_5q"
    fpath_tfim_80j_0hl = r"vqe_tfim_80j_0hl_5q"
    fpath_tfim_80j_0hl_run2 = r"vqe_tfim_80j_0hl_5q_run2"
    fpath_tfim_20j_0hl_ladder = r"vqe_tfim_20j_0hl_8q_ladder"

    o = []
    for items in (
        (fpath_tfim_80j_0hl, "TFIM_80j_0hl", "vqe", ""),
        (fpath_tfim_80j_0hl_run2, "TFIM_80j_0hl", "vqe", "alt. run"),
        (fpath_tfim_80j_0hl_sweep, "TFIM_80j_0hl", "sweep", "sweep"),
        (fpath_tfim_100j_0hl, "TFIM_100j_0hl", "vqe", ""),
        (fpath_tfim_20j_0hl_ladder, "TFIM_20j_0hl", "vqe", "ladder"),
    ):
        fpath, fig_name, alg, notes = items
        edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
        edm.use_calendar = False
        d = edm.load_last_saved_data_file
        if "model_dir" in d["inputs"]:
            which_model = d["inputs"]["model_dir"]
        else:
            which_model = d["inputs"]["which_model"]

        p = fig_name.split("_")
        j = float(p[1][:-1]) / 100
        hl = float(p[2][:-2]) / 100

        n_qubits = d["inputs"]["n_qubits"]

        for si in ("dict_l", "dict_s"):
            if si == "dict_s":
                n_qubits = n_qubits - 1
            if "ladder" in notes:
                qs = rf"{n_qubits} \times 2"
            else:
                qs = f"{n_qubits}"

            energy = d["results"][alg][si]["gsd"]["real_backend"]["energy"]
            err = d["results"][alg][si]["gsd"]["real_backend"]["error"]
            bp = d["results"][alg][si]["gsd"]["real_backend"]["bitstring_prop"]

            if notes != "":
                nn = f"({notes}) "
            else:
                nn = notes
            s = rf"$J/h={j}, h_l={hl}$ {nn}& ${qs}$ & ${etopow(err)}$ & ${bp*100:.1f}$\%"
            o.append(s)
    print(" \\\\\n".join(o))
