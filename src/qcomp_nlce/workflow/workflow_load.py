import copy
from data_manager import ExperimentDataManager
from qcomp_nlce.utils import (
    heff_backend_type,
    dispersion_all_backends,
    plot_multiple_dispersions,
    plot_all_backends_dispersions_UT,
)

import matplotlib.pyplot as plt


def average_matrices(dicts: list[dict]) -> dict:

    n_dicts = len(dicts)

    out_d = copy.deepcopy(dicts[0])
    ks = out_d["inputs"]["ks"]
    for si in ("dict_l", "dict_s"):
        for backend in out_d["results"]["vqe"][si]["eld"]:
            if backend != "system":
                out_d["results"]["vqe"][si]["eld"][backend]["values"] = (
                    out_d["results"]["vqe"][si]["eld"][backend]["values"] / n_dicts
                )
                out_d["results"]["vqe"][si]["exd"][backend]["values"] = (
                    out_d["results"]["vqe"][si]["exd"][backend]["values"] / n_dicts
                )
                out_d["results"]["vqe"][si]["gsd"][backend]["energy"] = (
                    out_d["results"]["vqe"][si]["gsd"][backend]["energy"] / n_dicts
                )

    for ii in range(1, len(dicts)):
        d = dicts[ii]
        for si in ("dict_l", "dict_s"):
            for backend in d["results"]["vqe"][si]["eld"]:
                if backend != "system":
                    out_d["results"]["vqe"][si]["eld"][backend]["values"] = (
                        out_d["results"]["vqe"][si]["eld"][backend]["values"]
                        + d["results"]["vqe"][si]["eld"][backend]["values"] / n_dicts
                    )
                    out_d["results"]["vqe"][si]["exd"][backend]["values"] = (
                        out_d["results"]["vqe"][si]["exd"][backend]["values"]
                        + d["results"]["vqe"][si]["exd"][backend]["values"] / n_dicts
                    )
                    out_d["results"]["vqe"][si]["gsd"][backend]["energy"] = (
                        out_d["results"]["vqe"][si]["gsd"][backend]["energy"]
                        + d["results"]["vqe"][si]["gsd"][backend]["energy"] / n_dicts
                    )
    for si in ("dict_l", "dict_s"):
        for backend in d["results"]["vqe"][si]["eld"]:
            if backend != "system":
                eld = out_d["results"]["vqe"][si]["eld"]
                gsd = out_d["results"]["vqe"][si]["gsd"]
                exd = out_d["results"]["vqe"][si]["exd"]
                heff = heff_backend_type(eld, gsd, exd, backend)
                out_d["results"]["vqe"][si]["heff"][backend] = heff
        disp = dispersion_all_backends(
            out_d["results"]["vqe"]["dict_l"], out_d["results"]["vqe"]["dict_s"], ks
        )
    out_d["results"]["vqe"]["disp"] = disp
    return out_d


def average_over_runs():
    runs = (
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00005",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00001",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00002",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00003",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00004",
    )

    dicts = []
    for r in runs:
        edm = ExperimentDataManager.load(r)
        d = edm.load_last_saved_data_file
        dicts.append(d)

    out_d = average_matrices(dicts)

    fig = plot_multiple_dispersions([out_d, dicts[0]])

    plt.show()


def load_and_plot(fpath):
    edm = ExperimentDataManager.load(fpath)
    d = edm.load_last_saved_data_file
    fig = plot_all_backends_dispersions_UT(d)
    plt.show()


if __name__ == "__main__":

    fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00005"
    load_and_plot(fpath)
