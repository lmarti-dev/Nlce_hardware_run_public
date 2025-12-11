import copy
from data_manager import ExperimentDataManager
from qcomp_nlce.utils.workflow_1d import (
    heff_backend_type,
    dispersion_all_backends,
    plot_multiple_dispersions,
    plot_all_backends_dispersions_UT,
)

from qcomp_nlce.utils.stats import mean_std

import matplotlib.pyplot as plt
import numpy as np


def plot_average_matrices(dicts: list[dict]) -> dict:

    n_dicts = len(dicts)

    out_d = copy.deepcopy(dicts[0])
    ks = out_d["inputs"]["ks"]
    out_d["inputs"]["n_shots"] = 2000 * n_dicts

    for si in ("dict_l", "dict_s"):
        for backend in out_d["results"]["vqe"][si]["eld"]:
            if backend != "system":
                for mnd in ("eld", "exd"):
                    out_d["results"]["vqe"][si][mnd][backend]["values"] = np.mean(
                        list(
                            d["results"]["vqe"][si][mnd][backend]["values"]
                            for d in dicts
                        ),
                        axis=0,
                    )
                    out_d["results"]["vqe"][si][mnd][backend]["stds"] = mean_std(
                        list(
                            d["results"]["vqe"][si][mnd][backend]["stds"] for d in dicts
                        ),
                        axis=0,
                    )

                out_d["results"]["vqe"][si]["gsd"][backend]["energy"] = np.mean(
                    np.array(
                        list(
                            d["results"]["vqe"][si]["gsd"][backend]["energy"]
                            for d in dicts
                        )
                    )
                )
                out_d["results"]["vqe"][si]["gsd"][backend]["std"] = mean_std(
                    np.array(
                        list(
                            d["results"]["vqe"][si]["gsd"][backend]["std"]
                            for d in dicts
                        )
                    )
                )

    for si in ("dict_l", "dict_s"):
        for backend in out_d["results"]["vqe"][si]["eld"]:
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


def plot_average_over_runs():
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

    out_d = plot_average_matrices(dicts)

    fig = plot_all_backends_dispersions_UT(out_d)

    plt.show()


def load_and_plot(fpath):
    edm = ExperimentDataManager.load(fpath)
    d = edm.load_last_saved_data_file
    fig = plot_all_backends_dispersions_UT(d)
    plt.show()


if __name__ == "__main__":

    fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_02\workflow_tfim_all_backends_test_00005"
    plot_average_over_runs()
