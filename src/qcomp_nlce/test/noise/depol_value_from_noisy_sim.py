from src.qcomp_nlce.utils import (
    dispersion_1d_thermodynamic,
    DATA_DIR,
    H_effective,
    get_ground_energy,
    TFIM,
)

from data_manager import ExperimentDataManager

import numpy as np

from pathlib import Path


def getdepol(noisy, ideal):
    return 1 - noisy / ideal


def getdepolcpx(noisy, ideal):
    return getdepol(noisy.real, ideal.real), getdepol(noisy.imag, ideal.imag)


def okdepol(m):
    return m[np.abs(m) <= 1]


def load_vqe_mats() -> dict:
    ledm = ExperimentDataManager.load(
        Path(
            DATA_DIR,
            "depol_vs_shots_80j/vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_WORSE_2_2000shots_x_20reps",
        )
    )

    jobj = ledm.load_last_saved_data_file

    for k in ("dict_l", "dict_s"):
        print(k)
        exd_fb = jobj["results"]["vqe"][k]["exd"]["fake_backend"]["values"]
        exd_sv = jobj["results"]["vqe"][k]["exd"]["statevector"]["values"]
        gsd_fb = jobj["results"]["vqe"][k]["gsd"]["fake_backend"]["energy"]
        gsd_sv = jobj["results"]["vqe"][k]["gsd"]["statevector"]["energy"]
        print(f"gsd: {getdepol(gsd_fb, gsd_sv)}")

        for f in (np.mean, np.std, np.min, np.max):
            print(
                f"{f.__name__} exd: {f(okdepol(getdepolcpx(exd_fb, exd_sv)[0]))},{f(okdepol(getdepolcpx(exd_fb, exd_sv)[1]))}"
            )


load_vqe_mats()
