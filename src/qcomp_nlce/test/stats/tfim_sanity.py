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

from scipy.linalg import inv, sqrtm


def load_vqe_mats() -> dict:
    ledm = ExperimentDataManager.load(Path(DATA_DIR, "1qp/vqe_tfim_80j_0hl_5q"))

    jobj = ledm.load_last_saved_data_file

    d = {
        "dict_l": {"eld": None, "exd": None, "gsd": None},
        "dict_s": {"eld": None, "exd": None, "gsd": None},
    }

    for k in ("dict_l", "dict_s"):
        d[k]["eld"] = jobj["results"]["vqe"][k]["eld"]["statevector"]["values"]
        d[k]["exd"] = jobj["results"]["vqe"][k]["exd"]["statevector"]["values"]
        d[k]["gsd"] = jobj["results"]["vqe"][k]["gsd"]["statevector"]["energy"]

    return d


def dispersion_from_d(ks: np.ndarray, d: dict) -> np.ndarray:
    heff_l = H_effective(d["dict_l"]["gsd"], d["dict_l"]["exd"], d["dict_l"]["eld"])
    heff_s = H_effective(d["dict_s"]["gsd"], d["dict_s"]["exd"], d["dict_s"]["eld"])
    return [dispersion_1d_thermodynamic(k, heff_l, heff_s) for k in ks]


n_qubits = 5
for J in np.linspace(0.1, 1, 100):
    model = TFIM(4, 1, J)
    e = get_ground_energy(model)
    # print(e)

N = n_qubits

ks = np.linspace(0, np.pi, 21)

for k in ks:
    mat_el = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # Calculate the phase factor exp(1im*k*(i-j)) and multiply with the matrix element
            # Here k is the momentum, and i, j are the site indices
            func = np.exp(1j * k * (i - j)) * e
            # Append the result to the "big" array
            # mat_el[i, j] = func
            # print(func)

    print(np.real(np.sum(mat_el)))
