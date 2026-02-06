from data_manager import ExtendedJSONDecoder

import io
import json

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from typing import Literal
from qcomp_nlce.utils import BACKEND_TYPES, reorder_matrix_elems
from qcomp_nlce.utils.results import pretty_print_diff, pretty_print
from data_manager import ExperimentDataManager
from fauplotstyle import style


def load_json(fpath: Path) -> list:
    with io.open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read(), cls=ExtendedJSONDecoder)
    return jobj


dry_run = True
edm = ExperimentDataManager(
    f"plot_error_mitigation_AQT", project="nlce", dry_run=dry_run
)


aqt_fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_02_06\error_mitigation_test_aqt_00001\run_00000\data\watermelon_00000.json"

jobj = load_json(aqt_fpath)


fig, ax = plt.subplots()


# {
#     "inputs": {"n_qubits_max": n_qubits_max, "J": J, "n_shots": n_shots},
#     "results": {"true_vals": true_vals, "fake_vals": fake_vals},
# }

# jobjfake_vals
# "with": [
#     np.zeros((n_qubits_max, n_qubits_max, n_qubits_max), dtype=complex),
#     np.zeros((n_qubits_max, n_qubits_max, n_qubits_max), dtype=complex),
#     0,
# ],
# "without": [
#     np.zeros((n_qubits_max, n_qubits_max, n_qubits_max), dtype=complex),
#     np.zeros((n_qubits_max, n_qubits_max, n_qubits_max), dtype=complex),
#     0,
# ],


true_vals = jobj["results"]["true_vals"]
for l in range(true_vals.shape[-1]):
    true_vals[:, :, l] = reorder_matrix_elems(true_vals[:, :, l])

fake_vals = jobj["results"]["fake_vals"]
n_qubits = jobj["inputs"]["n_qubits_max"]


errors = {
    "with": np.zeros((n_qubits, n_qubits)),
    "without": np.zeros((n_qubits, n_qubits)),
}

for ind_x in range(n_qubits):
    for ind_y in range(n_qubits):
        print(f"# {ind_x} -- {ind_y}")

        for k in fake_vals:
            print(f"-- {k}")
            fzz = fake_vals[k][0][ind_x, ind_y, :]
            fzz_std = fake_vals[k][0][ind_x, ind_y, :]
            tzz = true_vals[ind_x, ind_y, :]
            meanerr = np.mean(np.abs(fzz - tzz))
            errors[k][ind_x, ind_y] = meanerr
            pretty_print_diff(fzz, tzz)
        print(f"shots with: {fake_vals['with'][2]}, without: {fake_vals['without'][2]}")


for k in ("with", "without"):
    print(f"error {k}")
    pretty_print(errors[k])

error_improvement_perc = (errors["with"] - errors["without"]) / errors["without"]

pretty_print(error_improvement_perc)


print(f"MEAN PERCENTAGE IMPROVEMENT: {np.mean(error_improvement_perc)}")

jjj = 0

pretty_print_diff(true_vals[:, :, jjj], fake_vals["with"][0][:, :, jjj])
