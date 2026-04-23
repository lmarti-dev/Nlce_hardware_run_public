from data_manager import ExperimentDataManager
from pathlib import Path
from src.qcomp_nlce.utils import DATA_DIR
from src.qcomp_nlce.utils.results import pretty_print, pretty_print_diff
import numpy as np

ledm = ExperimentDataManager.load(Path(DATA_DIR, "1qp/vqe_tfim_80j_0hl_5q"))


def err_over_std(vals, tvals, stds):
    print("err over std")
    pretty_print(np.abs(vals - tvals) / stds)


def depolrate(vals, tvals):
    print("depol rate")
    pretty_print(vals / tvals)


jobj = ledm.load_last_saved_data_file
size = "dict_l"
which = "eld"

stds = jobj["results"]["vqe"][size][which]["real_backend"]["stds"]
vals = jobj["results"]["vqe"][size][which]["real_backend"]["values"]
tvals = jobj["results"]["vqe"][size][which]["fake_backend"]["values"]


print("REAL")

err_over_std(vals.real, tvals.real, stds.real)
depolrate(vals.real, tvals.real)
print("IMAG")
err_over_std(vals.imag, tvals.imag, stds.imag)
