from src.qcomp_nlce.utils.results import average_result_matrices
from src.qcomp_nlce.utils import DATA_DIR
from data_manager import ExperimentDataManager
from pathlib import Path

fpaths = [r"1qp/vqe_tfim_80j_0hl_5q", r"1qp/vqe_tfim_80j_0hl_5q_run2"]

dicts = []
for fpath in fpaths:
    edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    d = edm.load_last_saved_data_file
    dicts.append(d)

to_average = ["exd", "eld", "gsd"]
out_d = average_result_matrices(
    dicts,
    to_average=to_average,
    backends=["real_backend", "fake_backend", "statevector", "system"],
)
