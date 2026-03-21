from data_manager import ExperimentDataManager
from src.qcomp_nlce.utils.workflow_utils import (
    plot_all_backends_dispersions_montecarlo,
)

from src.qcomp_nlce.utils.results import average_result_matrices, show_diffs

from fauplotstyle import style

from data_manager import ExperimentDataManager
from pathlib import Path
from src.qcomp_nlce.utils import DATA_DIR


def load_diffs(fpath: Path):
    edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
    d = edm.load_last_saved_data_file
    show_diffs(d, "real_backend")


def plot_average_over_runs(
    m_edm: ExperimentDataManager,
    fpath: list[Path],
    model_name: str,
    notes: str = "",
    dry_run: bool = True,
):
    if notes != "":
        notes = "_" + notes

    dicts = []
    for fpath in fpath:
        edm = ExperimentDataManager.load(Path(DATA_DIR, fpath))
        d = edm.load_last_saved_data_file
        dicts.append(d)

    to_average = ["exd", "eld", "gsd"]
    out_d = average_result_matrices(
        dicts,
        to_average=to_average,
        backends=["real_backend", "fake_backend", "statevector", "system"],
    )

    fig = plot_all_backends_dispersions_montecarlo(out_d)

    m_edm.var_dump(files=fpath, to_average=to_average, ref_dict=fpath[0])

    m_edm.save_figure(
        fig,
        filename=f"{model_name}{notes}_mean",
        fig_shape="nice-y",
        add_timestamp=False,
    )

    # plt.show()


if __name__ == "__main__":
    style()
    dry_run = False

    m_edm = ExperimentDataManager("plot_average_runs", project="nlce", dry_run=dry_run)
    # fpath_tfim_100j_0hl = [r"vqe_tfim_100j_0hl_5q"]
    # fpath_tfim_80j_0hl_sweep = [r"sweep_tfim_80j_0hl_5q"]
    tfim_80j_0hl = (
        [r"vqe_tfim_80j_0hl_5q", r"vqe_tfim_80j_0hl_5q_run2"],
        "TFIM_80j_0hl",
        "vqe",
    )
    tfim_80j_0hl_sweep = (
        [r"sweep_tfim_80j_0hl_5q", r"sweep_tfim_80j_0hl_5q_run2"],
        "TFIM_80j_0hl",
        "sweep",
    )
    tfim_100j_0hl = (
        [r"vqe_tfim_100j_0hl_5q", r"vqe_tfim_100j_0hl_5q_run2"],
        "TFIM_100j_0hl",
        "vqe",
    )
    tfim_50j_10hl = (
        [r"vqe_tfim_50j_10hl_5q", r"vqe_tfim_50j_10hl_5q_run2"],
        "TFIM_50j_10hl",
        "vqe",
    )
    for item in (tfim_80j_0hl, tfim_80j_0hl_sweep, tfim_100j_0hl, tfim_50j_10hl):
        fpaths = item[0]
        model_name = item[1]
        notes = item[2]
        plot_average_over_runs(m_edm, fpaths, model_name, notes, dry_run=dry_run)
