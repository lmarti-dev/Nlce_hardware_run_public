import matplotlib.pyplot as plt
from src.qcomp_nlce.utils import TFIM, get_sweep_spectrum, model_str
import numpy as np
from fauplotstyle import style
from data_manager import ExperimentDataManager


def plot_spectrum(n_qubits: int, dry_run: bool):
    h = 1
    J = 1
    hl = 0
    ham_j, ham_h = TFIM(
        n_qubits=n_qubits, h=h, J=J, split_parts=True, longitudinal_field=hl
    )

    model_name = model_str(J, hl, ladder=False)

    n_steps = 2000
    spectrum = get_sweep_spectrum(ham_h, ham_j + ham_h, n_steps)

    fig, ax = plt.subplots()

    n_levels = 2**ham_j.num_qubits

    for n in range(n_levels):
        ax.plot(np.linspace(0, J, n_steps), spectrum[n, :])

    ax.set_xlabel(r"$J/h$")
    ax.set_ylabel(r"Energy")

    edm = ExperimentDataManager(
        f"plot_sweep_spectrum_{model_name}", project="nlce", dry_run=dry_run
    )

    edm.var_dump(j=J, h=h, n_qubits=n_qubits)
    edm.save_figure(
        fig,
        filename=f"sweep_spectrum_{model_name}",
        fig_shape="page-wide",
        add_timestamp=False,
    )

    plt.show()


if __name__ == "__main__":

    style()
    dry_run = True
    plot_spectrum(4, dry_run)
