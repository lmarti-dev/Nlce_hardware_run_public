from data_manager import ExperimentDataManager

from qcomp_nlce.utils import (
    load_optimized_vqe,
    load_sweep,
    get_tfim_ratio,
    pfeuty_energy,
    get_tfim_ham,
    get_ground_energy,
)
import matplotlib.pyplot as plt
import numpy as np

from fauplotstyle import style


def ED_nlce(q1: int, q2: int, ratio: float) -> float:

    ham1 = get_tfim_ham(n_qubits=q1, h=1, J=ratio)
    e1 = get_ground_energy(ham1)

    ham2 = get_tfim_ham(n_qubits=q2, h=1, J=ratio)
    e2 = get_ground_energy(ham2)

    return e2 - e1


def get_energy_and_system(which: str, backend: str = "real_backend"):
    if which == "vqe":
        filename = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_04_03\nlce_hardware_run_vqe_gs_00001"
    elif which == "sweep":
        filename = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_04_03\nlce_hardware_run_sweep_gs_00001"
    edm_load = ExperimentDataManager.load(filename)
    runs = edm_load.run_number
    data = {}

    data = {}

    for run in range(runs):
        vd = edm_load.load_var_dump(run)
        system_file = vd["system"]["name"]
        if which == "vqe":
            hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(
                system_file
            )
        elif which == "sweep":
            hamiltonian, circuit, ground_energy = load_sweep(system_file)
        ratio = get_tfim_ratio(hamiltonian)
        n_qubits = hamiltonian.num_qubits
        energy = vd[backend]["energy"]
        if ratio not in data.keys():
            data[ratio] = {}
        data[ratio][n_qubits] = energy

    out = []

    ratios = []
    for ratio in sorted(data.keys()):
        qubits = [k for k in data[ratio].keys()]
        out.append(data[ratio][max(qubits)] - data[ratio][min(qubits)])
        ratios.append(ratio)

    return ratios, out


style()

edm = ExperimentDataManager("nlce_figs", dry_run=False)

for backend in ("real_backend", "fake_backend", "statevector"):

    ratios_vqe, data_vqe = get_energy_and_system("vqe", backend)
    ratios_sweep, data_sweep = get_energy_and_system("sweep", backend)

    true_energies = np.array([pfeuty_energy(j) for j in ratios_vqe])

    fig, ax = plt.subplots()

    def relerr(e):
        return np.abs((e - true_energies) / true_energies)

    ax.plot(
        ratios_vqe,
        relerr(data_vqe),
        label="VQE",
        marker="x",
        linestyle="dashed",
    )
    ax.plot(
        ratios_sweep,
        relerr(data_sweep),
        label="Sweep",
        marker="o",
        linestyle="dashed",
    )

    ed = np.array([ED_nlce(9, 10, r) for r in ratios_vqe])
    ax.plot(
        ratios_sweep,
        relerr(ed),
        label="ED",
        marker="d",
        linestyle="dashed",
    )

    ax.legend()

    ax.set_ylabel("$\Delta E / E_0$")
    ax.set_xlabel("J/h")

    ax.set_yscale("log")

    ax.set_title(backend)

    edm.save_figure(fig)

    # plt.show()
