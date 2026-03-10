from qcomp_nlce.utils import TFIM, H_effective_ed
from qcomp_nlce.utils.workflow_utils import (
    dispersion_1d_thermodynamic,
    tfim_ed_dispersion,
)
from qiskit.quantum_info import SparsePauliOp
import numpy as np

import matplotlib.pyplot as plt


def tfim_ed_dispersion_trunc(
    n_qubits_max: int,
    J: float,
    ks: np.ndarray,
    trunc: int,
    hl: float = 0.0,
    ladder: bool = True,
):
    hamiltonian_l = TFIM(n_qubits_max, 1, J, longitudinal_field=hl, ladder=ladder)
    hamiltonian_s = TFIM(n_qubits_max - 1, 1, J, longitudinal_field=hl, ladder=ladder)
    return ed_dispersion_trunc(hamiltonian_l, hamiltonian_s, ks, trunc)


def ed_dispersion_trunc(
    hamiltonian_l: SparsePauliOp,
    hamiltonian_s: SparsePauliOp,
    ks: np.ndarray,
    trunc: int,
) -> list[float]:
    heff_l = np.round(H_effective_ed(hamiltonian_l), trunc)
    heff_s = np.round(H_effective_ed(hamiltonian_s), trunc)
    return [dispersion_1d_thermodynamic(k, heff_l, heff_s) for k in ks]


def tfim_ed_dispersion_noise(
    n_qubits_max: int,
    J: float,
    ks: np.ndarray,
    noise: float,
    hl: float = 0.0,
    ladder: bool = True,
):
    hamiltonian_l = TFIM(n_qubits_max, 1, J, longitudinal_field=hl, ladder=ladder)
    hamiltonian_s = TFIM(n_qubits_max - 1, 1, J, longitudinal_field=hl, ladder=ladder)
    return ed_dispersion_noise(hamiltonian_l, hamiltonian_s, ks, noise)


def ed_dispersion_noise(
    hamiltonian_l: SparsePauliOp,
    hamiltonian_s: SparsePauliOp,
    ks: np.ndarray,
    noise: float,
) -> list[float]:
    heff_l = H_effective_ed(hamiltonian_l) + np.random.normal(
        0, noise, (n_qubits_max, n_qubits_max)
    )
    heff_s = H_effective_ed(hamiltonian_s) + np.random.normal(
        0, noise, (n_qubits_max - 1, n_qubits_max - 1)
    )
    return [dispersion_1d_thermodynamic(k, heff_l, heff_s) for k in ks]


if __name__ == "__main__":
    n_samples = 100
    fig, ax = plt.subplots()
    ks = np.linspace(0, np.pi, 21)
    n_qubits_max = 5
    J = 1
    hl = 0
    disp = tfim_ed_dispersion(n_qubits_max, J, ks, hl, False)
    ax.plot(ks, disp, "o-", label="ED")
    for npow in np.linspace(-3, -0.25, 10):
        noise = 10**npow
        u = np.zeros((len(ks), n_samples))
        for n in range(n_samples):
            print(n, end="\r")
            disp = tfim_ed_dispersion_noise(n_qubits_max, J, ks, noise, hl, False)
            u[:, n] = disp
        yy = np.mean(u, axis=1)
        yy_std = np.std(u, axis=1)
        ax.errorbar(ks, yy, yy_std, fmt="+:", label=rf"$10^{{{npow:.2f}}}$")
    ax.legend()
    plt.show()
