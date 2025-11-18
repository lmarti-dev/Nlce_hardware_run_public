from qcomp_nlce.hw.quasiparticle.unitary_1qp_sector import measure_1qp_elements
from qcomp_nlce.hw.gs.subspace_sampling_hw import (
    measure_ground_state_energy_subspace_sampling,
)

from Nlce_hardware_run.src.qcomp_nlce.hw.quasiparticle.hamiltonian_1qp_expectation import (
    measure_1qp_expectations,
)

from qcomp_nlce.utils import (
    load_optimized_vqe,
    TFIM,
    get_min_gap,
    apply_parameters,
    get_tfim_sweep_circuit,
    get_ground_energy,
)
from qcomp_nlce.utils.dispersion import dispersion_1d_thermodynamic
from qcomp_nlce.utils.effective_hamiltonian import H_effective

from pathlib import Path
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from typing import Literal


def vqe_dispersion(
    gs_dirname: Path,
    elem_dirname: Path,
    run_on_hardware: bool,
    backend_name: str,
    n_shots: int,
):
    hamiltonian, elem_circuit, elem_weights, ground_energy = load_optimized_vqe(
        elem_dirname
    )
    elem_circuit = apply_parameters(elem_circuit, elem_weights[-1])
    hamiltonian, gs_circuit, gs_weights, ground_energy = load_optimized_vqe(gs_dirname)
    gs_circuit = apply_parameters(gs_circuit, gs_weights[-1])
    assert hamiltonian == hamiltonian

    get_dispersion(
        gs_circuit,
        elem_circuit,
        hamiltonian,
        ground_energy,
        run_on_hardware,
        backend_name,
        n_shots,
    )


def sweep_dispersion(
    n_qubits: int,
    J: float,
    mult: float,
    run_on_hardware: bool,
    backend_name: str,
    n_shots: int,
):
    ham_j, ham_h = TFIM(n_qubits, 1, J, periodic=False, split_parts=True)
    hamiltonian = ham_j + ham_h
    min_gap = get_min_gap(hamiltonian)
    # the most important thing is to have enough steps!!!
    # the total time can be low-ish

    total_time = 1 / (min_gap**2)
    n_steps = 1 * hamiltonian.num_qubits
    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=(mult * total_time),
        n_steps=int(mult * n_steps),
    )
    ground_energy = get_ground_energy(hamiltonian)
    get_dispersion(
        circuit,
        circuit,
        hamiltonian,
        ground_energy,
        run_on_hardware,
        backend_name,
        n_shots,
    )


def dispersion_backend_type(
    eld: dict,
    gsd: dict,
    exd: dict,
    which: Literal["statevector", "fake_backend", "real_backend"],
):
    overlap = eld[which]["values"]
    one_qp_expectation = exd[which]["values"]
    gs_expectation = gsd[which]["energy"]

    heff = H_effective(gs_expectation, one_qp_expectation, overlap)

    disp = dispersion_1d_thermodynamic()
    return disp


def get_dispersion(
    gs_circuit: QuantumCircuit,
    elem_circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    ground_energy: float,
    run_on_hardware: bool,
    backend_name: str = "IBM",
    n_shots: int = 1000,
):
    elements_dict = measure_1qp_elements(
        elem_circuit, backend_name, n_shots, run_on_hardware
    )

    ground_state_dict = measure_ground_state_energy_subspace_sampling(
        hamiltonian, ground_energy, gs_circuit, n_shots
    )

    expectations_dict = measure_1qp_expectations(
        elem_circuit, backend_name, n_shots, run_on_hardware, n_shots
    )


if __name__ == "__main__":

    ...
