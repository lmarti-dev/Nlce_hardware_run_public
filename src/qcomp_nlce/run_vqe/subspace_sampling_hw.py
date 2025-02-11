import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

from qcomp_nlce.utils import (
    backend_sample,
    subspace_diagonalize,
    get_counts_from_job,
    get_unique_bitstrings,
    estimate_observables,
    add_meas_layer,
    load_sim,
    get_optimal_parameters,
    get_appropriate_fake_backend,
)
from data_manager import ExperimentDataManager

from qcomp_nlce.utils.hardware import get_token


def get_expectations_from_sources(
    edm: ExperimentDataManager,
    dirname: str,
    n_shots: int = 1e3,
    run_on_ibmq: bool = False,
    override_params: bool = False,
):

    hamiltonian, circuit, weights, ground_energy = load_sim(dirname)
    fake_backend = get_appropriate_fake_backend(circuit.num_qubits)

    circuit_to_sample = add_meas_layer(circuit)

    def relerr(e):
        return np.abs(e - ground_energy) / np.abs(ground_energy)

    if override_params:
        parameters = get_optimal_parameters(dirname)
    else:
        parameters = weights[-1]

    job = backend_sample(
        backend=fake_backend,
        circuit=circuit_to_sample,
        parameters=parameters,
        n_shots=n_shots,
    )
    fake_counts = get_counts_from_job(job)
    fake_unique_bitstring = get_unique_bitstrings(fake_counts)
    fake_backend_bitstring_prop = len(fake_unique_bitstring) / (2**circuit.num_qubits)
    fake_backend_energy = subspace_diagonalize(
        bitstrings=fake_unique_bitstring, ham=hamiltonian
    )

    statevector_energy = estimate_observables(
        circuit=circuit, observables=hamiltonian, param_sweep=parameters
    ).data.evs

    if run_on_ibmq:
        token = get_token()
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        real_backend = service.least_busy(
            simulator=False, operational=True, min_num_qubits=circuit.num_qubits
        )

        real_job = backend_sample(
            backend=real_backend,
            circuit=circuit_to_sample,
            parameters=parameters,
            n_shots=n_shots,
        )
        real_counts = get_counts_from_job(real_job)
        real_unique_bitstring = get_unique_bitstrings(real_counts)
        real_bitstring_prop = len(real_unique_bitstring) / (2**circuit.num_qubits)
        real_energy = subspace_diagonalize(
            bitstrings=fake_unique_bitstring, ham=hamiltonian
        )

        print(
            f"RealBackend ({real_backend.name}) {relerr(real_energy)} sub. mat.: {len(real_unique_bitstring)} x {len(real_unique_bitstring)} %: {real_bitstring_prop}"
        )

    print(f"Statevector {relerr(statevector_energy)}")

    print(
        f"FakeBackend: {relerr(fake_backend_energy)} sub. mat.: {len(fake_unique_bitstring)} x {len(fake_unique_bitstring)} %: {fake_backend_bitstring_prop}"
    )

    jobj = {
        "fake_backend": {
            "name": fake_backend.name,
            "error": relerr(fake_backend_energy),
            "energy": fake_backend_energy,
            "bitstring_prop": fake_backend_bitstring_prop,
            "n_shots": n_shots,
        },
        "statevector": {
            "error": relerr(statevector_energy),
            "energy": statevector_energy,
        },
        "system": {
            "ground_energy": ground_energy,
            "n_qubits": circuit.num_qubits,
            "name": dirname,
        },
    }
    if run_on_ibmq:
        jobj["real_backend"] = {
            "name": real_backend.name,
            "error": relerr(real_energy),
            "energy": real_energy,
            "bitstring_prop": real_bitstring_prop,
            "n_shots": n_shots,
        }

    edm.var_dump(**jobj)


if __name__ == "__main__":
    dry_run = False
    edm = ExperimentDataManager(
        "nlce_hardware_run_small_system_hardware_run", dry_run=dry_run
    )
    run_on_ibmq = True
    get_expectations_from_sources(edm, "tfim_8q_4l_hva", run_on_ibmq=run_on_ibmq)
