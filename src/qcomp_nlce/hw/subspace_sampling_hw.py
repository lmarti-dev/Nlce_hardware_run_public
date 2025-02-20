import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

from qcomp_nlce.utils import (
    backend_sample,
    subspace_diagonalize,
    get_counts_from_job,
    get_unique_bitstrings,
    estimate_observables,
    add_meas_layer,
    load_optimized_vqe,
    load_sweep,
    get_optimal_parameters,
    get_appropriate_fake_backend,
)
from data_manager import ExperimentDataManager

from qcomp_nlce.utils.hardware import get_token
from pathlib import Path
import os


def get_expectations_from_sources(
    edm: ExperimentDataManager,
    dirname: str,
    n_shots: int = 1e3,
    run_on_ibmq: bool = False,
    override_params: bool = False,
    which: str = "vqe",
):

    if which == "vqe":
        hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(dirname)
        if override_params:
            parameters = get_optimal_parameters(dirname)
        else:
            parameters = weights[-1]
    elif which == "sweep":
        hamiltonian, circuit, ground_energy = load_sweep(dirname)
        weights = None
        parameters = None
    fake_backend = get_appropriate_fake_backend(circuit.num_qubits)

    circuit_to_sample = add_meas_layer(circuit)

    def relerr(e):
        return np.abs(e - ground_energy) / np.abs(ground_energy)

    fake_job = backend_sample(
        backend=fake_backend,
        circuit=circuit_to_sample,
        parameters=parameters,
        n_shots=n_shots,
    )
    fake_counts = get_counts_from_job(fake_job)
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
            bitstrings=real_unique_bitstring, ham=hamiltonian
        )

        print(
            f"RealBackend ({real_backend.name}) {relerr(real_energy)} sub. mat.: {len(real_unique_bitstring)} x {len(real_unique_bitstring)}: {real_bitstring_prop*100}%"
        )

    print(f"Statevector {relerr(statevector_energy)}")

    print(
        f"FakeBackend: {relerr(fake_backend_energy)} sub. mat.: {len(fake_unique_bitstring)} x {len(fake_unique_bitstring)}: {fake_backend_bitstring_prop*100}"
    )

    var_dict = {
        "fake_backend": {
            "name": fake_backend.name,
            "error": relerr(fake_backend_energy),
            "energy": fake_backend_energy,
            "unique_bitstrings": fake_unique_bitstring,
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
        var_dict["real_backend"] = {
            "name": real_backend.name,
            "error": relerr(real_energy),
            "energy": real_energy,
            "unique_bitstrings": real_unique_bitstring,
            "bitstring_prop": real_bitstring_prop,
            "n_shots": n_shots,
        }

    edm.var_dump(**var_dict)

    jobj = {"fake_result": fake_job.result()}
    if run_on_ibmq:
        jobj["real_result"] = real_job.result()
    edm.save_dict(jobj)


def run_all_vqes(edm: ExperimentDataManager, run_on_ibmq: bool):
    HOME = Path(__file__).parent
    dirnames = os.listdir(Path(HOME, "../res_sim/vqe"))
    for x in dirnames:
        get_expectations_from_sources(edm, x, run_on_ibmq=run_on_ibmq, which="vqe")
        edm.new_run()


def run_all_sweeps(edm: ExperimentDataManager, run_on_ibmq: bool):
    HOME = Path(__file__).parent
    dirnames = os.listdir(Path(HOME, "../res_sim/sweep"))
    for x in dirnames:
        print(f"Loading {x}")
        get_expectations_from_sources(edm, x, run_on_ibmq=run_on_ibmq, which="sweep")
        edm.new_run()


if __name__ == "__main__":
    dry_run = False
    which = "sweep"
    edm = ExperimentDataManager(f"nlce_hardware_run_{which}", dry_run=dry_run)
    run_on_ibmq = False
    if which == "sweep":
        run_all_sweeps(edm, run_on_ibmq)
    elif which == "vqe":
        run_all_vqes(edm, run_on_ibmq)
