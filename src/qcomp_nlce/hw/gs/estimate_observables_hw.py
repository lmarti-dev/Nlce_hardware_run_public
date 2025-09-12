import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

from qcomp_nlce.utils import (
    backend_estimate,
    estimate_observables,
    load_optimized_vqe,
    get_appropriate_fake_backend,
)
from data_manager import ExperimentDataManager

from qcomp_nlce.utils.hardware import get_token


def get_expectations_from_sources(
    edm: ExperimentDataManager,
    dirname: str,
    n_shots: int = 1e4,
    run_on_ibmq: bool = False,
    provider="AQT",
):
    backend_name = "AQT" if provider == "LRZ" else "IBM"
    hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(dirname)
    fake_backend = get_appropriate_fake_backend(circuit.num_qubits, backend_name)

    def relerr(e):
        return np.abs(e - ground_energy) / np.abs(ground_energy)

    parameters = weights[-1]

    job = backend_estimate(
        backend=fake_backend,
        hamiltonian=hamiltonian,
        parameters=parameters,
        circuit=circuit,
        n_shots=n_shots,
    )
    fake_backend_energy = job.result()[0].data.evs

    statevector_energy = estimate_observables(
        circuit=circuit, observables=hamiltonian, param_sweep=parameters
    ).data.evs

    if run_on_ibmq:
        token = get_token()
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        real_backend = service.least_busy(
            simulator=False, operational=True, min_num_qubits=circuit.num_qubits
        )

        real_job = backend_estimate(
            backend=real_backend,
            hamiltonian=hamiltonian,
            parameters=parameters,
            circuit=circuit,
            n_shots=n_shots,
        )

        hardware_energy = real_job.result()[0].data.evs
        print(f"RealBackend ({real_backend.name}) {relerr(hardware_energy)}")

    print(f"Statevector {relerr(statevector_energy)}")

    print(f"FakeBackend: {relerr(fake_backend_energy)}")

    jobj = {
        "fake_backend": {
            "name": fake_backend.name,
            "error": relerr(fake_backend_energy),
            "energy": fake_backend_energy,
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
            "error": relerr(hardware_energy),
            "energy": hardware_energy,
        }

    edm.var_dump(**jobj)


if __name__ == "__main__":
    dry_run = True
    edm = ExperimentDataManager(
        "nlce_hardware_run_small_system_hardware_run", dry_run=dry_run
    )
    run_on_ibmq = False
    get_expectations_from_sources(edm, "tfim_8q_4l_hva", run_on_ibmq=run_on_ibmq)
