import numpy as np
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2

from qcomp_nlce.utils import backend_estimate, estimate_observables, load_sim
import io
from pathlib import Path


def get_expectations_from_sources(dirname: str, n_shots: int = 10000):
    fake_backend = FakeJakartaV2()
    hamiltonian, circuit, weights, ground_energy = load_sim(dirname)

    job = backend_estimate(
        backend=fake_backend,
        hamiltonian=hamiltonian,
        parameters=weights[-1],
        circuit=circuit,
        n_shots=n_shots,
    )
    fake_backend_energy = job.result()[0].data.evs

    statevector_energy = estimate_observables(
        circuit=circuit, observables=hamiltonian, param_sweep=weights[-1]
    ).data.evs

    token_fpath = r"C:\Users\Moi4\Desktop\current\FAU\phd\code\nlce_hardware_run\Nlce_hardware_run\token.dat"
    token = io.open(
        token_fpath,
        "r",
        encoding="utf8",
    ).read()
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    real_backend = service.least_busy(
        simulator=False, operational=True, min_num_qubits=circuit.num_qubits
    )

    real_job = backend_estimate(
        backend=real_backend,
        hamiltonian=hamiltonian,
        parameters=weights[-1],
        circuit=circuit,
        n_shots=n_shots,
    )

    hardware_energy = real_job.result()[0].data.evs

    print(
        f"Statevector {np.abs(statevector_energy - ground_energy)/np.abs(ground_energy)}"
    )

    print(
        f"FakeBackend: {np.abs(fake_backend_energy - ground_energy)/np.abs(ground_energy)}"
    )

    print(
        f"RealBackend ({real_backend.name}) {np.abs(hardware_energy - ground_energy)/np.abs(ground_energy)}"
    )


if __name__ == "__main__":
    get_expectations_from_sources("4q_tfim_2_layers_hva")
    get_expectations_from_sources("3q_tfim_1_layers_hva")
