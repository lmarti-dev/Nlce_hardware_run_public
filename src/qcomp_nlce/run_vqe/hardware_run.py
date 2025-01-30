import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2

from qcomp_nlce.utils import (
    backend_estimate,
    estimate_observables,
    load_sim,
    get_optimal_parameters,
)
import io
from pathlib import Path
from data_manager import ExperimentDataManager


def get_expectations_from_sources(
    edm: ExperimentDataManager,
    dirname: str,
    n_shots: int = 1e4,
    run_on_ibmq: bool = False,
    override_params: bool = False,
):

    fake_backend = FakeJakartaV2()
    hamiltonian, circuit, weights, ground_energy = load_sim(dirname)

    def relerr(e):
        return np.abs(e - ground_energy) / np.abs(ground_energy)

    if override_params:
        parameters = get_optimal_parameters(dirname)
    else:
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
        home = Path(__file__).parent
        token_fpath = Path(home, "../files/token.dat")
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
    dry_run = False
    edm = ExperimentDataManager(
        "nlce_hardware_run_small_system_hardware_run", dry_run=dry_run
    )
    run_on_ibmq = True
    get_expectations_from_sources(edm, "tfim_3q_2l_hva", run_on_ibmq=run_on_ibmq)
    edm.new_run()
    get_expectations_from_sources(edm, "tfim_4q_2l_hva", run_on_ibmq=run_on_ibmq)
