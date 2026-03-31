from src.qcomp_nlce.utils.backends import depolarizing_backend
from src.qcomp_nlce.utils.sweep import sweep_time_from_steps, get_tfim_sweep_circuit
from src.qcomp_nlce.utils.hamiltonian import TFIM
from qiskit_ibm_runtime import EstimatorV2

if __name__ == "__main__":
    ham_j, ham_h = TFIM(n_qubits=5, h=1, J=0.8, split_parts=True)
    n_steps = 10
    hamiltonian = ham_h + ham_j
    total_time = sweep_time_from_steps(ham_h, hamiltonian, n_steps)
    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=total_time,
        n_steps=n_steps,
    )

    backend = depolarizing_backend(0.001, 0.01, "density_matrix")

    estimator = EstimatorV2(backend=backend)

    job = estimator.run([(circuit, hamiltonian)], precision=1e-6)
    res = job.result()
    res[0].data.evs[0]
