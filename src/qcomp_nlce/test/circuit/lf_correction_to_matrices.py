from src.qcomp_nlce.utils.workflow_utils import pick_model
from src.qcomp_nlce.utils import (
    get_tfim_sweep_circuit,
    sweep_time_from_steps,
    baptize_circuit,
    HAM_NAME_VAR,circuit_to_matrix
)

def print_lf_cdiff():

    n_qubits = 5
    J = 0.5

    hl=0.1
    ladder = False
    n_steps = 20


    ham_j, ham_h = pick_model(n_qubits, J, hl, ladder)
    hamiltonian = ham_j + ham_h
    # the most important thing is to have enough steps!!!
    # the total time can be low-ish

    total_time = sweep_time_from_steps(ham_h, hamiltonian, n_steps)
    circuit = get_tfim_sweep_circuit(
        ham_h,
        ham_j + ham_h,
        total_time=total_time,
        n_steps=n_steps,
    )
    

    umat = circuit.to_matrix()