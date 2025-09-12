from qiskit import QuantumCircuit, ClassicalRegister
import numpy as np
from qiskit_ibm_runtime import IBMBackend
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from qcomp_nlce.utils.measure import sample_counts_circuit
from qcomp_nlce.utils.hamiltonian import index_bits
from qcomp_nlce.utils.circuit import to_controlled_gate, apply_parameters
from qiskit.circuit.gate import Gate

BACKEND_ALIAS = str | FakeBackendV2 | IBMBackend


# computes the real part of overlap between psi and circuit(psi)
def hadamard_test_circuit(
    controlled_gate: Gate, real_part: bool = True
) -> QuantumCircuit:
    n_qubits = controlled_gate.num_qubits
    hadam_circuit = QuantumCircuit(n_qubits)
    if not real_part:
        hadam_circuit.s(0)
    hadam_circuit.h(0)
    hadam_circuit.append(controlled_gate, range(n_qubits))
    hadam_circuit.h(0)
    if not real_part:
        hadam_circuit.s(0)

    return hadam_circuit


def swap_test_circuit(
    circuit: QuantumCircuit, init_circuit: QuantumCircuit = None
) -> QuantumCircuit:
    n_qubits = circuit.num_qubits
    swap_circuit = QuantumCircuit(circuit.num_qubits * 2 + 1)
    swap_circuit.h(0)
    if init_circuit is not None:
        swap_circuit = swap_circuit.compose(init_circuit, range(1, n_qubits + 1))
        swap_circuit = swap_circuit.compose(
            init_circuit, range(n_qubits + 1, swap_circuit.num_qubits)
        )
    swap_circuit = swap_circuit.compose(circuit, range(1, n_qubits + 1))
    for x in range(1, n_qubits + 1):
        swap_circuit.cswap(0, x, x + n_qubits)

    swap_circuit.h(0)

    return swap_circuit


def swap_test(
    circuit: QuantumCircuit,
    init_circuit: QuantumCircuit = None,
    parameters: np.ndarray = None,
    backend: BACKEND_ALIAS = "statevector",
    n_shots: int = 1000,
):
    circuit = apply_parameters(circuit, parameters)
    swap_circuit = swap_test_circuit(circuit, init_circuit)
    swap_circuit.add_register(ClassicalRegister(1, name="meas"))
    swap_circuit.measure(0, 0)
    counts = sample_counts_circuit(swap_circuit, backend, n_shots)
    p0 = counts["0"] / n_shots
    return 2 * p0 - 1


def hadamard_test(
    circuit: QuantumCircuit,
    parameters: np.ndarray = None,
    backend: BACKEND_ALIAS = "statevector",
    n_shots: int = 1000,
    real_part: bool = True,
):

    controlled_gate = to_controlled_gate(circuit, parameters)
    hadam_circuit = hadamard_test_circuit(controlled_gate, real_part)
    hadam_circuit.add_register(ClassicalRegister(1, name="meas"))
    hadam_circuit.measure(0, 0)
    counts = sample_counts_circuit(hadam_circuit, backend, n_shots)
    p0 = counts["0"] / n_shots
    p1 = counts["1"] / n_shots
    return p0 - p1


def measure_matrix_element(
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
) -> complex:
    x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
    y_inds = index_bits(y, circuit.num_qubits, right_to_left=True)

    # initial state
    circ_in = QuantumCircuit(circuit.num_qubits)
    for ind_x in x_inds:
        circ_in.x(ind_x)
    circ_in = circ_in.compose(circuit)
    # X.compose(Y) append the Y after X
    circ_out = QuantumCircuit(circuit.num_qubits)
    # x is left bra and y right ket
    # if add x to balance both so that it becomes a diagonal value
    for ind_x, ind_y in zip(x_inds, y_inds):
        if ind_x != ind_y:
            circ_out.x(ind_x)
    circ_out = circ_in.compose(circ_out, range(circuit.num_qubits))
    re_part = hadamard_test(circ_out, parameters, backend, n_shots, real_part=True)
    im_part = hadamard_test(circ_out, parameters, backend, n_shots, real_part=False)

    return re_part + 1j * im_part


def measure_matrix_elements(
    circuit: QuantumCircuit,
    comp_states: list[str],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
) -> np.ndarray:
    out = np.zeros((len(comp_states), len(comp_states)), dtype=complex)
    for ind_x, x in enumerate(comp_states):
        for ind_y, y in enumerate(comp_states):
            out[ind_x, ind_y] = measure_matrix_element(
                circuit, x, y, parameters, backend, n_shots
            )
    return out
