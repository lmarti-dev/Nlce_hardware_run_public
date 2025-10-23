from qiskit import QuantumCircuit, ClassicalRegister
import numpy as np
from qiskit_ibm_runtime import IBMBackend
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from qcomp_nlce.utils.measure import (
    sample_counts_circuit,
    sample_circuit,
    get_counts_from_job,
)
from qcomp_nlce.utils.hamiltonian import index_bits
from qcomp_nlce.utils.circuit import (
    to_controlled_gate,
    apply_parameters,
    circuit_instruction_to_circuit,
)
from qiskit.circuit.gate import Gate

BACKEND_ALIAS = str | FakeBackendV2 | IBMBackend


def iterative_hadamard_test_circuits(
    circuit: QuantumCircuit, parameters: np.ndarray, real_part: bool = True
) -> list[QuantumCircuit]:
    circuit = apply_parameters(circuit, parameters)
    circuits = []
    for x in range(0, len(circuit)):
        hadam_circuit = QuantumCircuit(circuit.num_qubits + 1)
        hadam_circuit.h(0)
        if not real_part:
            hadam_circuit.sdg(0)

        circuit_stueckchen = circuit[:x]
        if len(circuit_stueckchen):
            c_temp = circuit_instruction_to_circuit(circuit_stueckchen)
            hadam_circuit.append(c_temp, range(1, hadam_circuit.num_qubits))

        controlled_gate = to_controlled_gate(
            circuit_instruction_to_circuit(circuit[x]), None
        )
        hadam_circuit.append(controlled_gate, range(circuit.num_qubits + 1))
        hadam_circuit.h(0)
        hadam_circuit.add_register(ClassicalRegister(1, name="meas"))
        hadam_circuit.measure(0, 0)

        circuits.append(hadam_circuit)
    return circuits


def iterative_hadamard_test(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
    backend: BACKEND_ALIAS = "statevector",
) -> float:
    circuits_re = iterative_hadamard_test_circuits(
        circuit=circuit, parameters=parameters, real_part=True
    )
    circuits_im = iterative_hadamard_test_circuits(
        circuit=circuit, parameters=parameters, real_part=False
    )
    circuits = [*circuits_re, *circuits_im]
    half_len = len(circuits_re)

    jobs = sample_circuit(circuits, parameters, backend, n_shots)
    gs = []
    for ind in range(0, half_len):
        re_counts = get_counts_from_job(jobs, ind)
        im_counts = get_counts_from_job(jobs, ind + half_len)
        gl = hadam_diff(re_counts, n_shots) + 1j * hadam_diff(im_counts, n_shots)
        gs.append(gl)
    up = gs[1::2]
    down = gs[0::2]
    return np.prod(up) / np.conj(np.prod(down))


# computes the real part of overlap between psi and circuit(psi)
def hadamard_test_circuit(
    circuit: QuantumCircuit, parameters: np.ndarray, real_part: bool = True
) -> QuantumCircuit:
    controlled_gate = to_controlled_gate(circuit, parameters)
    n_qubits = controlled_gate.num_qubits
    hadam_circuit = QuantumCircuit(n_qubits)
    hadam_circuit.h(0)
    if not real_part:
        hadam_circuit.sdg(0)
    hadam_circuit.append(controlled_gate, range(n_qubits))
    hadam_circuit.h(0)
    hadam_circuit.add_register(ClassicalRegister(1, name="meas"))
    hadam_circuit.measure(0, 0)
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
    hadam_circuit = hadamard_test_circuit(circuit, parameters, real_part)
    counts = sample_counts_circuit(hadam_circuit, backend, n_shots)
    return hadam_diff(counts, n_shots)


def measure_matrix_elements_merge_circuits(
    circuit: QuantumCircuit,
    comp_states: list[str],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
):
    circs: list[QuantumCircuit] = []
    for state_x in comp_states:
        for state_y in comp_states:
            re_circ, im_circ = measure_matrix_element_circuits(
                circuit=circuit, x=state_x, y=state_y, parameters=parameters
            )
            circs.append(re_circ)
            circs.append(im_circ)

    jobs = sample_circuit(circs, parameters, backend, n_shots)
    n_elems = len(comp_states) ** 2
    out = []
    for ind in range(0, 2 * n_elems - 1, 2):
        re_counts = get_counts_from_job(jobs, ind)
        im_counts = get_counts_from_job(jobs, ind + 1)
        v = hadam_diff(re_counts, n_shots) + 1j * hadam_diff(im_counts, n_shots)
        out.append(v)

    return np.reshape(out, shape=(len(comp_states), len(comp_states)))


def hadam_diff(counts: dict, n_shots) -> float:
    p0 = counts["0"] / n_shots
    p1 = counts["1"] / n_shots
    return p0 - p1


def circuit_with_init(
    circuit: QuantumCircuit,
    x: str,
    y: str,
) -> QuantumCircuit:
    x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
    y_inds = index_bits(y, circuit.num_qubits, right_to_left=True)
    circ_in = QuantumCircuit(circuit.num_qubits)
    for ind_y in y_inds:
        circ_in.x(ind_y)
    circ_in = circ_in.compose(circuit)
    # U1.compose(U2) gives U2 U1 |psi>
    circ_out = QuantumCircuit(circuit.num_qubits)
    # x is left bra and y right ket
    for ind_x in x_inds:
        circ_out.x(ind_x)
    circ_out = circ_in.compose(circ_out, range(circuit.num_qubits))
    return circ_out


def measure_matrix_element_circuits(
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
) -> tuple[QuantumCircuit, QuantumCircuit]:

    circ_out = circuit_with_init(circuit, x, y)
    re_circ = hadamard_test_circuit(circ_out, parameters, real_part=True)
    im_circ = hadamard_test_circuit(circ_out, parameters, real_part=False)
    return re_circ, im_circ


def measure_matrix_element(
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
    which="iterative",
) -> complex:

    if which == "full":
        re_circ, im_circ = measure_matrix_element_circuits(circuit, x, y, parameters)

        vals = []
        for tmp_circ in (re_circ, im_circ):
            counts = sample_counts_circuit(tmp_circ, parameters, backend, n_shots)
            v = hadam_diff(counts, n_shots)
            vals.append(v)

        return vals[0] + 1j * vals[1]
    elif which == "iterative":
        circ_out = circuit_with_init(circuit, x, y)
        return iterative_hadamard_test(circ_out, parameters, n_shots, backend)


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
                circuit, x, y, parameters, backend, n_shots, "iterative"
            )
            print(ind_x, ind_y, end="\r")
    return out
