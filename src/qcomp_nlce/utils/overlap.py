from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit_ibm_runtime import IBMBackend
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from qiskit.primitives.base.estimator_result import EstimatorResult
from qcomp_nlce.utils.measure import (
    sample_counts_circuit,
    sample_circuit,
    estimate_circuit,
    get_counts_from_job,
)
from qcomp_nlce.utils.hamiltonian import index_bits, projector_ham
from qcomp_nlce.utils.circuit import (
    to_controlled_gate,
    apply_parameters,
    circuit_instruction_to_circuit,
    prep_circ_superpos,
    from_bitstring,
    circuit_to_matrix,
)

BACKEND_ALIAS = str | FakeBackendV2 | IBMBackend


def custom_idx_groups_of_n(total_len: int, group_size: int = 2):
    return [
        list(range(x, min(x + group_size, total_len)))
        for x in range(0, total_len, group_size)
    ]


def iterative_hadamard_test_circuits(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    real_part: bool = True,
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
) -> list[QuantumCircuit]:
    circuit = apply_parameters(circuit, parameters)
    circuits = []
    if custom_idx is None:
        custom_idx = [[x] for x in range(0, len(circuit))]
    for x in range(len(custom_idx)):
        hadam_circuit = QuantumCircuit(circuit.num_qubits + 1)
        hadam_circuit.h(0)
        if False == real_part:
            hadam_circuit.sdg(0)
        stuekchen = [g for l in custom_idx[:x] for g in l]
        circuit_stueckchen = [circuit[c] for c in stuekchen]
        if len(circuit_stueckchen):
            c_temp = circuit_instruction_to_circuit(circuit_stueckchen)
            hadam_circuit.append(c_temp, range(1, hadam_circuit.num_qubits))

        for _gate in custom_idx[x]:
            controlled_gate = to_controlled_gate(circuit[_gate], None, handcraft)
            hadam_circuit.append(controlled_gate, range(circuit.num_qubits + 1))

        if real_part != "estimator":
            hadam_circuit.h(0)
            creg = ClassicalRegister(hadam_circuit.num_qubits, name="meas")
            hadam_circuit.add_register(creg)
            # we need to project the whole state back in ket(0)bra(0),
            # so we measure all the qubits of the state
            # because our initial state is ket(0) luckily
            for q in range(hadam_circuit.num_qubits):
                hadam_circuit.measure(q, q)

        circuits.append(hadam_circuit)
    return circuits


def iterative_hadamard_test_estimator(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
    backend: BACKEND_ALIAS = "statevector",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
):
    # !!! This only works for initial state ket(0)

    circuits = iterative_hadamard_test_circuits(
        circuit=circuit,
        parameters=parameters,
        real_part="estimator",
        custom_idx=custom_idx,
        handcraft=handcraft,
    )

    proj = projector_ham(circuit.num_qubits)
    hams = [
        proj ^ SparsePauliOp("I"),
        proj ^ SparsePauliOp("X"),
        proj ^ SparsePauliOp("Y"),
        proj ^ SparsePauliOp("Z"),
    ]

    jobs = estimate_circuit(
        backend=backend,
        observables=hams,
        circuit=circuits,
        parameters=None,
        n_shots=n_shots,
    )

    # gs = Re(g_l g*_l-1) + 1j * Im(g_l g*_l-1)
    gs = np.array([x.data.evs[1] + 1j * x.data.evs[2] for x in jobs.result()])

    # I-Z = 1/2 (g**2_l-1 + g**2_l) - 1/2 (g**2_l-1 - g**2_l) = g**2_l
    g_abs = np.array(
        [np.sqrt(np.abs(x.data.evs[0] - x.data.evs[3])) for x in jobs.result()]
    )
    # g_abs is the absolute value of g_l = <psi|U_1...U_l-1 U_l|psi>
    # g_m1 computes g_l-1 so that we get twice the results per shot.
    # g_m1[0] == 0.707 always if custom_inds = [[0],[1], etc]!
    # I+Z = 1/2 (g**2_l-1 + g**2_l) +  1/2 (g**2_l-1 - g**2_l) = g**2_l-1
    g_m1 = np.array(
        [np.sqrt(np.abs(x.data.evs[0] + x.data.evs[3])) for x in jobs.result()]
    )

    better_g_abs = gs[:-1] + g_m1[1:] / 2
    # print(f"|g_s|**2={g_abs**2}")

    if np.any(np.isclose(g_abs, 0)):
        raise ValueError(f"g_abs has null values, algorithm will fail: g_abs={g_abs}")

    if len(gs) == 1:
        final_g = gs[0]
    elif len(gs) % 2 == 0:
        final_g = np.prod(gs[1::2]) / np.prod(np.conj(gs[0::2]))
    elif len(gs) % 2 == 1:
        final_g = np.prod(gs[0::2]) / np.prod(np.conj(gs[1::2]))

    return final_g


def iterative_hadamard_test_off_diag(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
    x: str,
    y: str,
    backend: BACKEND_ALIAS = "statevector",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
):

    if x == y:
        return (
            iterative_hadamard_test_projecto_estimo(
                circuit, parameters, n_shots, x, y, backend, custom_idx, handcraft
            ),
            0,
        )

    # prepare
    # a = ket(x + iy)
    # b = ket(x - iy)
    # u = ket(x + y)
    # v = ket(x - y)

    # then
    # alpha = ia - i(u+v)/2
    # beta  = (u-v)/2
    # alpha+beta = 2 <k|U|j>
    # beta - alpha  = 2<j|U|k>

    d = {"+": 0 + 0j, "-": 0 + 0j, "+i": 0 + 0j, "-i": 0 + 0j}

    circuits = iterative_hadamard_test_circuits(
        circuit=circuit,
        parameters=parameters,
        real_part="estimator",
        custom_idx=custom_idx,
        handcraft=handcraft,
    )

    proj = projector_ham(circuit.num_qubits)
    hams = [
        proj ^ SparsePauliOp("I"),
        proj ^ SparsePauliOp("X"),
        proj ^ SparsePauliOp("Y"),
        proj ^ SparsePauliOp("Z"),
    ]
    for k in d:
        prep = prep_circ_superpos(x, y, k)

        out_circs = []

        for circ in circuits:
            prep_dag = prep.inverse()
            out = QuantumCircuit(circ.num_qubits)
            out.compose(prep, range(1, circ.num_qubits), inplace=True)
            out.compose(circ, range(circ.num_qubits), inplace=True)
            out.compose(prep_dag, range(1, circ.num_qubits), inplace=True)
            out_circs.append(out)

        jobs = estimate_circuit(
            backend=backend,
            observables=hams,
            circuit=out_circs,
            parameters=None,
            n_shots=n_shots,
        )

        # gs = Re(g_l g*_l-1) + 1j * Im(g_l g*_l-1)
        gs = np.array([x.data.evs[1] + 1j * x.data.evs[2] for x in jobs.result()])

        # I-Z = 1/2 (g**2_l-1 + g**2_l) - 1/2 (g**2_l-1 - g**2_l) = g**2_l
        g_abs = np.array(
            [np.sqrt(np.abs(x.data.evs[0] - x.data.evs[3])) for x in jobs.result()]
        )
        # g_abs is the absolute value of g_l = <psi|U_1...U_l-1 U_l|psi>
        # g_m1 computes g_l-1 so that we get twice the results per shot.
        # I+Z = 1/2 (g**2_l-1 + g**2_l) +  1/2 (g**2_l-1 - g**2_l) = g**2_l-1
        g_m1 = np.array(
            [np.sqrt(np.abs(x.data.evs[0] + x.data.evs[3])) for x in jobs.result()]
        )

        better_g_abs = gs[:-1] + g_m1[1:] / 2
        # print(f"({y,x} {k} min|g_s|**2={np.min(g_abs**2)}", end="\r")

        if np.any(np.isclose(g_abs, 0)):
            raise ValueError(
                f"g_abs has null values, algorithm will fail: g_abs={g_abs}"
            )

        if len(gs) == 1:
            final_g = gs[0]
        elif len(gs) % 2 == 0:
            final_g = np.prod(gs[1::2]) / np.prod(np.conj(gs[0::2]))
        elif len(gs) % 2 == 1:
            final_g = np.prod(gs[0::2]) / np.prod(np.conj(gs[1::2]))
        d[k] = final_g

    alpha = 1j * (d["+i"] - (d["+"] + d["-"]) / 2)
    beta = (d["+"] - d["-"]) / 2

    jUk = beta - alpha
    kUj = alpha + beta

    return jUk, kUj


def iterative_hadamard_test_projecto_estimo(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
    x: str,
    y: str,
    backend: BACKEND_ALIAS = "statevector",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
):

    # this one computes <x|U|y> without doing the x(j) in the iterative hadamard
    # which avoids some of the 0 overlap issue
    # but it's more annyoing
    # well, this was stupid. g0 = 0 so this method cannot in fact work for non-diagonal terms.
    # but it should work for diagonal terms though. Useless. Ack!

    circuits = iterative_hadamard_test_circuits(
        circuit=circuit,
        parameters=parameters,
        real_part="estimator",
        custom_idx=custom_idx,
        handcraft=handcraft,
    )

    ket = y
    bra = x
    init = QuantumCircuit(circuit.num_qubits + 1)
    ket_inds = index_bits(ket + "0", circuit.num_qubits + 1, right_to_left=True)
    for ket_ind in ket_inds:
        init.x(ket_ind)
    for idx in range(len(circuits)):
        circuits[idx] = init.compose(circuits[idx], inplace=False)

    proj = projector_ham(circuit.num_qubits, bitstring=bra)
    hams = [
        proj ^ SparsePauliOp("I"),
        proj ^ SparsePauliOp("X"),
        proj ^ SparsePauliOp("Y"),
        proj ^ SparsePauliOp("Z"),
    ]
    jobs = estimate_circuit(
        backend=backend,
        observables=hams,
        circuit=circuits,
        parameters=None,
        n_shots=n_shots,
    )

    # gs = Re(g_l g*_l-1) + 1j * Im(g_l g*_l-1)
    gs = np.array([x.data.evs[1] + 1j * x.data.evs[2] for x in jobs.result()])

    # I-Z = 1/2 (g**2_l-1 + g**2_l) - 1/2 (g**2_l-1 - g**2_l) = g**2_l
    g_abs = np.array(
        [np.sqrt(np.abs(x.data.evs[0] - x.data.evs[3])) for x in jobs.result()]
    )
    # g_abs is the absolute value of g_l = <psi|U_1...U_l-1 U_l|psi>
    # g_m1 computes g_l-1 so that we get twice the results per shot.
    # I+Z = 1/2 (g**2_l-1 + g**2_l) +  1/2 (g**2_l-1 - g**2_l) = g**2_l-1
    g_m1 = np.array(
        [np.sqrt(np.abs(x.data.evs[0] + x.data.evs[3])) for x in jobs.result()]
    )

    better_g_abs = gs[:-1] + g_m1[1:] / 2
    # print(f"({y,x} min|g_s|**2={np.min(g_abs**2)}", end="\r")

    if np.any(np.isclose(g_abs, 0)):
        raise ValueError(f"g_abs has null values, algorithm will fail: g_abs={g_abs}")

    if len(gs) == 1:
        final_g = gs[0]
    elif len(gs) % 2 == 0:
        final_g = np.prod(gs[1::2]) / np.prod(np.conj(gs[0::2]))
    elif len(gs) % 2 == 1:
        final_g = np.prod(gs[0::2]) / np.prod(np.conj(gs[1::2]))

    return final_g


def iterative_hadamard_test(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
    backend: BACKEND_ALIAS = "statevector",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
) -> float:

    # !!! This only works for initial state ket(0)
    circuits_re = iterative_hadamard_test_circuits(
        circuit=circuit,
        parameters=parameters,
        real_part=True,
        custom_idx=custom_idx,
        handcraft=handcraft,
    )
    circuits_im = iterative_hadamard_test_circuits(
        circuit=circuit,
        parameters=parameters,
        real_part=False,
        custom_idx=custom_idx,
        handcraft=handcraft,
    )
    circuits = [*circuits_re, *circuits_im]
    half_len = len(circuits_re)

    jobs = sample_circuit(circuits, parameters, backend, n_shots)
    gs = []
    # the first qubit is the last in the bitstring
    keys = {
        "p0": "0" * (circuit.num_qubits) + "0",
        "p1": "0" * (circuit.num_qubits) + "1",
    }
    for ind in range(0, half_len):
        re_counts = get_counts_from_job(jobs, ind)
        im_counts = get_counts_from_job(jobs, ind + half_len)
        gl = hadam_diff(re_counts, keys=keys) + 1j * hadam_diff(im_counts, keys=keys)
        gs.append(gl)

    gs = np.array(gs)

    if np.any(np.isclose(gs, 0.0)):
        # the zeros cancel out
        if np.sum(np.isclose(gs[::2], 0.0)) == np.sum(np.isclose(gs[1::2], 0.0)):
            gs[(np.isclose(gs, 0.0))] = 1.0
        else:
            # technically, there also could be a mistake somewhere,
            # but fuck this
            return 0.0

    if len(gs) == 1:
        final_g = gs[0]
    elif len(gs) % 2 == 0:
        final_g = np.prod(gs[1::2]) / np.prod(np.conj(gs[0::2]))
    elif len(gs) % 2 == 1:
        final_g = np.prod(gs[0::2]) / np.prod(np.conj(gs[1::2]))

    # assert np.isclose(np.abs(final_g), 1.0)
    return final_g


# computes the real part of overlap between psi and circuit(psi)
def hadamard_test_circuit(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    real_part: bool = True,
    handcraft: bool = True,
) -> QuantumCircuit:

    n_qubits = circuit.num_qubits + 1
    hadam_circuit = QuantumCircuit(n_qubits)
    hadam_circuit.h(0)
    if False == real_part:
        hadam_circuit.sdg(0)

    for ci in circuit:
        controlled_gate = to_controlled_gate(ci, parameters, handcraft)
        hadam_circuit.append(controlled_gate, range(n_qubits))
    if real_part != "estimator":
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
    handcraft: bool = True,
):

    circs: list[QuantumCircuit] = []
    for state_x in comp_states:
        for state_y in comp_states:
            re_circ, im_circ = measure_matrix_element_circuits(
                circuit=circuit,
                x=state_x,
                y=state_y,
                parameters=parameters,
                handcraft=handcraft,
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


def hadam_diff(
    counts: dict, n_shots: int = None, keys: dict[str, str] = {"p0": "0", "p1": "1"}
) -> float:

    if n_shots is None:
        # my brain
        n_shots = np.sum([counts[k] for k in keys.values() if k in counts.keys()])

    # 0 overlap
    if n_shots == 0:
        return 0
    if keys["p0"] not in counts.keys():
        return -1.0
    p0 = counts[keys["p0"]] / n_shots
    if keys["p1"] not in counts.keys():
        return 1.0
    p1 = counts[keys["p1"]] / n_shots
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
    handcraft: bool = True,
) -> tuple[QuantumCircuit, QuantumCircuit]:
    circ_out = circuit_with_init(circuit, x, y)
    re_circ = hadamard_test_circuit(
        circ_out, parameters, real_part=True, handcraft=handcraft
    )
    im_circ = hadamard_test_circuit(
        circ_out, parameters, real_part=False, handcraft=handcraft
    )
    return re_circ, im_circ


def get_hadam_estimator(res: EstimatorResult):
    # qiskit developers competing with python developers to see who can make the most breaking changes with a version chanege
    # anyway, here's wonderwall

    try:
        return res[0].data.evs[0] + 1j * res[0].data.evs[1]
    except Exception:
        return res.values[0] + 1j * res.values[1]


def measure_matrix_element(
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
    which="iterative",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
) -> complex:

    # This computes <x|U|y>

    if which == "projecto-estimo":

        ol = iterative_hadamard_test_projecto_estimo(
            circuit,
            parameters,
            n_shots,
            x,
            y,
            backend,
            custom_idx=custom_idx,
            handcraft=handcraft,
        )
        return ol

    elif which == "off-diag":

        jUk, kUj = iterative_hadamard_test_off_diag(
            circuit,
            parameters,
            n_shots,
            x,
            y,
            backend,
            custom_idx=custom_idx,
            handcraft=handcraft,
        )
        return jUk, kUj

    elif which == "full":
        circ_out = circuit_with_init(circuit, x, y)
        re_circ = hadamard_test_circuit(
            circ_out, parameters, real_part=True, handcraft=handcraft
        )
        im_circ = hadamard_test_circuit(
            circ_out, parameters, real_part=False, handcraft=handcraft
        )

        vals = []
        for tmp_circ in (re_circ, im_circ):
            counts = sample_counts_circuit(tmp_circ, parameters, backend, n_shots)
            v = hadam_diff(counts, n_shots)
            vals.append(v)
        ol = vals[0] + 1j * vals[1]
        return ol
    elif which == "estimator":
        circ_out = circuit_with_init(circuit, x, y)
        if custom_idx is None:
            custom_idx = custom_idx_groups_of_n(len([c for c in circ_out]), 1)
        n_qubits = circuit.num_qubits
        hadam_circ = hadamard_test_circuit(
            circ_out, parameters=parameters, real_part="estimator", handcraft=handcraft
        )
        hams = [
            SparsePauliOp(n_qubits * "I" + "X"),
            SparsePauliOp(n_qubits * "I" + "Y"),
        ]

        job = estimate_circuit(
            backend=backend,
            observables=hams,
            circuit=hadam_circ,
            parameters=None,
            n_shots=n_shots,
        )

        res = job.result()
        return get_hadam_estimator(res)

    elif which == "iterative":
        circ_out = circuit_with_init(circuit, x, y)

        ol = iterative_hadamard_test_estimator(
            circ_out,
            parameters,
            n_shots,
            backend,
            custom_idx=custom_idx,
            handcraft=handcraft,
        )
        return ol

    else:
        raise ValueError(f"Expected full, estimator, or iterative, got {which}")


def measure_matrix_elements(
    circuit: QuantumCircuit,
    comp_states: list[str],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
    which: str = "iterative",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
) -> np.ndarray:

    out = np.zeros((len(comp_states), len(comp_states)), dtype=complex)

    if which != "off-diag":
        for ind_x, x in enumerate(comp_states):
            for ind_y, y in enumerate(comp_states):
                out[ind_x, ind_y] = measure_matrix_element(
                    circuit,
                    x,
                    y,
                    parameters,
                    backend,
                    n_shots,
                    which,
                    custom_idx=custom_idx,
                    handcraft=handcraft,
                )
                # print(f"{ind_x} - {ind_y}: {out[ind_x,ind_y]:.4f}", end="\r")
    elif which == "off-diag":
        diag = []
        for ind_x, x in enumerate(comp_states):
            for ind_y, y in enumerate(comp_states[ind_x:]):
                ind_y = ind_x + ind_y
                jUk, kUj = measure_matrix_element(
                    circuit,
                    x,
                    y,
                    parameters,
                    backend,
                    n_shots,
                    which,
                    custom_idx=custom_idx,
                    handcraft=handcraft,
                )
                if ind_x == ind_y:
                    diag.append(jUk)
                else:
                    out[ind_x, ind_y] = jUk
                    out[ind_y, ind_x] = kUj
                # print(f"{ind_x} - {ind_y}: {jUk:.3f}, {kUj:.3f}", end="\r")
        # somehow indices are all messy
        out = out[::-1, ::-1] + np.diag(diag)
    return out


def trace_dist(u1: np.ndarray, u2: np.ndarray) -> float:
    diff = u1 - u2
    out = np.sqrt(diff.T.conj() @ diff)
    return 0.5 * np.real(np.trace(out))


def measure_hamiltonian_element(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
):
    if x == y:
        circ_out = QuantumCircuit(circuit.num_qubits)
        x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
        for x_ind in x_inds:
            circ_out.x(x_ind)
        circ_out.compose(circuit, inplace=True)
        job = estimate_circuit(circ_out, [hamiltonian], parameters, backend, n_shots)

        res = job.result()

        return res[0].data.evs

    else:
        d = ["+", "-", "+i", "-i"]
        circs = []
        for k in d:
            circ_p = prep_circ_superpos(x, y, k)
            circ_p.compose(circuit, inplace=True)
            circs.append(circ_p)

        job = estimate_circuit(circs, [hamiltonian], parameters, backend, n_shots)

        res = job.result()

        return (res[0].data.evs - res[1].data.evs) / 2 + 1j * (
            res[2].data.evs - res[3].data.evs
        ) / 2


def measure_hamiltonian_elements(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    states: list[str],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
):

    out = np.zeros((len(states), len(states)), dtype=complex)
    diag_items = np.zeros((len(states), len(states)), dtype=complex)
    for ind_x, x in enumerate(states):
        for ind_y, y in enumerate(states[ind_x:]):
            ind_y = ind_x + ind_y
            val = measure_hamiltonian_element(
                hamiltonian, circuit, x, y, parameters, backend, n_shots
            )
            if ind_x == ind_y:
                diag_items[ind_x, ind_y] = val
            else:
                out[ind_x, ind_y] = val.conj()
                out[ind_y, ind_x] = val
    out += diag_items[::-1, ::-1]
    return out


def circ_ham_expectation_values(
    hamiltonian: SparsePauliOp, circuit: QuantumCircuit, states: list[str]
) -> np.ndarray:
    # this returns the hamiltonian matrix for the 1 to n+1 first eigenstates
    # this is for the 1qp thing mostly
    out = np.zeros((len(states), len(states)), dtype=complex)
    inds = []
    for state in states:
        inds.append(from_bitstring(state, hamiltonian.num_qubits, right_to_left=True))

    umat = circuit_to_matrix(circuit)
    mat = umat.T.conj() @ hamiltonian.to_matrix() @ umat

    return mat[np.ix_(inds, inds)]
