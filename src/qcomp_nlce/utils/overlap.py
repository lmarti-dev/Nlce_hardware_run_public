from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit_ibm_runtime import IBMBackend
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from qiskit_aqt_provider.aqt_resource import OfflineSimulatorResource
from qcomp_nlce.utils.measure import (
    sample_counts_circuit,
    sample_circuit,
    estimate_circuit,
    get_counts_from_job,
)
from qcomp_nlce.utils.hamiltonian import index_bits, projector_ham, trace_dist
from qcomp_nlce.utils.circuit import (
    to_controlled_gate,
    apply_parameters,
    circuit_instruction_to_circuit,
    prep_circ_superpos,
    from_bitstring,
    circuit_to_matrix,
)

from qcomp_nlce.utils.save import (
    tmp_entry_exists,
    add_tmp_entry_and_save,
    load_tmp_item,
    rebuild_matrices_from_tmp,
    tmp_filename,
)

from qcomp_nlce.utils.hardware import is_hardware_run, BACKEND_ALIAS

from qcomp_nlce.utils.stats import sum_std

from qiskit.primitives import EstimatorResult, PrimitiveResult

from typing import Union, Tuple


def custom_idx_groups_of_n(total_len: int, group_size: int = 2):
    return [
        list(range(x, min(x + group_size, total_len)))
        for x in range(0, total_len, group_size)
    ]


def compare_with_true(true_vals: np.ndarray, vals: np.ndarray, name: str) -> float:
    td = -1
    if true_vals is not None:
        td = trace_dist(true_vals, vals)
        print(f"{name} -- {td:.5e}")
    else:
        print(np.round(vals, 3))
    return td


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


def get_overlaps_from_res(
    res: Union[EstimatorResult],
) -> Tuple[float, np.ndarray, np.ndarray]:
    # 1000 lines of code, and 999 of those are dedicated to babysit qiskit incompatibilities
    if isinstance(res, EstimatorResult):
        # this is a v1 primitive, so it depends on the order of your for loops :)
        # I'm going to be a goose farmer i stg
        n_obs = 4
        gs = []
        g_abs = []
        stds = []
        vals = res.values
        variances = [m["variance"] for m in res.metadata]
        for ind in range(0, len(vals), n_obs):
            gs.append(vals[ind + 1] + 1j * vals[ind + 2])
            stds.append(np.sqrt(variances[ind + 1]) + 1j * np.sqrt(variances[ind + 2]))
            g_abs.append(vals[ind] + 1j * vals[ind + 3])
    elif isinstance(res, PrimitiveResult):
        # gs = Re(g_l g*_l-1) + 1j * Im(g_l g*_l-1)
        gs = np.array([x.data.evs[1] + 1j * x.data.evs[2] for x in res])
        stds = np.array([x.data.stds[1] + 1j * x.data.stds[2] for x in res])

        # I-Z = 1/2 (g**2_l-1 + g**2_l) - 1/2 (g**2_l-1 - g**2_l) = g**2_l
        g_abs = np.array([np.sqrt(np.abs(x.data.evs[0] - x.data.evs[3])) for x in res])
        # g_abs is the absolute value of g_l = <psi|U_1...U_l-1 U_l|psi>
        # g_m1 computes g_l-1 so that we get twice the results per shot.
        # I+Z = 1/2 (g**2_l-1 + g**2_l) +  1/2 (g**2_l-1 - g**2_l) = g**2_l-1
        g_m1 = np.array([np.sqrt(np.abs(x.data.evs[0] + x.data.evs[3])) for x in res])

        better_g_abs = gs[:-1] + g_m1[1:] / 2
        # print(f"({y,x} {k} min|g_s|**2={np.min(g_abs**2)}", end="\r")

    if np.any(np.isclose(g_abs, 0)):
        raise ValueError(f"g_abs has null values, algorithm will fail: g_abs={g_abs}")

    if len(gs) == 1:
        final_g = gs[0]
    elif len(gs) % 2 == 0:
        final_g = np.prod(gs[1::2]) / np.prod(np.conj(gs[0::2]))
    elif len(gs) % 2 == 1:
        final_g = np.prod(gs[0::2]) / np.prod(np.conj(gs[1::2]))

    # first order error prop

    raise NotImplementedError(
        "This can be implemented via a UT, but I first need to get it to work"
    )
    final_std = ...

    return final_g, gs, g_abs, final_std


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
        jUj, std = iterative_hadamard_test_projecto_estimo(
            circuit, parameters, n_shots, x, y, backend, custom_idx, handcraft
        )
        return (jUj, 0, std)

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

        final_g, _, _, std = get_overlaps_from_res(jobs.result())
        d[k] = (final_g, std)

    alpha = 1j * (d["+i"][0] - (d["+"][0] + d["-"][0]) / 2)
    beta = (d["+"][0] - d["-"][0]) / 2

    jUk = beta - alpha
    kUj = alpha + beta

    # I wish I lived in a world without propagation of errors
    std = np.sqrt(0.5 * (d["+"][1] ** 2) + 0.5 * (d["-"][1] ** 2) + d["+i"] ** 2)

    return jUk, kUj, std


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

    final_g, _, _, std = get_overlaps_from_res(jobs.result())

    return final_g, std


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


def get_n0n1_hadam_counts(counts: dict, keys: dict[str, str] = {"p0": "0", "p1": "1"}):
    # make sure that keys are str and not int!
    out_counts = {}
    for k in counts:
        out_counts[str(k)] = counts[k]
    counts = out_counts

    if keys["p0"] not in counts.keys():
        n0 = 0
    else:
        n0 = counts[keys["p0"]]
    if keys["p1"] not in counts.keys():
        n1 = 0
    else:
        n1 = counts[keys["p1"]]
    return n0, n1


def hadam_diff(
    counts: dict, n_shots: int = None, keys: dict[str, str] = {"p0": "0", "p1": "1"}
) -> float:

    n0, n1 = get_n0n1_hadam_counts(counts, keys)
    if n_shots is None:
        n_shots = n0 + n1
    return (n0 - n1) / n_shots


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


def get_hadam_estimator(res: EstimatorResult):
    # qiskit developers competing with python developers to see who can make the most breaking changes with a version chanege
    # anyway, here's wonderwall

    try:
        return (
            res[0].data.evs[0] + 1j * res[0].data.evs[1],
            res[0].data.stds[0] + 1j * res[0].data.stds[1],
        )

    except Exception:
        return res.values[0] + 1j * res.values[1], np.sqrt(
            np.sqrt(res.metadata[0]["variance"])
            + 1j * +np.sqrt(res.metadata[1]["variance"])
        )


def full_hadamard_std(n_1: int, n_0: int) -> float:
    N = n_1 + n_0
    mu = n_1 / N
    sigma = np.sqrt((n_1 * ((mu - 1) ** 2)) / N)
    return sigma


def measure_matrix_element(
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
    which_hadamard_test="iterative",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
) -> Union[complex, tuple[complex, complex]]:

    # This computes <x|U|y>

    if which_hadamard_test == "projecto-estimo":

        ol, std = iterative_hadamard_test_projecto_estimo(
            circuit,
            parameters,
            n_shots,
            x,
            y,
            backend,
            custom_idx=custom_idx,
            handcraft=handcraft,
        )
        return ol, std

    elif which_hadamard_test == "off-diag":

        jUk, kUj, std = iterative_hadamard_test_off_diag(
            circuit,
            parameters,
            n_shots,
            x,
            y,
            backend,
            custom_idx=custom_idx,
            handcraft=handcraft,
        )
        return jUk, kUj, std

    elif which_hadamard_test == "full":
        circ_out = circuit_with_init(circuit, x, y)
        re_circ = hadamard_test_circuit(
            circ_out, parameters, real_part=True, handcraft=handcraft
        )
        im_circ = hadamard_test_circuit(
            circ_out, parameters, real_part=False, handcraft=handcraft
        )

        vals = []
        stds = []
        for tmp_circ in (re_circ, im_circ):
            counts = sample_counts_circuit(tmp_circ, parameters, backend, n_shots)
            v = hadam_diff(counts, n_shots)
            std = full_hadamard_std(*get_n0n1_hadam_counts(counts))
            stds.append(std)
            vals.append(v)

        ol = vals[0] + 1j * vals[1]
        std = stds[0] + 1j * stds[1]
        return ol, std
    elif which_hadamard_test == "estimator":
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

    elif which_hadamard_test == "iterative":
        raise ValueError("This area is out of service chief")
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
        raise ValueError(
            f"Expected full, estimator, or iterative, got {which_hadamard_test}"
        )


def reorder_matrix_elems(vals: np.ndarray) -> np.ndarray:
    # somehow indices are all messy
    out = np.zeros_like(vals)
    diag = np.diag(vals)

    out = vals[::-1, ::-1]

    out[range(out.shape[0]), range(out.shape[0])] = diag

    return out


def measure_matrix_elements(
    circuit: QuantumCircuit,
    comp_states: list[str],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    which_backend: str,
    n_shots: int,
    which_hadamard_test: str = "full",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
) -> np.ndarray:

    vals = np.zeros((len(comp_states), len(comp_states)), dtype=complex)
    stds = np.zeros((len(comp_states), len(comp_states)), dtype=complex)

    on_hardware = is_hardware_run(which_backend)
    filename = f"measure_matrix_elements_{tmp_filename(circuit,which_backend)}"

    if on_hardware:
        try:
            computed_values = load_tmp_item(filename)
        except FileNotFoundError:
            print("Starting from scratch (hadamard test)")
            computed_values = []

    if which_hadamard_test != "off-diag":
        for ind_x, x in enumerate(comp_states):
            for ind_y, y in enumerate(comp_states):
                if on_hardware and tmp_entry_exists(computed_values, ind_x, ind_y):
                    pass
                    print(f"Already computed {ind_x}, {ind_y}")
                else:
                    jUk, std = measure_matrix_element(
                        circuit,
                        x,
                        y,
                        parameters,
                        backend,
                        n_shots,
                        which_hadamard_test,
                        custom_idx=custom_idx,
                        handcraft=handcraft,
                    )
                    vals[ind_x, ind_y] = jUk
                    stds[ind_x, ind_y] = std
                    if on_hardware:
                        computed_values = add_tmp_entry_and_save(
                            filename, computed_values, jUk, std, ind_x, ind_y
                        )
                    print(f"{ind_x} - {ind_y}: {vals[ind_x,ind_y]:.4f}")
    elif which_hadamard_test == "off-diag":
        for ind_x, x in enumerate(comp_states):
            for ind_y, y in enumerate(comp_states[ind_x:]):
                ind_y = ind_x + ind_y
                if on_hardware and tmp_entry_exists(computed_values, ind_x, ind_y):
                    pass
                    print(f"Already computed {ind_x}, {ind_y}")
                else:
                    jUk, kUj, std = measure_matrix_element(
                        circuit,
                        x,
                        y,
                        parameters,
                        backend,
                        n_shots,
                        which_hadamard_test,
                        custom_idx=custom_idx,
                        handcraft=handcraft,
                    )
                    vals[ind_x, ind_y] = jUk
                    vals[ind_y, ind_x] = kUj
                    stds[ind_x, ind_y] = std
                    stds[ind_y, ind_x] = std
                    if on_hardware:
                        computed_values = add_tmp_entry_and_save(
                            filename, computed_values, jUk, std, ind_x, ind_y
                        )
                        computed_values = add_tmp_entry_and_save(
                            filename, computed_values, kUj, std, ind_y, ind_x
                        )
                    print(f"{ind_x} - {ind_y}: {jUk:.3f}, {kUj:.3f}")
    if on_hardware:
        vals, stds = rebuild_matrices_from_tmp(computed_values, len(comp_states))
    # somehow indices are all messy
    vals = reorder_matrix_elems(vals)
    stds = reorder_matrix_elems(stds)
    return vals, stds


def measure_hamiltonian_element(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
) -> tuple[complex, complex]:
    if x == y:
        circ_out = QuantumCircuit(circuit.num_qubits)
        x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
        for x_ind in x_inds:
            circ_out.x(x_ind)
        circ_out.compose(circuit, inplace=True)
        job = estimate_circuit(circ_out, [hamiltonian], parameters, backend, n_shots)

        res = job.result()
        if isinstance(res, EstimatorResult):
            return res.values[0], np.sqrt(res.metadata[0]["variance"])
        else:
            return res[0].data.evs, res[0].data.stds

    else:
        d = ["+", "-", "+i", "-i"]
        circs = []
        for k in d:
            circ_p = prep_circ_superpos(x, y, k)
            circ_p.compose(circuit, inplace=True)
            circs.append(circ_p)

        job = estimate_circuit(circs, [hamiltonian], parameters, backend, n_shots)

        res = job.result()

        if isinstance(res, EstimatorResult):
            val = (res.values[0] - res.values[1]) / 2 + 1j * (
                res.values[2] - res.values[3]
            ) / 2
            stds = [np.sqrt(res.metadata[k]["variance"]) for k in range(4)]
            std = sum_std(stds[0:2]) + 1j * sum_std(stds[2:4])
            return val, std

        else:
            val = (res[0].data.evs - res[1].data.evs) / 2 + 1j * (
                res[2].data.evs - res[3].data.evs
            ) / 2
            stds = [res[k].data.stds for k in range(4)]
            std = sum_std(stds[0:2]) + 1j * sum_std(stds[2:4])
            return val, std


def measure_hamiltonian_elements(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    states: list[str],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    which_backend: str,
    n_shots: int,
):

    vals = np.zeros((len(states), len(states)), dtype=complex)
    stds = np.zeros((len(states), len(states)), dtype=complex)

    on_hardware = is_hardware_run(which_backend)
    filename = f"measure_hamiltonian_elements_{tmp_filename(circuit,which_backend)}"

    if on_hardware:
        try:
            computed_values = load_tmp_item(filename)
        except FileNotFoundError:
            print("Starting from scratch (expectations)")
            computed_values = []

    for ind_x, x in enumerate(states):
        for ind_y, y in enumerate(states[ind_x:]):
            ind_y = ind_x + ind_y
            if on_hardware and tmp_entry_exists(computed_values, ind_x, ind_y):
                pass
                print(f"Already computed {ind_x}, {ind_y}")
            else:
                val, std = measure_hamiltonian_element(
                    hamiltonian, circuit, x, y, parameters, backend, n_shots
                )
                vals[ind_x, ind_y] = val.conj()
                vals[ind_y, ind_x] = val
                stds[ind_x, ind_y] = std
                stds[ind_y, ind_x] = std
                if on_hardware:
                    computed_values = add_tmp_entry_and_save(
                        filename, computed_values, val.conj(), std, ind_x, ind_y
                    )
                    computed_values = add_tmp_entry_and_save(
                        filename, computed_values, val, std, ind_y, ind_x
                    )
    # this is correct!
    if on_hardware:
        vals, stds = rebuild_matrices_from_tmp(computed_values, len(states))
    vals = reorder_matrix_elems(vals)[::-1, ::-1]
    stds = reorder_matrix_elems(stds)[::-1, ::-1]
    return vals, stds
