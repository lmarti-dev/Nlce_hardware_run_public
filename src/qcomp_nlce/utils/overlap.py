from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qcomp_nlce.utils.measure import (
    sample_counts_circuit,
    circuit_estimation_job,
    circuit_sampling_job,
    get_counts_from_res,
    std_from_shots,
    trace_out_estimation,
)
from qcomp_nlce.utils.hamiltonian import (
    index_bits,
    projector_ham,
    trace_dist,
    get_tfim_ratio,
    HAM_NAME_VAR,
)
from qcomp_nlce.utils.circuit import (
    to_controlled_gate,
    apply_parameters,
    circuit_instruction_to_circuit,
    prep_circ_superpos,
    circuit_to_matrix,
)

from qcomp_nlce.utils.save import (
    tmp_entry_exists,
    add_tmp_entry_and_save,
    load_tmp_item,
    tmp_get_item,
    tmp_dirname,
    TempListType,
    tmp_check_run_list,
    tmp_create_run_file,
    tmp_delete_run_file,
    tmp_load_run_list,
    tmp_save_run_file,
    init_run_split,
)

from qcomp_nlce.utils.unscented_transform import (
    unscented_transform,
    complex_to_double_real,
    vectorize_complex_func,
)

from qcomp_nlce.utils.hardware import (
    is_hardware_run,
    BACKEND_ALIAS,
    BACKEND_TYPES,
    manual_hamiltonian_circuits,
    circuits_to_sampler_payload,
)

from qcomp_nlce.utils.stats import sum_std, mean_std

from qiskit.primitives import EstimatorResult, PrimitiveResult

from typing import Union, Tuple, Literal, Dict, Any


OFFDIAG_SYMS = ["+", "-", "+i", "-i"]


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

        def prodfunc(gs: np.ndarray) -> np.ndarray:
            return np.prod(gs[1::2]) / np.prod(np.conj(gs[0::2]))

    elif len(gs) % 2 == 1:

        def prodfunc(gs: np.ndarray) -> np.ndarray:
            return np.prod(gs[0::2]) / np.prod(np.conj(gs[1::2]))

    final_g = prodfunc(gs)
    _, pmat = unscented_transform(
        vectorize_complex_func(prodfunc),
        complex_to_double_real(gs),
        complex_to_double_real(stds),
    )
    # first order error prop

    assert len(pmat) == 2
    final_std = pmat[0] + 1j * pmat[1]

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

        jobs = circuit_estimation_job(
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
    std = np.sqrt(0.5 * (d["+"][1] ** 2) + 0.5 * (d["-"][1] ** 2) + d["+i"][1] ** 2)

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
    jobs = circuit_estimation_job(
        backend=backend,
        observables=hams,
        circuit=circuits,
        parameters=None,
        n_shots=n_shots,
    )

    final_g, _, _, std = get_overlaps_from_res(jobs.result())

    return final_g, std


def build_stefan_circuit_estimator(
    circuit: QuantumCircuit, parameters: np.ndarray, x: str, y: str
) -> QuantumCircuit:
    n_qubits = circuit.num_qubits + 1
    hadam_circuit = QuantumCircuit(n_qubits)
    hadam_circuit.h(0)

    x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
    y_inds = index_bits(y, circuit.num_qubits, right_to_left=True)

    for y in y_inds:
        hadam_circuit.cx(0, y + 1)
    hadam_circuit.append(apply_parameters(circuit, parameters), range(1, n_qubits))
    for x in x_inds:

        hadam_circuit.cx(0, x + 1)

    return hadam_circuit


def build_stefan_circuits_sample(
    circuit: QuantumCircuit, parameters: np.ndarray, x: str, y: str
) -> list[QuantumCircuit]:
    hadam_circuits = []
    n_qubits = circuit.num_qubits + 1

    x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
    y_inds = index_bits(y, circuit.num_qubits, right_to_left=True)

    for t in ("real", "imag"):
        hadam_circuit = QuantumCircuit(n_qubits)
        hadam_circuit.h(0)
        if t == "imag":
            hadam_circuit.sdg(0)

        for y in y_inds:
            hadam_circuit.cx(0, y + 1)
        hadam_circuit.append(apply_parameters(circuit, parameters), range(1, n_qubits))
        for x in x_inds:
            hadam_circuit.cx(0, x + 1)

        hadam_circuit.h(0)
        hadam_circuit.measure_all()

        hadam_circuits.append(hadam_circuit)

    return hadam_circuits


def stefan_hadamard_test(
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
    x: str,
    y: str,
    which_backend: BACKEND_TYPES,
    backend: BACKEND_ALIAS = "statevector",
    counting_method: Literal["estimator", "samples"] = "estimator",
    n_reps: int = 10,
    reps_per_job: int = 5,
):

    reps_per_job = min(n_reps, reps_per_job)
    # this is Stefan Wolf's test, a bit modified though
    # because <0|u|0> in our case cancels perfectly, so
    # we ignore it from the start!

    # oUo = circuit_to_matrix(circuit)[0, 0]
    # print(f"Uncomputed overlap is {oUo:.5e}")

    if counting_method == "estimator":
        on_hardware = is_hardware_run(which_backend)
        _dirname = tmp_dirname(circuit, which_backend)

        n_runs = int(n_reps // reps_per_job)

        run_list = init_run_split(_dirname, on_hardware)

        remaining_runs = n_runs - len(run_list)

        for run_ind in range(remaining_runs):
            print(f"Run number {run_ind+1}/{remaining_runs}", end="\r")

            stef_circuit = build_stefan_circuit_estimator(circuit, parameters, x, y)
            proj = projector_ham(circuit.num_qubits, bitstring="0" * len(x))
            hams = [
                proj ^ SparsePauliOp("X"),
                proj ^ SparsePauliOp("Y"),
            ]
            jobs = circuit_estimation_job(
                backend=backend,
                observables=hams,
                circuit=[
                    stef_circuit,
                ]
                * reps_per_job,
                parameters=None,
                n_shots=n_shots,
            )
            # HARDWARE may fail HERE
            res = jobs.result()
            vals, stds = get_hadam_estimator_list(res, reps_per_job)
            run_list.append([vals, stds])
            if on_hardware:
                tmp_save_run_file(_dirname, run_list)
        ol, std = np.mean([v[0] for v in run_list]), mean_std(
            np.array([v[1] for v in run_list]).flatten()
        )
    elif counting_method == "samples":
        if n_reps > 1:
            raise NotImplementedError(
                "n_reps>1 not implemented for sampling Hadamard test"
            )
        n_qubits = circuit.num_qubits + 1
        stef_circuits = build_stefan_circuits_sample(circuit, parameters, x, y)
        jobs = circuit_sampling_job(stef_circuits, parameters, backend, n_shots)
        keys = {"p0": "0" * (len(x) + 1), "p1": "0" * (len(x)) + "1"}
        res = jobs.result()
        re, rstd = hadam_diff(get_counts_from_res(res, n_qubits, 0), n_shots, keys=keys)
        im, imstd = hadam_diff(
            get_counts_from_res(res, n_qubits, 1), n_shots, keys=keys
        )

        ol, std = re + 1j * im, rstd + 1j * imstd

    return ol, std


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
    val, std = hadam_diff(counts, n_shots)
    return val


def get_n0n1_hadam_counts(
    counts: dict[str, int], keys: dict[str, str] = {"p0": "0", "p1": "1"}
):
    # make sure that keys are str and not int!

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
    counts: dict, n_shots: int, keys: dict[str, str] = {"p0": "0", "p1": "1"}
) -> float:

    n0, n1 = get_n0n1_hadam_counts(counts, keys)
    if n0 == 0 and n1 == 0:
        return 0, 0

    std = std_from_shots(n1, n0, n_shots)

    return (n0 - n1) / n_shots, std


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


def get_hadam_estimator(
    res: EstimatorResult, n_reps: int = 1
) -> tuple[complex, complex]:
    # qiskit developers competing with python developers to see who can make the most breaking changes with a version chanege
    # anyway, here's wonderwall

    if n_reps == 1:
        try:
            return (
                res[0].data.evs[0] + 1j * res[0].data.evs[1],
                res[0].data.stds[0] + 1j * res[0].data.stds[1],
            )

        except Exception:
            return res.values[0] + 1j * res.values[1], np.sqrt(
                res.metadata[0]["variance"]
            ) + 1j * +np.sqrt(res.metadata[1]["variance"])
    else:
        vals_reps = []
        stds_reps = []
        for n in range(n_reps):
            jj = 2 * n
            if isinstance(res, EstimatorResult):
                val = res.values[jj] + 1j * res.values[jj + 1]
                std = np.sqrt(res.metadata[jj]["variance"]) + 1j * +np.sqrt(
                    res.metadata[jj + 1]["variance"]
                )
            else:
                val = res[n].data.evs[0] + 1j * res[n].data.evs[1]
                std = res[n].data.stds[0] + 1j * res[n].data.stds[1]

            vals_reps.append(val)
            stds_reps.append(std)
        return np.mean(vals_reps), mean_std(np.array(stds_reps))


def get_hadam_estimator_list(
    res: EstimatorResult, n_reps: int = 1
) -> tuple[list[complex], list[complex]]:
    # qiskit developers competing with python developers to see who can make the most breaking changes with a version chanege
    # anyway, here's wonderwall

    if n_reps == 1:
        try:
            return (
                [res[0].data.evs[0] + 1j * res[0].data.evs[1]],
                [res[0].data.stds[0] + 1j * res[0].data.stds[1]],
            )

        except Exception:
            return (
                [res.values[0] + 1j * res.values[1]],
                [
                    np.sqrt(res.metadata[0]["variance"])
                    + 1j * +np.sqrt(res.metadata[1]["variance"])
                ],
            )
    else:
        vals_reps = []
        stds_reps = []
        for n in range(n_reps):
            jj = 2 * n
            if isinstance(res, EstimatorResult):
                val = res.values[jj] + 1j * res.values[jj + 1]
                std = np.sqrt(res.metadata[jj]["variance"]) + 1j * +np.sqrt(
                    res.metadata[jj + 1]["variance"]
                )
            else:
                val = res[n].data.evs[0] + 1j * res[n].data.evs[1]
                std = res[n].data.stds[0] + 1j * res[n].data.stds[1]

            vals_reps.append(val)
            stds_reps.append(std)
        return vals_reps, stds_reps


HADAMARD_TESTS = Literal["stefan", "projecto-estimo", "off-diag", "full", "estimator"]


def measure_matrix_element(
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    which_backend: BACKEND_TYPES,
    backend: BACKEND_ALIAS,
    n_shots: int,
    which_hadamard_test: HADAMARD_TESTS = "estimator",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
    n_reps: int = 1,
) -> Union[complex, tuple[complex, complex]]:

    # This computes <x|U|y>

    if which_hadamard_test == "stefan" or which_hadamard_test == "stefan_samples":
        if which_hadamard_test == "stefan_samples":
            counting_method = "samples"
        else:
            counting_method = "estimator"
        ol, std = stefan_hadamard_test(
            circuit,
            parameters,
            n_shots,
            x,
            y,
            which_backend,
            backend,
            counting_method=counting_method,
            n_reps=n_reps,
        )
        return ol, std

    elif which_hadamard_test == "full":
        if n_reps > 1:
            raise NotImplementedError("n_reps>1 not implemented for full hadamard test")
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
            v, std = hadam_diff(counts, n_shots)
            stds.append(std)
            vals.append(v)

        ol = vals[0] + 1j * vals[1]
        std = stds[0] + 1j * stds[1]
        return ol, std
    elif which_hadamard_test == "estimator":
        if n_reps > 1:
            raise NotImplementedError(
                "n_reps>1 not implemented for estimator hadamard test"
            )
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

        job = circuit_estimation_job(
            backend=backend,
            observables=hams,
            circuit=hadam_circ,
            parameters=None,
            n_shots=n_shots,
        )

        res = job.result()
        return get_hadam_estimator(res)

    else:
        raise ValueError(
            f"Expected full, estimator, or stefan, got {which_hadamard_test}"
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
    which_hadamard_test: str,
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
    n_reps: int = 1,
) -> np.ndarray:

    vals = np.zeros((len(comp_states), len(comp_states)), dtype=complex)
    stds = np.zeros((len(comp_states), len(comp_states)), dtype=complex)

    on_hardware = is_hardware_run(which_backend)

    if on_hardware:
        filename = f"{tmp_dirname(circuit,which_backend)}/measure_matrix_elements.json"
        try:

            computed_values = load_tmp_item(filename)
        except FileNotFoundError:
            print(f"Starting from scratch (hadamard {which_hadamard_test} test)")
            computed_values = []

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
                    which_backend,
                    backend,
                    n_shots,
                    which_hadamard_test,
                    custom_idx=custom_idx,
                    handcraft=handcraft,
                    n_reps=n_reps,
                )
                vals[ind_x, ind_y] = jUk
                stds[ind_x, ind_y] = std
                if on_hardware:
                    computed_values = add_tmp_entry_and_save(
                        filename, computed_values, jUk, std, ind_x, ind_y
                    )
                    tmp_delete_run_file(tmp_dirname(circuit, which_backend))
                print(f"{ind_x} - {ind_y}: {jUk:.4f} ({std:.4f})")

    if on_hardware:
        vals, stds = rebuild_eld_matrices_from_tmp(computed_values, len(comp_states))
    # somehow indices are all messy
    vals = reorder_matrix_elems(vals)
    stds = reorder_matrix_elems(stds)
    return vals, stds


def estimation_from_samples(
    res: PrimitiveResult, hamiltonian: SparsePauliOp
) -> tuple[complex, complex]:

    counts = get_counts_from_res(res, hamiltonian.num_qubits)

    z_counts = counts[0]
    x_counts = counts[1]

    z_vals = []
    z_stds = []
    x_vals = []
    x_stds = []

    if "J" not in hamiltonian.__dict__:
        J = get_tfim_ratio(hamiltonian)
        h = 1
    else:
        J = hamiltonian.J
        h = hamiltonian.h

    for jj in range(hamiltonian.num_qubits):
        # set last arg to 1 to enable mitigation.
        val, std = trace_out_estimation(z_counts, [jj], 1)
        z_vals.append(val)
        z_stds.append(std)

    for jj in range(hamiltonian.num_qubits - 1):
        val, std = trace_out_estimation(x_counts, [jj, jj + 1], -1)
        x_vals.append(val)
        x_stds.append(std)

    ffval = -sum(x_vals) * J - sum(z_vals) * h
    ffstd = sum_std([sum_std(x_stds) * J, sum_std(z_vals) * h])

    return ffval, ffstd


def measure_manual_circuit(
    backend: BACKEND_ALIAS,
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    n_shots: int,
):
    [z_circ, x_circ, y_circ] = manual_hamiltonian_circuits(circuit, hamiltonian)

    job = circuit_sampling_job([z_circ, x_circ], None, backend, n_shots)
    return job


def measure_hamiltonian_element_manual(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
    n_reps: int = -1,
) -> tuple[complex, complex]:
    if x == y:
        circ_out = QuantumCircuit(circuit.num_qubits)
        x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
        for x_ind in x_inds:
            circ_out.x(x_ind)
        circ_out.compose(circuit, inplace=True)
        job = measure_manual_circuit(backend, circ_out, hamiltonian, n_shots)
        res = job.result()
        return estimation_from_samples(res, hamiltonian)
    else:
        d = ["+", "-", "+i", "-i"]
        stds = []
        vals = []
        for k in d:
            circ_p = prep_circ_superpos(x, y, k)
            circ_p.compose(circuit, inplace=True)
            job = measure_manual_circuit(backend, circ_p, hamiltonian, n_shots)
            res = job.result()
            val, std = estimation_from_samples(res, hamiltonian)
            vals.append(val)
            stds.append(std)

        val = (vals[0] - vals[1]) / 2 + 1j * (vals[2] - vals[3]) / 2
        std = sum_std(stds[0:2]) + 1j * sum_std(stds[2:4])
        return val, std


def estimate_ham_with_reps(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    n_shots: int,
    backend: BACKEND_ALIAS,
    n_reps: int,
    which_backend: BACKEND_TYPES,
    reps_per_job: int = 5,
):

    reps_per_job = min(n_reps, reps_per_job)
    if backend != "statevector" and ("IQM" in backend.name or "EQE" in backend.name):
        job = measure_manual_circuit(backend, circuit, hamiltonian, n_shots)
        res = job.result()
        return estimation_from_samples(res, hamiltonian)

    n_runs = int(n_reps // reps_per_job)

    on_hardware = is_hardware_run(which_backend)
    _dirname = tmp_dirname(circuit, which_backend)

    run_list = init_run_split(_dirname, on_hardware)

    remaining_runs = n_runs - len(run_list)

    for run_ind in range(remaining_runs):

        print(f"Run number {run_ind+1}/{remaining_runs}", end="\r")

        job = circuit_estimation_job(
            [circuit for _ in range(reps_per_job)],
            [hamiltonian],
            None,
            backend,
            n_shots,
        )

        # HARDWARE may fail HERE
        res = job.result()
        ...

        vals, stds = res_to_lists(res)
        run_list.append([vals, stds])
        if on_hardware:
            tmp_save_run_file(_dirname, run_list)

    return np.mean([v[0] for v in run_list]), mean_std(
        np.array([v[1] for v in run_list]).flatten()
    )


def res_to_lists(res: Union[EstimatorResult, Any]) -> tuple[list, list]:

    try:
        n_reps = len(res.values)
        vals = res.values
        stds = [np.sqrt(res.metadata[k]["variance"]) for k in range(n_reps)]
    except Exception:
        n_reps = len(res)
        vals = [res[k].data.evs for k in range(n_reps)]
        stds = [res[k].data.stds for k in range(n_reps)]

    return vals, stds


def average_res_over_reps(res: Union[EstimatorResult, Any]):

    if isinstance(res, EstimatorResult):
        n_reps = len(res.values)
    else:
        n_reps = len(res)

    if n_reps == 1:
        if isinstance(res, EstimatorResult):
            val, std = res.values[0], np.sqrt(res.metadata[0]["variance"])
        else:
            val, std = res[0].data.evs, res[0].data.stds
    else:
        if isinstance(res, EstimatorResult):
            val, std = np.mean([res.values[k] for k in range(n_reps)]), mean_std(
                np.array([np.sqrt(res.metadata[k]["variance"]) for k in range(n_reps)])
            )
        else:
            val, std = np.mean([res[k].data.evs for k in range(n_reps)]), mean_std(
                np.array([res[k].data.stds for k in range(n_reps)])
            )
    return val, std


def measure_hamiltonian_elements_v2(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    states: list[str],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    which_backend: str,
    n_shots: int,
    n_reps: int = 10,
):

    vals = np.zeros((len(states), len(states)), dtype=complex)
    stds = np.zeros((len(states), len(states)), dtype=complex)

    on_hardware = is_hardware_run(which_backend)

    if on_hardware:
        filename = (
            f"{tmp_dirname(circuit,which_backend)}/measure_hamiltonian_elements.json"
        )
        try:
            computed_values = load_tmp_item(filename)
        except FileNotFoundError:
            print("Starting from scratch (expectations)")
            computed_values = []

    for ind_x, x in enumerate(states):
        for ind_y, y in enumerate(states[ind_x:]):
            ind_y = ind_x + ind_y

            if ind_x == ind_y:
                if on_hardware and tmp_entry_exists(computed_values, ind_x, ind_y):
                    print(f"Already computed {ind_x}, {ind_y}")
                    val, std = tmp_get_item(computed_values, ind_x, ind_y)
                    pass

                else:
                    circ_out = QuantumCircuit(circuit.num_qubits)
                    x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
                    for x_ind in x_inds:
                        circ_out.x(x_ind)
                    circ_out.compose(circuit, inplace=True)
                    circ_out.metadata["tmpname"] = circuit.metadata["tmpname"]
                    val, std = estimate_ham_with_reps(
                        hamiltonian,
                        circ_out,
                        n_shots,
                        backend,
                        which_backend=which_backend,
                        n_reps=n_reps,
                    )

                    if on_hardware:
                        computed_values = add_tmp_entry_and_save(
                            filename, computed_values, val, std, ind_x, ind_y
                        )
                        tmp_delete_run_file(tmp_dirname(circuit, which_backend))
                    print(
                        f"{ind_x} - {ind_y} - {complex(val):.4f} ({complex(std):.4f})"
                    )
            else:
                d = OFFDIAG_SYMS
                vd = {}
                for k in d:
                    sind_x = str(ind_x) + k
                    sind_y = str(ind_y) + k
                    if on_hardware and tmp_entry_exists(
                        computed_values, sind_x, sind_y
                    ):
                        print(f"Already computed {sind_x}, {sind_y}")
                        vd[k] = tmp_get_item(computed_values, sind_x, sind_y)
                    else:
                        circ_p = prep_circ_superpos(x, y, k)
                        circ_p.compose(circuit, inplace=True)
                        circ_p.metadata["tmpname"] = circuit.metadata["tmpname"]
                        t_val, t_std = estimate_ham_with_reps(
                            hamiltonian,
                            circ_p,
                            n_shots,
                            backend,
                            which_backend=which_backend,
                            n_reps=n_reps,
                        )
                        vd[k] = (t_val, t_std)

                        if on_hardware:
                            computed_values = add_tmp_entry_and_save(
                                filename, computed_values, t_val, t_std, sind_x, sind_y
                            )
                            computed_values = add_tmp_entry_and_save(
                                filename,
                                computed_values,
                                t_val.conj(),
                                t_std,
                                sind_y,
                                sind_x,
                            )
                            tmp_delete_run_file(tmp_dirname(circuit, which_backend))
                        print(
                            f"{sind_x} - {sind_y} - {complex(val):.4f} ({complex(std):.4f})"
                        )
                val, std = compute_off_diagonal_element(vd)

            if ind_x != ind_y:
                vals[ind_x, ind_y] = np.conj(val)
                stds[ind_x, ind_y] = std
            vals[ind_y, ind_x] = val
            stds[ind_y, ind_x] = std

    # this is correct!
    vals = reorder_matrix_elems(vals)[::-1, ::-1]
    stds = reorder_matrix_elems(stds)[::-1, ::-1]
    return vals, stds


def rebuild_exd_matrices_from_tmp(l: TempListType, n: int) -> tuple[np.array, np.array]:
    # this function is wrong somehow
    # It  flips the complex sign of the upper triangle
    # don't know what causes it.
    vals = np.zeros((n, n), dtype=complex)
    stds = np.zeros((n, n), dtype=complex)

    for ind_x in range(n):
        for ind_y in range(n):
            if ind_x == ind_y:
                val, std = tmp_get_item(l, ind_x, ind_y)
                vals[ind_x, ind_y] = val
                stds[ind_x, ind_y] = std
            else:
                vd = {}
                for k in OFFDIAG_SYMS:
                    vd[k] = tmp_get_item(l, str(ind_x) + k, str(ind_y) + k)
                val, std = compute_off_diagonal_element(vd)
                vals[ind_x, ind_y] = val
                stds[ind_x, ind_y] = std
    return vals, stds


def rebuild_eld_matrices_from_tmp(l: TempListType, n: int) -> tuple[np.array, np.array]:
    vals = np.zeros((n, n), dtype=complex)
    stds = np.zeros((n, n), dtype=complex)

    for ind_x in range(n):
        for ind_y in range(n):
            val, std = tmp_get_item(l, ind_x, ind_y)
            vals[ind_x, ind_y] = val
            stds[ind_x, ind_y] = std

    return vals, stds


def compute_off_diagonal_element(
    d: Dict[str, tuple[complex, complex]],
) -> tuple[complex, complex]:
    val = (d["+"][0] - d["-"][0]) / 2 + 1j * (d["+i"][0] - d["-i"][0]) / 2
    stds = [np.sqrt(d[k][1]) for k in d]
    std = sum_std(stds[0:2]) + 1j * sum_std(stds[2:4])
    return val, std
