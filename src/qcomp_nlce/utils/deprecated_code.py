from src.qcomp_nlce.utils.stats import mean_std

import numpy as np


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

    final_g, _, _, std = get_overlaps_from_res(jobs.result())

    return final_g, std


def measure_hamiltonian_elements(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    states: list[str],
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    which_backend: str,
    n_shots: int,
    n_reps: int = 1,
):
    vals = np.zeros((len(states), len(states)), dtype=complex)
    stds = np.zeros((len(states), len(states)), dtype=complex)

    on_hardware = is_hardware_run(which_backend)

    if on_hardware:
        filename = (
            f"{tmp_filename(circuit, which_backend)}/measure_hamiltonian_elements.json"
        )
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
                if backend != "statevector" and (
                    "IQM" in backend.name or "EQE" in backend.name
                ):
                    val, std = measure_hamiltonian_element_manual(
                        hamiltonian, circuit, x, y, parameters, backend, n_shots, n_reps
                    )
                else:
                    val, std = measure_hamiltonian_element(
                        hamiltonian, circuit, x, y, parameters, backend, n_shots, n_reps
                    )
                vals[ind_x, ind_y] = val.conj()
                vals[ind_y, ind_x] = val
                stds[ind_x, ind_y] = std
                stds[ind_y, ind_x] = std

                print(f"{ind_x} - {ind_y} - {complex(val):.4f} ({complex(std):.4f})")
                if on_hardware:
                    computed_values = add_tmp_entry_and_save(
                        filename, computed_values, val.conj(), std, ind_x, ind_y
                    )
                    computed_values = add_tmp_entry_and_save(
                        filename, computed_values, val, std, ind_y, ind_x
                    )
    # this is correct!
    if on_hardware:
        vals, stds = rebuild_exd_matrices_from_tmp(computed_values, len(states))
    vals = reorder_matrix_elems(vals)[::-1, ::-1]
    stds = reorder_matrix_elems(stds)[::-1, ::-1]
    return vals, stds


def measure_hamiltonian_element(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
    n_reps: int = 10,
) -> tuple[complex, complex]:
    if x == y:
        if "statevector" != backend and (
            "IQM" in backend.name or "EQE" in backend.name
        ):
            return measure_hamiltonian_element_manual(
                hamiltonian, circuit, x, y, parameters, backend, n_shots
            )
        circ_out = QuantumCircuit(circuit.num_qubits)
        x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
        for x_ind in x_inds:
            circ_out.x(x_ind)
        circ_out.compose(circuit, inplace=True)
        return estimate_ham_with_reps(hamiltonian, circ_out, n_shots, backend, n_reps)

    # this doesn't happen in v2
    else:
        d = ["+", "-", "+i", "-i"]
        circs = []
        for _ in range(n_reps):
            for k in d:
                circ_p = prep_circ_superpos(x, y, k)
                circ_p.compose(circuit, inplace=True)
                circs.append(circ_p)

        job = circuit_estimation_job(circs, [hamiltonian], parameters, backend, n_shots)

        res = job.result()
        if n_reps == 1:
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
        else:
            vals_reps = []
            stds_reps = []
            for n in range(n_reps):
                jj = 4 * n
                if isinstance(res, EstimatorResult):
                    val = (res.values[jj] - res.values[jj + 1]) / 2 + 1j * (
                        res.values[jj + 2] - res.values[jj + 3]
                    ) / 2
                    stds = [np.sqrt(res.metadata[jj + k]["variance"]) for k in range(4)]
                    std = sum_std(stds[:2]) + 1j * sum_std(stds[2:4])

                else:
                    val = (res[jj].data.evs - res[jj + 1].data.evs) / 2 + 1j * (
                        res[jj + 2].data.evs - res[jj + 3].data.evs
                    ) / 2
                    stds = [res[jj + k].data.stds for k in range(4)]
                    std = sum_std(stds[0:2]) + 1j * sum_std(stds[2:4])
                vals_reps.append(val)
                stds_reps.append(std)

            return np.mean(vals_reps), mean_std(np.array(stds_reps))


def measure_matrix_element(
    circuit: QuantumCircuit,
    x: str,
    y: str,
    parameters: np.ndarray,
    backend: BACKEND_ALIAS,
    n_shots: int,
    which_hadamard_test: HADAMARD_TESTS = "estimator",
    custom_idx: list[list[int]] = None,
    handcraft: bool = True,
    n_reps: int = 1,
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

    elif which_hadamard_test == "stefan" or which_hadamard_test == "stefan_samples":
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
            backend,
            counting_method=counting_method,
            n_reps=n_reps,
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
            v, std = hadam_diff(counts, n_shots)
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
        filename = f"{tmp_dirname(circuit, which_backend)}/measure_matrix_elements.json"
        try:
            computed_values = load_tmp_item(filename)
        except FileNotFoundError:
            print(f"Starting from scratch (hadamard {which_hadamard_test} test)")
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
                    print(f"{ind_x} - {ind_y}: {jUk:.4f} ({std:.4f})")
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
                        n_reps=n_reps,
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
                    print(f"{ind_x} - {ind_y}: {jUk:.3f}, {kUj:.3f} ({std:.4f})")
    if on_hardware:
        vals, stds = rebuild_eld_matrices_from_tmp(computed_values, len(comp_states))
    # somehow indices are all messy
    vals = reorder_matrix_elems(vals)
    stds = reorder_matrix_elems(stds)
    return vals, stds


def measure_ground_state_energy_subspace_sampling(
    hamiltonian: SparsePauliOp,
    ground_energy: float,
    circuit: QuantumCircuit,
    n_shots: int = 1e3,
    run_on_hardware: bool = False,
    backend_name: str = "AQT",
) -> dict:
    fake_backend = get_appropriate_fake_backend(circuit.num_qubits, backend_name)
    ratio = get_tfim_ratio(hamiltonian)

    print(f"n_qubits: {hamiltonian.num_qubits}, j/h {ratio}")

    circuit_to_sample = add_meas_layer(circuit)

    def relerr(e):
        return np.abs(e - ground_energy) / np.abs(ground_energy)

    fake_job = backend_sample(
        backend=fake_backend,
        circuit=circuit_to_sample,
        parameters=[],
        n_shots=n_shots,
    )
    fake_counts = get_counts_from_job(fake_job)
    fake_unique_bitstring = get_unique_bitstrings(fake_counts)
    fake_backend_bitstring_prop = len(fake_unique_bitstring) / (2**circuit.num_qubits)
    fake_backend_energy = subspace_diagonalize(
        unique_bitstrings=fake_unique_bitstring, ham=hamiltonian
    )

    statevector_energy = sv_estimate_observables(
        circuit=circuit, observables=[hamiltonian], param_sweep=[], n_shots=n_shots
    ).data.evs[0]

    if run_on_hardware:
        real_backend = get_backend(circuit.num_qubits, backend_name)
        real_job = backend_sample(
            backend=real_backend,
            circuit=circuit_to_sample,
            parameters=[],
            n_shots=n_shots,
        )
        real_counts = get_counts_from_job(real_job)
        real_unique_bitstring = get_unique_bitstrings(real_counts)
        real_bitstring_prop = len(real_unique_bitstring) / (2**circuit.num_qubits)
        real_backend_energy = subspace_diagonalize(
            unique_bitstrings=real_unique_bitstring, ham=hamiltonian
        )

        print(
            f"RealBackend ({real_backend.name}) {relerr(real_backend_energy):.5e} sub. mat.: {len(real_unique_bitstring)} × {len(real_unique_bitstring)}: {real_bitstring_prop * 100:.3f}%"
        )

    print(f"Statevector {relerr(statevector_energy):.5e}")

    print(
        f"FakeBackend ({fake_backend.name}) {relerr(fake_backend_energy):.5e} sub. mat.: {len(fake_unique_bitstring)} × {len(fake_unique_bitstring)}: {fake_backend_bitstring_prop * 100:.3f}%"
    )

    var_dict = {
        "fake_backend": {
            "name": fake_backend.name,
            "error": relerr(fake_backend_energy),
            "energy": fake_backend_energy,
            "unique_bitstrings": fake_unique_bitstring,
            "bitstring_prop": fake_backend_bitstring_prop,
            "n_shots": n_shots,
            "job": encode_sampler_result(fake_job),
        },
        "statevector": {
            "error": relerr(statevector_energy),
            "energy": statevector_energy,
        },
        "system": {
            "ground_energy": ground_energy,
            "n_qubits": circuit.num_qubits,
            "hamiltonian": hamiltonian,
            "ratio": ratio,
        },
    }
    if run_on_hardware:
        var_dict["real_backend"] = {
            "name": real_backend.name,
            "error": relerr(real_backend_energy),
            "energy": real_backend_energy,
            "unique_bitstrings": real_unique_bitstring,
            "bitstring_prop": real_bitstring_prop,
            "n_shots": n_shots,
            "job": encode_sampler_result(real_job),
        }

    return var_dict
