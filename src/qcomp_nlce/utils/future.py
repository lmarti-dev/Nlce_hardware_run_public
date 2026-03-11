def measure_ground_state_energy_subspace_sampling_v3(
    hamiltonian: SparsePauliOp,
    ground_energy: float,
    circuit: QuantumCircuit,
    backend: BACKEND_ALIAS,
    which_backend: str,
    n_shots: int = 1e3,
) -> dict:

    def relerr(e):
        return np.abs(e - ground_energy) / np.abs(ground_energy)

    # assume that the std is sample dependent ish.
    # TODO: derive a better expression!!
    std = 1 / np.sqrt(n_shots)

    bn = get_backend_name(backend)
    job = backend_sample(backend, circuit, None, n_shots)
    counts = get_counts_from_res(job.result())
    unique_bitstring = get_unique_bitstrings(counts)
    bitstring_prop = len(unique_bitstring) / (2**circuit.num_qubits)
    comp_energy = subspace_diagonalize(
        unique_bitstrings=unique_bitstring, ham=hamiltonian
    )

    print(
        f"{which_backend} ({bn}) {relerr(comp_energy):.5e} sub. mat.: {len(unique_bitstring)} × {len(unique_bitstring)}: {bitstring_prop*100:.3f}%"
    )

    return {
        which_backend: {
            "name": bn,
            "error": relerr(comp_energy),
            "energy": comp_energy,
            "unique_bitstrings": unique_bitstring,
            "bitstring_prop": bitstring_prop,
            "n_shots": n_shots,
            "std": std,
            "counts": counts,
        }
    }
