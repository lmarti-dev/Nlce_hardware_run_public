from qcomp_nlce.utils import (
    load_optimized_vqe,
    prep_circ_superpos,
    apply_parameters,
    res_sim_dir,
    get_optimal_n_layers,
    get_appropriate_fake_backend,
    hamming_weight_states,
    index_bits,
    compute_off_diagonal_element,
    BACKEND_ALIAS,
    convert_count_keys,
    circuit_to_matrix,
    from_bitstring,
)

from pathlib import Path
from qiskit import QuantumCircuit, ClassicalRegister
import numpy as np

from qiskit_aqt_provider.primitives import AQTSampler
from data_manager import ExperimentDataManager

from qiskit.quantum_info import SparsePauliOp


def aqt_sample_job(backend, circuit: QuantumCircuit, n_shots: int):
    sampler = AQTSampler(backend)
    sampler.set_transpile_options(optimization_level=3)
    job = sampler.run(circuit, options={"shots": n_shots})
    return job


def estimate_z_field(counts, error_mitigation: bool) -> tuple[float, float, int]:
    n_qubits = len(list(counts.keys())[0])
    kept_counts = 0
    count_0 = np.zeros(n_qubits)
    count_1 = np.zeros(n_qubits)
    for ind in range(n_qubits):
        for ks in counts.keys():
            # if bitstring is even, z_parity is broken
            is_sim = True
            if error_mitigation and sum([int(kk) for kk in ks]) % 2 == 0:
                is_sim = False
            else:
                kept_counts += counts[ks]
            if is_sim:
                if ks[ind] == "0":
                    count_0[ind] += counts[ks]
                elif ks[ind] == "1":
                    count_1[ind] += counts[ks]
    e = (count_0 - count_1) / (count_0 + count_1)
    return e, (1 - e**2) / np.sqrt(n_shots), kept_counts


def get_counts(res, n_shots: int, n_qubits: int):
    qd = convert_count_keys(res.quasi_dists[0], n_qubits)
    for k in qd:
        qd[k] = int(qd[k] * n_shots)
    return qd


def measure_qubits(circ: QuantumCircuit):
    n_qubits = circ.num_qubits
    creg = ClassicalRegister(n_qubits)
    circ.add_register(creg)
    qreg = circ.qregs[0]
    for n in range(n_qubits):
        circ.measure(qreg[n], creg[n])
    return circ


def measure_jk(
    backend: BACKEND_ALIAS,
    n_shots: int,
    circuit: QuantumCircuit,
    x: str,
    y: str,
):
    print(f"Measuring {x} - {y}")
    n_qubits = circuit.num_qubits
    if x == y:
        circ = QuantumCircuit(n_qubits)
        x_inds = index_bits(x, circuit.num_qubits, right_to_left=True)
        for x_ind in x_inds:
            circ.x(x_ind)
        circ.compose(circuit, inplace=True)
        circ.measure_all()
        job = aqt_sample_job(backend, circ, n_shots)
        res = job.result()
        counts = get_counts(res, n_shots, n_qubits)
        return {
            "with": estimate_z_field(counts, True),
            "without": estimate_z_field(counts, False),
        }

    else:
        d = ["+", "-", "+i", "-i"]
        vd = {"with": {k: [] for k in d}, "without": {k: [] for k in d}}

        for k in d:
            circ_p = prep_circ_superpos(x, y, k)
            circ_p.compose(circuit, inplace=True)
            circ_p.measure_all()

            job = aqt_sample_job(backend, circ_p, n_shots)
            res = job.result()
            counts = get_counts(res, n_shots, n_qubits)

            (val, std, n_shots) = estimate_z_field(counts, True)
            vd["with"][k].append(val)
            vd["with"][k].append(std)
            vd["with"][k].append(n_shots)
            (val, std, n_shots) = estimate_z_field(counts, False)
            vd["without"][k].append(val)
            vd["without"][k].append(std)
            vd["without"][k].append(n_shots)
        return {
            "with": (*compute_off_diagonal_element(vd["with"]), vd["with"][2]),
            "without": (*compute_off_diagonal_element(vd["without"]), vd["without"][2]),
        }


def ideal_z_strings(circuit: QuantumCircuit, x: str, y: str):
    # <j|U* Z III U|k>
    mat = circuit_to_matrix(circuit)
    n_qubits = circuit.num_qubits
    ind_x = from_bitstring(x, n_qubits)
    ind_y = from_bitstring(y, n_qubits)
    start = np.zeros(mat.shape[0])
    start[ind_x] = 1
    end = np.zeros(mat.shape[0])
    end[ind_y] = 1
    v = mat @ start
    w = mat @ end
    vals = []
    for jj in range(n_qubits):
        ll = ["I" for _ in range(n_qubits)]
        ll[jj] = "Z"
        h = SparsePauliOp("".join(ll)).to_matrix()
        vals.append(w.conj().dot(h @ v))
    return vals


if __name__ == "__main__":

    dry_run = True

    edm = ExperimentDataManager(
        "error_mitigation_test_aqt", dry_run=dry_run, project="nlce"
    )

    n_qubits_max = 5
    J = 1
    n_shots = 2000
    backend = get_appropriate_fake_backend(n_qubits_max, "AQT")

    model_dir = f"vqe_{int(J*100)}j_0hl"
    vqe_dir = f"{n_qubits_max}q_{get_optimal_n_layers(n_qubits_max)}l"
    elem_dirname = Path(
        res_sim_dir(),
        f"elem_{model_dir}/vqe/cost_offdiag/{vqe_dir}",
    )

    hamiltonian1, circuit, elem_weights, ground_energy = load_optimized_vqe(
        elem_dirname
    )
    circuit = apply_parameters(circuit, elem_weights[-1])

    comp_states = hamming_weight_states(n_qubits_max, 1)
    true_vals = np.zeros((n_qubits_max, n_qubits_max, n_qubits_max))
    fake_vals = {
        "with": [
            np.zeros((n_qubits_max, n_qubits_max, n_qubits_max)),
            np.zeros((n_qubits_max, n_qubits_max, n_qubits_max)),
            0,
        ],
        "without": [
            np.zeros((n_qubits_max, n_qubits_max, n_qubits_max)),
            np.zeros((n_qubits_max, n_qubits_max, n_qubits_max)),
            0,
        ],
    }

    for ind_x, x in enumerate(comp_states):
        for ind_y, y in enumerate(comp_states):
            true_vals[ind_x, ind_y, :] = ideal_z_strings(circuit, x, y)
            d = measure_jk(backend, n_shots, circuit, x, y)
            for k in ("with", "without"):

                fake_vals[k][0][ind_x, ind_y, :] = d[k][0]
                fake_vals[k][1][ind_x, ind_y, :] = d[k][1]
                fake_vals[k][2] = d[k][2]

    edm.save_dict(
        {
            "inputs": {"n_qubits_max": n_qubits_max, "J": J, "n_shots": n_shots},
            "results": {"true_vals": true_vals, "fake_vals": fake_vals},
        }
    )
