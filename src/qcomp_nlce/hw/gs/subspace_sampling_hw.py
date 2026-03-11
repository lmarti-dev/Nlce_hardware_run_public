import numpy as np

from qcomp_nlce.utils.hardware import (
    backend_sample,
    get_appropriate_fake_backend,
    get_backend,
    encode_sampler_result,
    backend_estimate,
    mean_std_tuples_from_res,
    is_hardware_run,
    get_backend_name,
    BACKEND_ALIAS,
)
from qcomp_nlce.utils.statevector import (
    sv_estimate_observables,
    sv_expectation_observables,
)
from qcomp_nlce.utils.measure import (
    get_counts_from_job,
    get_counts_from_res,
    add_meas_layer,
    get_unique_bitstrings,
)
from qcomp_nlce.utils.circuit import (
    apply_parameters,
)
from qcomp_nlce.utils.hamiltonian import (
    get_tfim_ratio,
    get_ground_energy,
    subspace_diagonalize,
)
from qcomp_nlce.utils.save import (
    load_tmp_item,
    save_tmp_item,
    tmp_dirname,
    load_optimized_vqe,
    load_sweep,
    init_run_split,
    tmp_save_run_file,
)

from data_manager import ExperimentDataManager
from qiskit.primitives import StatevectorSampler

from pathlib import Path
import os
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from typing import Literal


def measure_ground_state_energy_estimator(
    hamiltonian: SparsePauliOp,
    target_energy: float,
    circuit: QuantumCircuit,
    which_backend: str,
    n_shots: int = 1e3,
    provider_name: str = "AQT",
):

    def relerr(e):
        return np.abs(e - target_energy) / np.abs(target_energy)

    if which_backend == "fake_backend":
        backend = get_appropriate_fake_backend(circuit.num_qubits, provider_name)
        job = backend_estimate(
            backend=backend,
            observables=hamiltonian,
            parameters=None,
            circuit=circuit,
            n_shots=n_shots,
        )
        energy, std = mean_std_tuples_from_res(job.result())
        bn = backend.name
    elif which_backend == "statevector_shots":
        res0 = sv_estimate_observables(
            circuit=circuit,
            observables=[hamiltonian],
            param_sweep=None,
            n_shots=n_shots,
        )
        energy = res0.data.evs[0]
        std = res0.data.stds[0]
        bn = which_backend

    elif which_backend == "statevector":
        res0 = sv_expectation_observables(
            circuit=circuit,
            observables=[hamiltonian],
            parameters=None,
        )
        energy = res0[0]
        std = 0

        bn = which_backend

    elif which_backend == "real_hardware":
        backend = get_backend(circuit.num_qubits, provider_name)

        job = backend_estimate(
            backend=backend_name,
            observables=hamiltonian,
            parameters=None,
            circuit=circuit,
            n_shots=n_shots,
        )

        energy, std = mean_std_tuples_from_res(job.result())
        bn = backend.name

    print(f"{bn} -- estimated energy: {energy:.5f}, err {relerr(energy):.5f}")

    return {
        which_backend: {
            "name": bn,
            "error": relerr(energy),
            "energy": energy,
            "n_shots": n_shots,
            "std": std,
        }
    }


def loop_measure_ground_state_energy_subspace_sampling_v2(
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    which_backends: list[str],
    n_shots: int = 1e3,
    provider_name: str = "AQT",
    solve: bool = False,
    n_reps: int = 1,
    reps_per_job: int = 5,
) -> dict:

    if solve:
        ground_energy = get_ground_energy(hamiltonian)
    else:
        ground_energy = 1000

    d = {
        "system": {
            "ground_energy": ground_energy,
            "n_qubits": circuit.num_qubits,
            "hamiltonian": hamiltonian,
        }
    }

    for which_backend in which_backends:
        on_hardware = is_hardware_run(which_backend)

        item = None
        if on_hardware:
            filename = (
                f"{tmp_dirname(circuit,which_backend)}/measure_ground_energy.json"
            )
            try:
                item = load_tmp_item(filename)
                print(f"Already Computed {circuit.metadata['tmpname']}")
            except FileNotFoundError:
                print("Starting from scratch (ground state)")

        if item is None:
            item = measure_ground_state_energy_subspace_sampling_v2(
                hamiltonian,
                ground_energy,
                circuit,
                which_backend,
                n_shots,
                provider_name,
                n_reps=n_reps,
                reps_per_job=reps_per_job,
            )
            if on_hardware:
                save_tmp_item(item, filename)
        d[which_backend] = item[which_backend]
    return d


def measure_ground_state_energy_subspace_sampling_v2(
    hamiltonian: SparsePauliOp,
    ground_energy: float,
    circuit: QuantumCircuit,
    which_backend: str,
    n_shots: int = 1e3,
    provider_name: str = "AQT",
    n_reps: int = 1,
    reps_per_job: int = 5,
) -> dict:

    reps_per_job = min(n_reps, reps_per_job)

    def relerr(e):
        return np.abs(e - ground_energy) / np.abs(ground_energy)

    circuit_to_sample = add_meas_layer(circuit)
    # assume that the std is sample dependent ish.
    # TODO: derive a better expression!!
    # TODO: this seems to be non-trivial
    # how does the uncertainty of subspace diagonalization work?
    std = 1 / np.sqrt(n_shots * n_reps)
    if which_backend == "statevector_shots" or which_backend == "statevector":
        sv_energy = sv_estimate_observables(
            circuit=circuit,
            observables=[hamiltonian],
            param_sweep=[],
            n_shots=(n_shots * n_reps),
        ).data.evs[0]
        print(f"Statevector {relerr(sv_energy):.5e}")

        bn = which_backend

        sv = StatevectorSampler()

        job = sv.run([circuit_to_sample], shots=n_shots * n_reps)

        counts = job.result()[0].join_data().get_counts()

        bn = which_backend

    elif which_backend == "fake_backend" or which_backend == "real_backend":

        if which_backend == "fake_backend":
            backend = get_appropriate_fake_backend(circuit.num_qubits, provider_name)

        elif which_backend == "real_backend":
            backend = get_backend(circuit.num_qubits, provider_name)

        on_hardware = is_hardware_run(which_backend)
        _dirname = tmp_dirname(circuit, which_backend)

        n_runs = int(n_reps // reps_per_job)

        run_list = init_run_split(_dirname, on_hardware)

        remaining_runs = n_runs - len(run_list)
        for run_ind in range(remaining_runs):
            print(f"Run number {run_ind+1}/{remaining_runs}", end="\r")
            job = backend_sample(
                backend=backend,
                circuit=circuit_to_sample,
                parameters=[],
                n_shots=n_shots,
            )
            counts = get_counts_from_res(job.result(), hamiltonian.num_qubits, 0)
            run_list.append(counts)
            bn = backend.name
            if on_hardware:
                tmp_save_run_file(_dirname, run_list)

        counts_out = {}
        for c in run_list:
            for k in c:
                if k in counts_out.keys():
                    counts_out[k] += c[k]
                else:
                    counts_out[k] = c[k]
        counts = counts_out
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


def save_ground_state_energy_run(
    edm: ExperimentDataManager,
    dirname: str,
    n_shots: int = 1e3,
    run_on_hardware: bool = False,
    which: str = "vqe",
    backend_name: str = "AQT",
):

    hamiltonian, ground_energy, circuit = load_which(dirname, which)

    var_dict = measure_ground_state_energy_subspace_sampling_v2(
        hamiltonian,
        ground_energy,
        circuit,
        n_shots,
        run_on_hardware,
        which,
        backend_name,
    )

    var_dict["system"]["dirname"] = dirname
    var_dict["system"]["which"] = which

    edm.save_dict(var_dict)

    return var_dict


def run_all(
    dirname: str,
    edm: ExperimentDataManager,
    run_on_hardware: bool,
    which: str,
    n_shots: int,
    backend_name: str,
):
    HOME = Path(__file__).parent
    dirname = Path(HOME, f"../../res_sim/{dirname}/{which}")
    experiments = os.listdir(dirname)
    for x in experiments:
        print(f"Loading {x}")
        save_ground_state_energy_run(
            edm, dirname, n_shots, run_on_hardware, which, backend_name
        )
        edm.new_run()


def load_which(dirname: Path, which: Literal["sweep", "vqe"]):
    if which == "vqe":
        hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(dirname)
        circuit = apply_parameters(circuit)
        parameters = weights[-1]
    elif which == "sweep":
        hamiltonian, circuit, ground_energy = load_sweep(dirname)

    return hamiltonian, ground_energy, circuit


if __name__ == "__main__":
    dry_run = False
    which = "sweep"
    edm = ExperimentDataManager(f"nlce_hardware_run_{which}_gs", dry_run=dry_run)
    run_on_hardware = True
    backend_name = "AQT"
    n_shots = 1e3
    if which == "sweep":
        run_all(
            "9q_10q_gs_aqt_midway",
            edm,
            run_on_hardware,
            which=which,
            backend_name=backend_name,
            n_shots=n_shots,
        )
    elif which == "vqe":
        run_all(
            "9q_10q_gs",
            edm,
            run_on_hardware,
            which=which,
            backend_name=backend_name,
            n_shots=n_shots,
        )
