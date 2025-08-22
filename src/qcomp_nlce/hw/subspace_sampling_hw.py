import numpy as np

from qcomp_nlce.utils import (
    backend_sample,
    subspace_diagonalize,
    get_unique_bitstrings,
    estimate_observables,
    load_optimized_vqe,
    load_sweep,
    get_appropriate_fake_backend,
    get_tfim_ratio,
    get_counts_from_job,
    get_backend,
    add_meas_layer,
    sampler_result_to_json,
)
from data_manager import ExperimentDataManager
from qiskit.primitives.base.sampler_result import SamplerResult

from pathlib import Path
import os

PROVIDER = "LRZ"


def encode_sampler_result(job):
    res = job.result()
    if isinstance(res, SamplerResult):
        return sampler_result_to_json(res)
    else:
        return res


def get_expectations_from_sources(
    edm: ExperimentDataManager,
    dirname: str,
    n_shots: int = 1e3,
    run_on_hardware: bool = False,
    which: str = "vqe",
):
    backend_name = "AQT" if PROVIDER == "LRZ" else "IBM"
    if which == "vqe":
        hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(dirname)

        parameters = weights[-1]
    elif which == "sweep":
        hamiltonian, circuit, ground_energy = load_sweep(dirname)
        weights = None
        parameters = None
    fake_backend = get_appropriate_fake_backend(circuit.num_qubits, backend_name)

    print(f"n_qubits: {hamiltonian.num_qubits}, j/h {get_tfim_ratio(hamiltonian)}")

    circuit_to_sample = add_meas_layer(circuit)

    def relerr(e):
        return np.abs(e - ground_energy) / np.abs(ground_energy)

    fake_job = backend_sample(
        backend=fake_backend,
        circuit=circuit_to_sample,
        parameters=parameters,
        n_shots=n_shots,
    )
    fake_counts = get_counts_from_job(fake_job)
    fake_unique_bitstring = get_unique_bitstrings(fake_counts)
    fake_backend_bitstring_prop = len(fake_unique_bitstring) / (2**circuit.num_qubits)
    fake_backend_energy = subspace_diagonalize(
        unique_bitstrings=fake_unique_bitstring, ham=hamiltonian
    )

    statevector_energy = estimate_observables(
        circuit=circuit, observables=[hamiltonian], param_sweep=parameters
    ).data.evs[0]

    if run_on_hardware:
        real_backend = get_backend(circuit.num_qubits, backend_name)
        real_job = backend_sample(
            backend=real_backend,
            circuit=circuit_to_sample,
            parameters=parameters,
            n_shots=n_shots,
        )
        real_counts = get_counts_from_job(real_job)
        real_unique_bitstring = get_unique_bitstrings(real_counts)
        real_bitstring_prop = len(real_unique_bitstring) / (2**circuit.num_qubits)
        real_energy = subspace_diagonalize(
            unique_bitstrings=real_unique_bitstring, ham=hamiltonian
        )

        print(
            f"RealBackend ({real_backend.name}) {relerr(real_energy):.5e} sub. mat.: {len(real_unique_bitstring)} × {len(real_unique_bitstring)}: {real_bitstring_prop*100:.3f}%"
        )

    print(f"Statevector {relerr(statevector_energy):.5e}")

    print(
        f"FakeBackend ({fake_backend.name}) {relerr(fake_backend_energy):.5e} sub. mat.: {len(fake_unique_bitstring)} × {len(fake_unique_bitstring)}: {fake_backend_bitstring_prop*100:.3f}%"
    )

    var_dict = {
        "fake_backend": {
            "name": fake_backend.name,
            "error": relerr(fake_backend_energy),
            "energy": fake_backend_energy,
            "unique_bitstrings": fake_unique_bitstring,
            "bitstring_prop": fake_backend_bitstring_prop,
            "n_shots": n_shots,
        },
        "statevector": {
            "error": relerr(statevector_energy),
            "energy": statevector_energy,
        },
        "system": {
            "ground_energy": ground_energy,
            "n_qubits": circuit.num_qubits,
            "name": dirname,
            "hamiltonian": hamiltonian,
            "ratio": get_tfim_ratio(hamiltonian),
        },
    }
    if run_on_hardware:
        var_dict["real_backend"] = {
            "name": real_backend.name,
            "error": relerr(real_energy),
            "energy": real_energy,
            "unique_bitstrings": real_unique_bitstring,
            "bitstring_prop": real_bitstring_prop,
            "n_shots": n_shots,
        }

    edm.var_dump(**var_dict)
    jobj = {"fake_result": encode_sampler_result(fake_job)}
    if run_on_hardware:
        jobj["real_result"] = encode_sampler_result(real_job)
    edm.save_dict(jobj)


def run_all(
    dirname: str, edm: ExperimentDataManager, run_on_hardware: bool, which: str
):
    HOME = Path(__file__).parent
    dirname = Path(HOME, f"../res_sim/{dirname}/{which}")
    experiments = os.listdir(dirname)
    for x in experiments:
        print(f"Loading {x}")
        get_expectations_from_sources(
            edm,
            Path(dirname, x),
            run_on_hardware=run_on_hardware,
            which=which,
            n_shots=200,
        )
        edm.new_run()


def run_single(edm: ExperimentDataManager, run_on_hardware: bool):
    HOME = Path(__file__).parent
    dirname = Path(HOME, f"../res_sim/9q_10q_gs/vqe")
    experiment = "6e4f28ef-ae5e-4f67-95e0-5c4575fcbdcf"
    get_expectations_from_sources(
        edm,
        Path(dirname, experiment),
        run_on_hardware=run_on_hardware,
        which="vqe",
        n_shots=200,
    )


if __name__ == "__main__":
    dry_run = False
    which = "sweep"
    edm = ExperimentDataManager(f"nlce_hardware_run_{which}_gs", dry_run=dry_run)
    run_on_hardware = True
    if which == "sweep":
        run_all("9q_10q_gs_aqt_midway", edm, run_on_hardware, which=which)
    elif which == "vqe":
        run_all("9q_10q_gs", edm, run_on_hardware, which=which)
    elif which == "test":
        run_single(edm, run_on_hardware)
