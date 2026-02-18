from qiskit.primitives import StatevectorEstimator
from qcomp_nlce.utils.folders import res_sim_dir
from pathlib import Path
from qcomp_nlce.utils.save import load_optimized_vqe
from qcomp_nlce.utils.circuit import (
    apply_parameters,
)

from qcomp_nlce.hw.gs.subspace_sampling_hw import (
    measure_ground_state_energy_subspace_sampling_v2,
)
from qcomp_nlce.utils.hardware import (
    BACKEND_ALIAS,
    PROVIDER_ALIAS,
    BACKEND_TYPES,
    compile_circuit_for_backend,
    get_appropriate_fake_backend,
    backend_estimate,
)
from qiskit_ibm_transpiler import generate_ai_pass_manager
from qcomp_nlce.utils.shadow_tomography import classical_shadow_expectation
from qcomp_nlce.utils.hamiltonian import get_ground_energy

from qiskit_ibm_runtime import SamplerV2
import numpy as np

from data_manager import ExperimentDataManager


def compare_expec_gs(provider_name: PROVIDER_ALIAS = "IBM_PERFECT"):
    dry_run = False
    edm = ExperimentDataManager(
        f"classical_shadows_vs_the_world_{str(provider_name)}",
        project="nlce",
        dry_run=dry_run,
    )
    dirname = Path(res_sim_dir(), f"gs_vqe_100j_0hl/vqe/5q_3l")

    hamiltonian, circ, weights, ground_energy = load_optimized_vqe(dirname)
    circ = apply_parameters(circ, weights[-1])
    backend = get_appropriate_fake_backend(hamiltonian.num_qubits, provider_name)
    sampler = SamplerV2(backend)
    backend_circ = compile_circuit_for_backend(circ, backend)
    sv_estim = StatevectorEstimator()

    classical_shadows = {"vals": [], "stds": []}
    statevecs = {"vals": [], "stds": []}
    estims = {"vals": [], "stds": []}
    subsamps = {"vals": [], "stds": []}

    n_shots_l = []
    bitstring_prop = []
    for spow in np.linspace(1, 5, 10):
        n_shots = int(10**spow)
        n_shots_l.append(n_shots)

        cs_gs, cs_std = classical_shadow_expectation(
            hamiltonian, backend_circ, n_shots, sampler
        )

        job = sv_estim.run(pubs=[(circ, hamiltonian)], precision=1 / np.sqrt(n_shots))
        res = job.result()
        sv_gs, sv_std = res[0].data.evs, res[0].data.stds

        job = backend_estimate(backend, [hamiltonian], circ, None, n_shots)
        res = job.result()

        fb_gs, fb_std = res[0].data.evs, res[0].data.stds

        out_d = measure_ground_state_energy_subspace_sampling_v2(
            hamiltonian,
            ground_energy,
            circ,
            which_backend="fake_backend",
            n_shots=n_shots,
            provider_name=provider_name,
        )
        ss_gs, ss_std = out_d["fake_backend"]["energy"], out_d["fake_backend"]["std"]
        bitstring_prop.append(out_d["fake_backend"]["bitstring_prop"])

        classical_shadows["vals"].append(cs_gs)
        classical_shadows["stds"].append(cs_std)

        estims["vals"].append(fb_gs)
        estims["stds"].append(fb_std)

        statevecs["vals"].append(sv_gs)
        statevecs["stds"].append(sv_std)

        subsamps["vals"].append(ss_gs)
        subsamps["stds"].append(ss_std)

    edm.save_dict(
        {
            "classical_shadows": classical_shadows,
            "estimator": estims,
            "statevector": statevecs,
            "subspace_diag": subsamps,
            "list_shots": n_shots_l,
            "hamiltonian": hamiltonian,
            "provider_name": provider_name,
            "ground_energy": ground_energy,
            "bitstring_prop": bitstring_prop,
        }
    )


if __name__ == "__main__":
    compare_expec_gs()
