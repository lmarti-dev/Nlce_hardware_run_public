"""
generate_dispersion_data.py
===========================
Compute H_eff matrices for TFIM clusters using VQE circuits and save to disk.
Generates ED, statevector, statevector-with-shots, and fake-backend results.

Data is saved per cluster size under:
    res_sim/dispersion_data/j{J}_hl{hl}/N{n}/
        H_eff_ED.npy
        H_eff_SV.npy
        H_eff_sv_shots_rep{i}.npy   (i = 0, 1, ...)
        H_eff_fake_backend_rep{i}.npy
        gs_energy.npz                (E_ED, E_SV)
        metadata.npz                 (J, hl, n_shots, provider, ...)

Usage:
    python generate_dispersion_data.py
"""

import numpy as np
from copy import deepcopy
from pathlib import Path

from qcomp_nlce.utils.save import load_optimized_vqe, get_save_directory, baptize_circuit
from qcomp_nlce.utils.circuit import apply_parameters, circuit_to_matrix
from qcomp_nlce.utils.statevector import hamming_weight_states, estimate_observable_1qp_initial_state
from qcomp_nlce.utils.hamiltonian import from_bitstring, TFIM, get_ground_energy, get_tfim_ratio, HAM_NAME_VAR
from qcomp_nlce.utils.effective_hamiltonian import H_effective, H_effective_ed
from qcomp_nlce.utils.vqe import get_optimal_n_layers
from qcomp_nlce.utils.workflow_tfim_1d import get_heff_values


# ── Paths ────────────────────────────────────────────────────────────────────

RES_SIM = Path(__file__).parent.parent.parent / "res_sim"
DATA_DIR = RES_SIM / "dispersion_data"


def _cluster_dir(N: int, J: float, hl: float) -> Path:
    """Return (and create) the data folder for cluster size N."""
    d = DATA_DIR / f"j{int(J*100)}_hl{int(hl*100)}" / f"N{N}"
    d.mkdir(exist_ok=True, parents=True)
    return d


# ── VQE data loader (cached per cluster) ────────────────────────────────────

def _load_vqe_data(N: int, J: float, hl: float) -> dict:
    """
    Load VQE circuits and weights once for cluster size N.
    Returns a dict with everything needed for all computation types.
    """
    n_layers = get_optimal_n_layers(N)
    gs_dir = str(get_save_directory(f"gs_vqe_{int(J*100)}j_{int(hl*100)}hl/vqe/{N}q_{n_layers}l"))
    elem_dir = str(get_save_directory(f"elem_vqe_{int(J*100)}j_{int(hl*100)}hl/vqe/cost_offdiag/{N}q_{n_layers}l"))

    ham_gs, gs_circuit, gs_weights, _ = load_optimized_vqe(gs_dir)
    ham_elem, elem_circuit, elem_weights, _ = load_optimized_vqe(elem_dir)

    gs_bound = apply_parameters(gs_circuit, gs_weights[-1])
    elem_bound = apply_parameters(elem_circuit, elem_weights[-1])

    hamiltonian = ham_elem

    return {
        "hamiltonian": hamiltonian,
        "gs_circuit": gs_circuit,
        "gs_weights": gs_weights,
        "gs_bound": gs_bound,
        "elem_circuit": elem_circuit,
        "elem_weights": elem_weights,
        "elem_bound": elem_bound,
    }



def compute_heff_ed(N: int, J: float, hl: float) -> np.ndarray:
    """Exact-diagonalisation H_eff for cluster of size N."""
    ham = TFIM(N, h=1, J=J, periodic=False, longitudinal_field=hl)
    return H_effective_ed(ham)


def compute_heff_sv_exact(vqe: dict) -> tuple[np.ndarray, float]:
    """
    Statevector (no noise) H_eff from pre-loaded VQE data.
    Returns (H_eff, gs_energy).
    """
    hamiltonian = vqe["hamiltonian"]
    n_qubits = hamiltonian.num_qubits

    # Ground-state energy
    umat_gs = circuit_to_matrix(vqe["gs_bound"])
    hmat = hamiltonian.to_matrix(False)
    gs_energy = float(np.real((umat_gs.T.conj() @ hmat @ umat_gs)[0, 0]))

    # 1qp expectation matrix
    one_qp_exp = estimate_observable_1qp_initial_state(
        vqe["elem_circuit"], vqe["elem_weights"][-1], hamiltonian
    )

    # Overlap matrix
    umat_elem = circuit_to_matrix(vqe["elem_bound"])
    comp_states = hamming_weight_states(n_qubits, 1)
    inds = [from_bitstring(b, n_qubits, True) for b in comp_states]
    overlap = umat_elem[np.ix_(inds, inds)]

    heff = H_effective(gs_energy, one_qp_exp, overlap)
    return heff, gs_energy


def compute_heff_backend(
    vqe: dict,
    which_backend: str,
    provider_name: str,
    n_shots: int,
    which_hadamard_test: str,
    rep_index: int,
) -> tuple[np.ndarray, dict]:
    """
    Compute H_eff via the measurement pipeline.
    Uses pre-loaded VQE data; only stamps a unique tmpname per rep.
    Returns (H_eff, mc_ingredients) where mc_ingredients contains the
    raw values and stds needed for Monte Carlo error propagation.
    """
    hamiltonian = vqe["hamiltonian"]
    hamiltonian.__dict__[HAM_NAME_VAR] = "TFIM"

    n_qubits = hamiltonian.num_qubits
    ratio = get_tfim_ratio(hamiltonian)
    base_name = baptize_circuit(
        "dim1d", "vqe", n_qubits, ratio, provider_name, n_shots, which_hadamard_test
    )
    tmpname = f"{base_name}_{which_backend}_rep{rep_index}"

    # Deep-copy bound circuits so tmpname doesn't leak between reps
    gs_circ = deepcopy(vqe["gs_bound"])
    elem_circ = deepcopy(vqe["elem_bound"])
    gs_circ.metadata["tmpname"] = tmpname
    elem_circ.metadata["tmpname"] = tmpname

    result = get_heff_values(
        gs_circ, elem_circ, hamiltonian,
        which_backends=[which_backend],
        provider_name=provider_name,
        n_shots=n_shots,
        group_size=None,
        which_hadamard_test=which_hadamard_test,
    )

    # Extract MC ingredients for error propagation
    # Note: expect_std and overlap_std are RAW stds, not error-on-the-mean.
    # Divide by sqrt(n_shots) before MC sampling (to_eotm correction).
    # gs_std is already EOTM from subspace sampling.
    mc_ingredients = {
        "expect": result["exd"][which_backend]["values"],
        "expect_std": result["exd"][which_backend]["stds"],
        "overlap": result["eld"][which_backend]["values"],
        "overlap_std": result["eld"][which_backend]["stds"],
        "gs_energy": result["gsd"][which_backend]["energy"],
        "gs_std": result["gsd"][which_backend]["std"],
        "n_shots": n_shots,
    }

    return result["heff"][which_backend], mc_ingredients


# ── Save helpers ─────────────────────────────────────────────────────────────

def save_heff(cdir: Path, label: str, heff: np.ndarray):
    np.save(cdir / f"{label}.npy", heff)


def save_gs_energies(cdir: Path, E_ED: float, E_SV: float):
    np.savez(cdir / "gs_energy.npz", E_ED=E_ED, E_SV=E_SV)


def save_mc_ingredients(cdir: Path, label: str, mc: dict):
    """Save MC error propagation ingredients (values + stds) as .npz."""
    np.savez(
        cdir / f"{label}.npz",
        expect=mc["expect"],
        expect_std=mc["expect_std"],
        overlap=mc["overlap"],
        overlap_std=mc["overlap_std"],
        gs_energy=mc["gs_energy"],
        gs_std=mc["gs_std"],
        n_shots=mc["n_shots"],
    )


def save_metadata(cdir: Path, **kwargs):
    np.savez(cdir / "metadata.npz", **kwargs)


# ── Main workflow ────────────────────────────────────────────────────────────

def generate_dispersion_data(
    J: float = 0.8,
    hl: float = 0.0,
    cluster_pairs: list[tuple[int, int]] = None,
    provider_name: str = "AQT",
    n_shots: int = 200,
    which_hadamard_test: str = "stefan",
    n_repetitions: int = 10,
):
    """
    Compute and save H_eff for each cluster size.

    Saved data per cluster N (in res_sim/dispersion_data/j{J}_hl{hl}/N{n}/):
        H_eff_ED.npy              — exact diagonalisation
        H_eff_SV.npy              — VQE statevector (exact, no noise)
        H_eff_sv_shots_rep{i}.npy — VQE statevector with shot noise
        H_eff_fake_backend_rep{i}.npy — VQE fake noisy backend
        gs_energy.npz             — ground state energies (E_ED, E_SV)
        metadata.npz              — run parameters
    """
    if cluster_pairs is None:
        cluster_pairs = [(4, 3)]

    all_n = sorted(set(n for pair in cluster_pairs for n in pair))

    # ── Per-cluster: load once, compute all, save ────────────────────────
    print(f"\n{'='*60}")
    print(f"Ground State Energy: VQE vs ED (J={J}, hl={hl})")
    print(f"{'='*60}")
    print(f"{'N':<6} {'E_VQE':<16} {'E_ED':<16} {'Error':<16} {'Rel Error':<12}")
    print(f"{'-'*60}")

    for N in all_n:
        cdir = _cluster_dir(N, J, hl)

        # Load VQE data once for this cluster size
        vqe = _load_vqe_data(N, J, hl)

        # ED
        ham = TFIM(N, h=1, J=J, periodic=False, longitudinal_field=hl)
        E_ED = get_ground_energy(ham)

        try:
            heff_ed = H_effective_ed(ham)
            save_heff(cdir, "H_eff_ED", heff_ed)
        except Exception as e:
            print(f"  N={N} ED failed: {e}")

        # Exact statevector (also gives us GS energy — no separate load needed)
        try:
            heff_sv, E_SV = compute_heff_sv_exact(vqe)
            save_heff(cdir, "H_eff_SV", heff_sv)
            save_gs_energies(cdir, E_ED, E_SV)

            err = E_SV - E_ED
            print(f"{N:<6} {E_SV:<16.10f} {E_ED:<16.10f} {err:<+16.10f} {abs(err/E_ED):<12.2e}")
        except Exception as e:
            print(f"{N:<6} {'FAILED':<16} {E_ED:<16.10f} {'---':<16} {'---':<12}  ({e})")

        # Statevector with shot noise (multiple reps)
        n_sv_done = 0
        for rep in range(n_repetitions):
            try:
                print(f"  N={N} SV shots rep {rep+1}/{n_repetitions}")
                heff, mc = compute_heff_backend(
                    vqe, "statevector", provider_name, n_shots, which_hadamard_test, rep,
                )
                save_heff(cdir, f"H_eff_sv_shots_rep{rep}", heff)
                save_mc_ingredients(cdir, f"mc_sv_shots_rep{rep}", mc)
                n_sv_done += 1
            except Exception as e:
                print(f"  N={N} SV shots rep {rep} failed: {e}")

        # Fake backend (multiple reps)
        n_fb_done = 0
        for rep in range(n_repetitions):
            try:
                print(f"  N={N} fake_backend rep {rep+1}/{n_repetitions}")
                heff, mc = compute_heff_backend(
                    vqe, "fake_backend", provider_name, n_shots, which_hadamard_test, rep,
                )
                save_heff(cdir, f"H_eff_fake_backend_rep{rep}", heff)
                save_mc_ingredients(cdir, f"mc_fake_backend_rep{rep}", mc)
                n_fb_done += 1
            except Exception as e:
                print(f"  N={N} fake_backend rep {rep} failed: {e}")

        # Save metadata
        save_metadata(
            cdir,
            J=J, hl=hl, N=N, n_shots=n_shots,
            provider_name=provider_name,
            which_hadamard_test=which_hadamard_test,
            n_repetitions=n_repetitions,
            n_sv_shots_completed=n_sv_done,
            n_fb_completed=n_fb_done,
        )
        print(f"  N={N} done ({n_sv_done} sv_shots, {n_fb_done} fb reps) -> {cdir}")

    print(f"{'='*60}\n")
    print("Data generation complete. Use plot_dispersion.py to visualize.")


if __name__ == "__main__":
    generate_dispersion_data(J=1.0, hl=0.0, cluster_pairs=[(5, 4)])