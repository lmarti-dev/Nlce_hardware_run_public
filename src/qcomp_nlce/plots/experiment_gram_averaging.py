"""
Experiment: compare two Gram matrix estimators.

Estimator A (current): G = X_bar @ X_bar^dagger
    Average overlap matrices first, then compute Gram.

Estimator B (proposed): G = mean_r( X_r @ X_r^dagger )
    Compute Gram per-rep, then average.

These differ by the covariance of the estimator.
B has a positive-definite bias on the diagonal, but preserves
hermiticity and positive-semidefiniteness at every rep.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.qcomp_nlce.utils import (
    hamming_weight_states,
    circuit_matrix_elements,
    circuit_to_matrix,
    get_appropriate_fake_backend,
    apply_parameters,
    DATA_DIR,
)
from src.qcomp_nlce.utils.save import load_optimized_vqe
from src.qcomp_nlce.utils.folders import res_sim_dir
from src.qcomp_nlce.utils.overlap import measure_matrix_elements_perrep, reorder_matrix_elems
from src.qcomp_nlce.utils.vqe import get_optimal_n_layers
from src.qcomp_nlce.utils.hamiltonian import model_str

from data_manager import ExperimentDataManager

try:
    from fauplotstyle import style
    style()
except ImportError:
    pass


def get_true_overlap(circuit, comp_states):
    true_vals = circuit_matrix_elements(circuit, comp_states)
    true_vals = true_vals[::-1, ::-1]
    oUo = circuit_to_matrix(circuit)[0, 0]
    true_vals = oUo.conj() * true_vals
    true_vals = reorder_matrix_elems(true_vals)
    return true_vals


def run_experiment(
    J: float = 0.8,
    hl: float = 0.0,
    ladder: bool = False,
    n_qubits: int = 5,
    provider_name: str = "AQT_WORSE_2",
    n_shots: int = 200,
    n_reps: int = 10,
):
    which_model = model_str(J, hl, ladder)
    model_dir = "vqe_" + which_model

    vqe_dir = f"{n_qubits}q_{get_optimal_n_layers(n_qubits, ladder, hl)}l"
    elem_dirname = Path(res_sim_dir(), f"elem_{model_dir}/vqe/cost_offdiag/{vqe_dir}")

    hamiltonian, circuit, weights, ground_energy = load_optimized_vqe(elem_dirname)
    circuit = apply_parameters(circuit, weights[-1])

    comp_states = hamming_weight_states(n_qubits, 1)
    true_overlap = get_true_overlap(circuit, comp_states)

    backend = get_appropriate_fake_backend(n_qubits, which=provider_name)

    print(f"Measuring {n_qubits}x{n_qubits} overlap matrix, {n_reps} reps x {n_shots} shots")
    perrep_matrices = measure_matrix_elements_perrep(
        circuit,
        comp_states,
        circuit.parameters,
        backend,
        "fake_backend",
        n_shots,
        n_reps=n_reps,
    )

    return perrep_matrices, true_overlap


def analyze_and_plot(perrep_matrices, true_overlap, title_suffix=""):
    n_reps = len(perrep_matrices)
    n = perrep_matrices[0].shape[0]

    # Estimator A: average X first, then Gram
    X_bar = np.mean(perrep_matrices, axis=0)
    gram_A = X_bar @ X_bar.conj().T

    # Estimator B: Gram per rep, then average
    gram_perrep = [X @ X.conj().T for X in perrep_matrices]
    gram_B = np.mean(gram_perrep, axis=0)

    # True Gram
    gram_true = true_overlap @ true_overlap.conj().T

    # Errors
    err_A = np.abs(gram_true - gram_A)
    err_B = np.abs(gram_true - gram_B)

    # Depolarization: 1 - measured/true (element-wise, real parts)
    def depol(measured, true):
        d = np.zeros_like(measured, dtype=float)
        for i in range(n):
            for j in range(n):
                if not np.isclose(true[i, j], 0):
                    d[i, j] = np.abs(1 - measured[i, j] / true[i, j])
                else:
                    d[i, j] = np.abs(measured[i, j])
        return d

    depol_A = depol(gram_A, gram_true)
    depol_B = depol(gram_B, gram_true)

    # Condition numbers
    cond_A = np.linalg.cond(gram_A)
    cond_B = np.linalg.cond(gram_B)
    cond_true = np.linalg.cond(gram_true)

    # Eigenvalues
    eigs_A = np.sort(np.linalg.eigvalsh(gram_A))[::-1]
    eigs_B = np.sort(np.linalg.eigvalsh(gram_B))[::-1]
    eigs_true = np.sort(np.linalg.eigvalsh(gram_true))[::-1]

    print(f"\n=== Results{title_suffix} ===")
    print(f"Condition numbers: true={cond_true:.2f}, A(X_bar X_bar†)={cond_A:.2f}, B(mean X_r X_r†)={cond_B:.2f}")
    print(f"Frobenius error:   A={np.linalg.norm(err_A):.6f}, B={np.linalg.norm(err_B):.6f}")
    print(f"Max element error: A={np.max(err_A):.6f}, B={np.max(err_B):.6f}")
    print(f"Eigenvalues true:  {eigs_true}")
    print(f"Eigenvalues A:     {eigs_A}")
    print(f"Eigenvalues B:     {eigs_B}")

    # Check positive semidefiniteness
    print(f"Min eigenvalue A:  {np.min(eigs_A):.6f} ({'PSD' if np.min(eigs_A) >= -1e-10 else 'NOT PSD'})")
    print(f"Min eigenvalue B:  {np.min(eigs_B):.6f} ({'PSD' if np.min(eigs_B) >= -1e-10 else 'NOT PSD'})")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    textcolors = ["k", "w"]

    def plot_matrix(ax, mat, title):
        amat = np.abs(mat)
        im = ax.imshow(amat)
        threshold = im.norm(np.max(amat)) / 2
        for i in range(n):
            for j in range(n):
                sign = "" if mat[i, j].imag >= 0 else "-"
                s = f"{mat[i, j].real:.3f}\n{sign}{np.abs(mat[i, j].imag):.3f}"
                ax.text(j, i, s, ha="center", va="center",
                        color=textcolors[int(im.norm(amat[i, j]) < threshold)],
                        fontsize=7)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.grid(False)

    def plot_error(ax, err, title):
        im = ax.imshow(err)
        threshold = im.norm(np.max(err)) / 2
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{err[i, j]:.4f}", ha="center", va="center",
                        color=textcolors[int(im.norm(err[i, j]) < threshold)],
                        fontsize=7)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.grid(False)

    plot_matrix(axes[0, 0], gram_true, r"True $O O^\dagger$")
    plot_matrix(axes[0, 1], gram_A, r"$\bar{X}\bar{X}^\dagger$ (current)")
    plot_matrix(axes[0, 2], gram_B, r"$\overline{X_r X_r^\dagger}$ (per-rep avg)")

    plot_error(axes[1, 0], depol_A, f"Depol A, cond={cond_A:.1f}")
    plot_error(axes[1, 1], depol_B, f"Depol B, cond={cond_B:.1f}")

    # Eigenvalue comparison
    ax = axes[1, 2]
    x = np.arange(n)
    w = 0.25
    ax.bar(x - w, eigs_true, w, label="True", color="C0")
    ax.bar(x, eigs_A, w, label=r"$\bar{X}\bar{X}^\dagger$", color="C1")
    ax.bar(x + w, eigs_B, w, label=r"$\overline{X_r X_r^\dagger}$", color="C2")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Gram eigenvalues")
    ax.legend(fontsize=8)
    ax.grid(False)

    fig.suptitle(f"Gram matrix estimator comparison{title_suffix}", fontsize=12)

    return fig


if __name__ == "__main__":
    edm = ExperimentDataManager(
        "gram_averaging_comparison",
        project="nlce",
        dry_run=False,
    )

    configs = [
        {"J": 0.8, "hl": 0.0, "n_qubits": 5, "provider_name": "AQT_WORSE_2", "n_shots": 200, "n_reps": 10},
    ]

    for cfg in configs:
        suffix = f" (J={cfg['J']}, {cfg['n_qubits']}q, {cfg['provider_name']}, {cfg['n_shots']}shots x {cfg['n_reps']}reps)"
        perrep_matrices, true_overlap = run_experiment(**cfg)
        fig = analyze_and_plot(perrep_matrices, true_overlap, title_suffix=suffix)
        edm.save_figure(fig, f"gram_comparison_J{cfg['J']}_{cfg['n_qubits']}q_{cfg['provider_name']}", fig_shape="double-size", add_timestamp=False)

    plt.show()
