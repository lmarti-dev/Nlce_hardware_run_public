"""Compute H_effective_ed and H_effective_ed_LF for a TFIM with
user-specified parameters and print the matrices."""

import numpy as np
from src.qcomp_nlce.utils import TFIM
from src.qcomp_nlce.utils.effective_hamiltonian import (
    H_effective_ed,
    H_effective_ed_LF,
)


N = 4
h = 1.0
J = 0.5
hl = 0.1

ham = TFIM(N, h=h, J=J, longitudinal_field=hl)
ham_ref = TFIM(N, h=h, J=0.0, longitudinal_field=hl)

H_eff = H_effective_ed(ham)
H_eff_LF = H_effective_ed_LF(ham, ham_ref)

np.set_printoptions(precision=6, suppress=True, linewidth=200)

print(f"N = {N}, h = {h}, J = {J}, hl = {hl}")
print()
print("H_effective_ed:")
print(H_eff)
print()
print("H_effective_ed_LF:")
print(H_eff_LF)
print()
print("Eigenvalues H_effective_ed:   ", np.sort(np.linalg.eigvalsh(H_eff)))
print("Eigenvalues H_effective_ed_LF:", np.sort(np.linalg.eigvalsh(H_eff_LF)))
