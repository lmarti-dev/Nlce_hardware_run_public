import numpy as np


# def Expectation(Hamiltonian, states):
#     return states.conj().T @ Hamiltonian @ states


# def H_effective(hamiltonian, ind_1, gs_unitary, one_qp_unitary):

#     gs_estimate = Expectation(hamiltonian, gs_unitary)
#     H_HVA = Expectation(hamiltonian, one_qp_unitary)

#     one_qp_states = one_qp_unitary[ind_1, :]
#     B = one_qp_states[ind_1, :].conj().T * (one_qp_states[ind_1, :] * one_qp_states[ind_1, :].conj().T)^(-1/2)
#     H_effective = B.conj().T @ H_HVA @ B - (np.eye(len(ind_1)), len(ind_1), len(ind_1)) * gs_estimate

#     return H_effective


# def H_effective_pcat(hamiltonian, ind_1, gs_unitary, one_qp_unitary,overlap,  gs_expectation, one_qp_expectation):

#     N = one_qp_expectation.shape[0]
#     one_qp_states = overlap
#     Help_matrix = (gs_unitary/gs_unitary[0,0] * one_qp_states[0:0, :]) 

#     A = one_qp_states - Help_matrix 
#     A = A[ind_1, :]
#     S_B = A
#     B = S_B.conj().conj().T @ (S_B @ S_B.conj().T)^(-1/2)

#     H_effective = B.conj().T * one_qp_expectation * B - np.eye(N) @ gs_expectation

#     return H_effective




def H_effective(gs_expectation, one_qp_expectation, overlap):
    
   # Get dimension from one_qp_expectation matrix
   N = one_qp_expectation.shape[0]

   # Compute V¹ = X̃†(X̃X̃†)^(-1/2) (Eq. 45)
   overlap_product = overlap @ overlap.conj().T
   B = overlap.conj().T @ np.linalg.inv(np.sqrtm(overlap_product))

   # Apply PCAT transformation (Eq. 47)
   H_effective = B.conj().T @ one_qp_expectation @ B - np.eye(N) @ gs_expectation

   return H_effective