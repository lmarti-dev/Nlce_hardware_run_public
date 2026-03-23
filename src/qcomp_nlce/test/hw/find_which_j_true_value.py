from src.qcomp_nlce.utils import TFIM, get_ground_energy


hw = {
    5: {
        "error": 0.007152582548913864,
        "energy": -5.611642920830822,
    },
    4: {
        "error": 0.0026626232462961933,
        "energy": -4.472847300786651,
    },
}
for n_qubits in (4, 5):
    tfim8 = get_ground_energy(TFIM(n_qubits, h=1, J=0.8, longitudinal_field=0.1))
    tfim5 = get_ground_energy(TFIM(n_qubits, h=1, J=0.5, longitudinal_field=0.1))
    tfimno = get_ground_energy(TFIM(n_qubits, h=1, J=0.8, longitudinal_field=0.0))
    print(
        f"{n_qubits}: {hw[n_qubits]['energy']:.5f}, true8 {tfim8:.5f}, true5 {tfim5:.5f} tfimno: {tfimno:.5f}"
    )
