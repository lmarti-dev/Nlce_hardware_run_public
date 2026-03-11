from qcomp_nlce.utils import (
    get_hva_vqe,
    backend_sample,
    get_appropriate_fake_backend,
    TFIM,
    apply_parameters,
    get_counts_from_res,
    circuits_to_sampler_payload,
    trace_out_estimation,
)
import numpy as np
from qiskit_aqt_provider.primitives import AQTSampler

from qiskit import generate_preset_pass_manager


J = 0.8
hl = 0
n_qubits = 5
n_shots = 2000
provider_name = "AQT"


tfim = TFIM(n_qubits, 1, J, longitudinal_field=hl)

circ = get_hva_vqe(hamiltonian=tfim, n_layers=2, unique_weights=True)
backend = get_appropriate_fake_backend(n_qubits, provider_name)

ap_circ = apply_parameters(circ, parameters=np.random.rand((len(circ.parameters))))
ap_circ.append(ap_circ.inverse(), range(n_qubits))
ap_circ.measure_all()


pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
isa_circuit = pm.run(ap_circ)
sampler = AQTSampler(backend)
job = sampler.run([isa_circuit], parameter_values=None, shots=n_shots)
res = job.result()

counts = get_counts_from_res(res, n_qubits)

z_vals = []
z_stds = []

for jj in range(tfim.num_qubits):
    # set last arg to 1 to enable mitigation.
    val, std = trace_out_estimation(counts, [jj], -1)
    z_vals.append(val)
    z_stds.append(std)


depol_rate = np.mean(z_vals)
depol_rate_unc = np.std(z_vals)
# there should be an std on the std, but I don't think it should make too much of a diff.

print(f"Depol rate: {depol_rate:.4f}, std: {depol_rate_unc:.4f}")
