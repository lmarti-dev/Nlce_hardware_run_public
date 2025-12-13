from mqt.predictor.rl import Predictor as RL_Predictor
from mqt.bench.targets import get_device


from bqskit.ir.gates.parameterized.u1q import U1qGate
from qiskit.qasm2 import CustomInstruction
from qiskit.circuit.library import RGate

import qiskit.qasm2

U1qGate._qasm_name = "r"

qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS = qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS + (
    CustomInstruction("r", 2, 1, RGate, builtin=True),
)


def train_iqm_model():
    device = get_device("iqm_crystal_20")
    rl_pred = RL_Predictor(device=device, figure_of_merit="expected_fidelity")
    rl_pred.train_model(timesteps=1000)
