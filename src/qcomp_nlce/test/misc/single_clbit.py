from qiskit import QuantumCircuit
from qiskit.circuit import Clbit, ClassicalRegister
from qiskit.primitives import StatevectorSampler


def add_reg():
    circuit = QuantumCircuit(1)

    clbits = ClassicalRegister(1, name="c0")
    circuit.add_register(clbits)

    circuit.measure(0, 0)
    return circuit


def add_clbit():
    circuit = QuantumCircuit(1)
    reg = ClassicalRegister(1, name="c0")
    clbits = Clbit(register=reg, index=0)
    circuit.add_bits([clbits])

    circuit.measure(0, 0)
    return circuit


def counts_from_sv(circuit):
    sv = StatevectorSampler()

    job = sv.run([circuit], shots=1000)

    data = job.result()[0].data

    key = list(data.keys())[0]

    counts = data[key].get_counts()

    print(counts)


circ_bit = add_clbit()
circ_reg = add_reg()

circuits_match = circ_bit.data == circ_reg.data

assert circuits_match  # passes

counts_from_sv(circ_reg)  # passes
counts_from_sv(circ_bit)  # fails
