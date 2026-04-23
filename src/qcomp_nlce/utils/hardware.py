from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, SamplerV2
from qiskit_ibm_runtime.base_runtime_job import BaseRuntimeJob
from qiskit.primitives import EstimatorResult, PrimitiveResult
from qiskit_ibm_transpiler import generate_ai_pass_manager
from iqm.iqm_client import CircuitCompilationOptions
from iqm.iqm_client.models import HeraldingMode, DDMode

from qiskit.providers.fake_provider import GenericBackendV2
from src.qcomp_nlce.utils.iqm_utils import (
    IQMFakePerfect,
    IQMFakePrettyGood,
    IQMFakeReallyGood,
)

from src.qcomp_nlce.utils.backends import depolarizing_backend

try:
    from iqm.qiskit_iqm.fake_backends.fake_deneb import (
        IQMFakeDeneb as IQMSmallFakeBackend,
    )
except Exception:
    from iqm.qiskit_iqm.fake_backends.fake_adonis import (
        IQMFakeAdonis as IQMSmallFakeBackend,
    )
from iqm.qiskit_iqm.fake_backends.fake_garnet import (
    IQMFakeGarnet,
)
from iqm.qiskit_iqm.fake_backends.fake_aphrodite import IQMFakeAphrodite
from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMFakeBackend
from iqm.qiskit_iqm import IQMProvider, IQMBackend, IQMJob


import io

from qiskit_ibm_runtime.fake_provider import (
    FakeLagosV2,
    FakeWashingtonV2,
    FakeMelbourneV2,
    FakeKolkataV2,
)
from typing import Literal, Union

from qiskit import transpile
from qiskit.primitives.base import SamplerResult

try:
    from qiskit_aqt_provider import AQTProvider
    from qiskit_aqt_provider.primitives import AQTSampler, AQTEstimator
    from qiskit_aqt_provider.aqt_resource import AQTResource
    from qiskit_aqt_provider.aqt_resource import OfflineSimulatorResource
except ModuleNotFoundError:
    print("qiskit_aqt_provider not found")
    # watch this

    class AQTResource: ...

    class AQTSampler: ...

    class AQTEstimator: ...

    class AQTProvider: ...

    class OfflineSimulatorResource: ...


try:
    raise ValueError
    from mqp.qiskit_provider import MQPProvider, MQPBackend

except Exception:
    print("mqp.qiskit_provider not found")
    # https://munich-quantum-software-stack.github.io/MQSS-Interfaces/qiskit/user_guide/getting_started/#usage
    try:
        # MQV people forcing me to write the nastiest code ever
        from mqss.qiskit_adapter import MQSSQiskitAdapter
        from mqss.qiskit_adapter.backend import MQSSQiskitBackend

        MQPProvider = MQSSQiskitAdapter
        MQPBackend = MQSSQiskitBackend
    except ModuleNotFoundError as e:
        print("Neither MQSS nor MQP working")
        raise e


class NotABackend: ...


from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

from src.qcomp_nlce.utils.circuit import (
    isa_circuit_param_dict,
    parameters_exist,
    circ_stats,
)

try:
    from src.qcomp_nlce.utils.aqt_utils import (
        aqt_better_fake_backend,
        aqt_pauli_damping_backend,
        aqt_pauli_fake_backend,
        OfflineSimulatorResourceBetter,
    )
except ModuleNotFoundError:
    aqt_better_fake_backend: lambda x: ...
    aqt_pauli_damping_backend: lambda x: ...
    aqt_pauli_fake_backend: lambda x: ...

    class OfflineSimulatorResourceBetter: ...


from src.qcomp_nlce.utils.hamiltonian import get_tfim_ratio, get_tfim_xx, HAM_NAME_VAR
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend


BACKEND_TYPES = Literal["statevector", "fake_backend", "real_backend"]

BACKEND_ALIAS = (
    Literal["statevector"] | FakeBackendV2 | IBMBackend | MQPBackend | IQMBackend
)

PROVIDER_ALIAS = Literal[
    "IBM",
    "AQT",
    "EQE",
    "IQM",
    "QEXA",
    "IBM_GENERIC",
    "IQM_PERFECT",
    "IQM_REALLYGOOD",
    "IQM_PRETTYGOOD",
    "AQT_BETTER",
]

BACKEND_LABELS = {
    "fake_backend": "Noisy",
    "real_backend": "QPU",
    "statevector": "SV",
    "ed": "ED",
    "analytic": "Analytic",
}

BACKEND_COLORS = {
    "fake_backend": "#228848",
    "real_backend": "#E87722",
    "statevector": "#971B2F",
    "ed": "#5FC5EC",
    "analytic": "#5F7CA3",
}


RUNS_POSTFIX = Literal["", "RUN2", "RUN3"]

[
    "#228848",
    "#E87722",
    "#0061A0",
    "#971B2F",
    "#5F7CA3",
    "#AAC3D1",
    "#FFD271",
    "#5FC5EC",
    "#89CD79",
    "#DC697C",
    "#41748D",
    "#041E42",
]


def get_aqt_backend(n_qubits: int) -> MQPBackend:
    if n_qubits > 14:
        raise ValueError(f"Maximum number of qubits is 12, requested: {n_qubits}")
    token = get_token("LRZ")
    provider = MQPProvider(token)
    backend = provider.get_backend("AQT20")
    return backend


def get_qexa_backend(n_qubits: int) -> MQPBackend:
    if n_qubits > 20:
        raise ValueError(f"Maximum number of qubits is 20, requested: {n_qubits}")
    token = get_token("LRZ")
    provider = MQPProvider(token)
    backend = provider.get_backend("QExa20")
    return backend


def get_eqe_backend(n_qubits: int) -> MQPBackend:
    if n_qubits > 53:
        raise ValueError(f"Maximum number of qubits is 53, requested: {n_qubits}")
    token = get_token("LRZ")
    provider = MQPProvider(token)
    backend = provider.get_backend("EQE1")
    return backend


def get_iqm_backend(n_qubits: int) -> "IQMBackend":
    if n_qubits < 20:
        device_name = "garnet"
    else:
        device_name = "emerald"
    url = f"https://resonance.meetiqm.com/"
    return IQMProvider(
        url, token=get_token("IQM"), quantum_computer=device_name
    ).get_backend()


def get_ibm_backend(n_qubits: int) -> IBMBackend:
    token = get_token("IBM")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    real_backend = service.least_busy(
        simulator=False, operational=True, min_num_qubits=n_qubits
    )
    return real_backend


def get_backend(n_qubits: int, which: PROVIDER_ALIAS) -> Union[IBMBackend, MQPBackend]:
    if which == "IBM":
        return get_ibm_backend(n_qubits)
    elif which == "AQT":
        return get_aqt_backend(n_qubits)
    elif which == "IQM":
        return get_iqm_backend(n_qubits)
    elif which == "QEXA":
        return get_qexa_backend(n_qubits)
    elif which == "EQE":
        return get_eqe_backend(n_qubits)
    else:
        raise ValueError(f"Requested invalid backend: {which}")


def get_appropriate_fake_backend(
    n_qubits: int, which: PROVIDER_ALIAS
) -> Union[FakeBackendV2, AQTResource]:
    if which == "IBM":
        if n_qubits <= 7:
            return FakeLagosV2()
        elif n_qubits <= 14:
            return FakeMelbourneV2()
        elif n_qubits <= 27:
            return FakeKolkataV2()
        else:
            return FakeWashingtonV2()
    elif which == "AQT":
        if n_qubits > 12:
            raise ValueError(f"Maximum number of qubits is 12, requested: {n_qubits}")
        return AQTProvider("ANY_TOKEN_WORKS_FOR_OFFLINE_SIM").get_backend(
            "offline_simulator_noise"
        )
    elif which == "AQT_PERFECT":
        if n_qubits > 12:
            raise ValueError(f"Maximum number of qubits is 12, requested: {n_qubits}")
        return AQTProvider("ANY_TOKEN_WORKS_FOR_OFFLINE_SIM").get_backend(
            "offline_simulator_no_noise"
        )
    elif which == "IQM" or which == "QEXA" or which == "EQE":
        if which == "EQE":
            # That's the architecture
            return IQMFakeAphrodite()
        n_qubits += 2
        # counting ancillas and resonators!
        if n_qubits <= 5:
            return IQMSmallFakeBackend()
        elif n_qubits <= 20:
            return IQMFakeGarnet()
        else:
            return IQMFakeAphrodite()
    elif which == "IBM_GENERIC":
        # I should've made the backend calling
        # after the circuit is created
        # because you might add ancillas and such
        # and this isn't taken into account here
        n_qubits += 4
        return GenericBackendV2(n_qubits)
    elif which == "IBM_PERFECT":
        n_qubits += 4
        backend = GenericBackendV2(n_qubits)
        backend.set_options(noise_model=None)
        return backend
    elif which == "IQM_PRETTYGOOD":
        return IQMFakePrettyGood()
    elif which == "IQM_REALLYGOOD":
        return IQMFakeReallyGood()
    elif which == "IQM_PERFECT":
        return IQMFakePerfect()
    elif "AQT_BETTER" in which:
        if len(which.split("_")) == 3:
            n_improve = float(which.split("_")[-1])
        else:
            n_improve = 2
        return aqt_better_fake_backend(n_improve)

    elif "AQT_DAMPING" in which:
        return aqt_pauli_damping_backend(0.001, 0.001)

    elif "AQT_PAULI" in which:
        return aqt_pauli_fake_backend(0.003)

    elif "AQT_WORSE" in which:
        if len(which.split("_")) == 3:
            n_improve = float(which.split("_")[-1])
        else:
            n_improve = 2
        return aqt_better_fake_backend(1 / n_improve)
    elif "AER" in which:
        # format AER_10_10
        if len(which.split("_")) == 3:
            fac1 = float(which.split("_")[1])
            fac2 = float(which.split("_")[2])
        else:
            raise ValueError(f"Need to specify the facs for AER backend, got: {which}")

        return depolarizing_backend(1 / fac1, 1 / fac2)

    else:
        raise ValueError(f"Expected a provider, got: {which}")


def mean_std_tuples_from_res(
    res: EstimatorResult | PrimitiveResult,
) -> list[tuple[complex, float]]:
    out = []
    if isinstance(res, EstimatorResult):
        for m, s in zip(res.values, res.metadata):
            out.append((m, s["variance"] ** (0.5)))
    elif isinstance(res, PrimitiveResult):
        for k in res:
            for m, s in zip(k.data.evs, k.data.stds):
                out.append((m, s))
    return out


def pending_jobs(n_qubits: int, provider_name: PROVIDER_ALIAS) -> int:
    backend = get_backend(n_qubits, provider_name)
    print(f"There are: {backend.num_pending_jobs} pending jobs on {provider_name}")
    return backend.num_pending_jobs


def compute_mean_circ_stats(circs: list | QuantumCircuit, verbose: bool = False):
    if isinstance(circs, QuantumCircuit):
        circ_list = [circs]
    elif isinstance(circs, list):
        circ_list = circs

    else:
        raise ValueError(f"circ is neither QuantumCircuit nor list, but: {type(circs)}")

    d_arr = []
    n_gates_arr = []
    nlg_arr = []
    for isa_c in circ_list:
        depth, n_gates, n_non_local_gates = circ_stats(isa_c)
        d_arr.append(depth)
        n_gates_arr.append(n_gates)
        nlg_arr.append(n_non_local_gates)

    mean_depth = np.mean(d_arr)
    mean_n_gates = np.mean(n_gates_arr)
    mean_nlg = np.mean(nlg_arr)

    if verbose:
        print(
            f"depth {mean_depth}, total gates: {mean_n_gates}, nonlocal (k>=2) gates: {mean_nlg} (mean of {len(circ_list)} circuits(s))"
        )

    return mean_depth, mean_n_gates, mean_nlg


def is_hardware_run(which_backend: str, debug: bool = False):
    if debug and which_backend == "fake_backend":
        return True
    return which_backend == "real_backend"


def compile_circuit_for_backend(
    circuit: QuantumCircuit,
    backend: BACKEND_ALIAS,
    options: dict = {},
    ai: bool = False,
):
    # IQMBackend, IQMFakeBackend
    if isinstance(backend, (IQMBackend, IQMFakeBackend)):
        initial_layout = list(range(circuit.num_qubits))
        isa_circuit = transpile(
            circuit,
            backend=backend,
            initial_layout=initial_layout,
            optimization_level=3,
            coupling_map=backend.coupling_map,
            scheduling_method="move_routing_rz_optimization_ignores_barriers",
        )

        # isa_circuit = transpile_to_IQM(
        #     circuit,
        #     backend=backend,
        #     optimization_level=3,
        #     perform_move_routing=True,
        #     optimize_single_qubits=True,
        #     remove_final_rzs=True,
        #     initial_layout=initial_layout,
        #     coupling_map=backend.coupling_map,
        # )
    else:
        if not ai:
            pm = generate_preset_pass_manager(backend=backend, **options)
        else:
            if "optimization_level" not in options:
                options["optimization_level"] = 3

            # never put coupling_map instead of backend
            # it will generate better circuit
            # but they will be broken
            pm = generate_ai_pass_manager(
                ai_optimization_level=3,
                ai_layout_mode="optimize",
                backend=backend,
                **options,
            )

        isa_circuit = pm.run(circuit)
    compute_mean_circ_stats(isa_circuit, True)
    return isa_circuit


def get_backend_name(backend: BACKEND_ALIAS) -> str:
    if backend == "statevector":
        return backend
    else:
        return backend.name


def circuits_to_sampler_payload(
    backend,
    circuit: Union[QuantumCircuit, list[QuantumCircuit]],
    parameters: np.ndarray,
    which: str = "v2",
):
    if isinstance(circuit, QuantumCircuit):
        circuit = [circuit]

    if which == "v0":
        return circuit

    options = {
        "optimization_level": 3,
        "initial_layout": list(range(circuit[0].num_qubits)),
    }

    isa_circuit = [compile_circuit_for_backend(c, backend, options) for c in circuit]
    if which == "v1":
        payload = []
        if isinstance(isa_circuit, QuantumCircuit):
            isa_circuit = [isa_circuit]
        for c in isa_circuit:
            # no parameters allowed!
            payload.append(c)
        return payload

    elif which == "v2":
        payload = []

        if isinstance(isa_circuit, QuantumCircuit):
            isa_circuit = [isa_circuit]

        for idx in range(len(isa_circuit)):
            toadd = []
            if parameters_exist(parameters):
                isa_params = isa_circuit_param_dict(
                    isa_circuit[idx], circuit[idx], parameters
                )
                toadd.append(isa_params)
            payload.append((isa_circuit[idx], *toadd))

    return payload


# from https://quantumcomputing.stackexchange.com/questions/25672/remove-inactive-qubits-from-qiskit-circuit
def count_gates(qc: QuantumCircuit):
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def remove_idle_wires(qc: QuantumCircuit):
    qc_out = qc.copy()
    gate_count = count_gates(qc_out)
    for qubit, count in gate_count.items():
        if count == 0:
            qc_out.qubits.remove(qubit)
    return qc_out


def get_provider_version(provider_name: PROVIDER_ALIAS) -> Literal["v1", "v2"]:
    if provider_name in ("AQT",):
        return "v1"
    else:
        return "v2"


def circuits_to_estimator_payload(
    backend,
    circuit: Union[QuantumCircuit, list[QuantumCircuit]],
    parameters: np.ndarray,
    observables: Union[list[SparsePauliOp], SparsePauliOp],
    which: str = "v2",
) -> list[tuple[QuantumCircuit, dict]]:
    if isinstance(circuit, QuantumCircuit):
        circuit = [circuit]

    isa_circuit = [
        compile_circuit_for_backend(c, backend, {"optimization_level": 3})
        for c in circuit
    ]
    if which == "v1":
        payload = [[], [], []]
        # assume that for each circuit you want to measure each parameters
        if isinstance(isa_circuit, QuantumCircuit):
            isa_circuit = [isa_circuit]
        if isinstance(observables, SparsePauliOp):
            observables = [observables]
        for c in isa_circuit:
            for o in observables:
                payload[0].append(c)
                isa_o = o.apply_layout(c.layout)
                payload[1].append(isa_o)
                if parameters_exist(parameters):
                    payload[2].append(parameters)
        return payload

    elif which == "v2":
        payload = []

        if isinstance(isa_circuit, QuantumCircuit):
            isa_circuit = [isa_circuit]

        for idx in range(len(isa_circuit)):
            toadd = []
            if observables is not None:
                toadd.append(
                    [o.apply_layout(isa_circuit[idx].layout) for o in observables]
                )
            if parameters_exist(parameters):
                isa_params = isa_circuit_param_dict(
                    isa_circuit[idx], circuit[idx], parameters
                )
                toadd.append(isa_params)
            payload.append((isa_circuit[idx], *toadd))

    return payload


def get_backend_sampler(
    backend: BACKEND_ALIAS,
) -> SamplerV2 | AQTSampler:
    if isinstance(backend, AQTResource) or backend.name == "AQT20":
        sampler = AQTSampler(backend)

    elif isinstance(
        backend,
        (
            IBMBackend,
            FakeBackendV2,
            IQMFakeBackend,
            IQMBackend,
            MQPBackend,
            GenericBackendV2,
        ),
    ):
        sampler = SamplerV2(mode=backend)
    else:
        raise ValueError(f"Unexpected backend: type {type(backend)}")
    return sampler


def backend_sample(
    backend: BACKEND_ALIAS,
    circuit: QuantumCircuit,
    parameters: np.ndarray,
    n_shots: int,
) -> BaseRuntimeJob:
    sampler = get_backend_sampler(backend)

    if isinstance(
        backend,
        (IBMBackend, FakeBackendV2, GenericBackendV2),
    ):
        payload = circuits_to_sampler_payload(backend, circuit, parameters, which="v2")

        job = sampler.run(
            payload,
            shots=int(n_shots),
        )
    elif "name" in backend.__dict__ and (backend.name == "EQE1"):
        # eqe1 is compiled server-side I believe
        # but the server compilation is brittle
        # so it's better to compile here and tell the server
        # not to touch anything!
        server_side = False
        if isinstance(circuit, list):
            payload = circuit
        else:
            payload = [circuit]
        if server_side:
            no_modify = False
            isa_payload = payload
        else:
            initial_layout = list(range(payload[0].num_qubits))
            isa_payload = transpile(
                payload,
                backend=backend,
                initial_layout=initial_layout,
                optimization_level=3,
            )
            no_modify = True
        job = backend.run(isa_payload, shots=n_shots, no_modify=no_modify)

    elif isinstance(backend, AQTResource) or backend.name == "AQT20":
        payload = circuits_to_sampler_payload(backend, circuit, parameters, which="v0")
        sampler.set_transpile_options(optimization_level=3)
        job = sampler.run(payload, parameter_values=None, shots=n_shots)

    elif isinstance(backend, (IQMBackend, IQMFakeBackend, MQPBackend)):
        payload = circuits_to_sampler_payload(backend, circuit, parameters, which="v1")

        job = backend.run(
            payload,
            shots=int(n_shots),
            circuit_compilation_options=CircuitCompilationOptions(
                heralding_mode=HeraldingMode.ZEROS,
                dd_mode=DDMode.ENABLED,
            ),
        )

    elif isinstance(backend, (NotABackend)):
        initial_layout = list(range(circuit.num_qubits))
        isa_circuits = transpile(
            circuit,
            backend=backend,
            initial_layout=initial_layout,
            optimization_level=3,
            coupling_map=backend.coupling_map,
        )
        job = backend.run(isa_circuits, shots=n_shots)
    else:
        raise ValueError(f"Unexpected backend: type {type(backend)}")
    return job


def backend_estimate(
    backend: BACKEND_ALIAS,
    observables: Union[SparsePauliOp, list[SparsePauliOp]],
    circuit: Union[QuantumCircuit | list[QuantumCircuit]],
    parameters: np.ndarray,
    n_shots: int,
) -> BaseRuntimeJob:
    # I believe only AQT now runs on V1, QExa can use V2 prims. (except the passmanager lol)
    # need MQPBackend in there because the AQT real backend is provided by MQP
    if isinstance(backend, (AQTResource, MQPBackend)) or backend.name == "AQT20":
        # can't replace it with EstimatorV1
        estimator = AQTEstimator(backend)
        estimator.set_transpile_options(optimization_level=3)
        (circuit, observables, parameters) = circuits_to_estimator_payload(
            backend, circuit, parameters, observables, which="v1"
        )
        # lose yourself in the moment, the music
        # the moment you own it you never let it go (oh)
        # you only get 2000 shots
        # if isinstance(backend, AQTResource):
        #     n_shots = min(n_shots, 2000)
        # I've commented out this part, to avoid getting
        # an unexpected number of shots
        kwargs = {
            "circuits": circuit,
            "observables": observables,
            "shots": n_shots,
        }
        if parameters != []:
            kwargs.update({"parameters": parameters})
        if not isinstance(
            backend, (OfflineSimulatorResource, OfflineSimulatorResourceBetter)
        ):
            kwargs.update({"qasm3": True, "queued": True})
        job = estimator.run(**kwargs)
    else:
        options = {}
        # avoid the warning when using a fakebackend
        if not isinstance(backend, (FakeBackendV2, IQMFakeBackend)):
            if "resilience_level" not in options.keys():
                options.update({"resilience_level": 2})
        estimator = EstimatorV2(backend, options=options)

        payload = circuits_to_estimator_payload(
            backend, circuit, parameters, observables
        )

        if n_shots == -1:
            precision = 1e-8
        else:
            precision = 1 / np.sqrt(n_shots)
        job = estimator.run(
            payload,
            precision=precision,
        )
    return job


def zxy_basis_hamiltonian_circuits(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
) -> list[QuantumCircuit]:
    n_qubits = hamiltonian.num_qubits
    # ZZZ
    z_circ = circuit.copy()
    z_circ.measure_all()

    y_circ = circuit.copy()
    y_circ.h(range(circuit.num_qubits))
    y_circ.s(range(circuit.num_qubits))
    y_circ.measure_all()

    # XX XX XX

    x_circ = circuit.copy()
    x_circ.h(range(circuit.num_qubits))
    x_circ.measure_all()

    return [z_circ, x_circ, y_circ]


def get_token(which: Literal["IBM", "LRZ"]):
    home = Path(__file__).parent
    if which == "IBM":
        token_fpath = Path(home, "../files/ibm_token.dat")
    elif which == "LRZ":
        token_fpath = Path(home, "../files/lrz_token.dat")
    elif which == "IQM":
        token_fpath = Path(home, "../files/iqm_token.dat")
    else:
        raise ValueError(f"Unallowed value: {which}")
    token = io.open(
        token_fpath,
        "r",
        encoding="utf8",
    ).read()

    return token


def sampler_result_to_json(sampler: SamplerResult):
    d = {}

    d["quasi_dists"] = [
        dict(sampler.quasi_dists[m]) for m in range(len(sampler.quasi_dists))
    ]
    d["metadata"] = sampler.metadata

    return d


def encode_sampler_result(job):
    res = job.result()
    if isinstance(res, SamplerResult):
        return sampler_result_to_json(res)
    else:
        return res
