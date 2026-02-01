from typing import (
    TYPE_CHECKING,
    TypeVar,
)
from uuid import UUID
from qiskit.transpiler import Target
from qiskit_aer import AerSimulator, noise
from typing_extensions import override

from qiskit_aqt_provider import api_client
from qiskit_aqt_provider.api_client import models as api_models
from qiskit_aqt_provider.aqt_job import AQTJob
from qiskit_aqt_provider.circuit_to_aqt import aqt_to_qiskit_circuit
from qiskit_aqt_provider.aqt_resource import (
    AQTResource,
    SimulatorJob,
    qubit_states_from_int,
)

from qiskit_aqt_provider.aqt_resource import (
    AQTResource,
    make_transpiler_target,
)
from qiskit_aqt_provider.api_client import Resource
from qiskit_aqt_provider.aqt_provider import AQTProvider, OfflineSimulator


TargetT = TypeVar("TargetT", bound=Target)


def aqt_better_fake_backend(improve_noise: float):
    simulator = OfflineSimulator(
        id="offline_simulator_noise",
        name="Offline noisy simulator (but better than actual)",
        noisy=True,
    )

    provider = AQTProvider("ANY_ACCESS_TOKEN_WORKS_FOR_OFFLINE_USE")
    backend = OfflineSimulatorResourceBetter(
        provider=provider,
        resource_id=Resource(
            workspace_id="default",
            resource_id=simulator.id,
            resource_name=simulator.name,
            resource_type="offline_simulator",
            available_qubits=20,
        ),
        with_noise_model=True,
        improve_noise=improve_noise,
    )
    backend._target = make_transpiler_target(Target, None)
    return backend


class OfflineSimulatorResourceBetter(AQTResource):
    """AQT-compatible offline simulator resource.

    Offline simulators expose the same interface and restrictions as hardware backends. If
    `with_noise_model` is true, a noise model approximating that of AQT hardware backends is used.

    .. tip::
      The simulator backend is provided by
      `Qiskit Aer <https://qiskit.github.io/qiskit-aer/>`_.
      The Qiskit Aer resource is exposed for detailed detuning as the
      ``OfflineSimulatorResource.simulator`` attribute.
    """

    def __init__(
        self,
        provider: "AQTProvider",
        resource_id: api_client.Resource,
        with_noise_model: bool,
        improve_noise: float,
    ) -> None:
        """Initialize an offline simulator resource.

        Args:
            provider: Qiskit provider that owns this backend.
            resource_id: identification of the offline simulator resource.
            with_noise_model: whether to configure a noise model in the simulator backend.
            available_qubits: size of the quantum register to simulate.
        """
        assert resource_id.resource_type == "offline_simulator"  # noqa: S101

        super().__init__(
            provider,
            resource_id=resource_id,
        )

        self.jobs: dict[UUID, SimulatorJob] = {}

        if not with_noise_model:
            noise_model = None
        else:
            # the transpiler lowers all operations to the gate set supported by the AQT API,
            # not to the resource target's one.
            noise_model = noise.NoiseModel(basis_gates=["r", "rz", "rxx"])
            noise_model.add_all_qubit_quantum_error(
                noise.depolarizing_error(0.003 / improve_noise, 1), ["r"]
            )
            noise_model.add_all_qubit_quantum_error(
                noise.depolarizing_error(0.01 / improve_noise, 2), ["rxx"]
            )

        self.simulator = AerSimulator(method="statevector", noise_model=noise_model)

    @property
    def with_noise_model(self) -> bool:
        """Whether the simulator includes a noise model."""
        return self.simulator.options.noise_model is not None

    @override
    def submit(self, job: AQTJob) -> UUID:
        """Submit a job for execution on the simulator.

        .. tip:: This is a low-level method. Use the :meth:`AQTResource.run()` method
            to submit a job and retrieve a :class:`AQTJob <qiskit_aqt_provider.aqt_job.AQTJob>`
            handle.

        Args:
            job: quantum circuits job to submit to the simulator.

        Returns:
            Unique identifier of the simulator job.
        """
        # Use the API payload such that the memory map is the same as that
        # of the remote devices.
        circuits = [
            aqt_to_qiskit_circuit(circuit.quantum_circuit, circuit.number_of_qubits)
            for circuit in job.api_submit_payload.payload.circuits
        ]

        sim_job = SimulatorJob(
            job=self.simulator.run(circuits, shots=job.options.shots),
            circuits=job.circuits,
            shots=job.options.shots,
        )

        # The Aer job is freshly created above, so its ID is unique
        # among the keys in self.jobs.
        self.jobs[sim_job.job_id] = sim_job

        return sim_job.job_id

    @override
    def result(self, job_id: UUID) -> api_models.JobResponse:
        """Query results for a simulator job.

        .. tip:: This is a low-level method. Use
            :meth:`AQTJob.result() <qiskit_aqt_provider.aqt_job.AQTJob.result>` instead.

        Args:
            job_id: identifier of the job to retrieve results for.

        Returns:
            AQT API payload with the job results.

        Raises:
            UnknownJobError: ``job_id`` doesn't correspond to a simulator job on this resource.
        """
        if (job := self.jobs.get(job_id)) is None:
            raise api_models.UnknownJobError(str(job_id))

        qiskit_result = job.job.result()

        results: dict[str, list[list[int]]] = {}
        for circuit_index, circuit in enumerate(job.circuits):
            samples: list[list[int]] = []

            # Use data()["counts"] instead of get_counts() to access the raw counts
            # in hexadecimal format.
            counts: dict[str, int] = qiskit_result.data(circuit_index)["counts"]

            for hex_state, occurrences in counts.items():
                samples.extend(
                    [
                        qubit_states_from_int(int(hex_state, 16), circuit.num_qubits)
                        for _ in range(occurrences)
                    ]
                )

            results[str(circuit_index)] = samples

        return api_models.Response.finished(
            job_id=job_id,
            workspace_id=self.resource_id.workspace_id,
            resource_id=self.resource_id.resource_id,
            results=results,
        )
