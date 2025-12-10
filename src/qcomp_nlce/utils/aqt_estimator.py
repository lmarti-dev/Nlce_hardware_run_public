from qiskit.primitives.backend_estimator_v2 import BackendEstimatorV2


class AQTEstimatorV2(BackendEstimatorV2):
    """:class:`BaseEstimatorV1 <qiskit.primitives.BaseEstimatorV1>` primitive for AQT backends."""

    _backend: AnyAQTResource

    def __init__(
        self,
        backend: AnyAQTResource,
        options: Optional[dict[str, Any]] = None,
        abelian_grouping: bool = True,
        skip_transpilation: bool = False,
    ) -> None:
        """Initialize an ``Estimator`` primitive using an AQT backend.

        See :class:`AQTSampler <qiskit_aqt_provider.primitives.sampler.AQTSampler>` for
        examples configuring run options.

        Args:
            backend: AQT resource to evaluate circuits on.
            options: options passed to through to the underlying
              :class:`BackendEstimator <qiskit.primitives.BackendEstimator>`.
            abelian_grouping:  whether the observable should be grouped into commuting parts.
            skip_transpilation: if :data:`True`, do not transpile circuits
              before passing them to the execution backend.
        """
        # Signal the transpiler to disable passes that require bound
        # parameters.
        # This allows the underlying sampler to apply most of
        # the transpilation passes, and cache the results.
        mod_backend = copy(backend)
        mod_backend._target = make_transpiler_target(
            transpiler_plugin.UnboundParametersTarget, backend.num_qubits
        )

        # if `with_progress_bar` is not explicitly set in the options, disable it
        options_copy = (options or {}).copy()
        options_copy.update(
            with_progress_bar=options_copy.get("with_progress_bar", False)
        )

        super().__init__(
            mod_backend,
            bound_pass_manager=transpiler_plugin.bound_pass_manager(),
            options=options_copy,
            abelian_grouping=abelian_grouping,
            skip_transpilation=skip_transpilation,
        )

    @property
    def backend(self) -> AnyAQTResource:
        """Computing resource used for circuit evaluation."""
        return self._backend
