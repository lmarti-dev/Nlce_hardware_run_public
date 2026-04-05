from src.qcomp_nlce.workflow.workflow_tfim_1qp_dispersion import (
    run_aqt_ladder,
    run_aqt_sweep_80j,
    run_aqt_sweep_30j,
    run_aqt_hl,
    run_aqt_100j,
    run_aqt_80j,
    run_aqt_30j,
    depol_vs_shots
)

from src.qcomp_nlce.workflow.try_until_success import keep_trying

from src.qcomp_nlce.plots.plot_offdiag_sv import plot_offdiag_elems

if __name__ == "__main__":
    # keep_trying(run_aqt_hl, "AQT")
    # keep_trying(run_aqt_ladder, "AQT")
    # keep_trying(run_aqt_j100, "AQT")
    # keep_trying(run_aqt_j80, "AQT")
    # keep_trying(run_aqt_80j, "AQT", postfix="RUN3")
    # keep_trying(run_aqt_sweep_80j, "AQT", postfix="RUN3")
    # keep_trying(run_aqt_j30, "AQT")

    # keep_trying(run_aqt_sweep_80j, "AQT")
    plot_offdiag_elems(False)
