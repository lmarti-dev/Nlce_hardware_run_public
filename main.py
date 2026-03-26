from src.qcomp_nlce.workflow.workflow_tfim_1qp_dispersion import (
    workflow_nq_aqt_ladder,
    workflow_nq_aqt_sweep,
    workflow_nq_aqt_hl,
    workflow_nq_aqt_j100,
    workflow_nq_aqt_j80,
    workflow_nq_aqt_j30,
)

from src.qcomp_nlce.workflow.try_until_success import keep_trying

if __name__ == "__main__":
    # keep_trying(workflow_nq_aqt_hl, "AQT")
    # keep_trying(workflow_nq_aqt_ladder, "AQT")
    # keep_trying(workflow_nq_aqt_j100, "AQT")
    keep_trying(workflow_nq_aqt_j80, "AQT")
    # keep_trying(workflow_nq_aqt_j30, "AQT")

    # keep_trying(workflow_nq_aqt_sweep, "AQT")
