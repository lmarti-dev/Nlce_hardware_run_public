from src.qcomp_nlce.workflow.workflow_tfim_1qp_dispersion import (
    workflow_nq_aqt_ladder,
    workflow_nq_aqt_sweep,
    workflow_nq_aqt_hl,
    workflow_nq_aqt_j100,
    workflow_nq_aqt_j80,
)
from src.qcomp_nlce.utils.hardware import pending_jobs
from typing import Callable
from time import sleep
from fauplotstyle import style


def count_to_awakening(secs: int):
    print("I sleep but I will awaken")
    for s in range(secs):
        print(f"in {secs-s:0>6} seconds", end="\r")
        sleep(1)
    print("I have awoken")


def keep_trying(func: Callable[[bool, str], None], provider_name: str,debug:bool=False):
    success = False
    while not success:
        try:
            n_jobs = pending_jobs(10, provider_name)
            if n_jobs > 1:
                raise Exception("Wait for jobs to finish!")
            func(True, which_backends=["real_backend"])
            success = True
        except Exception as e:
            if debug:
                raise e
            print(f"Encoutered exception: ({type(e).__name__}) {e}")
            count_to_awakening(30)

    style()
    func(False, which_backends=["real_backend", "fake_backend", "statevector"])
    print("🚀🪐🌕🌑☄️🛸")


if __name__ == "__main__":

    # keep_trying(workflow_nq_aqt_hl, "AQT")
    # keep_trying(workflow_nq_aqt_ladder, "AQT")
    # keep_trying(workflow_nq_aqt_j100, "AQT")
    # keep_trying(workflow_nq_aqt_j80, "AQT")

    keep_trying(workflow_nq_aqt_sweep, "AQT")
