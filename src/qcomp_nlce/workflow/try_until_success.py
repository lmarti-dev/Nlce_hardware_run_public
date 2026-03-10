from qcomp_nlce.workflow.workflow_tfim_1qp_dispersion import (
    workflow_nq_aqt,
    workflow_nq_qexa,
    workflow_nq_aqt_ladder,
    workflow_nq_aqt_sweeps,
    workflow_nq_aqt_hl,
)
from qcomp_nlce.utils.hardware import pending_jobs

from time import sleep


def count_to_awakening(secs: int):
    print("I sleep but I will awaken")
    for s in range(secs):
        print(f"in {secs-s:0>6} seconds", end="\r")
        sleep(1)
    print("I have awoken")


def keep_trying(func: callable, provider_name: str):
    success = False
    while not success:
        try:
            n_jobs = pending_jobs(10, provider_name)
            if n_jobs > 1:
                raise Exception("Wait for jobs to finish!")
            func(True)
            success = True
        except Exception as e:
            print(f"Encoutered exception: ({type(e).__name__}) {e}")
            count_to_awakening(30)

    print("🚀🪐🌕🌑☄️🛸")


if __name__ == "__main__":
    keep_trying(workflow_nq_aqt_hl, "AQT")
