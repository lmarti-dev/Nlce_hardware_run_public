from src.qcomp_nlce.workflow.workflow_tfim_1qp_dispersion import (
    run_aqt_ladder,
    run_aqt_sweep,
    run_aqt_hl,
    run_aqt_100j,
    run_aqt_80j,
    run_aqt_30j,
)
from src.qcomp_nlce.utils.hardware import pending_jobs, RUNS_POSTFIX
from typing import Callable
from time import sleep
from fauplotstyle import style


def count_to_awakening(secs: int):
    print("I sleep but I will awaken")
    for s in range(secs):
        print(f"in {secs - s:0>6} seconds", end="\r")
        sleep(1)
    print("I have awoken")


def keep_trying(
    func: Callable[[bool, str], None],
    provider_name: str,
    debug: bool = False,
    postfix: RUNS_POSTFIX = "",
):
    success = False
    while not success:
        try:
            n_jobs = pending_jobs(10, provider_name)
            if n_jobs > 1:
                raise Exception("Wait for jobs to finish!")
            func(True, which_backends=["real_backend"], postfix=postfix)
            success = True
        except Exception as e:
            if debug:
                raise e
            print(f"Encoutered exception: ({type(e).__name__}) {e}")
            count_to_awakening(30)

    style()
    func(
        False,
        which_backends=["real_backend", "fake_backend", "statevector"],
        postfix=postfix,
    )
    print("🚀🪐🌕🌑☄️🛸")


if __name__ == "__main__":
    # keep_trying(run_aqt_hl, "AQT")
    # keep_trying(run_aqt_ladder, "AQT")
    keep_trying(run_aqt_30j, "AQT", "RUN3")
    # keep_trying(run_aqt_j80, "AQT")

    # keep_trying(run_aqt_sweep, "AQT")
