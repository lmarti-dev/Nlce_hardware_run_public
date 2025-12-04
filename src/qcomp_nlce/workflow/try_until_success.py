from qcomp_nlce.workflow.workflow_1qp_dispersion import workflow_5q

from time import sleep


def keep_trying(func: callable):
    success = False
    while not success:
        try:
            func()
            success = True
        except Exception as e:
            print(f"Encoutered exception: {e}")
            print("Sleeping....")
            sleep(600)
    print("Τετέλεσται.")


if __name__ == "__main__":
    keep_trying(workflow_5q)
