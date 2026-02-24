"""
Load jobs from a list of job ids and get results
"""

from mqp.qiskit_provider.job import MQPJob
from mqp.qiskit_provider import MQPProvider
from qiskit.providers import JobStatus
from qcomp_nlce.utils import get_token


def load_jobs_from_ids(jobids: list):
    jobs = []
    provider = MQPProvider(token=get_token("LRZ"))
    [backend] = provider.backends(name="AQT20")
    for jobid in jobids:
        job = MQPJob(backend.client, jobid)
        jobs.append(job)
    return jobs


def print_jobs(job_ids: list[str]) -> None:
    jobs = load_jobs_from_ids(job_ids)
    for job in jobs:
        if job.status() == JobStatus.DONE:
            print(f"Job ID : {job.job_id()}, Status : {job.status()}")
            result_dict = job.result().to_dict()
            print(f"Submission Time: {result_dict['timestamps']['submitted']}")
            print(f"Scheduling Time: {result_dict['timestamps']['scheduled']}")
            print(f"Completion Time: {result_dict['timestamps']['completed']}")
            print(f"Counts : {job.result().get_counts()}")
        else:
            try:
                counts = job.result().get_counts()
                print(f"Counts : {counts}")
            except RuntimeError as err:
                print(
                    f"Job ID : {job.job_id()}, Status : {job.status()}, Reason : {err}"
                )


if __name__ == "__main__":
    print_jobs(["268514", "268512"])
