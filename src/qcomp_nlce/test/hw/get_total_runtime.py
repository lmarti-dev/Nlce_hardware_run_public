# this will only be for the ladder
# the firs job was on the 9th
# https://portal.quantum.lrz.de/jobs/262285
# go through all of them and do wall clock time + quantum device time.


from qcomp_nlce.utils import get_token

from mqp.qiskit_provider.job import MQPJob
from mqp.qiskit_provider import MQPProvider
from qiskit.providers import JobStatus


# Job IDs from the previous runs
# add your Job IDs here as a list
job_ids = [262634]
# Replace <YOUR TOKEN HERE> with your MQP TOKEN
TOKEN = get_token("LRZ")


def load_jobs_from_ids(jobids: list):
    jobs = []
    provider = MQPProvider(token=TOKEN)
    [backend] = provider.backends(name="AQT20")
    for jobid in jobids:
        job = MQPJob(backend.client, jobid)
        jobs.append(job)
    return jobs


if __name__ == "__main__":
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
