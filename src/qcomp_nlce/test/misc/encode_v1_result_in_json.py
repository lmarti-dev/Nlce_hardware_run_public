from mqp.qiskit_provider.job import MQPJob
from mqp.qiskit_provider import MQPProvider
from qiskit.providers import JobStatus


from qcomp_nlce.utils import get_token


TOKEN = get_token("LRZ")


def sampler_result_to_json(sampler: SamplerResult):
    d = {}

    d["quasi_dists"] = [
        dict(sampler.quasi_dists[m]) for m in range(len(sampler.quasi_dists))
    ]
    d["metadata"] = sampler.metadata

    return d


def load_jobs_from_ids(jobids: list):
    jobs = []
    provider = MQPProvider(token=TOKEN)
    [backend] = provider.backends(name="AQT20")
    for jobid in jobids:
        job = MQPJob(backend.client, jobid)
        jobs.append(job)
    return jobs


def display_jobs(job_ids):
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


def encode_result(job_ids):
    jobs = load_jobs_from_ids(job_ids)
    for job in jobs:
        print(type(job.result()))


if __name__ == "__main__":
    job_ids = ["144778"]
    encode_result(job_ids)
