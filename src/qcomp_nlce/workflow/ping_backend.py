from qcomp_nlce.utils import get_backend, PROVIDER_ALIAS


def pending_jobs(provider_name: PROVIDER_ALIAS):
    backend = get_backend(10, provider_name)
    print(f"There are: {backend.num_pending_jobs} pending jobs on {provider_name}")


if __name__ == "__main__":
    pending_jobs("EQE")
