from qiskit_ibm_runtime import QiskitRuntimeService
from qcomp_nlce.utils import get_ibm_token

service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="ibm-q/open/main",
    token=get_ibm_token(),
)
job = service.job("crx14dshazpg0084b9xg")
job_result = job.result()


job_result
# To get counts for a particular pub result, use
#
# pub_result = job_result[<idx>].data.<classical register>.get_counts()
#
# where <idx> is the index of the pub and <classical register> is the name of the classical register.
# You can use circuit.cregs to find the name of the classical registers.
