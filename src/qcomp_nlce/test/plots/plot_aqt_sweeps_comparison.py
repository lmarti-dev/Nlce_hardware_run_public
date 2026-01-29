from fauplotstyle import style
import matplotlib.pyplot as plt
from qcomp_nlce.utils import exact_dispersion_1qp_1d
from data_manager import ExperimentDataManager

style()

runs = (
    r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2026_01_29\vqe_workflow_TFIM_hadam_stefan_AQT_2000shots_00001",
    r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2026_01_29\vqe_workflow_TFIM_hadam_stefan_AQT_2000shots",
)
dry_run = False
m_edm = ExperimentDataManager(
    "plot_sweep_comparison_aqt_current", project="nlce", dry_run=dry_run
)

dicts = []
for r in runs:
    edm = ExperimentDataManager.load(r)
    d = edm.load_last_saved_data_file
    dicts.append(d)


d10, d20 = dicts
fig, ax = plt.subplots()


ks = d10["inputs"]["ks"]
j = d10["inputs"]["J"]

sv10 = d10["results"]["sweep"]["disp"]["statevector"]
sv20 = d20["results"]["sweep"]["disp"]["statevector"]
fb10 = d10["results"]["sweep"]["disp"]["fake_backend"]
fb20 = d20["results"]["sweep"]["disp"]["fake_backend"]
ed = d10["results"]["ed"]["disp"]["statevector"]
ana = exact_dispersion_1qp_1d(j, ks)

ax.plot(ks, sv10, "x-", label="SV, 10 steps")
ax.plot(ks, sv20, "x-", label="SV, 20 steps")
ax.plot(ks, fb10, "o-", label="Noisy, 10 steps")
ax.plot(ks, fb20, "o-", label="Noisy, 20 steps")
ax.plot(ks, ed, "+-", label="ED")
ax.plot(ks, ana, "d-", label="Analytic")

ax.set_ylabel(r"$\omega(k)$")
ax.set_xlabel("$k$")
# ax.set_yscale("log")
ax.legend()
m_edm.save_figure(fig, fig_shape="double-y")
plt.show()
