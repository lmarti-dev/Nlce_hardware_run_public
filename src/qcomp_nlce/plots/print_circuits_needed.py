from src.qcomp_nlce.utils import (
    heff_n_expectation_values_1d,
    heff_n_expectation_values_2d,
    heff_n_expectation_values_ladder,
)


def heff_total_shots_paper():
    d = {
        "tfim_1d_hl_80j": heff_n_expectation_values_1d(5, 1) * 2000,
        "tfim_1d_hl_80j_sweep": heff_n_expectation_values_1d(5, 1) * 2000,
        "tfim_1d_100j": heff_n_expectation_values_1d(5, 1) * 2000,
        "tfim_1d_80j_10hl": heff_n_expectation_values_1d(5, 1) * 2000,
        "tfim_1d_ladder": heff_n_expectation_values_ladder(8, 1) * 2000,
    }
    for k in d:
        print(f"{k}: {d[k]:.3e} shots")
    print(f"total: {sum(d.values()):.3e} shots")
    print(f"hours: {(1/6)*sum(d.values())/2000:.2f}")


def circuits_per_example():
    d = {
        "tfim_1d_hl_80j": heff_n_expectation_values_1d(5, 2),
        "tfim_1d_hl_80j_sweep": heff_n_expectation_values_1d(5, 2),
        "tfim_1d_100j": heff_n_expectation_values_1d(5, 2),
        "tfim_1d_80j_10hl": heff_n_expectation_values_1d(5, 3),
        "tfim_1d_ladder": heff_n_expectation_values_ladder(8, 2),
    }
    for k in d:
        print(f"{k}: {d[k]:.3f} circuits")
    print(f"total: {sum(d.values()):.3f} circuits")


def future_example():
    n_circs = heff_n_expectation_values_2d(100, 2)
    print(f"n circs: {n_circs:.4e}")


if __name__ == "__main__":
    heff_total_shots_paper()
    circuits_per_example()
    future_example()
