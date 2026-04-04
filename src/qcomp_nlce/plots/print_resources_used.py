from src.qcomp_nlce.utils import (
    heff_n_expectation_values_1d,
    heff_n_expectation_values_2d,
    heff_n_expectation_values_ladder,
)


def heff_total_shots_paper():
    shots = {
        "tfim_1d_hl_30j": 2000,
        "tfim_1d_hl_80j": 2000,
        "tfim_1d_hl_30j_sweep": 2000,
        "tfim_1d_hl_80j_sweep": 2000,
        "tfim_1d_100j": 2000,
        "tfim_1d_80j_10hl": 2000,
        "tfim_1d_ladder": 2000,
    }

    d = {
        "tfim_1d_hl_30j": heff_n_expectation_values_1d(5, 1) * 1,
        "tfim_1d_hl_30j_sweep": heff_n_expectation_values_1d(5, 1) * 1,
        "tfim_1d_hl_80j": heff_n_expectation_values_1d(5, 1) * 3,
        "tfim_1d_hl_80j_sweep": heff_n_expectation_values_1d(5, 1) * 3,
        "tfim_1d_100j": heff_n_expectation_values_1d(5, 1) * 3,
        "tfim_1d_80j_10hl": heff_n_expectation_values_1d(5, 1) * 3,
        "tfim_1d_ladder": heff_n_expectation_values_ladder(8, 1),
    }
    for k in d:
        print(f"{k}: {d[k] * shots[k]:.3e} shots")
    print(f"total: {sum(d[k] * shots[k] for k in d):.3e} shots")
    print(f"hours: {(1 / 6) * sum(d[k] for k in d):.2f}")


def circuits_per_example():
    d = {
        "tfim_1d_hl_30j": heff_n_expectation_values_1d(5, 2) * 3,
        "tfim_1d_hl_80j": heff_n_expectation_values_1d(5, 2) * 3,
        "tfim_1d_hl_30j_sweep": heff_n_expectation_values_1d(5, 2) * 3,
        "tfim_1d_hl_80j_sweep": heff_n_expectation_values_1d(5, 2) * 3,
        "tfim_1d_100j": heff_n_expectation_values_1d(5, 2) * 3,
        "tfim_1d_80j_10hl": heff_n_expectation_values_1d(5, 3) * 3,
        "tfim_1d_ladder": heff_n_expectation_values_ladder(8, 2),
    }
    for k in d:
        print(f"{k}: {d[k]:.3f} circuits")
    print(f"total: {sum(d.values()):.3f} circuits")


def future_example():
    n_circs = heff_n_expectation_values_2d(100, 2)
    print(f"in the future 100x100: n circs: {n_circs:.4e}")


if __name__ == "__main__":
    heff_total_shots_paper()
    circuits_per_example()
    future_example()
