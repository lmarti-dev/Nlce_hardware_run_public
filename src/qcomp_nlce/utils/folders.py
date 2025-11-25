import os
from pathlib import Path


def build_dict_of_sims():
    rsd = res_sim_dir()


def res_sim_dir() -> Path:
    HOME = Path(__file__).parent
    res_sim = Path(HOME, "../res_sim")
    return res_sim
