import os
from pathlib import Path
from data_manager import ExperimentDataManager, ExtendedJSONDecoder, ExtendedJSONEncoder
from src.qcomp_nlce.utils import (
    mc_disp_from_res_dict,
)
import numpy as np
import json


def load_json(fpath: Path):
    with open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read(), cls=ExtendedJSONDecoder)
    return jobj


def save_json(fpath: Path, jobj: dict):
    with open(fpath, "w+", encoding="utf8") as f:
        f.write(json.dumps(jobj, ensure_ascii=False, indent=2, cls=ExtendedJSONEncoder))


HOME = Path(__file__).parent
DATADR = Path(HOME, "../../data")
days = os.listdir(DATADR)
failures = 0
failed_experiments = []
for day in days:
    for experiment in os.listdir(Path(DATADR, day)):
        expath = Path(DATADR, day, experiment)
        edm = ExperimentDataManager.load(expath)
        fpath = edm.run_last_saved_data_file(-1)
        d = load_json(fpath)
        try:
            ks = d["inputs"]["ks"]
            j = d["inputs"]["J"]

            shots = d["inputs"]["n_reps"] * d["inputs"]["n_shots"]
            which_algorithms = d["inputs"]["which_algorithms"]
            n_samples = 1e4
            for alg in which_algorithms:
                mc_disp = mc_disp_from_res_dict(ks, d, alg, n_samples=4, to_eotm=True)
                d["results"][alg]["mc_disp"] = mc_disp
                save_json(fpath, d)
            print("SUCESS")
        except Exception as e:
            print(e)
            failures += 1
            failed_experiments.append(fpath)

print(f"Number of failures: {failures}")
print(f"Experiments: {failed_experiments}")
