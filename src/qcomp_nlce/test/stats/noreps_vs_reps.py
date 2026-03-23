from data_manager import ExtendedJSONDecoder
import json
from src.qcomp_nlce.utils.results import show_diffs


def load_json(fpath):
    with open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read(), cls=ExtendedJSONDecoder)
    return jobj


noreps_fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_21\vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_2000shots_x_1reps_00001\run_00000\data\hazelnuts_00000.json"
yesreps_fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2026_03_21\vqe_workflow_vqe_80j_0hl_hadam_stefan_AQT_200shots_x_10reps_00002\run_00000\data\peanuts_00000.json"


noreps_d = load_json(noreps_fpath)
yesreps_d = load_json(yesreps_fpath)


show_diffs(noreps_d, "fake_backend")
show_diffs(yesreps_d, "fake_backend")


for dd in ("dict_s", "dict_l"):
    errno = noreps_d["results"]["vqe"][dd]["gsd"]["fake_backend"]["error"]
    erryes = yesreps_d["results"]["vqe"][dd]["gsd"]["fake_backend"]["error"]
    print(f"{dd}: no: {errno:.4f}, erryes: {erryes:.4f}")
