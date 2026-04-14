from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import SparsePauliOp

import os
from pathlib import Path
import numpy as np
import json
from data_manager import ExtendedJSONDecoder, ExtendedJSONEncoder
import io


from src.qcomp_nlce.utils.constants import ALGORITHMS_ALIAS
from src.qcomp_nlce.utils.hardware import RUNS_POSTFIX


def get_save_directory(dirname: str):
    home = Path(__file__).parent.parent
    save_dir = Path(home, f"res_sim/{dirname}/")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def save_sweep(
    dirname: str,
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    ground_energy: float,
    dry_run: bool,
):
    if not dry_run:
        save_dir = get_save_directory(dirname)
        with open(Path(save_dir, "circuit.qpy"), "wb+") as fd:
            qpy.dump(circuit, fd)
        with open(Path(save_dir, "ground_energy.npy"), "wb+") as fd:
            np.save(fd, np.array(ground_energy))
        with open(Path(save_dir, "hamiltonian.npy"), "wb+") as fd:
            np.save(fd, hamiltonian.to_list(True))


def load_sweep(dirname: str):
    with open(Path(dirname, "circuit.qpy"), "rb") as fd:
        circuit = qpy.load(fd)[0]
    with open(Path(dirname, "ground_energy.npy"), "rb") as fd:
        ground_energy = np.load(fd)
    with open(Path(dirname, "hamiltonian.npy"), "rb") as fd:
        hamiltonian = SparsePauliOp.from_list(np.load(fd))

    return hamiltonian, circuit, ground_energy


def save_optimized_vqe(
    dirname: str,
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    weights: np.ndarray,
    ground_energy: float,
    dry_run: bool,
):
    if not dry_run:
        save_dir = get_save_directory(dirname)
        with open(Path(save_dir, "circuit.qpy"), "wb+") as fd:
            qpy.dump(circuit, fd)
        with open(Path(save_dir, "weights.npy"), "wb+") as fd:
            np.save(fd, weights)
        with open(Path(save_dir, "ground_energy.npy"), "wb+") as fd:
            np.save(fd, np.array(ground_energy))
        with open(Path(save_dir, "hamiltonian.npy"), "wb+") as fd:
            np.save(fd, hamiltonian.to_list(True))


def load_optimized_vqe(
    dirname: str,
) -> tuple[SparsePauliOp, QuantumCircuit, np.ndarray, float]:
    with open(Path(dirname, "circuit.qpy"), "rb") as fd:
        circuit = qpy.load(fd)[0]
    with open(Path(dirname, "weights.npy"), "rb") as fd:
        weights = np.load(fd)
    with open(Path(dirname, "ground_energy.npy"), "rb") as fd:
        ground_energy = np.load(fd)
    with open(Path(dirname, "hamiltonian.npy"), "rb") as fd:
        hamiltonian = SparsePauliOp.from_list(np.load(fd))

    return hamiltonian, circuit, weights, ground_energy


TempListType = list[tuple[tuple[int | str, int | str], tuple[complex, complex]]]
# l[ii] = ((ind_x,ind_y),(val,std))


def save_tmp_item(d, filename: str):
    fpath = Path(Path(__file__).parent, f"../tmp/{filename}")

    if not fpath.parent.exists():
        os.makedirs(fpath.parent)

    with io.open(fpath, "w+", encoding="utf8") as fd:
        fd.write(json.dumps(d, cls=ExtendedJSONEncoder, ensure_ascii=False, indent=4))


def load_tmp_item(filename: str) -> TempListType:
    fpath = Path(Path(__file__).parent, f"../tmp/{filename}")

    with io.open(fpath, "r", encoding="utf8") as fd:
        jobj = json.loads(fd.read(), cls=ExtendedJSONDecoder)
    return jobj


def add_tmp_entry_and_save(
    filename: str, l: TempListType, val: complex, std: complex, ind_x: int, ind_y: int
) -> TempListType:
    l.append(((ind_x, ind_y), (val, std)))
    save_tmp_item(l, filename)
    return l


def tmp_get_item(l: TempListType, ind_x: int, ind_y: int) -> tuple[complex, complex]:
    if not tmp_entry_exists(l, ind_x, ind_y):
        raise ValueError("Item doesn't exist")
    else:
        for k in l:
            if ind_x == k[0][0] and ind_y == k[0][1]:
                return k[1]


def tmp_entry_exists(l: TempListType, ind_x: int, ind_y: int):
    for k in l:
        if ind_x == k[0][0] and ind_y == k[0][1]:
            return True
    return False


def tmp_dirname(
    circuit: QuantumCircuit, which_backend: str, postfix: RUNS_POSTFIX = ""
):
    if "tmpname" not in circuit.metadata.keys():
        raise ValueError(f"You did not give a tmpname to the circuit: {circuit.name}")

    s = f"{circuit.metadata['tmpname']}_{which_backend}"
    if postfix != "":
        return s + "_" + postfix
    return s


def load_result_matrices(fpath: Path = None):
    if fpath is None:
        fpath = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2025_12_10\workflow_tfim_hadam_stefan_AQT_2000shots_00004\run_00000\data\green_00000.json"
        n_shots = 2000
        fac = np.sqrt(n_shots)
    with open(fpath, "r", encoding="utf8") as f:
        jobj = json.loads(f.read(), cls=ExtendedJSONDecoder)

    be = "fake_backend"

    data_l = jobj["results"]["vqe"]["dict_l"]
    data_s = jobj["results"]["vqe"]["dict_s"]

    expect_l = data_l["exd"][be]["values"]
    expect_std_l = data_l["exd"][be]["stds"] / fac
    overlap_l = data_l["eld"][be]["values"]
    overlap_std_l = data_l["eld"][be]["stds"] / fac
    gs_l = data_l["gsd"][be]["energy"]
    gs_std_l = data_l["gsd"][be]["std"] / fac

    expect_s = data_s["exd"][be]["values"]
    expect_std_s = data_s["exd"][be]["stds"] / fac
    overlap_s = data_s["eld"][be]["values"]
    overlap_std_s = data_s["eld"][be]["stds"] / fac
    gs_s = data_s["gsd"][be]["energy"]
    gs_std_s = data_s["gsd"][be]["std"] / fac

    return (
        expect_l,
        expect_std_l,
        overlap_l,
        overlap_std_l,
        gs_l,
        gs_std_l,
        expect_s,
        expect_std_s,
        overlap_s,
        overlap_std_s,
        gs_s,
        gs_std_s,
    )


def baptize_circuit(
    prefix: str,
    method: ALGORITHMS_ALIAS,
    n_qubits: int,
    provider_name: str,
    n_shots: int,
    which_hadamard_test: str,
) -> str:
    return f"{prefix}_{method}_{n_qubits}_{provider_name}_{n_shots}shots_{which_hadamard_test}_test"


RUN_FILENAME = "runlist.json"


def init_run_split(_dirname: Path, on_hardware: bool) -> list:
    if on_hardware:
        if tmp_check_run_list(_dirname):
            run_list = tmp_load_run_list(_dirname)
            print(f"Loading file at {_dirname}, {len(run_list)} already done")
        else:
            tmp_create_run_file(_dirname)
            print(f"Starting runs at {_dirname} from scratch")
            run_list = []
    else:
        run_list = []
    return run_list


def dirname_to_tmp(_dirname: Path) -> Path:
    return Path(Path(__file__).parent, "../tmp", _dirname)


def tmp_delete_run_file(_dirname: Path):
    _fpath = Path(dirname_to_tmp(_dirname), RUN_FILENAME)
    if _fpath.is_file():
        os.remove(_fpath)


def tmp_check_run_list(_dirname: Path):
    return Path(dirname_to_tmp(_dirname), RUN_FILENAME).is_file()


def tmp_create_run_file(_dirname: Path):
    fpath = Path(dirname_to_tmp(_dirname), RUN_FILENAME)
    if tmp_check_run_list(_dirname):
        raise ValueError(f"Run file at {_dirname} already exists!")
    else:
        if not fpath.parent.exists():
            os.makedirs(fpath.parent)
        with open(fpath, "w+", encoding="utf8") as f:
            f.write("[]")
        assert fpath.is_file()


def tmp_save_run_file(_dirname: Path, run_list: list[list, list]):
    with open(Path(dirname_to_tmp(_dirname), RUN_FILENAME), "w+", encoding="utf8") as f:
        f.write(
            json.dumps(run_list, cls=ExtendedJSONEncoder, ensure_ascii=False, indent=2)
        )


def tmp_load_run_list(_dirname: Path):
    with open(Path(dirname_to_tmp(_dirname), RUN_FILENAME), "r", encoding="utf8") as f:
        jobj = json.loads(f.read(), cls=ExtendedJSONDecoder)
    return jobj


DATA_DIR = Path(Path(__file__).parent, "../data")
