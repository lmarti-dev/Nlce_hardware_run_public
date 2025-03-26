from data_manager import ExperimentDataManager


file_a = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_03_24\nlce_hardware_run_vqe_gs"
)
file_b = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2025_03_25\nlce_hardware_run_sweep_gs"
)

edm_load = ExperimentDataManager.load(file_a)

edm_load
