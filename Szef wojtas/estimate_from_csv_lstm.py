import glob
import os
from typing import Dict

import numpy as np
import pandas as pd

from KalmanEstimator import KalmanEstimator
from marker_groups import MARKER_GROUPS
from estimators import KalmanEstimatorWrapper, LSTMXYZEstimator

# -----------------------
# KONFIGURACJA (FINAL)
# -----------------------
ROOT = r"E:\MODEL LSTM"

GAP_GENERATION_DIR = os.path.join(ROOT, "gap_generation_output")
ESTIMATION_OUTPUT_DIR = os.path.join(ROOT, "estimated_output")

# wybierz 1 konkretny plik do testu (ustaw None, żeby brać z folderu)
CSV_FILE_PATH = None

# ile plików z folderu przetwarzać, gdy CSV_FILE_PATH=None
N_FILES_TO_ESTIMATE = 5


def get_estimator(mode: str, model_path: str | None = None):
    if mode == "kalman":
        print("[INFO] Using Kalman Estimator")
        core = KalmanEstimator(marker_groups=MARKER_GROUPS)
        return KalmanEstimatorWrapper(core)

    if mode == "lstm_xyz":
        if not model_path:
            raise ValueError("Model path required for LSTM_XYZ mode")
        print(f"[INFO] Using LSTM-XYZ Estimator (Model: {model_path})")
        return LSTMXYZEstimator(model_path=model_path, root_marker="LASI", seq_len=60)


def estimate_file(file_path: str, estimator):
    os.makedirs(ESTIMATION_OUTPUT_DIR, exist_ok=True)

    output_file_path = os.path.join(ESTIMATION_OUTPUT_DIR, "filled_" + os.path.basename(file_path))
    base_filename = os.path.basename(file_path).replace(".csv", "")
    print(f"Loading file: {base_filename}")

    df_gap = pd.read_csv(file_path)

    def missing_frames_for_marker(df, m):
        cols = [f"{m}_X", f"{m}_Y", f"{m}_Z"]
        tmp = df[cols].apply(pd.to_numeric, errors="coerce")
        return tmp.isna().any(axis=1).to_numpy()

    miss_lasi = missing_frames_for_marker(df_gap.copy(), "LASI")
    print("LASI missing frames:", int(miss_lasi.sum()), " / ", len(df_gap))

    # (opcjonalnie) pokaż pierwsze indeksy gdzie znika
    idxs = np.where(miss_lasi)[0][:20]
    print("First LASI-missing indices:", idxs)

    # >>> LSTM-XYZ działa OFFLINE na całym pliku
    if hasattr(estimator, "estimate_dataframe"):
        df_filled = estimator.estimate_dataframe(df_gap)
        df_filled.to_csv(output_file_path, index=False)
        print(f"\n✅ Estimation of file {base_filename} complete (LSTM-XYZ offline).")
        return

    # >>> Kalman / legacy LSTM-bone: klatka po klatce
    filled_data = []

    if hasattr(estimator, "history_buffer"):
        estimator.history_buffer.clear()
        if hasattr(estimator, "has_valid_root"):
            estimator.has_valid_root = False

    for _, row in df_gap.iterrows():
        frame_data: Dict[str, np.ndarray] = {}
        for marker in MARKER_GROUPS.keys():
            position = row[[f"{marker}_X", f"{marker}_Y", f"{marker}_Z"]].values
            frame_data[marker] = position

        estimated_positions = estimator.estimate_frame(frame_data)

        filled_row = {}
        for marker, pos in estimated_positions.items():
            if pos is not None:
                filled_row[f"{marker}_X"] = pos[0]
                filled_row[f"{marker}_Y"] = pos[1]
                filled_row[f"{marker}_Z"] = pos[2]

        filled_data.append(filled_row)

    df_filled = pd.DataFrame(filled_data, columns=df_gap.columns)
    df_filled.to_csv(output_file_path, index=False)
    print(f"\n✅ Estimation of file {base_filename} complete.")


def run_estimation():
    MODE = "lstm_xyz"
    LSTM_MODEL = os.path.join(ROOT, "best_lstm_xyz_model_constrained.pth")  # <<< NOWY MODEL XYZ

    try:
        estimator = get_estimator(MODE, LSTM_MODEL if MODE != "kalman" else None)
    except Exception as e:
        print(f"Failed to initialize estimator: {e}")
        return

    if CSV_FILE_PATH:
        estimate_file(CSV_FILE_PATH, estimator)
        return

    gap_files = glob.glob(os.path.join(GAP_GENERATION_DIR, "*.csv"))
    if not gap_files:
        print("ERROR: No files found in GAP_GENERATION_DIR")
        return

    for file_path in gap_files[:N_FILES_TO_ESTIMATE]:
        estimate_file(file_path, estimator)


if __name__ == "__main__":
    run_estimation()
