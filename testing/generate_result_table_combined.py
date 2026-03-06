import os
import re
import time
import glob
import argparse
import numpy as np
import pandas as pd

# >>> IMPORTS DLA OBU ESTYMATORÓW
from estimators import LSTMXYZEstimator, KalmanEstimatorWrapper
from KalmanEstimator import KalmanEstimator
from marker_groups import MARKER_GROUPS

# =========================
# KONFIGURACJA
# =========================

DIR_ORIGINAL = '../augmented_output'
DIR_GAPPED = './test_data'

MODEL_PATH = os.path.join("../best_lstm_xyz_model_constrained.pth")
ROOT_MARKER = "LASI"
SEQ_LEN = 60

CONFIGURATIONS = [
    "2m_100f",
    "2m_300f", "2m_500f",
    "6m_100f", "6m_300f", "6m_500f",
    "10m_100f", "10m_300f", "10m_500f",
    "14m_100f", "14m_300f", "14m_500f"
]

TARGET_MARKERS = [
    'LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'RBAK', 'LSHO', 'LUPA', 'LELB', 'LFRM',
    'LWRA', 'LWRB', 'LFIN', 'RSHO', 'RUPA', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RFIN', 'LASI', 'RASI',
    'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK',
    'RHEE', 'RTOE'
]

EXPECTED_COLS = [f"{m}_{ax}" for m in TARGET_MARKERS for ax in ["X", "Y", "Z"]]


# =========================
# POMOCNICZE
# =========================
def ensure_numeric_and_reindex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.reindex(columns=EXPECTED_COLS, fill_value=np.nan)
    for c in EXPECTED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def missing_mask_from_gap(df_gap: pd.DataFrame) -> np.ndarray:
    T = len(df_gap)
    M = len(TARGET_MARKERS)
    miss = np.zeros((T, M), dtype=bool)
    for i, m in enumerate(TARGET_MARKERS):
        cols = [f"{m}_X", f"{m}_Y", f"{m}_Z"]
        miss[:, i] = df_gap[cols].isna().any(axis=1).to_numpy()
    return miss


def df_to_xyz(df: pd.DataFrame) -> np.ndarray:
    T = len(df)
    M = len(TARGET_MARKERS)
    xyz = np.zeros((T, M, 3), dtype=np.float32)
    for i, m in enumerate(TARGET_MARKERS):
        xyz[:, i, 0] = df[f"{m}_X"].to_numpy(np.float32)
        xyz[:, i, 1] = df[f"{m}_Y"].to_numpy(np.float32)
        xyz[:, i, 2] = df[f"{m}_Z"].to_numpy(np.float32)
    return xyz


def compute_rmse_mm(xyz_pred: np.ndarray, xyz_true: np.ndarray, miss: np.ndarray):
    """
    Safely calculates RMSE using np.nanmean to ignore frames where the estimator failed (returned NaN),
    preventing artificial error spikes.
    """
    # Do NOT replace NaNs with 0.0. Calculate squared error directly.
    err = xyz_pred - xyz_true
    se = err ** 2  # [T,M,3]

    # RMSE na całości (All markers, all frames)
    mse_all = np.nanmean(se)
    rmse_all = float(np.sqrt(mse_all))

    # RMSE tylko tam gdzie marker był missing (w tej klatce)
    mask3 = np.repeat(miss[:, :, None], 3, axis=2)
    if mask3.any():
        se_missing = se[mask3]
        mse_missing = np.nanmean(se_missing)
        rmse_missing = float(np.sqrt(mse_missing))
    else:
        rmse_missing = float("nan")

    return rmse_missing, rmse_all


def orig_name_from_gap_filename(gap_filename: str) -> str:
    m = re.match(r"(.+?)_(\d+m)_(\d+f)_c\d+\.csv$", gap_filename)
    if not m:
        raise ValueError(f"Nie rozpoznano nazwy gap: {gap_filename}")
    return m.group(1) + ".csv"


def gap_pattern_from_filename(gap_filename: str) -> str:
    m = re.search(r"_(\d+m)_(\d+f)_c\d+\.csv$", gap_filename)
    if not m:
        return "unknown"
    return f"{m.group(1)}_{m.group(2)}"


# =========================
# GŁÓWNY SKRYPT
# =========================
def main(estimator_type="lstm"):
    print(f"Initializing {estimator_type.upper()} Estimator...")

    # 1. Initialize the selected estimator
    if estimator_type == "lstm":
        estimator = LSTMXYZEstimator(model_path=MODEL_PATH, root_marker=ROOT_MARKER, seq_len=SEQ_LEN)
    elif estimator_type == "kalman":
        core = KalmanEstimator(marker_groups=MARKER_GROUPS)
        estimator = KalmanEstimatorWrapper(core)
    else:
        raise ValueError("Invalid estimator type. Choose 'lstm' or 'kalman'.")

    detailed_rows = []
    summary_rows = []

    print("Processing files grouped by configuration...\n")

    for cfg in CONFIGURATIONS:
        search_pattern = os.path.join(DIR_GAPPED, f"*{cfg}*_c*.csv")
        gap_paths = sorted(glob.glob(search_pattern))

        if not gap_paths:
            continue

        cfg_rmse_missing = []
        cfg_rmse_all = []
        cfg_times = []

        print(f"=== CONFIG: {cfg} | files: {len(gap_paths)} ===")

        for j, gap_path in enumerate(gap_paths, 1):
            gap_fn = os.path.basename(gap_path)

            try:
                orig_fn = orig_name_from_gap_filename(gap_fn)
            except ValueError:
                continue

            orig_path = os.path.join(DIR_ORIGINAL, orig_fn)
            if not os.path.exists(orig_path):
                continue

            # Load Data
            df_gap_raw = pd.read_csv(gap_path)
            df_orig_raw = pd.read_csv(orig_path)

            df_gap = ensure_numeric_and_reindex(df_gap_raw)
            df_orig = ensure_numeric_and_reindex(df_orig_raw)
            miss = missing_mask_from_gap(df_gap)

            # --- ESTIMATION LOGIC ---
            t0 = time.perf_counter()

            if estimator_type == "lstm":
                df_pred = estimator.estimate_dataframe(df_gap_raw)

            elif estimator_type == "kalman":
                # Reset Kalman state for new sequence
                if hasattr(estimator, 'history_buffer'):
                    estimator.history_buffer.clear()
                if hasattr(estimator, 'has_valid_root'):
                    estimator.has_valid_root = False

                filled_data = []
                for index, row in df_gap_raw.iterrows():
                    frame_data = {}
                    for marker in TARGET_MARKERS:
                        if f'{marker}_X' in row and not pd.isna(row[f'{marker}_X']):
                            frame_data[marker] = row[[f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
                        else:
                            # Pass NaNs so the estimator knows it's missing
                            frame_data[marker] = np.array([np.nan, np.nan, np.nan])

                    est_pos = estimator.estimate_frame(frame_data)

                    filled_row = {}
                    for marker, pos in est_pos.items():
                        if pos is not None:
                            filled_row[f'{marker}_X'] = pos[0]
                            filled_row[f'{marker}_Y'] = pos[1]
                            filled_row[f'{marker}_Z'] = pos[2]
                    filled_data.append(filled_row)

                df_pred = pd.DataFrame(filled_data, columns=df_gap_raw.columns)

            dt = time.perf_counter() - t0
            # --- END ESTIMATION ---

            df_pred = ensure_numeric_and_reindex(df_pred)

            xyz_true = df_to_xyz(df_orig)
            xyz_pred = df_to_xyz(df_pred)

            rmse_missing, rmse_all = compute_rmse_mm(xyz_pred, xyz_true, miss)

            detailed_rows.append({
                "gap_file": gap_fn,
                "gap_pattern": cfg,
                "rmse_missing_mm": rmse_missing,
                "rmse_all_mm": rmse_all,
                "time_sec": dt,
                "frames": len(df_gap_raw)
            })

            if np.isfinite(rmse_missing): cfg_rmse_missing.append(rmse_missing)
            if np.isfinite(rmse_all): cfg_rmse_all.append(rmse_all)
            cfg_times.append(dt)

            print(
                f"  [{j}/{len(gap_paths)}] {gap_fn} | RMSE_miss={rmse_missing:.2f}mm | RMSE_all={rmse_all:.2f}mm | time={dt:.2f}s")

        # Summary Generation
        if cfg_rmse_missing or cfg_rmse_all:
            mean_miss = float(np.mean(cfg_rmse_missing)) if cfg_rmse_missing else float("nan")
            std_miss = float(np.std(cfg_rmse_missing, ddof=1)) if len(cfg_rmse_missing) > 1 else float("nan")
            max_miss = float(np.max(cfg_rmse_missing)) if cfg_rmse_missing else float("nan")

            mean_all = float(np.mean(cfg_rmse_all)) if cfg_rmse_all else float("nan")
            std_all = float(np.std(cfg_rmse_all, ddof=1)) if len(cfg_rmse_all) > 1 else float("nan")

            mean_time = float(np.mean(cfg_times)) if cfg_times else float("nan")

            summary_rows.append({
                "Configuration": cfg,
                "Mean RMSE (Missing)": round(mean_miss, 3),
                "SD RMSE (Missing)": round(std_miss, 3),
                "Max RMSE (Missing)": round(max_miss, 3),
                "Thesis Format (Missing)": f"{mean_miss:.2f} ± {std_miss:.2f}" if np.isfinite(mean_miss) else "",

                "Mean RMSE (All)": round(mean_all, 3),
                "SD RMSE (All)": round(std_all, 3),
                "Thesis Format (All)": f"{mean_all:.2f} ± {std_all:.2f}" if np.isfinite(mean_all) else "",

                "Mean Est. Time (s)": round(mean_time, 4)
            })

    # Save outputs dynamically based on estimator type
    df_detailed = pd.DataFrame(detailed_rows)
    df_summary = pd.DataFrame(summary_rows)

    out_summary = f"{estimator_type}_rmse_summary.csv"
    df_summary.to_csv(out_summary, index=False)

    out_detailed = f"{estimator_type}_rmse_detailed.csv"
    df_detailed.to_csv(out_detailed, index=False)

    print(f"✅ Saved summary table: {out_summary}")

    print(f"\n--- FINAL THESIS TABLE ({estimator_type.upper()}) ---")
    if not df_summary.empty:
        print(df_summary[
            ["Configuration", "Thesis Format (Missing)", "Thesis Format (All)", "Mean Est. Time (s)"]].to_string(
            index=False))


if __name__ == "__main__":
    main()
