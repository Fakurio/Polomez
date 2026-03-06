import os
import re
import time
import glob
import numpy as np
import pandas as pd

# >>> UPEWNIJ SIĘ, że ten import wskazuje na Twoje estimators.py z LSTMXYZEstimator
from estimators import LSTMXYZEstimator

# =========================
# KONFIGURACJA (U CIEBIE)
# =========================
ROOT = r"E:\MODEL LSTM"

DIR_ORIGINAL = os.path.join(ROOT, "augmented_output")   # oryginały (bez dziur)
DIR_GAPPED   = os.path.join(ROOT, "test_data")          # losowa próbka gapów (np. z pick_random_data.py)
OUT_DIR      = os.path.join(ROOT, "rmse_results")       # gdzie zapisać wyniki

MODEL_PATH = os.path.join(ROOT, "best_lstm_xyz_model_constrained.pth")

ROOT_MARKER = "LASI"
SEQ_LEN = 60

# Kolejność grup jak we "wzorze"
CONFIGURATIONS = [
    "2m_100f",
    "2m_300f", "2m_500f",
    "6m_100f", "6m_300f", "6m_500f",
    "10m_100f", "10m_300f", "10m_500f",
    "14m_100f", "14m_300f", "14m_500f"
]

# Musi odpowiadać temu, co używa estimator (w Twoim estimators.py to TARGET_MARKERS)
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
    """Gwarantuje, że df ma wszystkie kolumny markerów i są numeric."""
    df = df.copy()
    df = df.reindex(columns=EXPECTED_COLS, fill_value=np.nan)
    for c in EXPECTED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def missing_mask_from_gap(df_gap: pd.DataFrame) -> np.ndarray:
    """Zwraca miss[T,M] True jeśli marker ma NaN w X/Y/Z w pliku gap."""
    T = len(df_gap)
    M = len(TARGET_MARKERS)
    miss = np.zeros((T, M), dtype=bool)
    for i, m in enumerate(TARGET_MARKERS):
        cols = [f"{m}_X", f"{m}_Y", f"{m}_Z"]
        miss[:, i] = df_gap[cols].isna().any(axis=1).to_numpy()
    return miss


def df_to_xyz(df: pd.DataFrame) -> np.ndarray:
    """df -> xyz[T,M,3] w mm (float32)."""
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
    xyz_*: [T,M,3] w mm
    miss:  [T,M] (True = marker missing w gap)
    Zwraca:
      rmse_missing_mm, rmse_all_mm
    """
    pred = np.nan_to_num(xyz_pred.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    true = np.nan_to_num(xyz_true.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    err = pred - true
    se = err ** 2  # [T,M,3]

    # RMSE na całości
    mse_all = se.mean()
    rmse_all = float(np.sqrt(mse_all))

    # RMSE tylko tam gdzie marker był missing (w tej klatce)
    mask3 = np.repeat(miss[:, :, None], 3, axis=2)
    if mask3.any():
        mse_missing = se[mask3].mean()
        rmse_missing = float(np.sqrt(mse_missing))
    else:
        rmse_missing = float("nan")

    return rmse_missing, rmse_all


def orig_name_from_gap_filename(gap_filename: str) -> str:
    """
    gap:  coś_2m_100f_c1.csv
    orig: coś.csv  (w augmented_output)
    """
    m = re.match(r"(.+?)_(\d+m)_(\d+f)_c\d+\.csv$", gap_filename)
    if not m:
        raise ValueError(f"Nie rozpoznano nazwy gap: {gap_filename}")
    return m.group(1) + ".csv"


def gap_pattern_from_filename(gap_filename: str) -> str:
    """Zwraca np. '2m_100f' z nazwy pliku."""
    m = re.search(r"_(\d+m)_(\d+f)_c\d+\.csv$", gap_filename)
    if not m:
        return "unknown"
    return f"{m.group(1)}_{m.group(2)}"


# =========================
# GŁÓWNY SKRYPT
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # inicjalizacja estymatora LSTM-XYZ (offline)
    estimator = LSTMXYZEstimator(model_path=MODEL_PATH, root_marker=ROOT_MARKER, seq_len=SEQ_LEN)

    detailed_rows = []
    summary_rows = []

    print("Processing files grouped by configuration...\n")

    for cfg in CONFIGURATIONS:
        search_pattern = os.path.join(DIR_GAPPED, f"*{cfg}*_c*.csv")
        gap_paths = sorted(glob.glob(search_pattern))

        if not gap_paths:
            print(f"[WARN] Brak plików dla konfiguracji: {cfg} (pattern: {search_pattern})")
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
                print(f"[WARN] Pomijam (zła nazwa): {gap_fn}")
                continue

            orig_path = os.path.join(DIR_ORIGINAL, orig_fn)
            if not os.path.exists(orig_path):
                print(f"[WARN] Brak oryginału dla: {gap_fn} -> {orig_fn}")
                continue

            # wczytaj
            df_gap_raw = pd.read_csv(gap_path)
            df_orig_raw = pd.read_csv(orig_path)

            # ujednolicenie kolumn + numeric
            df_gap = ensure_numeric_and_reindex(df_gap_raw)
            df_orig = ensure_numeric_and_reindex(df_orig_raw)

            # maska braków z GAP (na NaN!)
            miss = missing_mask_from_gap(df_gap)

            # estymacja LSTM (offline na całym df)
            t0 = time.perf_counter()
            df_pred = estimator.estimate_dataframe(df_gap_raw)  # estimator sam liczy swoje maski itd.
            dt = time.perf_counter() - t0

            # ujednolicenie pred
            df_pred = ensure_numeric_and_reindex(df_pred)

            xyz_true = df_to_xyz(df_orig)
            xyz_pred = df_to_xyz(df_pred)

            rmse_missing, rmse_all = compute_rmse_mm(xyz_pred, xyz_true, miss)

            detailed_rows.append({
                "gap_file": gap_fn,
                "orig_file": orig_fn,
                "gap_pattern": cfg,
                "rmse_missing_mm": rmse_missing,
                "rmse_all_mm": rmse_all,
                "time_sec": dt,
                "frames": len(df_gap_raw)
            })

            if np.isfinite(rmse_missing):
                cfg_rmse_missing.append(rmse_missing)
            if np.isfinite(rmse_all):
                cfg_rmse_all.append(rmse_all)
            cfg_times.append(dt)

            print(f"  [{j}/{len(gap_paths)}] {gap_fn} | RMSE_missing={rmse_missing:.3f} mm | RMSE_all={rmse_all:.3f} mm | time={dt:.3f}s")

        # summary per cfg
        if cfg_rmse_missing or cfg_rmse_all:
            mean_missing = float(np.mean(cfg_rmse_missing)) if cfg_rmse_missing else float("nan")
            std_missing = float(np.std(cfg_rmse_missing, ddof=1)) if len(cfg_rmse_missing) > 1 else float("nan")
            max_missing = float(np.max(cfg_rmse_missing)) if cfg_rmse_missing else float("nan")

            mean_all = float(np.mean(cfg_rmse_all)) if cfg_rmse_all else float("nan")
            std_all = float(np.std(cfg_rmse_all, ddof=1)) if len(cfg_rmse_all) > 1 else float("nan")

            mean_time = float(np.mean(cfg_times)) if cfg_times else float("nan")

            summary_rows.append({
                "Configuration": cfg,
                "Count": len(gap_paths),
                "Mean RMSE_missing (mm)": round(mean_missing, 3) if np.isfinite(mean_missing) else np.nan,
                "SD RMSE_missing (mm)": round(std_missing, 3) if np.isfinite(std_missing) else np.nan,
                "Max RMSE_missing (mm)": round(max_missing, 3) if np.isfinite(max_missing) else np.nan,
                "Mean RMSE_all (mm)": round(mean_all, 3) if np.isfinite(mean_all) else np.nan,
                "SD RMSE_all (mm)": round(std_all, 3) if np.isfinite(std_all) else np.nan,
                "Mean Est. Time (s)": round(mean_time, 4) if np.isfinite(mean_time) else np.nan,
                "Thesis Format (missing)": (f"{mean_missing:.2f} ± {std_missing:.2f}"
                                           if np.isfinite(mean_missing) and np.isfinite(std_missing) else "")
            })

        print()

    if not detailed_rows:
        raise RuntimeError("Nie policzono żadnych wyników (sprawdź ścieżki, nazwy plików i konfiguracje).")

    df_detailed = pd.DataFrame(detailed_rows)
    df_summary = pd.DataFrame(summary_rows)

    # Zapis szczegółowy
    out_detailed = os.path.join(OUT_DIR, "lstm_rmse_detailed_grouped.csv")
    df_detailed.to_csv(out_detailed, index=False)
    print("✅ Zapisano szczegółowe wyniki:", out_detailed)

    # Zapis tabeli podsumowań (jak w kalmanowym wzorze)
    out_summary = os.path.join(OUT_DIR, "lstm_rmse_summary_by_config.csv")
    df_summary.to_csv(out_summary, index=False)
    print("✅ Zapisano podsumowanie konfiguracji:", out_summary)

    print("\n--- FINAL THESIS TABLE (by config) ---")
    if not df_summary.empty:
        print(df_summary.to_string(index=False))
    else:
        print("[WARN] Brak konfiguracji z wynikami (wszystko puste).")


if __name__ == "__main__":
    main()