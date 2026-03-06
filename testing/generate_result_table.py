import os
import glob
import time
from typing import Dict
import numpy as np
import pandas as pd
import re

from KalmanEstimator import KalmanEstimator
from marker_groups import MARKER_GROUPS
from estimators import KalmanEstimatorWrapper

# --- Configuration ---
DIR_ORIGINAL = '../augmented_output'
DIR_GAPPED = './test_data'

CONFIGURATIONS = [
    "2m_100f",
    "2m_300f", "2m_500f",
    "6m_100f", "6m_300f", "6m_500f",
    "10m_100f", "10m_300f", "10m_500f",
    "14m_100f", "14m_300f", "14m_500f"
]


def calculate_masked_rmse(orig_df, est_df, gap_df):
    """Calculates RMSE exclusively on the artificially generated gaps."""
    mask_array = gap_df.isna().values

    orig_gaps = orig_df.values[mask_array]
    est_gaps = est_df.values[mask_array]

    if len(orig_gaps) == 0:
        return np.nan

    squared_errors = (orig_gaps - est_gaps) ** 2
    mse = np.nanmean(squared_errors)
    rmse = np.sqrt(mse)
    return rmse


def generate_thesis_table():
    summary_results = []
    print("Initializing Kalman Estimator...")

    # Initialize the estimator once
    core = KalmanEstimator(marker_groups=MARKER_GROUPS)
    estimator = KalmanEstimatorWrapper(core)

    print("Processing files, calculating RMSE, and profiling execution time...")

    for config in CONFIGURATIONS:
        search_pattern = os.path.join(DIR_GAPPED, f"*{config}*.csv")
        gap_files = glob.glob(search_pattern)

        if len(gap_files) != 12:
            print(f"Warning: Found {len(gap_files)} files for {config} (Expected 12).")

        config_rmse_values = []
        config_time_values = []

        for gap_path in gap_files:
            filename = os.path.basename(gap_path)
            orig_filename = re.sub(r'_\d+m_\d+f_c\d+', '', filename)
            orig_path = os.path.join(DIR_ORIGINAL, orig_filename)
            print(f"Processing file: {filename}, Original file: {orig_filename}")

            try:
                # 1. Load Data
                df_gap = pd.read_csv(gap_path)
                df_orig = pd.read_csv(orig_path)

                filled_data = []

                # --- START PROFILING TIMER ---
                # We use perf_counter because it is the most accurate clock for profiling algorithms
                start_time = time.perf_counter()

                # 3. Estimation Loop
                for index, row in df_gap.iterrows():
                    frame_data: Dict[str, np.ndarray] = {}
                    for marker in MARKER_GROUPS.keys():
                        frame_data[marker] = row[[f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values

                    # Estimate
                    estimated_positions = estimator.estimate_frame(frame_data)

                    # Flatten results
                    filled_row = {}
                    for marker, pos in estimated_positions.items():
                        if pos is not None:
                            filled_row[f'{marker}_X'] = pos[0]
                            filled_row[f'{marker}_Y'] = pos[1]
                            filled_row[f'{marker}_Z'] = pos[2]

                    filled_data.append(filled_row)

                # --- STOP PROFILING TIMER ---
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                config_time_values.append(execution_time)

                # 4. Reconstruct DataFrame and Calculate RMSE
                df_est = pd.DataFrame(filled_data, columns=df_gap.columns)
                file_rmse = calculate_masked_rmse(df_orig, df_est, df_gap)

                config_rmse_values.append(file_rmse)

            except FileNotFoundError:
                print(f"Missing corresponding original file for {filename}")

        # Aggregate results for the configuration
        if config_rmse_values:
            mean_rmse = np.mean(config_rmse_values)
            std_rmse = np.std(config_rmse_values, ddof=1)
            max_rmse = np.max(config_rmse_values)
            mean_time = np.mean(config_time_values)

            summary_results.append({
                "Configuration": config,
                "Mean RMSE": round(mean_rmse, 3),
                "SD": round(std_rmse, 3),
                "Max Error": round(max_rmse, 3),
                "Mean Est. Time (s)": round(mean_time, 4),  # <-- NEW METRIC
                "Thesis Format": f"{mean_rmse:.2f} ± {std_rmse:.2f}"
            })

    final_table = pd.DataFrame(summary_results)
    return final_table


if __name__ == "__main__":
    results_df = generate_thesis_table()

    print("\n--- FINAL THESIS TABLE ---")
    print(results_df.to_string(index=False))

    results_df.to_csv("kalman_final_results_with_time.csv", index=False)
    print("\nResults successfully saved to 'kalman_final_results_with_time.csv'")
