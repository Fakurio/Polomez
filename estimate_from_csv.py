import glob
import os
from typing import Dict
import numpy as np
import pandas as pd
from KalmanEstimator import KalmanEstimator
from marker_groups import MARKER_GROUPS

GAP_GENERATION_DIR = "gap_generation_output"
ESTIMATION_OUTPUT_DIR = "estimated_output"
N_FILES_TO_ESTIMATE = 1


def estimate_file(file_path: str):
    output_file_path = os.path.join(ESTIMATION_OUTPUT_DIR, "filled_" + os.path.basename(file_path))
    base_filename = os.path.basename(file_path).replace('.csv', '')
    print(f"Loading file: {base_filename}")

    df_gap = pd.read_csv(file_path)
    filled_data = []
    estimator = KalmanEstimator(marker_groups=MARKER_GROUPS)

    for index, row in df_gap.iterrows():
        # Restructure data from flat row (LFHD_X, LFHD_Y, LFHD_Z, ...)
        # to dict{marker_name: [x, y, z]} for the estimator
        frame_data: Dict[str, np.ndarray] = {}
        for marker in estimator.marker_groups.keys():
            position = row[[f'{marker}_X', f'{marker}_Y', f'{marker}_Z']].values
            frame_data[marker] = position

        estimated_positions = estimator.estimate_frame(frame_data)

        # Flatten the results back into a dictionary matching the CSV columns
        filled_row = {}
        for marker, pos in estimated_positions.items():
            if pos is not None:
                filled_row[f'{marker}_X'] = pos[0]
                filled_row[f'{marker}_Y'] = pos[1]
                filled_row[f'{marker}_Z'] = pos[2]

        filled_data.append(filled_row)

    df_filled = pd.DataFrame(filled_data, columns=df_gap.columns)
    df_filled.to_csv(output_file_path, index=False)

    print(f"\n✅ Estimation of file {base_filename} complete.")


def run_estimation():
    os.makedirs(ESTIMATION_OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory created/verified: **{ESTIMATION_OUTPUT_DIR}**")

    gap_files = glob.glob(os.path.join(GAP_GENERATION_DIR, '*.csv'))
    if not gap_files:
        print("ERROR: No files found in GAP_GENERATION_DIR")
        return

    for file_path in gap_files[:N_FILES_TO_ESTIMATE]:
        estimate_file(file_path)


if __name__ == '__main__':
    run_estimation()
