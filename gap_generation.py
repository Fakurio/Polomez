import numpy as np
import pandas as pd
import os
import glob
from typing import List, Tuple, Callable

# ----------------------------------------------------------------------
# 1. Configuration and Marker List
# ----------------------------------------------------------------------

AUGMENTED_OUTPUT_DIR = "augmented_output"
GAP_OUTPUT_DIR = "gap_generation_output"

TARGET_MARKERS = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'RBAK', 'LSHO', 'LUPA', 'LELB', 'LFRM',
                  'LWRA', 'LWRB', 'LFIN', 'RSHO', 'RUPA', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RFIN', 'LASI', 'RASI',
                  'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK',
                  'RHEE', 'RTOE']

SCENARIOS = []
GAP_DURATIONS = [100, 300, 500]
MISSING_MARKER_COUNTS = [2, 6, 10, 14]
N_MARKER_COMBINATION_ITERATIONS = 4
N_GAPS_PER_SEQUENCE = 4

for d in GAP_DURATIONS:
    for m in MISSING_MARKER_COUNTS:
        for k in range(N_MARKER_COMBINATION_ITERATIONS):
            suffix = f'{m}m_{d}f_c{k + 1}'
            SCENARIOS.append((suffix, m, d))  # (suffix, num_markers, duration_frames)


# ----------------------------------------------------------------------
# 2. Utility Functions
# ----------------------------------------------------------------------

def load_data_from_csv(file_path: str) -> np.ndarray:
    """Loads CSV data and reshapes it to (N_frames, N_markers, 3)."""
    df = pd.read_csv(file_path)

    expected_cols = [f'{m}_{axis}' for m in TARGET_MARKERS for axis in ['X', 'Y', 'Z']]
    df_reindexed = df.reindex(columns=expected_cols, fill_value=np.nan)
    marker_data_flat = df_reindexed[expected_cols].values

    N_frames = marker_data_flat.shape[0]
    num_markers = len(TARGET_MARKERS)

    # Reshape from (N_frames, 3*N_markers) to (N_frames, N_markers, 3)
    positions = marker_data_flat.reshape(N_frames, num_markers, 3)

    return positions


def select_non_overlapping_windows(N_frames: int, duration_frames: int, n_gaps: int) -> List[Tuple[int, int]]:
    """Selects n_gaps non-overlapping start/end frame windows."""

    # Min number of frames between generated gaps
    min_buffer = 5
    total_space_needed = n_gaps * duration_frames + (n_gaps - 1) * min_buffer

    if total_space_needed > N_frames:
        return []

    frame_windows = []
    available_indices = np.arange(N_frames)

    for i in range(n_gaps):
        possible_starts = available_indices[available_indices <= N_frames - duration_frames]

        if len(possible_starts) == 0:
            print(f"    -> WARNING: Could only place {i} gaps. Skipping remaining.")
            break

        start_frame = np.random.choice(possible_starts)
        end_frame = start_frame + duration_frames
        frame_windows.append((start_frame, end_frame))

        removal_start = max(0, start_frame - min_buffer)
        removal_end = min(N_frames, end_frame + min_buffer)
        available_indices = available_indices[
            ~((available_indices >= removal_start) & (available_indices < removal_end))]

    return frame_windows


def create_gaps(
        positions: np.ndarray,
        num_markers: int,
        duration_frames: int,
        n_gaps: int = N_GAPS_PER_SEQUENCE
) -> np.ndarray:
    """
    Simulates N non-overlapping missing marker data blocks,
    where all N gaps use the same randomly chosen set of markers.

    Args:
        positions: The input marker data (N_frames, N_markers, 3).
        num_markers: How many markers to make missing for each gap.
        duration_frames: How many frames each gap lasts.
        n_gaps: The total number of non-overlapping gaps to create.

    Returns:
        The data array with NaNs inserted, or the original copy if skipping.
    """
    N_frames, N_markers, _ = positions.shape
    corrupted_data = positions.copy()

    frame_windows = select_non_overlapping_windows(N_frames, duration_frames, n_gaps)

    if len(frame_windows) != n_gaps:
        # This check is important as select_non_overlapping_windows might return fewer than n_gaps
        print(f"    -> SKIPPING file: Could not place {n_gaps} non-overlapping gaps.")
        return positions.copy()

    if num_markers > N_markers:
        num_markers = N_markers

    missing_marker_indices = np.random.choice(N_markers, num_markers, replace=False)
    marker_names = [TARGET_MARKERS[i] for i in missing_marker_indices]
    print(f"    -> Selected {num_markers} markers for all gaps: {', '.join(marker_names)}")

    for i in range(n_gaps):
        start_frame, end_frame = frame_windows[i]
        corrupted_data[start_frame:end_frame, missing_marker_indices, :] = np.nan
        print(
            f"    -> Gap {i + 1}: Frames {start_frame}-{end_frame} ({duration_frames}f) applied.")

    return corrupted_data


def save_data_to_csv(corrupted_data: np.ndarray, output_path: str):
    """Saves the corrupted data back to a CSV file."""

    N_frames, N_markers, _ = corrupted_data.shape

    # Flatten the corrupted data back to (N_frames, 3*N_markers)
    corrupted_flat = corrupted_data.reshape(N_frames, N_markers * 3)

    header = [f'{m}_{axis}' for m in TARGET_MARKERS for axis in ['X', 'Y', 'Z']]
    corrupted_df = pd.DataFrame(corrupted_flat, columns=header)
    corrupted_df.to_csv(output_path, index=False)


# ----------------------------------------------------------------------
# 3. Main Execution
# ----------------------------------------------------------------------

def run_gap_generation():
    """Main function to iterate over all files and scenarios."""

    os.makedirs(GAP_OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory created/verified: **{GAP_OUTPUT_DIR}**")

    csv_files = glob.glob(os.path.join(AUGMENTED_OUTPUT_DIR, '*.csv'))
    print(f"Found {len(csv_files)} CSV files in {AUGMENTED_OUTPUT_DIR} for gap generation.")

    if not csv_files:
        print(f"No files found in {AUGMENTED_OUTPUT_DIR}. Cannot proceed.")
        return

    total_files_generated = 0
    for file_path in csv_files:
        try:
            base_filename = os.path.basename(file_path).replace('.csv', '')
            print(f"\nProcessing file: {base_filename}")

            positions_data = load_data_from_csv(file_path)
            N_frames = positions_data.shape[0]

            for suffix, num_markers, duration_frames in SCENARIOS:

                # Min number of frames between generated gaps
                min_buffer = 5
                total_space_needed = N_GAPS_PER_SEQUENCE * duration_frames + (N_GAPS_PER_SEQUENCE - 1) * min_buffer
                if total_space_needed > N_frames:
                    print(
                        f"    -> SKIPPING scenario {suffix}: File too short ({N_frames}f) for {N_GAPS_PER_SEQUENCE} gaps of {duration_frames}f.")
                    continue

                corrupted_data = create_gaps(
                    positions=positions_data,
                    num_markers=num_markers,
                    duration_frames=duration_frames,
                    n_gaps=N_GAPS_PER_SEQUENCE
                )

                if np.isnan(corrupted_data).any():
                    output_filename = os.path.join(GAP_OUTPUT_DIR, f'{base_filename}_{suffix}.csv')
                    save_data_to_csv(corrupted_data, output_filename)
                    print(f"    -> File saved: {os.path.basename(output_filename)}")
                    total_files_generated += 1
                else:
                    print(f"    -> SKIPPING scenario {suffix}: Gap generation failed or file was too short internally.")
        except Exception as e:
            print(f"Fatal error processing file {file_path}: {e}")
            continue

    print(
        f"\n✅ Gap generation complete. Generated {total_files_generated} new CSV files in the **{GAP_OUTPUT_DIR}** directory.")


if __name__ == '__main__':
    run_gap_generation()
