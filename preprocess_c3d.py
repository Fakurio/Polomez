import ezc3d
import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
import glob

# ----------------------------------------------------------------------
# 1. Configuration and Marker List
# ----------------------------------------------------------------------

DATA_DIR = "original_data"
OUTPUT_DIR = "augmented_output"

try:
    FILE_PATHS = glob.glob(os.path.join(DATA_DIR, '*.c3d'))
    if not FILE_PATHS:
        print(f"Warning: No .c3d files found in the directory: {DATA_DIR}. Please ensure files exist.")
    else:
        print(f"Found {len(FILE_PATHS)} C3D files in {DATA_DIR}.")
except Exception as e:
    print(f"Error reading directory {DATA_DIR}: {e}")
    FILE_PATHS = []

TARGET_MARKERS = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'RBAK', 'LSHO', 'LUPA', 'LELB', 'LFRM',
                  'LWRA', 'LWRB', 'LFIN', 'RSHO', 'RUPA', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RFIN', 'LASI', 'RASI',
                  'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK',
                  'RHEE', 'RTOE']


# ----------------------------------------------------------------------
# 2. Augmentation Functions
# ----------------------------------------------------------------------

def augment_rotation(marker_data: np.ndarray, axis: str) -> tuple[np.ndarray, str]:
    """
    Performs data augmentation by applying a random rotation around a specified axis.

    The input marker_data must be (3, N_markers, N_frames).
    Returns the augmented data and a description string.
    """
    angle_deg = np.random.uniform(-15, 15)

    r = R.from_euler(axis.lower(), angle_deg, degrees=True)
    rotation_matrix = r.as_matrix()  # Shape (3, 3)

    # Apply the rotation to all frames and all markers
    augmented_data = np.einsum('ij,jkl->ikl', rotation_matrix, marker_data)

    description = f"Rotation around {axis}-axis by {angle_deg:.2f}°"
    return augmented_data, description


def augment_translation(marker_data: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Performs data augmentation by applying a random translation (shift) in X, Y, Z.

    The input marker_data must be (3, N_markers, N_frames).
    Returns the augmented data and a description string.
    """
    # Choose random translation values in mm
    tx = np.random.uniform(-100, 100)
    ty = np.random.uniform(-100, 100)
    tz = np.random.uniform(-100, 100)

    translation_vector = np.array([tx, ty, tz])  # Shape (3,)
    augmented_data = marker_data + translation_vector[:, np.newaxis, np.newaxis]

    description = f"Translation (Tx={tx:.1f}, Ty={ty:.1f}, Tz={tz:.1f})"
    return augmented_data, description


# ----------------------------------------------------------------------
# 3. Standardization and Saving Utility
# ----------------------------------------------------------------------

def save_data_to_csv(marker_data: np.ndarray, base_filename: str, file_labels: list,
                     target_markers: list, standard_header: list, output_dir: str, suffix: str):
    """
    Helper function to standardize marker order, flatten data, add metadata, and save to CSV.
    """

    if marker_data.shape[0] != 3:
        raise ValueError("Marker data must have shape (3, N_markers, N_frames).")

    N_frames = marker_data.shape[2]
    standard_frame_data = np.full((3 * len(target_markers), N_frames), np.nan)

    for i, target_marker in enumerate(target_markers):
        try:
            source_index = file_labels.index(target_marker)
            marker_ts_data = marker_data[:, source_index, :]  # (3, N_frames)
            standard_frame_data[3 * i: 3 * (i + 1), :] = marker_ts_data
        except ValueError:
            # If a marker is missing in the file, it remains NaN (default fill value)
            continue

    # Flatten marker data: transpose to (N_frames, 3*N_markers)
    marker_df = pd.DataFrame(standard_frame_data.T, columns=standard_header)

    output_filename = os.path.join(output_dir, f'{base_filename}_{suffix}.csv')
    marker_df.to_csv(output_filename, index=False)
    print(f"  -> File saved: {output_filename} ({N_frames} frames) [Type: {suffix.upper()}]")


# ----------------------------------------------------------------------
# 4. Main Processing Function
# ----------------------------------------------------------------------

def process_c3d_files(file_paths: list, target_markers: list, output_dir: str):
    """
    Loads C3D files, applies 4 types of augmentation (Rx, Ry, Rz, T),
    reorders markers, and saves 4 augmented CSV files PLUS 1 original CSV file per C3D file.
    """

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory created/verified: **{output_dir}**")

    AUGMENTATION_TYPES = [
        ('rot_x', lambda data: augment_rotation(data, 'X')),
        ('rot_y', lambda data: augment_rotation(data, 'Y')),
        ('rot_z', lambda data: augment_rotation(data, 'Z')),
        ('trans', lambda data: augment_translation(data)),
    ]

    standard_header = []
    for marker in target_markers:
        standard_header.extend([f'{marker}_X', f'{marker}_Y', f'{marker}_Z'])

    print(f"Starting processing of {len(file_paths)} files, generating {len(AUGMENTATION_TYPES) + 1} files each...")
    total_files_generated = 0
    for file_path in file_paths:
        try:
            base_filename = os.path.basename(file_path).replace('.c3d', '')
            print(f"\nProcessing file: {file_path}")

            c3d = ezc3d.c3d(file_path)
            points_data = c3d['data']['points']
            file_labels = c3d['parameters']['POINT']['LABELS']['value']

            # Pruning the 4th dimension added by Vicon
            if points_data.shape[0] > 3:
                # Assuming the first 3 rows are X, Y, Z coordinates
                points_data = points_data[0:3, :, :]
                print(f"  -> WARNING: Pruned first dimension from {c3d['data']['points'].shape[0]} to 3 (XYZ).")
            elif points_data.shape[0] != 3:
                raise ValueError(f"Unexpected point data dimension: {points_data.shape[0]}. Expected 3 or more.")

            # Save the original file to CSV
            save_data_to_csv(
                points_data, base_filename, file_labels,
                target_markers, standard_header, output_dir,
                suffix='orig'
            )
            total_files_generated += 1

            # Augmentation loop
            for suffix, aug_func in AUGMENTATION_TYPES:
                augmented_points, description = aug_func(points_data)
                print(f"  -> Applying: {description} (Suffix: {suffix})")
                save_data_to_csv(
                    augmented_points, base_filename, file_labels,
                    target_markers, standard_header, output_dir,
                    suffix=suffix
                )
                total_files_generated += 1
        except Exception as e:
            print(f"Fatal error processing file {file_path}: {e}")
            continue

    print(
        f"\n✅ Processing complete. Generated {total_files_generated} CSV files (Original + 4 Augmented) in the **{output_dir}** directory.")


if __name__ == '__main__':
    process_c3d_files(FILE_PATHS, TARGET_MARKERS, output_dir=OUTPUT_DIR)
