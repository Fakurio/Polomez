import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
import pandas as pd

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

CSV_FILE_PATH = "estimated_output/filled_k_krok_podstaw_uklon_polonez_1_orig_10m_100f_c3.csv"
TARGET_MARKERS = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'RBAK', 'LSHO', 'LUPA', 'LELB', 'LFRM',
                  'LWRA', 'LWRB', 'LFIN', 'RSHO', 'RUPA', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RFIN', 'LASI', 'RASI',
                  'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK',
                  'RHEE', 'RTOE']

# ----------------------------------------------------------------------
# Data Loading and Reshaping
# ----------------------------------------------------------------------
try:
    df = pd.read_csv(CSV_FILE_PATH)
    marker_cols = [col for col in df.columns if col not in ['File_ID', 'Frame_Index']]

    # Check if we have the expected number of columns (3 * N_markers)
    if len(marker_cols) != len(TARGET_MARKERS) * 3:
        expected_cols = [f'{m}_{axis}' for m in TARGET_MARKERS for axis in ['X', 'Y', 'Z']]
        df = df.reindex(columns=expected_cols, fill_value=np.nan)
        marker_data_flat = df[expected_cols].values
    else:
        marker_data_flat = df[marker_cols].values

    frames_count = marker_data_flat.shape[0]
    num_markers = len(TARGET_MARKERS)

    # Reshape the flat data into (N_frames, N_markers, 3)
    # The data is currently (N_frames, 3*N_markers)
    positions = marker_data_flat.reshape(frames_count, num_markers, 3)

    print(f"Data loaded from {CSV_FILE_PATH}")
    print(f"Total Frames: {frames_count}, Markers: {num_markers}")

except FileNotFoundError:
    print(f"ERROR: The file '{CSV_FILE_PATH}' was not found. Please verify the path.")
    exit()

except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

x = positions[:, :, 0]
y = positions[:, :, 1]
z = positions[:, :, 2]

# ----------------------------------------------------------------------
# Visualization Setup
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set consistent axes limits based on the min/max across all frames
x_min, x_max = np.nanmin(x), np.nanmax(x)
y_min, y_max = np.nanmin(y), np.nanmax(y)
z_min, z_max = np.nanmin(z), np.nanmax(z)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Initialize the scatter plot with the first frame
sc = ax.scatter(x[0, :], y[0, :], z[0, :], c='blue', marker='o', s=20)


# ----------------------------------------------------------------------
#  Run Animation
# ----------------------------------------------------------------------
def update(frame):
    sc._offsets3d = (x[frame, :], y[frame, :], z[frame, :])
    return sc,


# interval=30: Delay between frames in ms (approx 33 FPS)
ani = FuncAnimation(fig, update, frames=frames_count, interval=30, blit=False)

plt.tight_layout()
plt.show()
