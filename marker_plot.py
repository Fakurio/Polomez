import numpy as np
from KalmanEstimator import KalmanEstimator
from marker_groups import MARKER_GROUPS
import json
import matplotlib.pyplot as plt
import ezc3d

FILE_NAME = "k_krok_podstaw_uklon_polonez_1.3"
MARKER_NAME = "LFHD"

c3d = ezc3d.c3d(f"{FILE_NAME}.c3d")
marker_names = c3d['parameters']['POINT']['LABELS']['value']
markers = c3d['data']['points']  # (4, n_markers, n_frames)

marker_index = marker_names.index(MARKER_NAME)
marker_data = markers[:, marker_index, :]  # (4, n_frames), (x,y,z,1)
x = marker_data[0, :]
y = marker_data[1, :]
z = marker_data[2, :]

data = json.load(open("sequence.json", "r"))
ke = KalmanEstimator(MARKER_GROUPS)
positions = np.full((marker_data.shape[1], markers.shape[1], 3), np.nan)
for f_idx, frame in enumerate(list(data.keys())):
    estimated_frame = ke.estimate_frame(data[frame])
    for idx, val in enumerate(estimated_frame.values()):
        positions[f_idx, idx] = val
marker_pos_in_estimation_data = list(list(data.values())[0].keys()).index(MARKER_NAME)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(range(marker_data.shape[1]), x, label="Original")
ax[0].plot(range(marker_data.shape[1]), positions[:, marker_pos_in_estimation_data, 0], label="Estimated")
ax[0].set_title('X coord')
ax[0].legend()
ax[1].plot(range(marker_data.shape[1]), y, label="Original")
ax[1].plot(range(marker_data.shape[1]), positions[:, marker_pos_in_estimation_data, 1], label="Estimated")
ax[1].set_title('Y coord')
ax[1].legend()
ax[2].plot(range(marker_data.shape[1]), z, label="Original")
ax[2].plot(range(marker_data.shape[1]), positions[:, marker_pos_in_estimation_data, 2], label="Estimated")
ax[2].set_title('Z coord')
ax[2].legend()
plt.show()
