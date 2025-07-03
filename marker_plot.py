import numpy as np
import json
import matplotlib.pyplot as plt
import ezc3d

FILE_NAME = "m_krok_podstaw_uklon_polonez_1b.3"
MARKER_NAME = "CLAV"

c3d = ezc3d.c3d(f"{FILE_NAME}.c3d")
marker_names = c3d['parameters']['POINT']['LABELS']['value']
markers = c3d['data']['points']  # (4, n_markers, n_frames)
marker_index = marker_names.index(MARKER_NAME)
marker_data = markers[:, marker_index, :]  # (4, n_frames), (x,y,z,1)

x_original = marker_data[0, :]
y_original = marker_data[1, :]
z_original = marker_data[2, :]

filtered_data = json.load(open("sequence_with_kalman_online.json", "r"))
positions = np.full((marker_data.shape[1], markers.shape[1], 3), np.nan)  # (n_frames, n_markers, 3)
for f_idx, frame in enumerate(list(filtered_data.keys())):
    for idx, val in enumerate(filtered_data[frame].values()):
        positions[f_idx, idx] = val
marker_pos_in_filtered_data = list(list(filtered_data.values())[0].keys()).index(MARKER_NAME)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(range(marker_data.shape[1]), x_original, label="Original")
ax[0].plot(range(marker_data.shape[1]), positions[:, marker_pos_in_filtered_data, 0], label="Estimated")
ax[0].set_title('X coord')
ax[0].legend()
ax[1].plot(range(marker_data.shape[1]), y_original, label="Original")
ax[1].plot(range(marker_data.shape[1]), positions[:, marker_pos_in_filtered_data, 1], label="Estimated")
ax[1].set_title('Y coord')
ax[1].legend()
ax[2].plot(range(marker_data.shape[1]), z_original, label="Original")
ax[2].plot(range(marker_data.shape[1]), positions[:, marker_pos_in_filtered_data, 2], label="Estimated")
ax[2].set_title('Z coord')
ax[2].legend()
plt.show()
