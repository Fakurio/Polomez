import ezc3d
import json
import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt


def load_c3d_marker_data(c3d_filepath, marker_name):
    c3d = ezc3d.c3d(c3d_filepath)
    marker_names = c3d['parameters']['POINT']['LABELS']['value']
    marker_index = marker_names.index(marker_name)
    marker_data = []

    for i in range(c3d["header"]["points"]["last_frame"] + 1):
        x = c3d["data"]["points"][0, marker_index, i]
        y = c3d["data"]["points"][1, marker_index, i]
        z = c3d["data"]["points"][2, marker_index, i]
        marker_data.append(np.array([x, y, z]))

    return np.array(marker_data)


def load_json_marker_data(json_filepath, marker_name):
    json_data = json.load(open(json_filepath, "r"))
    marker_data = []

    for _, markers in json_data.items():
        marker_data.append(np.array(markers[marker_name]))

    return np.array(marker_data)


ORIGINAL_FILE = "m_krok_podstaw_uklon_polonez_1b.3.c3d"
JSON_FILE = "sequence_with_kalman_online.json"
MARKER_NAME = "CLAV"

original_sequence = load_c3d_marker_data(ORIGINAL_FILE, MARKER_NAME)
filtered_sequence = load_json_marker_data(JSON_FILE, MARKER_NAME)

alignment = dtw(original_sequence, filtered_sequence, keep_internals=True)
original_sequence_mapped = original_sequence[alignment.index1]
filtered_sequence_mapped = filtered_sequence[alignment.index2]

# Cut mapped sequences to length of original sequence
original_sequence_mapped = original_sequence_mapped[:original_sequence.shape[0]]
filtered_sequence_mapped = filtered_sequence_mapped[:original_sequence.shape[0]]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(range(original_sequence_mapped.shape[0]), original_sequence_mapped[:, 0], label="Original", color="red")
ax[0].plot(range(filtered_sequence_mapped.shape[0]), filtered_sequence_mapped[:, 0], label="Estimated")
ax[0].set_title('X coord')
ax[0].legend()
ax[1].plot(range(original_sequence_mapped.shape[0]), original_sequence_mapped[:, 1], label="Original", color="red")
ax[1].plot(range(filtered_sequence_mapped.shape[0]), filtered_sequence_mapped[:, 1], label="Estimated")
ax[1].set_title('Y coord')
ax[1].legend()
ax[2].plot(range(original_sequence_mapped.shape[0]), original_sequence_mapped[:, 2], label="Original", color="red")
ax[2].plot(range(filtered_sequence_mapped.shape[0]), filtered_sequence_mapped[:, 2], label="Estimated")
ax[2].set_title('Z coord')
ax[2].legend()
plt.show()
