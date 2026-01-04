import json
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

data = json.load(open("sequence_with_kalman_online.json", "r"))
frame_list = list(data.keys())
marker_list = list(data[frame_list[0]].keys())
frames_count = len(frame_list)

positions = np.full((len(frame_list), len(marker_list), 3), np.nan)
for f_idx, frame in enumerate(frame_list):
    for m_idx, marker in enumerate(marker_list):
        positions[f_idx, m_idx, 0] = data[frame][marker][0][0]
        positions[f_idx, m_idx, 1] = data[frame][marker][0][1]
        positions[f_idx, m_idx, 2] = data[frame][marker][0][2]
x = positions[:, :, 0]
y = positions[:, :, 1]
z = positions[:, :, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_min, x_max = np.nanmin(x), np.nanmax(x)
y_min, y_max = np.nanmin(y), np.nanmax(y)
z_min, z_max = np.nanmin(z), np.nanmax(z)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

sc = ax.scatter(x[0, :], y[0, :], z[0, :])


def update(frame):
    sc._offsets3d = (x[frame, :], y[frame, :], z[frame, :])
    return sc,


ani = FuncAnimation(fig, update, frames=frames_count, interval=10, blit=False)
plt.show()
