import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
filename2read = "./data/data_0.npz"
data_in = np.load(filename2read)

timestamps_in = data_in["timestamps"]
directions_in = data_in["directions"]
distances_in = data_in["distances"]
x_coords_in = data_in["x_coords"]
y_coords_in = data_in["y_coords"]


idx_sel = 0

x_coords_sel = x_coords_in[idx_sel]
y_coords_sel = y_coords_in[idx_sel]
distances_sel = distances_in[idx_sel]

theta_range = np.arange(-np.pi, np.pi, np.pi/360)

x_coords_recover = []
y_coords_recover = []
for i in range(theta_range.shape[0]):
    theta_sel = theta_range[i]
    d_sel = distances_sel[i]
    x_sel = d_sel * np.cos(theta_sel)
    y_sel = d_sel * np.sin(theta_sel)
    x_coords_recover.append(x_sel)
    y_coords_recover.append(y_sel)

import matplotlib.pyplot as plt

plt.plot(x_coords_sel[0:12], y_coords_sel[0:12], 'bo')
plt.plot(x_coords_recover, y_coords_recover, 'rx')
plt.axis('equal')
plt.show()