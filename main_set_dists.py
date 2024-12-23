import os
import numpy as np
import io

NUM_DATA = 50
TIME_INTERVAL = 0.05
MAX_D = 15


if __name__ == '__main__':
    DIR2SAVE = "./data/dists"
    if not os.path.exists(DIR2SAVE):
        os.makedirs(DIR2SAVE)
    for data_idx in range(NUM_DATA):
        data_in = np.load("./data/data_{:d}.npz".format(data_idx))

        timestamps_in = data_in["timestamps"]
        directions_in = data_in["directions"]
        distances_in = data_in["distances"]
        x_coords_in = data_in["x_coords"]
        y_coords_in = data_in["y_coords"]

        # 첫 번째 시간에서 시작하여 타겟 간격에 가장 가까운 값을 선택
        selected_indices = []
        current_time = timestamps_in[0]

        for i, t in enumerate(timestamps_in):
            if t >= current_time:
                selected_indices.append(i)  # 현재 인덱스 추가
                current_time = t + TIME_INTERVAL  # 다음 목표 시간 갱신

        # 선택된 timesteps와 coords
        distances_sel = distances_in[selected_indices]

        # np.inf 처리
        infinite_indices = np.isinf(distances_sel)
        distances_sel[infinite_indices] = MAX_D

        # 저장
        np.save(os.path.join(DIR2SAVE, "dist_{:d}.npy".format(data_idx)), distances_sel)
        print("SAVED: dist_{:d}.npy".format(data_idx))