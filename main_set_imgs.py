import os
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

NUM_DATA = 50
TIME_INTERVAL = 0.05
PLOT_SIZE = [4.0, 4.0]
X_PLOT_RANGE = [-7, +7]
Y_PLOT_RANGE = [-7, +7]
IMG_SIZE = [100, 100]

# Pillow 버전에 따른 리샘플링 필터 설정
try:
    resample_filter = Image.Resampling.BICUBIC  # CUBIC 필터 사용
except AttributeError:
    resample_filter = Image.BICUBIC  # 또는 Image.ANTIALIAS (구버전에서)


if __name__ == '__main__':
    DIR2SAVE = "./data/imgs"
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
        x_selected_coords = x_coords_in[selected_indices]
        y_selected_coords = y_coords_in[selected_indices]

        xy_selected_coords = np.stack((x_selected_coords, y_selected_coords), axis=-1)

        # 저장장소
        data_out = np.zeros((xy_selected_coords.shape[0], IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)

        # DPI 계산
        dpi_value = IMG_SIZE[0] / PLOT_SIZE[0]

        # 각 데이터 포인트에 대해 이미지 생성 및 배열에 저장
        for j in range(xy_selected_coords.shape[0]):
            xy_selected_coords_sel = xy_selected_coords[j, :, :]

            fig = plt.figure(figsize=(PLOT_SIZE[0], PLOT_SIZE[1]), dpi=dpi_value)

            # 플롯 생성
            plt.plot(
                xy_selected_coords_sel[:, 0],
                xy_selected_coords_sel[:, 1],
                marker='.',
                linestyle='none',
                markersize=1.5,
                color='k'
            )
            plt.xlim(X_PLOT_RANGE[0], X_PLOT_RANGE[1])
            plt.ylim(Y_PLOT_RANGE[0], Y_PLOT_RANGE[1])

            # 축 가져오기
            ax = plt.gca()
            ax.axis('off')  # 축 숨기기
            ax.set_position([0, 0, 1, 1])  # 축 위치 조정
            ax.margins(0)  # 마진 제거

            # 이미지를 메모리 버퍼에 저장
            with io.BytesIO() as buf:
                plt.savefig(buf, format='png')  # bbox_inches와 pad_inches 제거
                buf.seek(0)
                with Image.open(buf) as img:
                    img = img.convert('L')  # 그레이스케일로 변환
                    img_array = np.array(img)

                    # 필요에 따라 이미지 크기 조정
                    if img_array.shape != (IMG_SIZE[0], IMG_SIZE[1]):
                        img_array = np.array(
                            Image.fromarray(img_array).resize(
                                (IMG_SIZE[1], IMG_SIZE[0]), resample=resample_filter
                            )
                        )

                    # 데이터 타입 확인 및 조정
                    if img_array.dtype != np.uint8:
                        img_array = img_array.astype(np.uint8)

                    data_out[j, :, :] = img_array / 255.0  # 0~1 범위로 정규화

            # plt.matshow(data_out[j, :, :])
            # plt.show()
            plt.close(fig)

        np.save(os.path.join(DIR2SAVE, "img_{:d}.npy".format(data_idx)), data_out)
        print("SAVED: img_{:d}.npy".format(data_idx))
