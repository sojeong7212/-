import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# 데이터 로드
filename2read = "./data/imgs/img_1.npy"
data_in = np.load(filename2read)
print(data_in.shape)

# 초기 인덱스
idx_0 = 0

# 플롯 생성
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
data_in_sel = data_in[idx_0, :, :]
image = ax.imshow(data_in_sel, cmap='gray')
plt.title(f"Index: {idx_0}")

# 콜백 함수 정의
def next_image(event):
    global idx_0
    if idx_0 < data_in.shape[0] - 1:
        idx_0 += 1
    update_image()

def prev_image(event):
    global idx_0
    if idx_0 > 0:
        idx_0 -= 1
    update_image()

def update_image():
    data_in_sel = data_in[idx_0, :, :]
    image.set_data(data_in_sel)
    ax.set_title(f"Index: {idx_0}")
    plt.draw()

# 버튼 생성
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bprev = Button(axprev, 'Previous')
bnext.on_clicked(next_image)
bprev.on_clicked(prev_image)

plt.show()
