import scipy.signal
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import datetime


def show_config(config):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in config.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def plot(data1, data2, save_path):
    iters = range(len(data1))

    plt.figure()
    plt.plot(iters, data1, 'red', linewidth=2, label='Training loss')
    plt.plot(iters, data2, 'coral', linewidth=2, label='Dice coefficient')
    # try:
    #     if len(data1) < 25:
    #         num = 5
    #     else:
    #         num = 15
    #
    #     plt.plot(iters, scipy.signal.savgol_filter(data1, num, 3), 'green', linestyle='--', linewidth=2,
    #              label='Smooth training loss')
    #     plt.plot(iters, scipy.signal.savgol_filter(data2, num, 3), '#8B4513', linestyle='--', linewidth=2,
    #              label='Smooth dice coefficient')
    # except:
    #     pass

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    plt.savefig(save_path, dpi=1000, format="svg")  # Wiley,折线图dpi=600,图像dpi=300

    plt.cla()
    plt.close("all")


def show_label(label):
    # img = Image.fromarray(np.uint8(label))  # abel是数组形式
    img = label.convert('RGBA')
    x, y = img.size
    for i in range(x):
        for j in range(y):
            color = img.getpixel((i, j))
            Mean = np.mean(list(color[:-1]))
            if Mean < 255:  # 我的标签区域为白色，非标签区域为黑色
                color = color[:-1] + (0,)  # 若非标签区域则设置为透明
            else:
                color = (255, 97, 0, 255)  # 标签区域设置为橙色，前3位为RGB值，最后一位为透明度情况，255为完全不透明，0为完全透明
            img.putpixel((i, j), color)
    return img
