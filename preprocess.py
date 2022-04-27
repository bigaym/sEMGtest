# 预处理的一些方法
import pandas as pd
import numpy as np
import glob
import os
from scipy.signal import butter, sosfilt, sosfreqz

# 从单个Excel文件中加载数据，返回numpy数组
def load_excel_file( filename):
    sheet = pd.read_excel(filename)
    data = np.array(sheet)
    return data

#从b包含多个Excel文件的文件夹中加载数组，返回外层list，内层numpy数组
def load_excel_folder( folder ):
    files = glob.glob(os.path.join(folder, "*.xlsx"))
    num = len(files)
    datas = list(list([]) for i in range(num))
    for name in files:
        index = int(os.path.basename(name)[:-5]) - 1
        datas[index].append(load_excel_file(name))
        print("loading ", name)
    return datas

#读取文件夹内的Excel转为txt文件,保存在和Excel文件父目录同级的txt文件夹
def excel_to_txt_folder(folder):
    files = glob.glob(os.path.join(folder, "*.xlsx"))
    savepath = os.path.join(os.path.dirname(folder), "txt")
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for name in files:
        data = load_excel_file(name)
        index = int(os.path.basename(name)[:-5])
        fname = str(index) + ".txt"
        savename = os.path.join(savepath, fname)
        np.savetxt(savename, data, fmt='%d')
        print("save ", savename, "successfully.")

#从b包含多个txt文件的文件夹中加载数组，返回外层list，内层numpy数组
def load_txt_folder(folder):
    files = glob.glob(os.path.join(folder, "*.txt"))
    num = len(files)
    datas = list(list([]) for i in range(num))
    for name in files:
        index = int(os.path.basename(name)[:-4]) - 1
        if os.path.getsize(name) < 1:
            datas[index] = np.zeros((1, 16))
        # datas[index].append(np.loadtxt(name))
        datas[index] = np.loadtxt(name, dtype=np.float32)
        print("loading ", name)
    return datas

# 归一化
def normalization_y(data_arry):
    m1 = [25, 60, 105, 30]
    m2 = [-105, -30, -25, -60]
    data = np.array(data_arry)
    for i in range(4):
        # m1 = np.max(data[:, i])
        # m2 = np.min(data[:, i])
        # print(m1, m2)
        data[:, i] = (data[:, i] - m2[i]) / (m1[i] - m2[i])
    return data

# 巴特沃斯带通滤波器参数计算
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

# 巴特沃斯带通滤波器使用
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

# 将信号剪成一截一截的,并打标签
def cut_signal(data_arr,cut_length):
    times = data_arr.shape[0]

# 测试
# import matplotlib.pyplot as plt
# workplace = parent = os.path.dirname(os.path.realpath(__file__))
# # file = workplace + r"\data\上坡\excel" + "\\1.xlsx"
# base_folder = workplace + r"\data\上坡\txt"
# # load_excel_folder(base_folder)
# # excel_to_txt_folder(base_folder)
# datas = load_txt_folder(base_folder)
# print(datas[0].shape)
# channels = 16
# degrees = np.array(datas[0][:, 12:channels])
# for i in range(1, len(datas)):
#     if len(datas[i]) < 100:
#         continue
#     else:
#         degrees = np.vstack((degrees, np.array(datas[i][:, 12:channels])))
# print(degrees.shape)
# degrees = degrees / 22.22222222
# print(np.max(np.max(degrees)), np.min(np.min(degrees)))
# plt.figure(1)
# degree_1 = degrees[:, 0]
# length = len(degree_1)
# window_length = 50  # 毫秒
# avg_x = []
# avg_y = []
# times = length // window_length
# for i in range(times):
#     index = (i+1)*window_length
#     avg_x.append(index)
#     avg = np.average(degree_1[i*window_length:(i+1)*window_length])
#     avg_y.append(avg)
# plt.plot(degree_1)
# plt.plot(avg_x, avg_y)
# plt.show()


# # 绘制频率响应
# sos = butter_bandpass(lowcut, highcut, fs, order=order)
# w, h = sosfreqz(sos, worN=2000)
