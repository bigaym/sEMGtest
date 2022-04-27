from scipy.signal import butter, lfilter
import numpy as np
import math



# 计算RMS
def RMS(seqs, win_len, step):
    n = len(seqs)
    temp = np.power(seqs[:win_len], 2)
    rmss = [0] * (n//step - win_len//step + 1)
    rmss[0] = np.sqrt(sum(temp / win_len))
    for s in range(n//step - win_len//step):
        squares = np.power(seqs[win_len+s*step:win_len+(s+1)*step], 2)
        temp[:win_len-step] = temp[step:]
        temp[win_len-step:] = squares
        rms = np.sqrt(sum(temp / win_len))
        rmss[s+1] = rms
    return rmss


# 计算过零率
def ZeroCR(waveData, win_len, step):
    wlen = len(waveData)
    frameNum = (wlen - win_len + step) // step
    zcr = [0] * frameNum
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step, i*step+win_len)]
        curFrame = curFrame - np.mean(curFrame) #  zero-justified
        zcr[i] = sum(curFrame[0:-1]*curFrame[1::]<=0)
    return zcr


# 计算近似熵
def ApEn(wave, m, r):

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[wave[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(wave)

    return abs(_phi(m+1) - _phi(m))


# 计算样本熵
def SampEn(U, m, r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        B = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1.0) / (N - m + 1) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(B)

    N = len(U)
    if _phi(m) != 0:
        return -np.log(_phi(m + 1) / _phi(m))
    else:
        return -np.log(_phi(m + 1) / 1)

# 计算巴特沃斯带通滤波参数b, a
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# 滤波函数
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# 利用样本熵和近似熵之差阈值检测活动段
def point_detect_en(wave, win_len, step, en_m, en_r, low_ratio, high_ratio, belif_ratio=0.7):
    """
    利用样本熵和近似熵之差阈值检测活动段
    :param wave: 一维波
    :param win_len: 滑动窗长
    :param step: 滑动步长
    :param en_m: 近似熵和样本熵的m参数，一般2~3
    :param en_r: 近似熵和样本熵的r参数，一般0.1~0.25*std
    :param low_ratio: 双门限的低门限比率，用于检测终点
    :param high_ratio: 双门限的高门限比率，用于检测起点
    :return: 活跃段坐标集
    """
    action = 0        # 非活跃标记
    action_list = []  # 活跃段坐标集
    wlen = len(wave)  # 样本点数
    count = wlen//step - (win_len-step)//step  # 窗口滑动次数
    suben = [0] * count  # 所有近似熵和样本熵之差的绝对值

    # 计算熵差
    for i in range(count):
        apen = ApEn(wave[i*step:(i+1)*step+(win_len-step)], en_m, en_r)      # 近似熵
        sampen = SampEn(wave[i*step:(i+1)*step+(win_len-step)], en_m, en_r)  # 样本熵
        if sampen == float('inf'):
            sampen = 0
        suben[i] = abs(apen-sampen)
    plt.plot(suben)
    plt.title('complex En')
    plt.show()
    # 活动段检测
    lowcut = max(suben) * low_ratio  # 低门限阈值
    hightcut = lowcut * high_ratio / low_ratio
    check_steps = 10                # 滑动窗长
    belif = math.ceil(belif_ratio*check_steps)   # 置信度
    for j in range(count-check_steps+1):
        seqs = suben[j*2:j*2+check_steps]
        if action == 0:
            high_num = len([i for i in seqs if i >= hightcut])
            if high_num >= belif:
                action_list.append((j-belif)*step)
                action = 1
            else:
                continue
        elif action == 1:
            low_num = len([i for i in seqs if i <= lowcut])
            if low_num >= belif:
                action_list.append((j-belif)*step)
                action = 0
            else:
                continue
        else:
            continue
    # for j in range(count):
    #     if (suben[j] > hightcut) and (action == 0):
    #         point_start = j*10
    #         action_list.append(point_start)
    #         action = 1
    #     elif (suben[j] < lowcut) and (action == 1):
    #         point_end = j*10
    #         action_list.append(point_end)
    #         action = 0
    #     else:
    #         pass
    return action_list


# main
if __name__ == '__main__':
    import glob
    import time
    import matplotlib.pyplot as plt

    start = time.time()

    # 参数设置
    fs = 1000       # 采样频率
    lowcut = 51     # 低频截止频率
    hightcut = 150  # 高频截止频率
    order = 4       # 滤波器阶数
    win_len = 100   # 窗口长度
    step = 10       # 窗口滑动步长

    # 加载数据
    datafolder = 'data/平地/txt/'
    datanames = glob.glob('data/平地/txt/*.txt')
    data = np.loadtxt(datanames[0])
    channel0 = data[2000:5000, 0]
    end_load = time.time()
    print("数据加载时间：{}秒".format(end_load - start))

    # 带通滤波
    channel0_bandpass = butter_bandpass_filter(channel0, lowcut, hightcut, fs, order)
    end_bandpass = time.time()
    print("带通滤波用时：{}秒".format(end_bandpass - end_load))

    # 求RMS
    rmses = RMS(channel0_bandpass, win_len, step)
    print(len(rmses))
    end_rms = time.time()
    print("计算RMS时间：{}秒".format(end_rms - end_bandpass))
    
    # 求过零率
    zcr = ZeroCR(channel0_bandpass, win_len, step)
    print(len(zcr))
    end_zcr = time.time()
    print("计算过零率时间：{}秒".format(end_zcr - end_rms))

    # 计算熵
    # frames = 20            # 帧长
    # steps_frame = 10       # 移动步长
    # shang = [0] * len(rmses)
    # for i in range(len(rmses)):
    #     apen = ApEn(channel0_bandpass[i*steps_frame:(i+1)*steps_frame+frames-steps_frame], m=2, r=3)
    #     sampen = SampEn(channel0_bandpass[i*steps_frame:(i+1)*steps_frame+frames-steps_frame], m=2, r=3)
    #     sub = abs(apen-sampen)
    #     shang[i] = sub
    # print(len(shang))
    # end_apen = time.time()
    # endPointDetect(channel0_bandpass, rmses, zcr)

    # 端点检测
    action_list = point_detect_en(channel0_bandpass,
                                  win_len=20,
                                  step=1,
                                  en_m=2,
                                  en_r=3,
                                  low_ratio=0.1,
                                  high_ratio=0.5)
    print("端点检测结果：{}".format(action_list))
    print(len(action_list))
    # 运算结束
    end = time.time()
    print("本次运行用时{}秒".format(end - start))

    # 绘图
    plt.figure(1)
    plt.subplot(4, 1, 1)
    plt.plot(data[:, -1])
    plt.title('raw signal')
    plt.subplot(4, 1, 2)
    plt.plot(channel0_bandpass)
    plt.title('filtered signal')
    plt.subplot(4, 1, 3)
    plt.plot(rmses)
    plt.title('rms')
    plt.subplot(4, 1, 4)
    plt.plot(channel0_bandpass)
    for point in action_list:
        plt.vlines(point, 0, 100, 'r', '--')
    plt.title('action detection')
    plt.show()





# count = 0
# for dataname in datanames:
#     count += 1
#     data1 = np.loadtxt(dataname)
#     print('{}的数据规模为：{}'.format(dataname, data1.shape))
#     plt.figure(count)
#     for i in range(1, 13):
#         plt.subplot(14, 1, i)
#         plt.plot(data1[:, i - 1])
#     plt.subplot(14, 1, 13)
#     plt.plot(data1[:, 13])
#     plt.subplot(14, 1, 14)
#     plt.plot(data1[:, 15])
#     plt.show()


