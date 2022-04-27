"""
@func: 在这里测试一些代码
"""


# 测试小波分解和重构
def dwt_test():
    import pywt
    import matplotlib.pyplot as plt
    import numpy as np
    import test1

    file = "./data/平地/txt/1.txt"
    datas = np.loadtxt(file, dtype=np.float32)
    # print(datas.shape)
    data0 = datas[:5000, 0]
    # 滤波
    # # 滤波器参数设置
    fs = 1000       # 采样频率
    lowcut = 51     # 低频截止频率
    hightcut = 300  # 高频截止频率
    order = 4       # 滤波器阶数
    data1 = test1.butter_bandpass_filter(data0, lowcut, hightcut, fs, order)
    # 滤波后信号显示
    plt.figure(1)
    plt.plot(data1)
    # 小波分解
    base_wave = 'db4'  # 小波基函数
    # print(pywt.Modes.modes)
    #*['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect'] *#
    mode = pywt.Modes.smooth
    a = data1
    cA = []  # 近似分量
    cD = []  # 细节分量

    for i in range(5):
        (a, d) = pywt.dwt(a, base_wave, mode)  # 进行5阶离散小波变换
        cA.append(a)
        cD.append(d)

    # for i in range(5):
    #     print(len(cA[i]))

    # 用于重组
    rec_a = []
    rec_d = []

    # 有时可能希望在waverec省略某些系数集的情况下运行。这可以通过将相应的数组设置为匹配形状和
    # dtype的零数组来最好地完成。不支持显式删除列表条目或将其设置为None。
    # 具体来说，要忽略第2级的细节系数，可以这样做：
    # coeffs[-2] == np.zeros_like(coeffs[-2])
    # coeffs is a list like [cAn, cDn, cDn - 1, …, cD2, cD1]

    for i, coeff in enumerate(cA):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, base_wave))  # 重构
        # print(len(rec_a[i]))

    for i, coeff in enumerate(cD):
        coeff_list = [None, coeff] + [None] * i
        if i ==3:
            print(len(coeff))
            print(len(coeff_list))
        rec_d.append(pywt.waverec(coeff_list, base_wave))

    fig = plt.figure(2)
    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))
    # print(pywt.families())
    # ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
    # print(pywt.wavelist('sym'))
    plt.show()


# 测试缩放
def zoom_test(zoom):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-50, 51, 100)
    y = -x*x
    y_zoom = -x*x / (np.power(zoom, 0.5)*zoom*zoom)

    plt.rcParams['font.family'] = ['sans-serif']  # 让 plt显示汉字
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 让 plt显示汉字
    plt.subplot(2, 1, 1)
    plt.plot(x, y)
    plt.title('原信号')
    plt.subplot(2, 1, 2)
    plt.title('缩放后信号')
    plt.plot(x, y_zoom)
    plt.show()


# 测试连续小波变换
def cwt_test():
    import numpy as np
    import pywt
    import matplotlib.pyplot as plt

    # wav = pywt.ContinuousWavelet('cmor1.5-1.0')
    wav = pywt.ContinuousWavelet('cgau6')
    # print the range over which the wavelet will be evaluated
    print("Continuous wavelet will be evaluated over the range [{}, {}]".format(
        wav.lower_bound, wav.upper_bound))

    width = wav.upper_bound - wav.lower_bound

    scales = [1, 2, 3, 4, 10, 15]

    max_len = int(np.max(scales) * width + 1)
    t = np.arange(max_len)
    fig, axes = plt.subplots(len(scales), 2, figsize=(12, 6))
    for n, scale in enumerate(scales):

        # The following code is adapted from the internals of cwt
        int_psi, x = pywt.integrate_wavelet(wav, precision=10)
        step = x[1] - x[0]
        j = np.floor(
            np.arange(scale * width + 1) / (scale * step))
        if np.max(j) >= np.size(int_psi):
            j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
        j = j.astype(np.int)

        # normalize int_psi for easier plotting
        int_psi /= np.abs(int_psi).max()

        # discrete samples of the integrated wavelet
        filt = int_psi[j][::-1]

        # The CWT consists of convolution of filt with the signal at this scale
        # Here we plot this discrete convolution kernel at each scale.

        nt = len(filt)
        t = np.linspace(-nt // 2, nt // 2, nt)
        axes[n, 0].plot(t, filt.real, t, filt.imag)
        axes[n, 0].set_xlim([-max_len // 2, max_len // 2])
        axes[n, 0].set_ylim([-1, 1])
        axes[n, 0].text(50, 0.35, 'scale = {}'.format(scale))

        f = np.linspace(-np.pi, np.pi, max_len)
        filt_fft = np.fft.fftshift(np.fft.fft(filt, n=max_len))
        filt_fft /= np.abs(filt_fft).max()
        axes[n, 1].plot(f, np.abs(filt_fft) ** 2)
        axes[n, 1].set_xlim([-np.pi, np.pi])
        axes[n, 1].set_ylim([0, 1])
        axes[n, 1].set_xticks([-np.pi, 0, np.pi])
        axes[n, 1].set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        axes[n, 1].grid(True, axis='x')
        axes[n, 1].text(np.pi / 2, 0.5, 'scale = {}'.format(scale))

    axes[n, 0].set_xlabel('time (samples)')
    axes[n, 1].set_xlabel('frequency (radians)')
    axes[0, 0].legend(['real', 'imaginary'], loc='upper left')
    axes[0, 1].legend(['Power'], loc='upper left')
    axes[0, 0].set_title('filter')
    axes[0, 1].set_title(r'|FFT(filter)|$^2$')
    plt.show()


# 测试连续小波变换2
def cwt_test2():
    import numpy as np
    import matplotlib.pyplot as plt
    import pywt

    sampling_rate = 1024  # 采样频率
    t = np.arange(0, 1.0, 1.0 / sampling_rate)  # 0-1.0之间的数，步长为1.0/sampling_rate
    f1 = 100  # 频率
    f2 = 200
    f3 = 300
    data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
                        [lambda t: np.sin(2 * np.pi * f1 * t),
                         lambda t: np.sin(2 * np.pi * f2 * t),
                         lambda t: np.sin(2 * np.pi * f3 * t)])
    print(len(data))
    wavename = "cgau8"  # 小波函数
    totalscal = 256  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
    fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    cparam = 2 * fc * totalscal  # 常数c
    scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)  # 连续小波变换模块

    plt.figure(figsize=(8, 4))
    plt.subplot(211)  # 第一整行
    plt.plot(t, data)
    plt.xlabel(u"time(s)")
    plt.title(u"300Hz 200Hz 100Hz Time spectrum")
    plt.subplot(212)  # 第二整行

    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)  # 调整边距和子图的间距 hspace为子图之间的空间保留的高度，平均轴高度的一部分
    plt.show()
    # ————————————————
    # 版权声明：本文为CSDN博主「小橙子喜欢吃果冻」的原创文章，遵循CC
    # 4.0
    # BY - SA版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https: // blog.csdn.net / weixin_50888378 / article / details / 111871131


# 测试连续小波变换，用肌电信号
def semg_cwt_test():
    import pywt
    import matplotlib.pyplot as plt
    import numpy as np
    import test1

    file = "./data/平地/txt/1.txt"
    datas = np.loadtxt(file, dtype=np.float32)
    # print(datas.shape)
    data0 = datas[:5000, 0]
    t = np.linspace(0, 5000, 5000)
    # 滤波
    # # 滤波器参数设置
    fs = 1000  # 采样频率
    lowcut = 51  # 低频截止频率
    hightcut = 300  # 高频截止频率
    order = 4  # 滤波器阶数
    data1 = test1.butter_bandpass_filter(data0, lowcut, hightcut, fs, order)
    # 滤波后信号显示
    plt.figure(1)
    plt.plot(data1)

    # 连续小波变换
    wavename = "morl"  # 小波函数
    totalscal = 256  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
    fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    cparam = 2 * fc * totalscal  # 常数c
    scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
    [cwtmatr, frequencies] = pywt.cwt(data1, scales, wavename, 1.0 / fs)  # 连续小波变换模块

    # 绘图
    plt.figure(figsize=(8, 4))
    plt.subplot(211)  # 第一整行
    plt.plot(t, data1)
    plt.xlabel(u"time(s)")
    plt.title(u"raw signal")
    plt.subplot(212)  # 第二整行

    plt.contourf(t, frequencies, abs(cwtmatr))
    print(np.array(frequencies).shape)
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)  # 调整边距和子图的间距 hspace为子图之间的空间保留的高度，平均轴高度的一部分
    plt.show()


def entropy_test():
    from pyentrp import entropy as ent
    import test1
    import numpy as np
    import matplotlib.pyplot as plt

    file = "./data/平地/txt/1.txt"
    datas = np.loadtxt(file, dtype=np.float32)
    # print(datas.shape)
    data0 = datas[:5000, 0]
    # 滤波
    # # 滤波器参数设置
    fs = 1000  # 采样频率
    lowcut = 51  # 低频截止频率
    hightcut = 300  # 高频截止频率
    order = 4  # 滤波器阶数
    data1 = test1.butter_bandpass_filter(data0, lowcut, hightcut, fs, order)
    # 滤波后信号显示
    plt.figure(1)
    plt.plot(data1)

    # 计算样本熵
    win_len = 20
    step = 1
    count = len(data1) // step - (win_len - step) // step
    sampen = []

    for i in range(count):
        ts = data1[i*step:(i+1)*step+(win_len-step)]
        temp = ent.sample_entropy(ts, 1, 1*np.std(data1, axis=-1))
        sampen.append(temp)
    sampen = np.array(sampen)
    print(sampen.shape)
    print(np.std(data1))

    # 绘图
    plt.figure(2)
    plt.plot(sampen)
    plt.title('sample entropy')
    plt.show()


# 主函数
if __name__ == '__main__':

    # 测试小波分解和重构
    # dwt_test()

    # 测试缩放
    # zoom_test(2)

    # 测试连续小波变换
    # cwt_test()

    # 测试连续小波变换2
    # cwt_test2()

    # 肌电信号连续小波变换测试
    # semg_cwt_test()

    # 测试样本熵计算
    entropy_test()

    print('over')

