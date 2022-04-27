import numpy as np


def SampEn(U, m, r):
    """
    用于量化时间序列的可预测性
    :param U: 时间序列
    :param m: 模板向量维数
    :param r: 距离容忍度，一般取0.1~0.25倍的时间序列标准差，也可以理解为相似度的度量阈值
    :return: 返回一个-np.log(A/B)，该值越小预测难度越小
    """

    def _maxdist(x_i, x_j):
        """
         Chebyshev distance
        :param x_i:
        :param x_j:
        :return:
        """
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)
    return -np.log(_phi(m + 1) / _phi(m))


if __name__ == '__main__':
    _U = [0.2, 0.6, 0.7, 1.2, 55, 66]
    rand_small = np.random.randint(0, 100, size=120)
    rand_big = np.random.randint(0, 100, size=136)
    m = 2
    print(SampEn(_U, m, r=0.2 * np.std(_U)))
    print(SampEn(rand_small, m, r=0.2 * np.std(rand_small)))
    print(SampEn(rand_big, m, r=0.2 * np.std(rand_big)))
