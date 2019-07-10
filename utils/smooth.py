# -*- coding: utf-8 --*--
# @Author: Zessay
# @time: 2019.05.05 19:23
# @File: smooth.py
# @Software: PyCharm

import time
import scipy.special as special
import numpy as np


# 定义贝叶斯平滑的函数
## 用于平滑单词到实体的转化率

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        #         print(sample)
        I = []
        C = []
        for clk_rt in sample:
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):

        t = time.time()
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta
            t2 = time.time()
            if i % 50 == 0 and i != 0: 
                print(f'\t\t已迭代到第{i}轮 | 用时：{t2 - t}s')

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
        #         print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        for i in range(0, len(imps), 1):  # 步长复用时去掉
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))
        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)


# 第一个输入I是总次数，第二个输入C是出现次数
def bys(I, C, iter_num=500, epsilon=0.00001):
    bs = BayesianSmoothing(1, 1)
    # I, C = bs.sample(500, 500, 10, 1000)
    bs.update(I, C, iter_num=iter_num, epsilon=epsilon)  #
    # print(bs.alpha, bs.beta)
    ctr = []
    for i in range(len(I)):
        ctr.append((C[i] + bs.alpha) / (I[i] + bs.alpha + bs.beta))
    return ctr