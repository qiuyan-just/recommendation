from problem import Problem
from solver import Solver
import time #计时单位为秒/s
import numpy as np
import sys  # 需要引入的包


p = Problem()
p.build()
s = Solver(p)


# 以下为包装好的 Logger 类的定义
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



if __name__ == "__main__":

    filename = 'log_T=4_d=3_maximum_V2_HS.txt'
    starttime = time.time()
    log = Logger(filename)
    sys.stdout = log

    """求解primal dual模型"""
    # for lr in np.arange(0,1.01,0.01):
    #     lr=round(lr,2)
    #     p = Problem()
    #     p.build()
    #     s = Solver(p)
    #     print('现在学习率是:',lr)
    #     s.solve_primal_dual_model(lr)


    """求解完整的模型"""
    # s.solve_total_model()
    """greedy方案"""
    # s.greedy_solver()
    """threshold方案"""
    # s.threshold_solver()


    """求解primal dual 滚动时域模型"""
    for lr in np.arange(0,1.01,0.01):
        lr=round(lr,2)
        p = Problem()
        p.build()
        s = Solver(p)
        print('现在学习率是:',lr)
        s.solve_primal_dual_model_v2(lr)


    endtime = time.time()
    print('运行时间：', endtime - starttime)
