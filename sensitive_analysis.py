# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

"英文显示的图片使用下述字体"
plt.rcParams['font.sans-serif']=['Times New Roman']  #用来显示英文标签
#定义一种字体属性
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20}

names = ['CIM_cmin', 'HS_cmin', 'CIM_cmax', 'HS_cmax', 'CIM_cavg', 'HS_cavg']
x = range(len(names))

y_1 = [0.872, 0.891, 0.891, 0.91, 0.88, 0.902]
y_2 = [0.88, 0.902, 0.921, 0.921, 0.884, 0.908]
y_3 = [0.881, 0.903, 0.943, 0.94, 0.903, 0.907]
y_4 = [0.884, 0.917, 0.966, 0.96, 0.906, 0.922]

plt.plot(x, y_1, color='grey', marker='*', linestyle='-', label='T=1')
plt.plot(x, y_2, color='blue', marker='o', linestyle='-.', label='T=2')
plt.plot(x, y_3, color='green', marker='x', linestyle=':', label='T=3')
plt.plot(x, y_4, color='red', marker='s', linestyle='--', label='T=4')
plt.legend()  # 显示图例
plt.xticks(x, names,fontproperties='Times New Roman', rotation=25)
plt.ylabel("CR values")  # Y轴标签

plt.grid()

plt.savefig('./images/sensitive_analysis.tiff', format='tiff', dpi=600)
plt.show()

