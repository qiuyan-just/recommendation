
'''
患者到达时间跟患者人数之间的散点图绘制
'''

"""
第一步
根据arrival time统计重复项的出现次数
"""
# import pandas as pd
#
# df = pd.read_csv(r'./data/到达记录/肝脏外科/10_dryanjianjun.haodf.com.csv')
#
# df_count = df['arrival time'].value_counts().reset_index(name='出现次数').rename({"index":"日期"},axis='columns')
#
# df_count.to_csv('./output/到达统计/肝脏外科/10_dryanjianjun.haodf.com_统计.csv', encoding='ANSI')



"""
第二步
基于第一步的相同时间点出现的次数，将日期改为具体的数字，作为输入
"""


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./output/到达统计/96刘海波_不同时间患者人数统计 - 副本.csv',encoding='ANSI')

x = df['日期']
y = df['出现次数']

plt.figure(figsize=(13,10))

"英文显示的图片使用下述字体"
plt.rcParams['font.sans-serif']=['Times New Roman']  #用来显示英文标签
#定义一种字体属性
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20}

"中文显示的图片使用下述字体"
from matplotlib import rcParams

# config = {
#             "font.family": 'serif',
#             "font.size": 16,# 相当于小四大小
#             "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
#             "font.serif": ['SimSun'],#宋体
#             'axes.unicode_minus': False # 处理负号，即-号
#          }
# rcParams.update(config)


plt.xlabel("Days", fontsize=20, weight='bold') #设置x轴名称
# plt.xlabel("时间", fontsize=16) #设置x轴名称
plt.xticks(fontproperties='Times New Roman', rotation=30, fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.ylabel("Number of Arrivals", fontsize=20, weight='bold')
# plt.ylabel("患者到达数量", fontsize=16)

# plt.title('The distribution of patient arrivals for physician 3', fontdict=font1)
# plt.title('患者到达的分布情况')

# plt.scatter(x,y, c='r', marker='o')
plt.scatter(x,y)

plt.grid()


plt.savefig('./images/physician A.tiff', format='tiff', dpi=600)
plt.show()

# plt.savefig('./images/Physician3.pdf', format = 'pdf', dpi=600)
# plt.show()
