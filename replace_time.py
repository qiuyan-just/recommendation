
#输出nump数组时，元素过多会自动省略中间的元素输出，为查看所有元素的值，使用numpy自带的set_printoptions来输出完整数组

import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)


### 在2019.12.24-2019.12.30中间有636位患者，考虑到计算需要，需要提前进行聚类

data = pd.read_csv(r"./data/肝脏外科/final_liver_time.csv", encoding='utf-8')

# #时间跨度为7，考虑设置的匹配轮数为2，
data['arrival time'] = data['arrival time'].replace('2019.12.24', '2019-12-24')
data['arrival time'] = data['arrival time'].replace('2019.12.25', '2019-12-25')
data['arrival time'] = data['arrival time'].replace('2019.12.26', '2019-12-26')
data['arrival time'] = data['arrival time'].replace('2019.12.27', '2019-12-27')
data['arrival time'] = data['arrival time'].replace('2019.12.28', '2019-12-28')
data['arrival time'] = data['arrival time'].replace('2019.12.29', '2019-12-29')
data['arrival time'] = data['arrival time'].replace('2019.12.30', '2019-12-30')

data.to_csv('./output/716_patients.csv', encoding='ANSI')










