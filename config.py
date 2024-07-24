import os

'''
心内科数据：

T=2, d=1时
minimum capacity对应的最优学习率是0.25
maximum capacity对应的最优学习率是0.13
avgerage capacity对应的最优学习率是0.2

T=3, d=1时
minimum capacity对应的最优学习率是0.19
maximum capacity对应的最优学习率是0.11
avgerage capacity对应的最优学习率是0.24

T=4, d=1时
minimum capacity对应的最优学习率是0.21
maximum capacity对应的最优学习率是0.11
avgerage capacity对应的最优学习率是0.23

'''


'''
肝脏外科数据：
T=2, d=1时
minimum capacity对应的最优学习率是0.39
maximum capacity对应的最优学习率是0.15
avgerage capacity对应的最优学习率是0.36

T=3, d=1时
minimum capacity对应的最优学习率是0.39
maximum capacity对应的最优学习率是0.14
avgerage capacity对应的最优学习率是0.39

T=4, d=1时
minimum capacity对应的最优学习率是0.39
maximum capacity对应的最优学习率是0.16
avgerage capacity对应的最优学习率是0.39

'''

class Config(object):

    """学习率"""
    gamma = 0.34
    """决策周期的个数"""
    period = 1
    """阈值"""
    threshold = 0.6
    """滚动时域最大滚动次数"""
    # iteration = 1 ## 延迟决策，表示在period决策周期再往后推一天
    iteration = 1


    """总的时域个数"""
    total_period = period + iteration
    """文件路径"""
    root_folder_path = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(root_folder_path, "data")
    """数据文件"""
    patient_file_path = os.path.join(data_folder_path, "./肝脏外科/716_sequence_patients.csv")
    doctor_file_path = os.path.join(data_folder_path, "./肝脏外科/capacity_avg.csv")
    matching_file_path = os.path.join(data_folder_path, "./肝脏外科/matching_quality_716-36.csv")

    # patient_file_path = os.path.join(data_folder_path, "./心内科/1436_sequence_patients.csv")
    # doctor_file_path = os.path.join(data_folder_path, "./心内科/capacity_avg.csv")
    # matching_file_path = os.path.join(data_folder_path, "./心内科/matching_quality_1436-60.csv")



















