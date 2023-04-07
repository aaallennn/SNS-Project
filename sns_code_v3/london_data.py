import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
#from prophet import Prophet
import numpy as np


# 提取指定路径文件夹下的所有文件路径
def listdir(path):
    list_name =[]
    file_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name.loc

data_london = pd.read_csv('london_weather.csv')
print(data_london.columns)
print(data_london.info())
print(data_london.isna().sum())



# data_london.loc[data_london['cloud_cover']<-9998,'cloud_cover'] = np.nan
# data_london.loc[data_london['sunshine']<-9998,'sunshine'] = np.nan
# data_london.loc[data_london['global_radiation']<-9998,'global_radiation'] = np.nan
# data_london.loc[data_london['max_temp']<-9998,'max_temp'] = np.nan
# data_london.loc[data_london['min_temp']<-9998,'min_temp'] = np.nan
# data_london.loc[data_london['precipitation']<-9998,'precipitation'] = np.nan
################################################################################################

data_london.TX = data_london.TX * 10
data_london.TG = data_london.TG * 10
data_london.TN = data_london.TN * 10

###################################################################################################
# 绘制箱线图 和 删除异常值之后的散点图
# 对于异常值用红色标注
# 同时将异常值进行处理 3σ原则：如果数据服从正态分布，异常值被定义为一组测定值中与平均值的偏差超过3倍的值 → p(|x - μ| > 3σ) ≤ 0.003

def dealData(col):
    data = data_london[col]
    u = data.mean()  # 计算均值
    std = data.std()  # 计算标准差
    stats.kstest(data, 'norm', (u, std))
    print('均值为：%.3f，标准差为：%.3f' % (u, std))
    print('------')
    # 异常值替换逻辑
    error = data[np.abs(data - u) > 3 * std]
    data_c = data[np.abs(data - u) <= 3 * std]
    print('异常值共%i条' % len(error))
    # 筛选出异常值error、剔除异常值之后的数据data_c
    data_london[col] = data_c
    data_london[col] = data_london[col].interpolate()  # 补充缺失值


def drawImg1(col):
    data = data_london[col]
    u = data.mean()  # 计算均值
    std = data.std()  # 计算标准差
    stats.kstest(data, 'norm', (u, std))
    # 异常值替换逻辑
    error = data[np.abs(data - u) > 3 * std]
    data_c = data[np.abs(data - u) <= 3 * std]
    print('异常值共%i条' % len(error))
    # 筛选出异常值error、剔除异常值之后的数据data_c
    # datadf_ESKDALEMUIR[col] = data_c

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    data.plot.box(vert=False, grid=True, color=color, ax=ax1, label=col)
    # 箱线图分析

    ax2 = fig.add_subplot(2, 1, 2)
    plt.scatter(data_c.index, data_c, color='k', marker='.', alpha=0.3)
    plt.scatter(error.index, error, color='r', marker='.', alpha=0.5)
    plt.xlim([-10, 10010])
    plt.grid()
    plt.show()

# #########################################################################################
# def box_plot_drop(data_series):
#     q_abnormal_low = data_series.quantile(0.25) - 1.5 * (data_series.quantile(0.75) - data_series.quantile(0.25))
#     q_abnormal_up = data_series.quantile(0.75) + 1.5 * (data_series.quantile(0.75) - data_series.quantile(0.25))
#     index = (data_series < q_abnormal_low) | (data_series > q_abnormal_up)
#     data_series.loc[(data_series < q_abnormal_low) | (data_series > q_abnormal_up)] = np.nan
#     outliers = data_series.loc[index]
#     return data_series
#
#     sorted_data = sorted(data)
#     data_series = pd.Series(sorted_data)
#     outliers = box_plot(data_series)
#
# def box_plot(data_series):
#     q_abnormal_low = data_series.quantile(0.25) - 1.5 * (data_series.quantile(0.75) - data_series.quantile(0.25))
#     q_abnormal_up = data_series.quantile(0.75) + 1.5 * (data_series.quantile(0.75) - data_series.quantile(0.25))
#     index = (data_series < q_abnormal_low) | (data_series > q_abnormal_up)
#     outliers = data_series.loc[index]
#     return outliers.tolist()
#     sorted_data = sorted(data)
#     data_series = pd.Series(sorted_data)
#     outliers = box_plot(data_series)
#
# ########################################################################################
col_ls = ['CC','SS','QQ','TX','TG','TN','precipitation','pressure','snow_depth']
for col in col_ls :
    print(col)
    dealData(col)
    drawImg1(col)

data_london.to_csv('./data_copy.csv', sep=',', index=False, header=True)

################################################################################################
#################heat map#####################
test_corr = data_london.corr()

fig = plt.figure()
plt.title("Heat Map of London Features ")

sns.heatmap(test_corr,cmap = 'RdBu_r')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

################################################



