# -*- coding:utf-8 -*-
# @Time : 2022/7/10 12:58 下午
# @Author : Bin Bin Xue
# @File : data_process
# @Project : tradaboostTest

# 这里只对B训练集进行了处理

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
import os
import gc

#
sns.set(style='white', context='notebook', palette='deep')
sns.set_style('white')
pd.set_option('display.max_columns', 500)

# EDA
# 导入数据
train_A = pd.read_csv('data/A_train.csv')
train_B = pd.read_csv('data/B_train.csv')
test = pd.read_csv('data/B_test.csv')
sample = pd.read_csv('data/submit_sample.csv')

if __name__ == '__main__':
    # 观察数据(看到数据量和数据类型）
    # print(sample.shape)
    # print(sample.head)
    # print(train_A.info())
    # print(train_B.info())
    # print(test.info())
    #
    # # 探索数据
    # # 查看数据维度
    # print('shape of train_A：', train_A.shape)
    # print('shape of train_B：', train_B.shape)
    # print('shape of test：', test.shape)

    # 对A分析
    # print(train_A.head())
    # 特征分析(特征数据范围很大，需要进行归一化处理）
    # print(train_A.describe())
    # 查看特征缺失值（缺失值很多，需要进行缺失值处理）
    # print(train_A.isnull().sum().sort_values(ascending=False))

    # 对B和test像对A一样进行分析

    # 进一步特征分析（查看int型数据的特征情况-已知no为用户标识，flag为标签）
    # print(train_A.dtypes[train_A.dtypes == np.int64])
    # # 发现B中多了一个特征，且其所有值都是0，所以将它删除
    # print(train_B.dtypes[train_B.dtypes == np.int64])
    # print(train_B['UserInfo_170'].value_counts())
    train_B = train_B.drop('UserInfo_170', axis=1)
    # c中也发现了这个特征，但c为测试集，所以暂时不动
    # print(test.dtypes[test.dtypes == np.int64])
    # print(test['UserInfo_170'].value_counts())

    # 对train_B缺失值较多的特征进行处理（缺失值较多会带来极大的噪声，所以要对有较大噪声的数据进行删除 - 将特征缺失率大于99%的特征进行删除）
    train_B_info = train_B.describe()
    # 统计表中count的值表示的是不含缺失值的总数，所以可以用这一行做缺失值判断
    train_B_info.head()
    useful_col = []
    for col in train_B_info.columns:
        # 只要不含缺失值的总数大于总数的0.01，就把该特征加入列表中
        if train_B_info.loc['count', col] > train_B.shape[0] * 0.01:
            useful_col.append(col)

    # 将只含这些特征的列表复制给另一个
    train_B_1 = train_B[useful_col].copy()
    # print(train_B.shape)
    # print(train_B_1.shape)

    # train_B缺失值填充（这里使用-999进行填充）
    train_B_1 = train_B_1.fillna(-999)

    # 线性相关特征处理
    # 计算train_B特征间的相关性
    relation = train_B_1.corr()
    print(relation)

    # 删除存在线性关系的特征
    length = relation.shape[0]
    high_corr = list()
    final_cols = []
    del_cols = []
    for i in range(length):
        if relation.columns[i] not in del_cols:
            final_cols.append(relation.columns[i])
            for j in range(i + 1, length):
                if (relation.iloc[i, j] > 0.98) and (relation.columns[j] not in del_cols):
                    del_cols.append(relation.columns[j])

    # 去除客户标识和标签，标签存为单独的序列
    train_B_1 = train_B_1[final_cols]
    train_B_flag = train_B_1['flag']
    train_B_1.drop('no', axis=1, inplace=True)
    train_B_1.drop('flag', axis=1, inplace=True)

    # 保存处理过的train_B数据
    train_B_1.to_csv('data_after/B_train.csv', index=False)
    train_B_flag.to_csv('data_after/B_train_flag.csv', index=False)
