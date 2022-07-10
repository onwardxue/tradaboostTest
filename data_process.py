# -*- coding:utf-8 -*-
# @Time : 2022/7/10 12:58 下午
# @Author : Bin Bin Xue
# @File : data_process
# @Project : tradaboostTest

# 这里设置的是一个通用的预处理，也对源域A进行了处理

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


# 创建一个分析和处理数据集情况的类
class preprocess:
    def __init__(self, site):
        self.ds = site

    # 输出特征缺失值数量
    def miss_count(self):
        print(self.ds.isnull().sum().sort_values(ascending=False))

    # 查看int型的特征有哪些
    def int_feature(self):
        ser = self.ds
        print(self.ds.dtypes[train_A.dtypes == np.int64])

    # 删除某个特征（某列）：
    def delete_feature(self, str):
        self.ds = self.ds.drop(str, axis=1)

    # 缺失值处理：
    def miss_process(self):
        ds_2 = self.ds.describe()
        # 统计表中count的值表示的是不含缺失值的总数，所以可以用这一行做缺失值判断
        # ds_2.head()
        useful_col = []
        for col in ds_2.columns:
            # 只要不含缺失值的总数大于总数的0.01，就把该特征加入列表中
            if ds_2.loc['count', col] > self.ds.shape[0] * 0.01:
                useful_col.append(col)

        ds_2 = self.ds[useful_col].copy()
        # train_B缺失值填充（这里使用-999进行填充）
        ds_2 = ds_2.fillna(-999)
        return ds_2

    # 对线性相关特征进行处理：
    def linear_process(self,ds_2):
        relation = ds_2.corr()
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
        ds_2 = ds_2[final_cols]
        return ds_2

    # 从数据集划出标签列和去除标识列，并保存到指定路径
    def general_flags(self,ds_2,data_name):
        ds_2_flag = ds_2['flag']
        ds_2.drop('no', axis=1, inplace=True)
        ds_2.drop('flag', axis=1, inplace=True)

        ds_2.to_csv('data_after/'+data_name+'.csv', index=False)
        ds_2_flag.to_csv('data_after/'+data_name+'_flag.csv', index=False)


if __name__ == '__main__':
    # 对A进行数据处理
    des =preprocess(train_A)
    # des.int_feature()
    des_2 = des.miss_process()
    des_2 = des.linear_process(des_2)
    des.general_flags(des_2,'A_train')
