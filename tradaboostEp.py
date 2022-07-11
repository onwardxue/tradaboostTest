# -*- coding:utf-8 -*-
# @Time : 2022/7/10 4:23 下午
# @Author : Bin Bin Xue
# @File : tradaboostEp
# @Project : tradaboostTest
# Tradaboost实验
# 参考：这里用的是王晋东的Tradaboost代码，要求特征空间一致，因此没有实现，存在问题。(还是存在缺失值的问题。。）
from sklearn.model_selection import train_test_split

import Tradaboost
import pandas as pd

# train_A = pd.read_csv('data_after/A_train.csv')
# train_A_flag = pd.read_csv('data_after/A_train_flag.csv')
# train_B = pd.read_csv('data_after/B_train.csv')
# train_B_flag = pd.read_csv('data_after/B_train_flag.csv')
# test = pd.read_csv('data/B_test.csv')

train_A = pd.read_csv('data/A_train.csv')
train_B = pd.read_csv('data/B_train.csv')
test = pd.read_csv('data/B_test.csv')
sample = pd.read_csv('data/submit_sample.csv')

# 使用tradaboostEp_2中的数据处理过程
# 预处理
# 1.去除b_train中缺失率大于99%的特征
train_B_info = train_B.describe()
useful_col = []
for col in train_B_info.columns:
    if train_B_info.loc['count', col] > train_B.shape[0] * 0.01:
        useful_col.append(col)

train_B_1 = train_B[useful_col].copy()
# train_B_1 = train_B_1.fillna(-999)
relation = train_B_1.corr()

# 2.保留train_A中和train_B一样的特征
train_A_1 = train_A[useful_col].copy()
# train_A_1 = train_A_1.fillna(-999)

# 3.对线性相关特征进行处理
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

train_B_1 = train_B_1[final_cols]
train_A_1 = train_A_1[final_cols]

# 4.取出标签列和去除标识列
train_B_flag = train_B_1['flag']
train_B_1.drop('no', axis=1, inplace=True)
train_B_1.drop('flag', axis=1, inplace=True)

train_A_flag = train_A_1['flag']
train_A_1.drop('no', axis=1, inplace=True)
train_A_1.drop('flag', axis=1, inplace=True)

# 划分目标域train_B_1训练集为valid和test（5：5划分）
train_B_1_valid, train_B_1_test, train_B_1_valid_y, train_B_1_test_y = train_test_split(train_B_1, train_B_flag,
                                                                                        test_size=0.5)

if __name__ == '__main__':
    clf = Tradaboost.TrAdaBoost()
    print(len(train_A.columns))
    print(len(train_B.columns))
    print(len(test.columns))

    # 取共有特征
    # same_features = train_A.columns.intersection(train_B.columns)
    # train_A_2 = train_A[same_features]
    # train_B_2 = train_B[same_features]
    # test_2= test[same_features]

    predict =clf.fit_predict(train_A_1,train_B_1_valid,train_A_flag,train_B_1_valid_y,train_B_1_test)
    print(predict)

