# -*- coding:utf-8 -*-
# @Time : 2022/7/10 4:23 下午
# @Author : Bin Bin Xue
# @File : tradaboostEp
# @Project : tradaboostTest
# Tradaboost实验
# 参考：https://zhuanlan.zhihu.com/p/109540481

import Tradaboost
import pandas as pd

train_A = pd.read_csv('data_after/A_train.csv')
train_A_flag = pd.read_csv('data_after/A_train_flag.csv')
train_B = pd.read_csv('data_after/B_train.csv')
train_B_flag = pd.read_csv('data_after/B_train_flag.csv')
test = pd.read_csv('data/B_test.csv')


if __name__ == '__main__':
    clf = Tradaboost.TrAdaBoost()
    print(len(train_A.columns))
    print(len(train_B.columns))
    print(len(test.columns))

    same_features = train_A.columns.intersection(train_B.columns)
    train_A_2 = train_A[same_features]
    train_B_2 = train_B[same_features]
    test_2= test[same_features]

    predict =clf.fit_predict(train_A_2,train_B_2,train_A_flag,train_B_flag,test_2)

