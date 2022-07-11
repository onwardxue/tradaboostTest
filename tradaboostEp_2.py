# -*- coding:utf-8 -*-
# @Time : 2022/7/10 4:23 下午
# @Author : Bin Bin Xue
# @File : tradaboostEp
# @Project : tradaboostTest
# Tradaboost实验
# 参考：https://zhuanlan.zhihu.com/p/109540481
# 这里用的是这篇文章中的代码，作者说github上的代码有点问题，在文章中做了调整
# 这里使用的基分类器是xgb，而王晋东那里的tradaboost基分类器是决策树

import copy
import random
import os
import numpy as np
from sklearn.model_selection import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from Tradaboost_2 import Tradaboost

warnings.filterwarnings('ignore')
# 读取数据

train_A = pd.read_csv('data/A_train.csv')
train_B = pd.read_csv('data/B_train.csv')
test = pd.read_csv('data/B_test.csv')
sample = pd.read_csv('data/submit_sample.csv')

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


# 5.设置可以复现的随机种子 ?问题：这一步有什么用？
def seed_everything(seed=0):
    # random.seed(seed)
    # 获取环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(456)

# 6.划分目标域train_B_1训练集为valid和test（5：5划分）
train_B_1_valid, train_B_1_test, train_B_1_valid_y, train_B_1_test_y = train_test_split(train_B_1, train_B_flag,
                                                                                        test_size=0.5)

# 7.用xgb做一个简单的base_line
# dtrain_B = xgb.DMatrix(data = train_B_1, label = train_B_flag)
Trate = 0.25
params = {'booster': 'gbtree',
          'eta': 0.1,
          'max_depth': 20,
          'max_delta_step': 0,
          'subsample': 1,
          'colsample_bytree': 1,
          'base_score': Trate,
          'objective': 'binary:logistic',
          'lambda': 5,
          'alpha': 8,
          'n_estimators': 500,
          'random_state': 100,
          'n_jobs': -1
          }
# 设置参数，并用源域数据train_A训练模型
clf = xgb.XGBClassifier(**params)
clf.fit(train_A_1, train_A_flag)

# 源域数据训练的模型预测源域数据（不是100%，只有0.984）
y_pred_A = clf.predict_proba(train_A_1)[:, 1]
# 用源域数据训练的模型预测目标训练集中的验证集部分（结果只有0.58，说明源和目标差异较大）
y_pred_B_valid = clf.predict_proba(train_B_1_valid)[:, 1]
# 用源域数据训练的模型预测目标训练集中的测试集部分（结果只有0.53，说明源域模型预测目标域测试集效果较差）
y_pred_B_test = clf.predict_proba(train_B_1_test)[:, 1]

print(f" train AUC = {roc_auc_score(train_A_flag, y_pred_A)}")
print(f" valid AUC = {roc_auc_score(train_B_1_valid_y, y_pred_B_valid)}")
print(f" test AUC = {roc_auc_score(train_B_1_test_y, y_pred_B_test)}")

# 8.用Tradaboost来处理数据
# 初始化Tradaboost（迭代次数200，基分类器使用xgb，阈值为0.92975，评价标准为roc）
clf = Tradaboost(N=200, base_estimator=xgb.XGBClassifier(**params), threshold=0.92975, score=roc_auc_score)
# 训练模型：源域数据train_A和目标域数据train_B的验证集数据训练模型（最早结束迭代次数为50）
clf.fit(train_A_1.values, train_B_1_valid.values, train_A_flag, train_B_1_valid_y.values, 50)

estimators = clf.estimators
i = 0
# 输出每个基分类器的表现
for estimator in estimators:
    print('The ' + str(i + 1) + ' estimator:')
    i += 1

    y_pred_A = estimator.predict_proba(train_A_1.values)[:, 1]
    y_pred_B_valid = estimator.predict_proba(train_B_1_valid.values)[:, 1]
    y_pred_B_test = estimator.predict_proba(train_B_1_test.values)[:, 1]

    print(f" train AUC = {roc_auc_score(train_A_flag, y_pred_A)}")
    print(f" valid AUC = {roc_auc_score(train_B_1_valid_y, y_pred_B_valid)}")
    print(f" test AUC = {roc_auc_score(train_B_1_test_y, y_pred_B_test)}")
    print('\n')
    print('==============================================================')
