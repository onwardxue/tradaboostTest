# -*- coding:utf-8 -*-
# @Time : 2022/7/10 3:25 下午
# @Author : Bin Bin Xue
# @File : baseline_model
# @Project : tradaboostTest

import xgboost as xgb
import pandas as pd

# 用train_b直接训练xgb模型，对test_b数据进行预测

# 读取数据
train_B = pd.read_csv('data_after/B_train.csv')
train_B_flag = pd.read_csv('data_after/B_train_flag.csv')
test = pd.read_csv('data/B_test.csv')

# 转成适合xgb内部使用的数据
dtrain_B = xgb.DMatrix(data = train_B, label = train_B_flag)

Trate = 0.25
params = {'booster': 'gbtree', 'eta': 0.1, 'max_depth': 4, 'max_delta_step': 0, 'subsample': 0.9,
          'colsample_bytree': 0.9, 'base_score': Trate, 'objective': 'binary:logistic', 'lambda': 5, 'alpha': 8,
          'random_seed': 100, 'eval_metric': 'auc'}
xgb_model = xgb.train(params, dtrain_B, num_boost_round=200, maximize = True,
                      verbose_eval= True )

prediction = xgb_model.predict(xgb.DMatrix(test[train_B.columns].fillna(-999)))
test['pred'] = prediction
test[['no','pred']].to_csv('result/submission.csv', index = None)

for x in test['pred']:
    if(x>0.5):
        test['label'] = 1
    else:
        test['label'] = 0

test[['no','label']].to_csv('result/rs.csv',index = None)
