import sys
sys.path.append('..')
from get_essays import getEssays
from essay_processor import EssayProcessor
from main_processor import Preprocessor
from time_processor import TimeProcessor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import lightgbm as lgb
from sklearn import metrics, model_selection, preprocessing, linear_model, ensemble, decomposition, tree
import warnings
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import regex as re
import torch
import optuna
import pandas as pd
warnings.filterwarnings('ignore')

train_logs = pd.read_csv("../train_logs.csv")
train_scores = pd.read_csv("../train_scores.csv")
test_logs = pd.read_csv("../test_logs.csv")
train_scores = pd.read_csv("../train_scores.csv")
print(train_logs.head())

train_essays = pd.read_csv('../train_essays_02.csv')
train_essays.index = train_essays["Unnamed: 0"]
train_essays.index.name = None
train_essays.drop(columns=["Unnamed: 0"], inplace=True)
print(train_essays.head())

essay_processor = EssayProcessor()
train_sent_agg_df = essay_processor.sentence_processor(df=train_essays)
train_paragraph_agg_df = essay_processor.paragraph_processor(df=train_essays)

test_essays = getEssays(test_logs)
test_sent_agg_df = essay_processor.sentence_processor(df=test_essays)
test_paragraph_agg_df = essay_processor.paragraph_processor(df=test_essays)

preprocessor = Preprocessor(seed=42)
train_feats = preprocessor.make_feats(train_logs)
test_feats = preprocessor.make_feats(test_logs)
print("The shape of train_feats after mainprocessor: ", train_feats.shape)
print("The shape of test_feats after mainprocessor: ",test_feats.shape)

timeprocessor = TimeProcessor()
# train_agg_fe_df1 = timeprocessor.train_processor1(train_logs)
# train_agg_fe_df2 = timeprocessor.train_processor2(train_logs)
# test_agg_fe_df1 = timeprocessor.test_processor1(test_logs)
# test_agg_fe_df2 = timeprocessor.test_processor2(test_logs)
# train_feats = train_feats.merge(train_agg_fe_df1, on="id", how="left")
# train_feats = train_feats.merge(train_agg_fe_df2, on='id', how='left')
# test_feats = test_feats.merge(test_agg_fe_df1, on='id', how='left')
# test_feats = test_feats.merge(test_agg_fe_df2, on='id', how='left')
# print("The shape of train_feats after timeprocessor: ", train_feats.shape)
# print("The shape of test_feats after timeprocessor: ", test_feats.shape)

train_pause_features, test_pause_features = timeprocessor.additional_processor(train_logs, test_logs)
train_feats = train_feats.merge(train_pause_features, on='id', how='left')
train_feats = train_feats.merge(train_scores, on='id', how='left')
test_feats = test_feats.merge(test_pause_features, on='id', how='left')

train_feats = train_feats.merge(train_sent_agg_df, on="id", how="left")
train_feats = train_feats.merge(train_paragraph_agg_df, on="id", how="left")
train_feats = train_feats.fillna(0.0)
print("The final shape of train_feats:", train_feats.shape)

test_feats = test_feats.merge(test_sent_agg_df, on='id', how='left')
test_feats = test_feats.merge(test_paragraph_agg_df, on='id', how='left')
test_feats = test_feats.fillna(0.0)
print("The final shape of test_feats:", test_feats.shape)

nan_cols = train_feats.columns[train_feats.isna().any()].tolist()
print("nan_cols:", nan_cols)

train_feats = train_feats.drop(columns=nan_cols)
test_feats = test_feats.drop(columns=nan_cols)
print(train_feats.shape)
print(test_feats.shape)

target_col = ['score']
drop_cols = ['id']
train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

count_label0 = 0
count_label1 = 0
def create_binary_score(score):
    global count_label0, count_label1
    if score <= 1.5 or score >= 4.5:
        count_label0 += 1
        return 0
    else:
        count_label1 += 1
        return 1
    
train_feats['score'] = train_feats['score'].apply(create_binary_score)
train_feats['score'] = train_feats['score'].astype('category')
print("Number of label0:", count_label0)
print("Number of label1:", count_label1)

TEST_PREDS = np.zeros((len(test_feats), 1))

os.makedirs('./cls_lgb_models', exist_ok=True)

EPOCHS = 5
SPLIT = 2

model_dict = {}
scores = []
preds = np.zeros((len(train_feats), 1))

best_params = {
    'reg_alpha': 0.6016917340618352, 
    'reg_lambda': 3.8071290717767194, 
    'colsample_bytree': 0.45216556596658897, 
    'subsample': 0.4832292138435902, 
    'learning_rate': 0.001,
    'num_leaves': 11, 
    'max_depth': 27, 
    'min_child_samples': 17,
    'n_jobs': 4
}

for i in range(EPOCHS):
    kf = model_selection.KFold(n_splits=SPLIT, random_state=42 + i * 10, shuffle=True)
    valid_preds = np.zeros(train_feats.shape[0])
    X_test = test_feats[train_cols]
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
        print(f'Epoch: {i + 1} Fold: {fold + 1}')
        X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
        X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]
        params = {
            "objective": "binary" if len(np.unique(y_train)) == 2 else "multiclass",
            "metric": "binary_logloss" if len(np.unique(y_train)) == 2 else "multi_logloss",
            "random_state": 42,
            "n_estimators": 11_861,
            "verbosity": 1,
            **best_params
        }
        weights = np.where(y_train == 0, 2, 1)

        model = lgb.LGBMClassifier(**params)
        early_stopping_callback = lgb.early_stopping(100, first_metric_only=True, verbose=True)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[early_stopping_callback],
            sample_weight=weights
        )
        
        valid_predict = model.predict(X_valid)
        valid_preds[valid_idx] = valid_predict
        preds[valid_idx, 0] += valid_predict.astype(float) / EPOCHS
        
        test_predict = model.predict(X_test)
        test_predict = test_predict.astype(float)
        TEST_PREDS[:, 0] += test_predict / EPOCHS / SPLIT
        
        score = metrics.accuracy_score(y_valid, valid_predict)
        model_dict[f'Epoch{i + 1}-Fold{fold + 1}'] = model
        model.booster_.save_model(f'./cls_lgb_models/lgbm_model_epoch{i + 1}_fold{fold + 1}.txt')
        
    final_score = metrics.accuracy_score(train_feats[target_col], valid_preds)
    scores.append(final_score)

preds = np.where(preds < 0.5, 0, 1)
preds = preds.astype(int)
    
print("Avg Acc:", np.mean(scores))

print('metric LGBM = {:.5f}'.format(metrics.accuracy_score(train_feats[target_col], preds[:, 0])))