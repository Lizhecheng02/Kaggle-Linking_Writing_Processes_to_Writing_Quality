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
import catboost as cb
import optuna
import pandas as pd
import xgboost as xgb
import json
warnings.filterwarnings('ignore')

train_logs = pd.read_csv("../train_logs.csv")
train_scores = pd.read_csv("../train_scores.csv")
test_logs = pd.read_csv("../test_logs.csv")
train_scores = pd.read_csv("../train_scores.csv")
print(train_logs.head())

for column in ['down_time', 'up_time', 'action_time']:
    train_logs[column] = train_logs[column] / 1000.0
    test_logs[column] = test_logs[column] / 1000.0

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

nan_cols = train_feats.columns[train_feats.isna().any()].tolist()
print("nan_cols:", nan_cols)

train_feats = train_feats.drop(columns=nan_cols)
test_feats = test_feats.drop(columns=nan_cols)
print(train_feats.shape)
print(test_feats.shape)

timeprocessor = TimeProcessor()
train_agg_fe_df1 = timeprocessor.train_processor1(train_logs)
train_agg_fe_df2 = timeprocessor.train_processor2(train_logs)
test_agg_fe_df1 = timeprocessor.test_processor1(test_logs)
test_agg_fe_df2 = timeprocessor.test_processor2(test_logs)
train_feats = train_feats.merge(train_agg_fe_df1, on="id", how="left")
train_feats = train_feats.merge(train_agg_fe_df2, on='id', how='left')
test_feats = test_feats.merge(test_agg_fe_df1, on='id', how='left')
test_feats = test_feats.merge(test_agg_fe_df2, on='id', how='left')
print("The shape of train_feats after timeprocessor: ", train_feats.shape)
print("The shape of test_feats after timeprocessor: ", test_feats.shape)

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

target_col = ['score']
drop_cols = ['id']
train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

TEST_PREDS = np.zeros((len(test_feats), 1))

os.makedirs('../cb_models_optuna', exist_ok=True)

def objective(trial):
    EPOCHS = 1
    SPLIT = 10

    test_prediction_list = []
    model_dict = {}
    scores = []
    preds = np.zeros((len(train_feats), 1))

    best_params = {
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'depth': trial.suggest_int('depth', 1, 10),
        'iterations': trial.suggest_int('iterations', 1000, 15000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
        'thread_count': 4
    }

    for i in range(EPOCHS):
        kf = model_selection.KFold(n_splits=SPLIT, random_state=42 + i * 10, shuffle=True)
        valid_preds = np.zeros(train_feats.shape[0])
        X_test = test_feats[train_cols]
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
            print(f'Epoch: {i + 1} Fold: {fold + 1}')
            X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
            X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]
            
            model = cb.CatBoostRegressor(
                loss_function='RMSE',
                random_seed=2023,
                verbose=True,
                **best_params
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=100,
                verbose=True
            )

            valid_predict = model.predict(X_valid)
            valid_preds[valid_idx] = valid_predict
            preds[valid_idx, 0] += valid_predict / EPOCHS
            
            score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
            model_dict[f'Epoch{i + 1}-Fold{fold + 1}'] = model
            model.save_model(f'../cb_models_optuna/cb_model_epoch{i + 1}_fold{fold + 1}.cbm')
            
        final_score = metrics.mean_squared_error(train_feats[target_col], valid_preds, squared=False)
        scores.append(final_score)
        
    print("Avg Loss:", np.mean(scores))
    return np.mean(scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")

with open('../cb_best_params.json', 'w') as json_file:
    json.dump(trial.params, json_file, indent=4)

print("Save catboost best_params to json file")