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

train_logs = pd.read_csv("train_logs.csv")
train_scores = pd.read_csv("train_scores.csv")
test_logs = pd.read_csv("test_logs.csv")
train_scores = pd.read_csv("train_scores.csv")
print(train_logs.head())

for column in ['down_time', 'up_time', 'action_time']:
    train_logs[column] = train_logs[column] / 1000.0
    test_logs[column] = test_logs[column] / 1000.0

train_essays = pd.read_csv('train_essays_02.csv')
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

train_feats.to_csv("features.csv", index=False)
print("Save successfully to features.csv")

columns = train_feats.columns
with open('columns.txt', 'w') as file:
    file.write('[')
    for idx, column in enumerate(columns):
        if idx != len(columns) - 1:
            file.write("'" + column + "'" + ", ")
        else:
            file.write("'" + column + "'" + "]")
print("Save successfully to columns.txt")