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

essay_processr = EssayProcessor()
train_sent_agg_df = essay_processr.sentence_processor(df=train_essays)
train_paragraph_agg_df = essay_processr.paragraph_processor(df=train_essays)

test_essays = getEssays(test_logs)
test_sent_agg_df = essay_processr.sentence_processor(df=test_essays)
test_paragraph_agg_df = essay_processr.paragraph_processor(df=test_essays)

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

os.makedirs('../tabnet_models', exist_ok=True)

EPOCHS = 3
SPLIT = 10

losses = []

for i in range(EPOCHS):
    kf = model_selection.KFold(n_splits=SPLIT, random_state=42 + i * 30, shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
        print(f"Training model ... Epoch {i + 1} Fold {fold + 1}")
        X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
        X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]

        model = TabNetRegressor(
            n_d=4,
            n_a=4,
            n_steps=4,
            optimizer_params={'lr': 0.005},
            verbose=1
        )

        model.fit(
            X_train=X_train.values, 
            y_train=y_train.values,
            eval_set=[(X_valid.values, y_valid.values)],
            eval_name=['valid'],
            eval_metric=['mse'],
            max_epochs=300, 
            patience=30,
            batch_size=32,
            virtual_batch_size=16,
            num_workers=0,
            drop_last=False
        )
        model.save_model(f"../tabnet_models/tabnet_model_epoch{i + 1}_fold{fold + 1}")
#         model.load_model(f"tabnet_models/tabnet_model_epoch{i + 1}_fold{fold + 1}.zip")

        val_preds = model.predict(X_valid.values)
        loss = metrics.mean_squared_error(y_valid, val_preds, squared=False)
        losses.append(loss)
        print("Single Loss:", loss)
        
        X_test = test_feats[train_cols]
        test_predict = model.predict(X_test.values).squeeze(1)
        TEST_PREDS[:, 0] += test_predict / SPLIT / EPOCHS

print("Avg Loss:", np.mean(losses))