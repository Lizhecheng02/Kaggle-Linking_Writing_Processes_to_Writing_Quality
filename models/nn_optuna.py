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

os.makedirs('../nn_models_optuna', exist_ok=True)

DO_SCALER = False

for column in tqdm(train_feats[train_cols].columns):
    try:
        scaler = StandardScaler()
        if DO_SCALER:
            train_feats[column] = scaler.fit_transform(train_feats[column].values.reshape(-1, 1))
            test_feats[column] = scaler.transform(test_feats[column].values.reshape(-1, 1))
            print("Successfully Excute!")
        else:
            output = scaler.fit_transform(train_feats[column].values.reshape(-1, 1))
            print("Successfully Output!")
    except:
        print(column)

class RMSE(tf.keras.metrics.Metric):
    def __init__(self, name='rmse', **kwargs):
        super(RMSE, self).__init__(name=name, **kwargs)
        self.mse = tf.keras.metrics.MeanSquaredError()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mse.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return tf.sqrt(self.mse.result())

    def reset_states(self):
        self.mse.reset_states()

input_dim = len(train_cols)
output_dim = 1

def objective(trial):
    lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    first_layer_units = trial.suggest_categorical('first_layer_units', [128, 256, 512])
    second_layer_units = trial.suggest_categorical('second_layer_units', [128, 256, 512])
    third_layer_units = trial.suggest_categorical('third_layer_units', [128, 256, 512])
    fourth_layer_units = trial.suggest_categorical('fourth_layer_units', [128, 256, 512])
    fifth_layer_units = trial.suggest_categorical('fifth_layer_units', [64, 128, 256])
    
    def My_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(first_layer_units, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(second_layer_units, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(third_layer_units, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(fourth_layer_units, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(fifth_layer_units, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[RMSE()])
        return model

    model = My_model()
    
    EPOCHS = 5
    SPLIT = 10

    losses = []

    for i in range(EPOCHS):
        kf = model_selection.KFold(n_splits=SPLIT, random_state=42 + i * 20, shuffle=True)

        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
            print(f"Training on Epoch {i + 1} Fold {fold + 1}...")

            X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
            X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]

            model = My_model()
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                f'../nn_models_optuna/best_model_epoch{i + 1}_fold{fold + 1}.h5',           
                monitor='val_rmse',       
                verbose=1,               
                save_best_only=True,        
                mode='min'                
            )
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_rmse',   
                patience=20,           
                verbose=-1,            
                mode='min',           
                restore_best_weights=True 
            )
            model.fit(
                X_train, y_train, epochs=200, 
                batch_size=32, validation_data=(X_valid, y_valid),
                callbacks=[checkpoint, early_stopping]
            )
        
            best_model = tf.keras.models.load_model(f'../nn_models_optuna/best_model_epoch{i + 1}_fold{fold + 1}.h5', custom_objects={'RMSE': RMSE})

            valid_predict = best_model.predict(X_valid).squeeze(1)
            best_valid_loss = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
            losses.append(best_valid_loss)

    print("Avg Loss:", np.mean(losses))
    return np.mean(losses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")