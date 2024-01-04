import polars as pl
import pandas as pd
import numpy as np
import re
import os
from sklearn import metrics, model_selection
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from scipy.stats import skew, kurtosis
import warnings
import json
warnings.filterwarnings("ignore")


class CFG:
    is_train_lgbm_model = True
    is_train_lgbm_optuna = False
    is_train_xgb_model = True
    is_train_xgb_optuna = False
    is_train_cb_model = True
    is_train_cb_optuna = False


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


AGGREGATIONS = ['count', 'mean', 'min', 'max',
                'first', 'last', q1, 'median', q3, 'sum']

num_cols = ['down_time', 'up_time', 'action_time',
            'cursor_position', 'word_count']

activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']

events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft',
          '.', ',', 'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']

text_changes = ['q', ' ', '.', ',', '\n', "'",
                '"', '-', '?', ';', '=', '/', '\\', ':']


def count_by_values(df, colname, values):
    fts = df.select(pl.col('id').unique(maintain_order=True))
    for i, value in enumerate(values):
        tmp_df = df.group_by('id').agg(pl.col(colname).is_in(
            [value]).sum().alias(f'{colname}_{i}_cnt'))
        fts = fts.join(tmp_df, on='id', how='left')
    return fts


def dev_feats(df):

    print("< Count by values features >")

    feats = count_by_values(df, 'activity', activities)
    feats = feats.join(count_by_values(df, 'text_change',
                       text_changes), on='id', how='left')
    feats = feats.join(count_by_values(
        df, 'down_event', events), on='id', how='left')
    feats = feats.join(count_by_values(
        df, 'up_event', events), on='id', how='left')

    print("< Input words stats features >")

    temp = df.filter((~pl.col('text_change').str.contains('=>'))
                     & (pl.col('text_change') != 'NoChange'))
    temp = temp.group_by('id').agg(
        pl.col('text_change').str.concat('').str.extract_all(r'q+'))
    temp = temp.with_columns(
        input_word_count=pl.col('text_change').list.lengths(),
        input_word_length_mean=pl.col('text_change').apply(
            lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)
        ),
        input_word_length_max=pl.col('text_change').apply(
            lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)
        ),
        input_word_length_std=pl.col('text_change').apply(
            lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)
        ),
        input_word_length_median=pl.col('text_change').apply(
            lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0)
        ),
        input_word_length_skew=pl.col('text_change').apply(
            lambda x: skew([len(i) for i in x] if len(x) > 0 else 0)
        )
    )
    temp = temp.drop('text_change')
    feats = feats.join(temp, on='id', how='left')

    print("< Numerical columns features >")

    temp = df.group_by("id").agg(
        pl.sum('action_time').suffix('_sum'),
        pl.mean(num_cols).suffix('_mean'),
        pl.std(num_cols).suffix('_std'),
        pl.median(num_cols).suffix('_median'),
        pl.min(num_cols).suffix('_min'),
        pl.max(num_cols).suffix('_max'),
        pl.quantile(num_cols, 0.25).suffix('_quantile25'),
        pl.quantile(num_cols, 0.75).suffix('_quantile75')
    )
    feats = feats.join(temp, on='id', how='left')

    print("< Categorical columns features >")

    temp = df.group_by("id").agg(
        pl.n_unique(['activity', 'down_event', 'up_event', 'text_change'])
    )
    feats = feats.join(temp, on='id', how='left')

    print("< Idle time features >")

    temp = df.with_columns(pl.col('up_time').shift().over(
        'id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col(
        'down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.group_by("id").agg(
        inter_key_largest_lantency=pl.max('time_diff'),
        inter_key_median_lantency=pl.median('time_diff'),
        mean_pause_time=pl.mean('time_diff'),
        std_pause_time=pl.std('time_diff'),
        total_pause_time=pl.sum('time_diff'),
        pauses_zero_sec=pl.col('time_diff').filter(  # 新增特征
            pl.col('time_diff') < 0.5).count(),
        pauses_zero_sec_mean=pl.col('time_diff').filter(
            pl.col('time_diff') < 0.5).mean(),
        pauses_zero_sec_std=pl.col('time_diff').filter(
            pl.col('time_diff') < 0.5).std(),
        pauses_zero_sec_quantile=pl.col('time_diff').filter(
            pl.col('time_diff') < 0.5).quantile(0.5),
        pauses_half_sec=pl.col('time_diff').filter(
            (pl.col('time_diff') > 0.5) & (pl.col('time_diff') < 1)
        ).count(),
        pauses_half_sec_mean=pl.col('time_diff').filter(
            (pl.col('time_diff') > 0.5) & (pl.col('time_diff') < 1)).mean(),
        pauses_half_sec_std=pl.col('time_diff').filter(
            (pl.col('time_diff') > 0.5) & (pl.col('time_diff') < 1)).std(),
        pauses_half_sec_quantile=pl.col('time_diff').filter(
            (pl.col('time_diff') > 0.5) & (pl.col('time_diff') < 1)).quantile(0.5),
        pauses_1_sec=pl.col('time_diff').filter(
            (pl.col('time_diff') > 1) & (pl.col('time_diff') < 1.5)
        ).count(),
        pauses_1_sec_mean=pl.col('time_diff').filter(
            (pl.col('time_diff') > 1) & (pl.col('time_diff') < 1.5)).mean(),
        pauses_1_sec_std=pl.col('time_diff').filter(
            (pl.col('time_diff') > 1) & (pl.col('time_diff') < 1.5)).std(),
        pauses_1_sec_quantile=pl.col('time_diff').filter(
            (pl.col('time_diff') > 1) & (pl.col('time_diff') < 1.5)).quantile(0.5),
        pauses_1_half_sec=pl.col('time_diff').filter(
            (pl.col('time_diff') > 1.5) & (pl.col('time_diff') < 2)
        ).count(),
        pauses_1_half_sec_mean=pl.col('time_diff').filter(
            (pl.col('time_diff') > 1.5) & (pl.col('time_diff') < 2)).mean(),
        pauses_1_half_sec_std=pl.col('time_diff').filter(
            (pl.col('time_diff') > 1.5) & (pl.col('time_diff') < 2)).std(),
        pauses_1_half_sec_quantile=pl.col('time_diff').filter(
            (pl.col('time_diff') > 1.5) & (pl.col('time_diff') < 2)).quantile(0.5),
        pauses_2_sec=pl.col('time_diff').filter(
            (pl.col('time_diff') > 2) & (pl.col('time_diff') < 3)
        ).count(),
        pauses_2_sec_mean=pl.col('time_diff').filter(
            (pl.col('time_diff') > 2) & (pl.col('time_diff') < 3)).mean(),
        pauses_2_sec_std=pl.col('time_diff').filter(
            (pl.col('time_diff') > 2) & (pl.col('time_diff') < 3)).std(),
        pauses_2_sec_quantile=pl.col('time_diff').filter(
            (pl.col('time_diff') > 2) & (pl.col('time_diff') < 3)).quantile(0.5),
        pauses_3_sec=pl.col('time_diff').filter(
            pl.col('time_diff') > 3).count(),
        pauses_3_sec_mean=pl.col('time_diff').filter(
            pl.col('time_diff') > 3).mean(),
        pauses_3_sec_std=pl.col('time_diff').filter(
            pl.col('time_diff') > 3).std(),
        pauses_3_sec_quantile=pl.col('time_diff').filter(
            pl.col('time_diff') > 3).quantile(0.5)
    )
    feats = feats.join(temp, on='id', how='left')

    print("< P-bursts features >")

    temp = df.with_columns(pl.col('up_time').shift().over(
        'id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col(
        'down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('time_diff') < 2)
    temp = temp.with_columns(pl.when(pl.col("time_diff") & pl.col("time_diff").is_last(
    )).then(pl.count()).over(pl.col("time_diff").rle_id()).alias('P-bursts'))
    temp = temp.drop_nulls()
    temp = temp.group_by("id").agg(
        pl.mean('P-bursts').suffix('_mean'),
        pl.std('P-bursts').suffix('_std'),
        pl.count('P-bursts').suffix('_count'),
        pl.median('P-bursts').suffix('_median'),
        pl.max('P-bursts').suffix('_max'),
        pl.first('P-bursts').suffix('_first'),
        pl.last('P-bursts').suffix('_last')
    )
    feats = feats.join(temp, on='id', how='left')

    print("< R-bursts features >")

    temp = df.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
    temp = temp.with_columns(pl.col('activity').is_in(['Remove/Cut']))
    temp = temp.with_columns(pl.when(pl.col("activity") & pl.col("activity").is_last(
    )).then(pl.count()).over(pl.col("activity").rle_id()).alias('R-bursts'))
    temp = temp.drop_nulls()
    temp = temp.group_by("id").agg(
        pl.mean('R-bursts').suffix('_mean'),
        pl.std('R-bursts').suffix('_std'),
        pl.median('R-bursts').suffix('_median'),
        pl.max('R-bursts').suffix('_max'),
        pl.first('R-bursts').suffix('_first'),
        pl.last('R-bursts').suffix('_last')
    )
    feats = feats.join(temp, on='id', how='left')

    return feats


def reconstruct_essay(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        if Input[0] == 'Replace':
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + \
                essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + \
                Input[2] + essayText[Input[1] - len(Input[2]):]
            continue
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + \
                essayText[Input[1] + len(Input[2]):]
            continue
        if "M" in Input[0]:
            croppedTxt = Input[0][10:]
            splitTxt = croppedTxt.split(' To ')
            valueArr = [item.split(', ') for item in splitTxt]
            moveData = (int(valueArr[0][0][1:]), int(
                valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
            if moveData[0] != moveData[2]:
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + \
                        essayText[moveData[1]:moveData[3]] + \
                        essayText[moveData[0]:moveData[1]] + \
                        essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + \
                        essayText[moveData[0]:moveData[1]] + \
                        essayText[moveData[2]:moveData[0]] + \
                        essayText[moveData[1]:]
            continue
        essayText = essayText[:Input[1] - len(Input[2])] + \
            Input[2] + essayText[Input[1] - len(Input[2]):]
    return essayText


def get_essay_df(df):
    df = df[df.activity != 'Nonproduction']
    temp = df.groupby('id').apply(lambda x: reconstruct_essay(
        x[['activity', 'cursor_position', 'text_change']]))
    essay_df = pd.DataFrame({'id': df['id'].unique().tolist()})
    essay_df = essay_df.merge(temp.rename('essay'), on='id')
    return essay_df


def word_feats(df):
    essay_df = df
    df['word'] = df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!', x))
    df = df.explode('word')
    df['word_len'] = df['word'].apply(lambda x: len(x))
    df = df[df['word_len'] != 0]

    word_agg_df = df[['id', 'word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df


def sent_feats(df):
    df['sent'] = df['essay'].apply(lambda x: re.split('\\.|\\?|\\!', x))
    df = df.explode('sent')
    df['sent'] = df['sent'].apply(lambda x: x.replace('\n', '').strip())
    # Number of characters in sentences
    df['sent_len'] = df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    df['sent_word_count'] = df['sent'].apply(lambda x: len(x.split(' ')))
    df = df[df.sent_len != 0].reset_index(drop=True)

    sent_agg_df = pd.concat([df[['id', 'sent_len']].groupby(['id']).agg(AGGREGATIONS),
                             df[['id', 'sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1)
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count": "sent_count"})
    return sent_agg_df


def parag_feats(df):
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    # Number of characters in paragraphs
    df['paragraph_len'] = df['paragraph'].apply(lambda x: len(x))
    # Number of words in paragraphs
    df['paragraph_word_count'] = df['paragraph'].apply(
        lambda x: len(x.split(' ')))
    df = df[df.paragraph_len != 0].reset_index(drop=True)

    paragraph_agg_df = pd.concat([df[['id', 'paragraph_len']].groupby(['id']).agg(AGGREGATIONS),
                                  df[['id', 'paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1)
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(
        columns={"paragraph_len_count": "paragraph_count"})
    return paragraph_agg_df


def product_to_keys(logs, essays):
    essays['product_len'] = essays.essay.str.len()
    tmp_df = logs[logs.activity.isin(['Input', 'Remove/Cut'])].groupby(['id']).agg(
        {'activity': 'count'}).reset_index().rename(columns={'activity': 'keys_pressed'})
    essays = essays.merge(tmp_df, on='id', how='left')
    essays['product_to_keys'] = essays['product_len'] / essays['keys_pressed']
    return essays[['id', 'product_to_keys']]


def get_keys_pressed_per_second(logs):
    temp_df = logs[logs['activity'].isin(['Input', 'Remove/Cut'])].groupby(
        ['id']).agg(keys_pressed=('event_id', 'count')).reset_index()
    temp_df_2 = logs.groupby(['id']).agg(min_down_time=(
        'down_time', 'min'), max_up_time=('up_time', 'max')).reset_index()
    temp_df = temp_df.merge(temp_df_2, on='id', how='left')
    temp_df['keys_per_second'] = temp_df['keys_pressed'] / \
        ((temp_df['max_up_time'] - temp_df['min_down_time']) / 1000)
    return temp_df[['id', 'keys_per_second']]


data_path = './'
train_logs = pl.scan_csv(data_path + 'train_logs.csv')
train_feats = dev_feats(train_logs)
train_feats = train_feats.collect().to_pandas()

print('< Essay Reconstruction >')
train_logs = train_logs.collect().to_pandas()
train_essays = get_essay_df(train_logs)
train_feats = train_feats.merge(word_feats(train_essays), on='id', how='left')
train_feats = train_feats.merge(sent_feats(train_essays), on='id', how='left')
train_feats = train_feats.merge(parag_feats(train_essays), on='id', how='left')
train_feats = train_feats.merge(
    get_keys_pressed_per_second(train_logs), on='id', how='left')
train_feats = train_feats.merge(product_to_keys(
    train_logs, train_essays), on='id', how='left')

print('< Mapping >')
train_scores = pd.read_csv(data_path + 'train_scores.csv')
data = train_feats.merge(train_scores, on='id', how='left')
data.to_csv("baseline_features.csv", index=False)

x = data.drop(['id', 'score'], axis=1)
y = data['score'].values
print(f'Number of features: {len(x.columns)}')


print('< Testing Data >')
test_logs = pl.scan_csv(data_path + 'test_logs.csv')
test_feats = dev_feats(test_logs)
test_feats = test_feats.collect().to_pandas()

test_logs = test_logs.collect().to_pandas()
test_essays = get_essay_df(test_logs)
test_feats = test_feats.merge(word_feats(test_essays), on='id', how='left')
test_feats = test_feats.merge(sent_feats(test_essays), on='id', how='left')
test_feats = test_feats.merge(parag_feats(test_essays), on='id', how='left')
test_feats = test_feats.merge(
    get_keys_pressed_per_second(test_logs), on='id', how='left')
test_feats = test_feats.merge(product_to_keys(
    test_logs, test_essays), on='id', how='left')

test_ids = test_feats['id'].values
testin_x = test_feats.drop(['id'], axis=1)


def train_valid_split(data_x, data_y, train_idx, valid_idx):
    x_train = data_x.iloc[train_idx]
    y_train = data_y[train_idx]
    x_valid = data_x.iloc[valid_idx]
    y_valid = data_y[valid_idx]
    return x_train, y_train, x_valid, y_valid


def evaluate(data_x, data_y, model, random_state=42, n_splits=5, test_x=None):
    skf = model_selection.StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )
    test_y = np.zeros(len(data_x)) if (
        test_x is None) else np.zeros((len(test_x), n_splits))
    for i, (train_index, valid_index) in enumerate(skf.split(data_x, data_y.astype(str))):
        train_x, train_y, valid_x, valid_y = train_valid_split(
            data_x, data_y, train_index, valid_index)
        model.fit(train_x, train_y)
        if test_x is None:
            test_y[valid_index] = model.predict(valid_x)
        else:
            test_y[:, i] = model.predict(test_x)
    return test_y if (test_x is None) else np.mean(test_y, axis=1)


target_col = ['score']
drop_cols = ['id']
train_cols = [
    col for col in train_feats.columns if col not in target_col + drop_cols
]

print('< Learning and Evaluation >')


def train_lgbm_model(train_feats, test_feats):
    TEST_PREDS = np.zeros((len(test_feats), 1))

    os.makedirs('../baseline_lgb_models', exist_ok=True)

    EPOCHS = 1
    SPLIT = 10

    test_prediction_list = []
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
        "n_estimators": 11_861,
        'n_jobs': 4
    }

    for i in range(EPOCHS):
        kf = model_selection.KFold(
            n_splits=SPLIT, random_state=42 + i * 10, shuffle=True)
        valid_preds = np.zeros(train_feats.shape[0])
        X_test = test_feats[train_cols]

        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
            print(f'Epoch: {i + 1} Fold: {fold + 1}')
            X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
            X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]
            params = {
                "objective": "regression",
                "metric": "rmse",
                "random_state": 42,
                "verbosity": 1,
                **best_params
            }
            model = lgb.LGBMRegressor(**params)
            early_stopping_callback = lgb.early_stopping(
                100, first_metric_only=True, verbose=True)

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[early_stopping_callback]
            )

            valid_predict = model.predict(X_valid)
            valid_preds[valid_idx] = valid_predict
            preds[valid_idx, 0] += valid_predict / EPOCHS

            test_predict = model.predict(X_test)
            TEST_PREDS[:, 0] += test_predict / EPOCHS / SPLIT
            test_prediction_list.append(test_predict)

            score = metrics.mean_squared_error(
                y_valid, valid_predict, squared=False)
            model_dict[f'Epoch{i + 1}-Fold{fold + 1}'] = model
            model.booster_.save_model(
                f'../baseline_lgb_models/lgbm_model_epoch{i + 1}_fold{fold + 1}.txt')

        final_score = metrics.mean_squared_error(
            train_feats[target_col], valid_preds, squared=False)
        scores.append(final_score)

    print("Avg Loss:", np.mean(scores))

    print('metric LGBM = {:.5f}'.format(metrics.mean_squared_error(
        train_feats[target_col], preds[:, 0], squared=False)))

    return TEST_PREDS


def train_xgb_model(train_feats, test_feats):
    TEST_PREDS = np.zeros((len(test_feats), 1))

    os.makedirs('../baseline_xgb_models', exist_ok=True)

    EPOCHS = 1
    SPLIT = 10

    test_prediction_list = []
    model_dict = {}
    scores = []
    preds = np.zeros((len(train_feats), 1))

    best_params = {
        'reg_alpha': 0.6016917340618352,
        'reg_lambda': 3.8071290717767194,
        'colsample_bytree': 0.45216556596658897,
        'subsample': 0.4832292138435902,
        'learning_rate': 0.002,
        'max_depth': 27,
        'min_child_weight': 1.0,
        "n_estimators": 11_861,
        'n_jobs': 4
    }

    for i in range(EPOCHS):
        kf = model_selection.KFold(
            n_splits=SPLIT, random_state=42 + i * 10, shuffle=True)
        valid_preds = np.zeros(train_feats.shape[0])
        X_test = test_feats[train_cols]

        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
            print(f'Epoch: {i + 1} Fold: {fold + 1}')
            X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
            X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "random_state": 42,
                "verbosity": 0,
                **best_params
            }
            model = xgb.XGBRegressor(**params)

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=100,
                verbose=True
            )

            valid_predict = model.predict(X_valid)
            valid_preds[valid_idx] = valid_predict
            preds[valid_idx, 0] += valid_predict / EPOCHS

            test_predict = model.predict(X_test)
            TEST_PREDS[:, 0] += test_predict / EPOCHS / SPLIT
            test_prediction_list.append(test_predict)

            score = metrics.mean_squared_error(
                y_valid, valid_predict, squared=False)
            model_dict[f'Epoch{i + 1}-Fold{fold + 1}'] = model
            model.save_model(
                f'../baseline_xgb_models/xgb_model_epoch{i + 1}_fold{fold + 1}.json')
            # model.load_model(f'../baseline_xgb_models/xgb_model_epoch{i + 1}_fold{fold + 1}.json')

        final_score = metrics.mean_squared_error(
            train_feats[target_col], valid_preds, squared=False)
        scores.append(final_score)

    print("Avg Loss:", np.mean(scores))

    print('metric XGB = {:.5f}'.format(metrics.mean_squared_error(
        train_feats[target_col], preds[:, 0], squared=False)))

    return TEST_PREDS


def train_cb_model(train_feats, test_feats):
    TEST_PREDS = np.zeros((len(test_feats), 1))

    os.makedirs('../baseline_cb_models', exist_ok=True)

    EPOCHS = 1
    SPLIT = 10

    test_prediction_list = []
    model_dict = {}
    scores = []
    preds = np.zeros((len(train_feats), 1))

    best_params = {
        'l2_leaf_reg': 3.8071290717767194,
        'colsample_bylevel': 0.45216556596658897,
        'subsample': 0.4832292138435902,
        'learning_rate': 0.002,
        'depth': 6,
        'thread_count': 4,
        'min_child_samples': 7,
        'iterations': 11_861,
    }

    for i in range(EPOCHS):
        kf = model_selection.KFold(
            n_splits=SPLIT, random_state=42 + i * 10, shuffle=True)
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

            test_predict = model.predict(X_test)
            TEST_PREDS[:, 0] += test_predict / EPOCHS / SPLIT
            test_prediction_list.append(test_predict)

            score = metrics.mean_squared_error(
                y_valid, valid_predict, squared=False)
            model_dict[f'Epoch{i + 1}-Fold{fold + 1}'] = model
            model.save_model(
                f'../baseline_cb_models/cb_model_epoch{i + 1}_fold{fold + 1}.cbm')
            # model.load_model(f'../baseline_cb_models/cb_model_epoch{i + 1}_fold{fold + 1}.cbm')

        final_score = metrics.mean_squared_error(
            train_feats[target_col], valid_preds, squared=False)
        scores.append(final_score)

    print("Avg Loss:", np.mean(scores))

    print('metric CB = {:.5f}'.format(metrics.mean_squared_error(
        train_feats[target_col], preds[:, 0], squared=False)))

    return TEST_PREDS


def train_lgbm_optuna(train_feats, test_feats):
    os.makedirs('../baseline_lgb_models_optuna', exist_ok=True)

    def objective(trial):
        EPOCHS = 1
        SPLIT = 10

        model_dict = {}
        scores = []
        preds = np.zeros((len(train_feats), 1))

        best_params = {
            'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 1.0),
            'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 5.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.4, 1.0),
            'subsample': trial.suggest_float("subsample", 0.4, 1.0),
            'learning_rate': trial.suggest_float("learning_rate", 1e-4, 1e-1),
            'num_leaves': trial.suggest_int("num_leaves", 5, 50),
            'max_depth': trial.suggest_int("max_depth", 5, 30),
            'min_child_samples': trial.suggest_int("min_child_samples", 2, 30),
            'n_jobs': 4,
            "n_estimators": trial.suggest_int("n_estimators", 1000, 20000)
        }

        for i in range(EPOCHS):
            kf = model_selection.KFold(
                n_splits=SPLIT, random_state=42 + i * 10, shuffle=True)
            valid_preds = np.zeros(train_feats.shape[0])

            for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
                print(f'Epoch: {i + 1} Fold: {fold + 1}')
                X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
                X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]
                params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "random_state": 42,
                    "verbosity": -1,
                    **best_params
                }
                model = lgb.LGBMRegressor(**params)
                early_stopping_callback = lgb.early_stopping(
                    100, first_metric_only=True, verbose=True)

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    callbacks=[early_stopping_callback]
                )

                valid_predict = model.predict(X_valid)
                valid_preds[valid_idx] = valid_predict
                preds[valid_idx, 0] += valid_predict / EPOCHS

                score = metrics.mean_squared_error(
                    y_valid, valid_predict, squared=False)
                model_dict[f'Epoch{i + 1}-Fold{fold + 1}'] = model
                model.booster_.save_model(
                    f'../baseline_lgb_models_optuna/lgbm_model_epoch{i + 1}_fold{fold + 1}.txt')

            final_score = metrics.mean_squared_error(
                train_feats[target_col], valid_preds, squared=False)
            scores.append(final_score)

        print("Avg Loss:", np.mean(scores))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("LightGBM Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    with open('lgbm_best_params.json', 'w') as json_file:
        json.dump(trial.params, json_file, indent=4)

    print("Save LightGBM best_params to json file")


def train_xgb_optuna(train_feats):
    os.makedirs('../baseline_xgb_models_optuna', exist_ok=True)

    def objective(trial):
        EPOCHS = 1
        SPLIT = 10

        model_dict = {}
        scores = []
        preds = np.zeros((len(train_feats), 1))

        best_params = {
            'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 1.0),
            'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 5.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.4, 1.0),
            'subsample': trial.suggest_float("subsample", 0.4, 1.0),
            'learning_rate': trial.suggest_float("learning_rate", 1e-4, 1e-1),
            'max_depth': trial.suggest_int("max_depth", 5, 30),
            'min_child_weight': trial.suggest_float("min_child_weight", 1.0, 5.0),
            'gamma': trial.suggest_float("gamma", 0.0, 10.0),
            'max_delta_step': trial.suggest_int("max_delta_step", 1, 5),
            'n_jobs': 4,
            "n_estimators": trial.suggest_int("n_estimators", 1000, 20000)
        }

        for i in range(EPOCHS):
            kf = model_selection.KFold(
                n_splits=SPLIT, random_state=42 + i * 10, shuffle=True)
            valid_preds = np.zeros(train_feats.shape[0])

            for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
                print(f'Epoch: {i + 1} Fold: {fold + 1}')
                X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
                X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]
                params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "random_state": 42,
                    "verbosity": 0,
                    **best_params
                }
                model = xgb.XGBRegressor(**params)

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=100,
                    verbose=True
                )

                valid_predict = model.predict(X_valid)
                valid_preds[valid_idx] = valid_predict
                preds[valid_idx, 0] += valid_predict / EPOCHS

                score = metrics.mean_squared_error(
                    y_valid, valid_predict, squared=False)
                model_dict[f'Epoch{i + 1}-Fold{fold + 1}'] = model
                model.save_model(
                    f'../baseline_xgb_models_optuna/xgb_model_epoch{i + 1}_fold{fold + 1}.json')

            final_score = metrics.mean_squared_error(
                train_feats[target_col], valid_preds, squared=False)
            scores.append(final_score)

        print("Avg Loss:", np.mean(scores))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("XGBoost Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    with open('xgb_best_params.json', 'w') as json_file:
        json.dump(trial.params, json_file, indent=4)

    print("Save XGBoost best_params to json file")


def train_cb_optuna(train_feats):
    os.makedirs('../baseline_cb_models_optuna', exist_ok=True)

    def objective(trial):
        EPOCHS = 1
        SPLIT = 10

        model_dict = {}
        scores = []
        preds = np.zeros((len(train_feats), 1))

        best_params = {
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'depth': trial.suggest_int('depth', 1, 6),
            'iterations': trial.suggest_int('iterations', 1000, 15000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
            'thread_count': 4
        }

        for i in range(EPOCHS):
            kf = model_selection.KFold(
                n_splits=SPLIT, random_state=42 + i * 10, shuffle=True)
            valid_preds = np.zeros(train_feats.shape[0])

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

                score = metrics.mean_squared_error(
                    y_valid, valid_predict, squared=False)
                model_dict[f'Epoch{i + 1}-Fold{fold + 1}'] = model
                model.save_model(
                    f'../baseline_cb_models_optuna/cb_model_epoch{i + 1}_fold{fold + 1}.cbm')

            final_score = metrics.mean_squared_error(
                train_feats[target_col], valid_preds, squared=False)
            scores.append(final_score)

        print("Avg Loss:", np.mean(scores))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("CatBoost Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    with open('cb_best_params.json', 'w') as json_file:
        json.dump(trial.params, json_file, indent=4)

    print("Save catboost best_params to json file")


if CFG.is_train_lgbm_optuna:
    train_lgbm_optuna(train_feats=data)

if CFG.is_train_xgb_optuna:
    train_xgb_optuna(train_feats=data)

if CFG.is_train_cb_optuna:
    train_cb_optuna(train_feats=data)

if CFG.is_train_lgbm_model:
    lgbm_preds = train_lgbm_model(train_feats=data, test_feats=test_feats)

if CFG.is_train_xgb_model:
    xgb_preds = train_xgb_model(train_feats=data, test_feats=test_feats)

if CFG.is_train_cb_model:
    cb_preds = train_cb_model(train_feats=data, test_feats=test_feats)
