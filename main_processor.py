import torch
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import regex as re


def kurtosis_func(x): return x.kurt()


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


class Preprocessor:
    def __init__(self, seed):
        self.seed = seed

        self.activities = ['Input', 'Remove/Cut',
                           'Nonproduction', 'Replace', 'Paste']

#         self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft',
#                        'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', 'Delete', 'Unidentified', 'Control',
#                        'Tab', 'Rightclick', 'ContextMenu', 'End', 'Meta', 'Alt', 'NumLock', 'Insert', 'Home',
#                        'Escape']

        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft',
                       'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', 'Delete', 'Unidentified']

        self.events2 = ['q', 'Space', 'Backspace']

        self.text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'",
                             '"', '-', '?', ';', '=', '/', '\\', ':']
        
        # self.text_changes = ['q', ' ', 'NoChange', ',']

        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                             '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '`', '~',
                             '|', '!', '\\']

        self.gaps = [1, 5, 15, 30, 50, 100]
        # self.gaps = []

        self.idf = defaultdict(float)

        self.device = "cuda" if torch.cuda.is_available else "cpu"

    def activity_counts(self, df):
        tmp_df = df.groupby('id').agg({'activity': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['activity'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.activities:
                di[k] = 0
            di["Move"] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
                else:
                    di["Move"] += v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'activity_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)
        epsilon = 1e-15

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log((ret[col] + epsilon) / (cnts + epsilon))
            ret[col] *= idf

        # cnts = ret.sum(axis=1)
        # for col in cols:
        #     ret[col] = ret[col] / cnts

        return ret

    def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret

    def text_change_counts(self, df):
        tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['text_change'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes:
                di[k] = 0
            di['Change'] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
                elif k.find('q') != -1 and not k.find('=>') != -1:
                    di['q'] += v
                elif k.find('=>') != -1:
                    di['Change'] += v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'text_change_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)
        epsilon = 1e-15

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log((ret[col] + epsilon) / (cnts + epsilon))
            ret[col] *= idf

        # cnts = ret.sum(axis=1)
        # for col in cols:
        #     ret[col] = ret[col] / cnts

        return ret

    def event_counts(self, df, colname):
        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.events:
                di[k] = 0
            di['Other'] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
                else:
                    di['Other'] += v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)
        epsilon = 1e-15

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log((ret[col] + epsilon) / (cnts + epsilon))
            ret[col] *= idf

        # cnts = ret.sum(axis=1)
        # for col in cols:
        #     ret[col] = ret[col] / cnts

        return ret

    def get_input_words(self, df):
        tmp_df = df[(~df['text_change'].str.contains('=>')) & (
            df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        tmp_df['text_change'] = tmp_df['text_change'].apply(
            lambda x: ''.join(x))
        tmp_df['text_change'] = tmp_df['text_change'].apply(
            lambda x: re.findall(r'q+', x))
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(
            lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(
            lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(
            lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_median'] = tmp_df['text_change'].apply(
            lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df

    # 这里是我完全新加的特征，考察的是text_change中含有=>的情况，左侧会出现很多q，右边通常只有一个q，因此我就没有对右边进行考察
    def get_change_words(self, df):
        tmp_df = df[df['text_change'].str.contains(
            '=>')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        tmp_df['text_change'] = tmp_df['text_change'].apply(
            lambda x: ''.join(x))
        tmp_df['left_word'] = tmp_df['text_change'].apply(
            lambda x: x.split('=>')[0])
        tmp_df['left_word'] = tmp_df['left_word'].apply(
            lambda x: re.findall(r'q+', x))
        tmp_df['origin_word_count'] = tmp_df['left_word'].apply(len)
        tmp_df['origin_word_length_mean'] = tmp_df['left_word'].apply(
            lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['origin_word_length_max'] = tmp_df['left_word'].apply(
            lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['origin_word_length_std'] = tmp_df['left_word'].apply(
            lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['origin_word_length_median'] = tmp_df['left_word'].apply(
            lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df = tmp_df.fillna(0.0)
        tmp_df.drop(['text_change', 'left_word'], axis=1, inplace=True)
        return tmp_df

    def action_time_events_activities_all(self, df):
        def action_time_events_activities(group):
            features = {}
            # for event in self.events2:
            #     event_group = group[group['down_event'] == event]
            #     features[f'down_{event}_id_mean'] = event_group['action_time'].mean(
            #     )
            #     features[f'down_{event}_id_std'] = event_group['action_time'].std()
            #     features[f'down_{event}_id_min'] = event_group['action_time'].min()
            #     features[f'down_{event}_id_max'] = event_group['action_time'].max()
            #     features[f'down_{event}_id_median'] = event_group['action_time'].median(
            #     )
            #     features[f'down_{event}_id_skew'] = event_group['action_time'].skew(
            #     )
            #     features[f'down_{event}_id_kurt'] = event_group['action_time'].kurt(
            #     )
            #     features[f'down_{event}_id_25%'] = event_group['action_time'].quantile(
            #         0.25)
            #     features[f'down_{event}_id_75%'] = event_group['action_time'].quantile(
            #         0.75)
            #     features[f'down_{event}_id_sum'] = event_group['action_time'].sum()

            for event in self.events2:
                event_group = group[group['up_event'] == event]
                features[f'up_{event}_id_mean'] = event_group['action_time'].mean()
                # features[f'up_{event}_id_std'] = event_group['action_time'].std()
                # features[f'up_{event}_id_min'] = event_group['action_time'].min()
                # features[f'up_{event}_id_max'] = event_group['action_time'].max()
                features[f'up_{event}_id_median'] = event_group['action_time'].median(
                )
                # features[f'up_{event}_id_skew'] = event_group['action_time'].skew()
                # features[f'up_{event}_id_kurt'] = event_group['action_time'].kurt()
                features[f'up_{event}_id_25%'] = event_group['action_time'].quantile(
                    0.25)
                features[f'up_{event}_id_75%'] = event_group['action_time'].quantile(
                    0.75)
                features[f'up_{event}_id_sum'] = event_group['action_time'].sum()

            for activity in self.activities:
                activity_group = group[group['activity'] == activity]
                features[f'{activity}_id_mean'] = activity_group['action_time'].mean()
                # features[f'{activity}_id_std'] = activity_group['action_time'].std()
                # features[f'{activity}_id_min'] = activity_group['action_time'].min()
                # features[f'{activity}_id_max'] = activity_group['action_time'].max()
                features[f'{activity}_id_median'] = activity_group['action_time'].median()
                # features[f'{activity}_id_skew'] = activity_group['action_time'].skew()
                # features[f'{activity}_id_kurt'] = activity_group['action_time'].kurt()
                features[f'{activity}_id_25%'] = activity_group['action_time'].quantile(
                    0.25)
                features[f'{activity}_id_75%'] = activity_group['action_time'].quantile(
                    0.75)
                features[f'{activity}_id_sum'] = activity_group['action_time'].sum()

            return pd.Series(features)

        return df.groupby('id').apply(action_time_events_activities)

    def idle_time_events_activities_all(self, df):

        df['idle_time'] = df.groupby('id')['down_time'].shift(-1)
        df['idle_time'] = df['idle_time'].fillna(0.0)

        def idle_time_events_activities(group):
            features = {}
            for event in self.events2:
                event_group = group[group['down_event'] == event]
                features[f'idle_down_{event}_id_mean'] = event_group['idle_time'].mean(
                )
                features[f'idle_down_{event}_id_std'] = event_group['idle_time'].std(
                )
                features[f'idle_down_{event}_id_min'] = event_group['idle_time'].min(
                )
                features[f'idle_down_{event}_id_max'] = event_group['idle_time'].max(
                )
                features[f'idle_down_{event}_id_median'] = event_group['idle_time'].median(
                )
                features[f'idle_down_{event}_id_skew'] = event_group['idle_time'].skew(
                )
                features[f'idle_down_{event}_id_kurt'] = event_group['idle_time'].kurt(
                )
                features[f'idle_down_{event}_id_25%'] = event_group['idle_time'].quantile(
                    0.25)
                features[f'idle_down_{event}_id_75%'] = event_group['idle_time'].quantile(
                    0.75)

            for event in self.events2:
                event_group = group[group['up_event'] == event]
                features[f'idle_up_{event}_id_mean'] = event_group['idle_time'].mean(
                )
                features[f'idle_up_{event}_id_std'] = event_group['idle_time'].std(
                )
                features[f'idle_up_{event}_id_min'] = event_group['idle_time'].min(
                )
                features[f'idle_up_{event}_id_max'] = event_group['idle_time'].max(
                )
                features[f'idle_up_{event}_id_median'] = event_group['idle_time'].median(
                )
                features[f'idle_up_{event}_id_skew'] = event_group['idle_time'].skew(
                )
                features[f'idle_up_{event}_id_kurt'] = event_group['idle_time'].kurt(
                )
                features[f'idle_up_{event}_id_25%'] = event_group['idle_time'].quantile(
                    0.25)
                features[f'idle_up_{event}_id_75%'] = event_group['idle_time'].quantile(
                    0.75)

            for activity in self.activities:
                activity_group = group[group['activity'] == activity]
                features[f'idle_{activity}_id_mean'] = activity_group['idle_time'].mean(
                )
                features[f'idle_{activity}_id_std'] = activity_group['idle_time'].std(
                )
                features[f'idle_{activity}_id_min'] = activity_group['idle_time'].min(
                )
                features[f'idle_{activity}_id_max'] = activity_group['idle_time'].max(
                )
                features[f'idle_{activity}_id_median'] = activity_group['idle_time'].median(
                )
                features[f'idle_{activity}_id_skew'] = activity_group['idle_time'].skew(
                )
                features[f'idle_{activity}_id_kurt'] = activity_group['idle_time'].kurt(
                )
                features[f'idle_{activity}_id_25%'] = activity_group['idle_time'].quantile(
                    0.25)
                features[f'idle_{activity}_id_75%'] = activity_group['idle_time'].quantile(
                    0.75)

            return pd.Series(features)

        return df.groupby('id').apply(idle_time_events_activities)

    def make_feats(self, df):
        feats = pd.DataFrame({'id': df['id'].unique().tolist()})

        print("Engineering time data")
        for gap in self.gaps:
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - \
                df[f'up_time_shift{gap}']
        df.drop(
            columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering cursor position data")
        for gap in self.gaps:
            df[f'cursor_position_shift{gap}'] = df.groupby(
                'id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - \
                df[f'cursor_position_shift{gap}']
            # df[f'cursor_position_abs_change{gap}'] = np.abs(
            #     df[f'cursor_position_change{gap}'])
        df.drop(
            columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering word count data")
        for gap in self.gaps:
            df[f'word_count_shift{gap}'] = df.groupby(
                'id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - \
                df[f'word_count_shift{gap}']
            # df[f'word_count_abs_change{gap}'] = np.abs(
            #     df[f'word_count_change{gap}'])
        df.drop(
            columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering statistical summaries for features")

        if self.device == "cpu":
            feats_stat = [
                ('event_id', ['max']),
                ('up_time', ['max']),
                ('action_time', ['mean', 'median', 'sem', 'sum', 'skew']),
                # ('activity', ['nunique']),
                # ('down_event', ['nunique']),
                # ('up_event', ['nunique']),
                # ('text_change', ['nunique']),
                ('cursor_position', ['nunique', 'max', 'last',
                 'median', 'sem', q1, q3]),
                ('word_count', ['nunique', 'max', 'last',
                 'median', 'sem', q1, q3])
            ]

            for gap in self.gaps:
                if gap == 1:
                    feats_stat.extend([
                        (f'action_time_gap{gap}', [
                         'sum', 'mean', 'std', 'median', 'skew']),
                        (f'cursor_position_change{gap}', [
                         'sum', 'max', 'min', 'mean', 'std', 'skew']),
                        # (f'word_count_change{gap}', [
                        #  'sum', 'max', 'min', 'mean', 'std', 'median', 'skew', kurtosis_func, q1, q3])
                    ])
                else:
                    feats_stat.extend([
                        (f'action_time_gap{gap}', [
                         'mean', 'std', 'median', 'skew']),
                        (f'cursor_position_change{gap}', [
                         'max', 'min', 'mean', 'std', 'skew']),
                        # (f'word_count_change{gap}', [
                        #  'max', 'min', 'mean', 'std', 'median', 'skew', kurtosis_func, q1, q3])
                    ])

        if self.device == "cuda":
            feats_stat = [
                ('event_id', ['max']),
                ('up_time', ['max']),
                ('action_time', ['mean', 'median', 'sem', 'sum', 'skew']),
                # ('activity', ['nunique']),
                # ('down_event', ['nunique']),
                # ('up_event', ['nunique']),
                # ('text_change', ['nunique']),
                ('cursor_position', ['nunique', 'max', 'last',
                 'median', 'sem', q1, q3]),
                ('word_count', ['nunique', 'max', 'last',
                 'median', 'sem', q1, q3])
            ]

            for gap in self.gaps:
                if gap == 1:
                    feats_stat.extend([
                        (f'action_time_gap{gap}', [
                         'sum', 'mean', 'std', 'median', 'skew']),
                        (f'cursor_position_change{gap}', [
                         'sum', 'max', 'min', 'mean', 'std', 'skew']),
                    #     (f'word_count_change{gap}', [
                    #      'sum', 'max', 'min', 'mean', 'std', 'median', 'skew', kurtosis_func])
                    ])
                else:
                    feats_stat.extend([
                        (f'action_time_gap{gap}', [
                         'mean', 'std', 'median', 'skew']),
                        (f'cursor_position_change{gap}', [
                         'max', 'min', 'mean', 'std', 'skew']),
                        # (f'word_count_change{gap}', [
                        #  'max', 'min', 'mean', 'std', 'median', 'skew', kurtosis_func])
                    ])

        pbar = tqdm(feats_stat)
        for item in pbar:
            colname, methods = item[0], item[1]
            for method in methods:
                pbar.set_postfix()
                if isinstance(method, str):
                    method_name = method
                else:
                    method_name = method.__name__
                pbar.set_postfix(column=colname, method=method_name)
                tmp_df = df.groupby(['id']).agg({colname: method}).reset_index().rename(
                    columns={colname: f'{colname}_{method_name}'})
                feats = feats.merge(tmp_df, on='id', how='left')

        print("Engineering activity counts data")
        tmp_df = self.activity_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering event counts data")
        tmp_df = self.event_counts(df, 'down_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        tmp_df = self.event_counts(df, 'up_event')
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering text change counts data")
        tmp_df = self.text_change_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering punctuation counts data")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        # print("Engineering input words data")
        # tmp_df = self.get_input_words(df)
        # feats = pd.merge(feats, tmp_df, on='id', how='left')

        # print("Engineering change words data")
        # tmp_df = self.get_change_words(df)
        # feats = pd.merge(feats, tmp_df, on='id', how='left')

        print("Engineering action time features")
        tmp_df = self.action_time_events_activities_all(df)
        tmp_df = tmp_df.reset_index()
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        # print("Engineering idle time features")
        # tmp_df = self.idle_time_events_activities_all(df)
        # tmp_df = tmp_df.reset_index()
        # feats = pd.merge(feats, tmp_df, on='id', how='left')

        print("Engineering ratios data")
        feats['word_time_ratio'] = feats['word_count_last'] / \
            feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_last'] / \
            feats['event_id_max']
        feats['position_time_ratio'] = feats['cursor_position_last'] / \
            feats['up_time_max']
        feats['position_event_ratio'] = feats['cursor_position_last'] / \
            feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max'] / feats['up_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_sum'] / \
            feats['up_time_max']
        feats['word_per_action_time'] = feats['word_count_max'] / feats['action_time_sum']
        feats['position_per_action_time'] = feats['cursor_position_last'] / feats['action_time_sum']
        feats.drop(columns=['up_time_max', 'event_id_max'], inplace=True)

        return feats
