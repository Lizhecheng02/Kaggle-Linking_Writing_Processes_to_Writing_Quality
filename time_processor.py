import pandas as pd


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


class TimeProcessor:
    def __init__(self):
        pass

    def train_processor1(self, train_logs):
        train_agg_fe_df1 = train_logs.groupby("id")[['action_time']].agg(
            ['mean', 'std', 'min', 'max', 'median', q1, q3]
        )
        train_agg_fe_df1.columns = [
            '_'.join(x) for x in train_agg_fe_df1.columns]
        train_agg_fe_df1 = train_agg_fe_df1.add_prefix("tmp_")
        train_agg_fe_df1.reset_index(inplace=True)
        return train_agg_fe_df1

    def train_processor2(self, train_logs):
        train_agg_fe_df2 = train_logs.groupby("id")[
            ['down_time', 'up_time', 'cursor_position', 'word_count']].agg([q1, 'median', q3, 'max'])
        train_agg_fe_df2.columns = [
            '_'.join(x) for x in train_agg_fe_df2.columns]
        train_agg_fe_df2 = train_agg_fe_df2.add_prefix("tmp_")
        train_agg_fe_df2.reset_index(inplace=True)
        return train_agg_fe_df2

    def test_processor1(self, test_logs):
        test_agg_fe_df1 = test_logs.groupby("id")[['action_time']].agg(
            ['mean', 'std', 'min', 'max', 'median', q1, q3]
        )
        test_agg_fe_df1.columns = ['_'.join(x)
                                   for x in test_agg_fe_df1.columns]
        test_agg_fe_df1 = test_agg_fe_df1.add_prefix("tmp_")
        test_agg_fe_df1.reset_index(inplace=True)
        return test_agg_fe_df1

    def test_processor2(self, test_logs):
        test_agg_fe_df2 = test_logs.groupby("id")[['down_time', 'up_time', 'cursor_position', 'word_count']].agg(
            [q1, 'median', q3, 'max']
        )
        test_agg_fe_df2.columns = ['_'.join(x)
                                   for x in test_agg_fe_df2.columns]
        test_agg_fe_df2 = test_agg_fe_df2.add_prefix("tmp_")
        test_agg_fe_df2.reset_index(inplace=True)
        return test_agg_fe_df2

    def additional_processor(self, train_logs, test_logs):
        data = []

        for logs in [train_logs, test_logs]:
            logs['up_time_lagged'] = logs.groupby(
                'id')['up_time'].shift(1).fillna(logs['down_time'])
            logs['time_diff'] = abs(logs['down_time'] - logs['up_time_lagged'])

            group = logs.groupby('id')['time_diff']
            largest_lantency = group.max()
            smallest_lantency = group.min()
            median_lantency = group.median()
            std_lantency = group.std()
            initial_pause = logs.groupby('id')['down_time'].first()
            last_pause = logs.groupby('id')['up_time'].last()
            pause_zero_sec = group.apply(lambda x: (x < 0.5).sum())
            pauses_half_sec = group.apply(
                lambda x: ((x > 0.5) & (x < 1)).sum())
            pauses_1_sec = group.apply(lambda x: ((x > 1) & (x < 1.5)).sum())
            pauses_1_half_sec = group.apply(
                lambda x: ((x > 1.5) & (x < 2)).sum())
            pauses_2_sec = group.apply(lambda x: ((x > 2) & (x < 3)).sum())
            pauses_3_sec = group.apply(lambda x: (x > 3).sum())

            data.append(pd.DataFrame({
                'id': logs['id'].unique(),
                'largest_lantency': largest_lantency,
                'smallest_lantency': smallest_lantency,
                'median_lantency': median_lantency,
                'std_lantency': std_lantency,
                # 'initial_pause': initial_pause,
                'pause_zero_sec': pause_zero_sec,
                'pauses_half_sec': pauses_half_sec,
                'pauses_1_sec': pauses_1_sec,
                'pauses_1_half_sec': pauses_1_half_sec,
                'pauses_2_sec': pauses_2_sec,
                'pauses_3_sec': pauses_3_sec,
            }).reset_index(drop=True))

        return data
