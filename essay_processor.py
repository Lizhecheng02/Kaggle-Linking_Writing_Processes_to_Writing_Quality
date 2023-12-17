import torch
import pandas as pd
import regex as re


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


class EssayProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        def kurtosis_func(x): return x.kurt()

        if self.device == "cpu":
            self.SENT_AGGREGATIONS = [
                'count', 'mean', 'std', 'min', 'max',
                q1, 'median', q3, 'skew', kurtosis_func
            ]
            self.PARA_AGGREGATIONS = [
                'count', 'mean', 'std', 'min', 'max', 'first',
                'last', 'sem', q1, 'median', q3, 'skew', 'sum', kurtosis_func
            ]
        elif self.device == "cuda":
            self.SENT_AGGREGATIONS = [
                'count', 'mean', 'std', 'min',
                'max', 'median', 'skew', 'quantile'
            ]
            self.PARA_AGGREGATIONS = [
                'count', 'mean', 'std', 'min', 'max',
                'first', 'last', 'sem', 'median', 'quantile', 'skew', 'sum'
            ]

    def split_essays_into_sentences(self, df):
        essay_df = df
        essay_df['id'] = essay_df.index
        essay_df['sent'] = essay_df['essay'].apply(
            lambda x: re.split('\\.|\\?|\\!', x))
        essay_df = essay_df.explode('sent')
        essay_df['sent'] = essay_df['sent'].apply(
            lambda x: x.replace('\n', '').strip())
        essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
        essay_df['sent_word_count'] = essay_df['sent'].apply(
            lambda x: len(x.split(' ')))
        essay_df = essay_df[essay_df.sent_len != 0].reset_index(drop=True)
        return essay_df

    def compute_sentence_aggregations(self, df):
        sent_agg_df = pd.concat([
            df[['id', 'sent_len']].groupby(['id']).agg(self.SENT_AGGREGATIONS),
            df[['id', 'sent_word_count']].groupby(
                ['id']).agg(self.SENT_AGGREGATIONS)
        ], axis=1)
        sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
        sent_agg_df['id'] = sent_agg_df.index
        sent_agg_df = sent_agg_df.reset_index(drop=True)
        sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
        sent_agg_df = sent_agg_df.rename(
            columns={"sent_len_count": "sent_count"})
        return sent_agg_df

    def split_essays_into_paragraphs(self, df):
        essay_df = df
        essay_df['id'] = essay_df.index
        essay_df['paragraph'] = essay_df['essay'].apply(
            lambda x: x.split('\n'))
        essay_df = essay_df.explode('paragraph')
        essay_df['paragraph_len'] = essay_df['paragraph'].apply(
            lambda x: len(x))
        essay_df['paragraph_word_count'] = essay_df['paragraph'].apply(
            lambda x: len(x.split(' ')))
        essay_df = essay_df[essay_df.paragraph_len != 0].reset_index(drop=True)
        return essay_df

    def compute_paragraph_aggregations(self, df):
        paragraph_agg_df = pd.concat([
            df[['id', 'paragraph_len']].groupby(
                ['id']).agg(self.PARA_AGGREGATIONS),
            df[['id', 'paragraph_word_count']].groupby(
                ['id']).agg(self.PARA_AGGREGATIONS)
        ], axis=1)
        paragraph_agg_df.columns = [
            '_'.join(x) for x in paragraph_agg_df.columns]
        paragraph_agg_df['id'] = paragraph_agg_df.index
        paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
        paragraph_agg_df.drop(
            columns=["paragraph_word_count_count"], inplace=True)
        paragraph_agg_df = paragraph_agg_df.rename(
            columns={"paragraph_len_count": "paragraph_count"})
        return paragraph_agg_df

    def sentence_processor(self, df):
        sent_df = self.split_essays_into_sentences(df)
        sent_agg_df = self.compute_sentence_aggregations(sent_df)
        print("The shape of sent agg:", sent_agg_df.shape)
        return sent_agg_df

    def paragraph_processor(self, df):
        paragraph_df = self.split_essays_into_paragraphs(df)
        paragraph_agg_df = self.compute_paragraph_aggregations(paragraph_df)
        print("The shape of paragraph agg:", paragraph_agg_df.shape)
        return paragraph_agg_df
