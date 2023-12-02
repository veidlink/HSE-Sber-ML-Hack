import pandas as pd
import re
from tqdm import tqdm

tqdm.pandas()


def add_time(df):
    df['day'] = df['trans_time'].str.split().apply(lambda x: int(x[0]) % 7)
    # день недели, в который была совершена транзакция
    df['weekday'] = df['day'] % 7 + 1
    df['hour'] = df['trans_time'].apply(
        lambda x: re.search(' \d*', x).group(0)).astype(int)
    df['night'] = ~df['hour'].between(6, 22).astype(int)


def features_creation_advanced(x):
    features = []
    features.append(pd.Series(x['day'].value_counts(
        normalize=True).add_prefix('day_')))
    features.append(pd.Series(x['hour'].value_counts(
        normalize=True).add_prefix('hour_')))
    features.append(pd.Series(x['night'].value_counts(
        normalize=True).add_prefix('night_')))

    features.append(pd.Series(x[x['amount'] > 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count', 'skew', 'kurt'])
                              .add_prefix('positive_transactions_')).fillna(0))
    features.append(pd.Series(x[x['amount'] < 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count', 'skew', 'kurt'])
                              .add_prefix('negative_transactions_')).fillna(0))

    features.append(pd.Series(x['mcc_code'].value_counts(
        normalize=True).add_prefix('mcc_')).fillna(0))
    features.append(pd.Series(x['trans_type'].value_counts(
        normalize=True).add_prefix('trans_')).fillna(0))

    return pd.concat(features)


def mcc_segmented_stats(x):
    agg_funcs = ['min', 'max', 'mean', 'median',
                 'std', 'count', 'skew', 'kurt']
    return pd.Series(x['amount'].agg(agg_funcs).add_prefix('mcc_code_')).fillna(0)


def trans_segmented_stats(x):
    agg_funcs = ['min', 'max', 'mean', 'median',
                 'std', 'count', 'skew', 'kurt']
    return pd.Series(x['amount'].agg(agg_funcs).add_prefix('trans_code_')).fillna(0)


def create_features(df):
    advanced_features = df.groupby(df.index).progress_apply(
        features_creation_advanced).unstack(-1).fillna(0)
    mcc_features = df.groupby([df.index, 'mcc_code']).progress_apply(
        mcc_segmented_stats).unstack(-1).fillna(0)
    mcc_features.columns = mcc_features.columns.map(
        lambda tup: '_'.join(map(str, tup)))
    trans_features = df.groupby([df.index, 'trans_type']).progress_apply(
        trans_segmented_stats).unstack(-1).fillna(0)
    trans_features.columns = trans_features.columns.map(
        lambda tup: '_'.join(map(str, tup)))
    # rtf = df.groupby(df.index).progress_apply(repetitive_transaction_features).unstack(-1).fillna(0)
    print('Генерация фичей готова. Начилась склейка')

    # Объединение всех признаков
    final_features = advanced_features
    final_features = pd.concat(
        [advanced_features, mcc_features, trans_features], axis=1, join='inner')
    # final_features = add_time_features(df, final_features)
    return final_features
