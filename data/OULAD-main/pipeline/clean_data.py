import logging
import os
import pandas as pd
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


_LOG = logging.getLogger(__name__)


def feature_eng_clean(df, base_wd, name):

    # transform code_presentation into year and semester columns
    df['year'] = df['code_presentation'].str.strip().str[0:4]
    df['semester'] = df['code_presentation'].str.strip().str[-1]
    df2 = df.copy(deep=True)

    # one hot encoding for OOS regression (RQ2)
    # ohe_cols = ['final_result', 'age_band', 'imd_band', 'disability', 'gender', 'region', 'highest_education',
    #             'code_module', 'assessment_type']
    # ohe_df = pd.get_dummies(df, columns=ohe_cols,  drop_first=True)  # drop_first to avoid singularity
    # df2 = pd.concat([df, ohe_df], axis=1)
    # df2 = df2.T.drop_duplicates().T

    # label encoding categorical variables
    le_cols = ['final_result', 'age_band', 'imd_band', 'disability', 'gender', 'region', 'highest_education',
               'code_module', 'semester']

    label_encoder = preprocessing.LabelEncoder()

    for col in le_cols:
        df2[col] = label_encoder.fit_transform(df2[col])

    # create overall total clicks column
    df2['overall_total_clicks'] = df2['total_n_days'] * df2['avg_total_sum_clicks']

    return df2


def cap_outliers(col):

    col_capped = col.clip(upper=col.quantile(0.98))

    return col_capped


def outlier_boxplt(df, name, base_wd):

    plt.clf()
    fig_outliers = sns.boxplot(x='variable', y='value', data=pd.melt(df))
    fig_outliers.set_xticklabels(fig_outliers.get_xticklabels(), rotation=90)
    plt.tight_layout()
    fig_outliers.figure.savefig(base_wd + '\\outputs\\plots\\cleaning\\' + name + '_outlier_boxplots.png')


def basic_clean(df, base_wd, name):

    # missingness
    print(f'Number of missing records for {df}:')
    print(df.isnull().sum())

    nulls = df.isnull().sum()
    drop_cols = []

    for i, v in nulls.items():
        if v == len(df):
            drop_cols.append(i)

    df = df.drop(columns=drop_cols)

    # drop missing records for non-numeric columns
    df = df.dropna(axis=0, subset=['imd_band'])

    # imputations
    df = df.fillna(0)

    # duplicates
    dups = len(df) - len(df.drop_duplicates())
    print(f'Number of duplicate records: {dups}')
    df = df.drop_duplicates()

    # outliers - continuous variables
    prefixes = ['n_day', 'avg_sum']
    num_vars = ['num_of_prev_attempts', 'studied_credits'] + \
               list(compress(df.columns, df.columns.str.startswith(tuple(prefixes))))

    outlier_boxplt(df[num_vars], name, base_wd)

    for col in num_vars:
        if col != 'score':
            df[col] = cap_outliers(df[col])

    outlier_boxplt(df[num_vars], name + '_capped', base_wd)

    return df, num_vars


def clean_data(df):

    base_wd = os.path.normpath(os.getcwd())

    dir = 'outputs\\plots\\cleaning'
    if not os.path.exists(dir):
        os.makedirs(dir)

    df2, num_vars = basic_clean(df, base_wd, df.name)
    df_cleaned = feature_eng_clean(df2, base_wd, df.name)

    # save cleaned df
    df_cleaned.to_csv(base_wd + f'\\outputs\\dataframes\\' + df.name + '_cleaned.csv')

    return df_cleaned, df2
