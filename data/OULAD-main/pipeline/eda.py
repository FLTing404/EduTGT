import logging
import os
import numpy as np
import pandas as pd
from itertools import compress
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats

_LOG = logging.getLogger(__name__)


def vle_eda(df, base_wd):

    vle_df = pd.DataFrame({'vle_var': [],
                           'average': [],
                           'pct_utilized': []
                           })

    prefixes = ['n_day', 'avg_sum']
    vle_vars = ['total_n_days', 'avg_total_sum_clicks'] + \
               list(compress(df.columns, df.columns.str.startswith(tuple(prefixes))))

    vle_vars.append('final_result')
    df2 = df[vle_vars]

    for col in df2.columns:
        if col in ['final_result', 'score']:
            continue

        col_avg = df2[col].mean()
        df2[col] = np.where(df2[col] > 0, 1, 0)
        col_pct = df2[col].sum() / len(df2[col])

        df_temp = {'vle_var': col,
                   'average': col_avg,
                   'pct_utilized': col_pct
                   }

        vle_df = vle_df.append(df_temp, ignore_index=True)

    vle_df.to_csv(base_wd + f"\\outputs\\dataframes\\vle_df_rq1.csv")


def draw_histograms(df, variables, base_wd, subfolder):
    for i, var_name in enumerate(variables):
        plt.figure()
        df[var_name].hist(bins=10)
        plt.title(var_name + "Distribution")
        # plt.show()
        plt.savefig(base_wd + f'\\outputs\\plots\\{subfolder}\\histogram_{var_name}.png')


def draw_boxplots(df, variables, outcome, base_wd, subfolder):

    for i, var_name in enumerate(variables):
        if var_name == outcome:
            continue
        plt.figure()
        sns.set_style("whitegrid")
        sns.boxplot(x=outcome, y=var_name, data=df, palette="Set3")
        plt.tight_layout()
        # plt.show()
        name = f'{outcome}_by_{var_name}'
        plt.savefig(base_wd + f'\\outputs\\plots\\{subfolder}\\' + name + '.png')


def draw_barplots(df, variables, base_wd, subfolder, group=False):
    plt.clf()

    for i, var_name in enumerate(variables):

        if group:
            if var_name == 'final_result':
                continue
            df_grp = pd.crosstab(index=df[var_name], columns=df['final_result'], normalize="index")
            df_grp.plot(kind='bar', stacked=True, colormap='tab10')
            plt.legend(loc="upper left", ncol=4)
            plt.xlabel(var_name)
            plt.ylabel("Proportion")
            plt.title(f'{var_name} by final results')
            name = f'final_result_by_{var_name}'

        else:
            df[var_name].value_counts().plot(kind="bar")
            plt.title(var_name)
            name = f'barplot_{var_name}'

        # plt.show()
        plt.savefig(base_wd + f'\\outputs\\plots\\{subfolder}\\' + name + '.png')


def draw_corrplots(df, base_wd, subfolder):

    plt.clf()
    df = df.apply(pd.to_numeric, errors='coerce')
    matrix = df.corr()
    sns.heatmap(matrix, annot=True)
    # plt.show()
    plt.savefig(base_wd + f'\\outputs\\plots\\{subfolder}\\correlation_matrix.png')


def scatterplots(df, variables, outcome, base_wd, subfolder):

    for i, var_name in enumerate(variables):
        if var_name == outcome:
            continue

        plt.figure()
        pc = stats.pearsonr(df[var_name], df[outcome])[0]
        ax = sns.scatterplot(x=var_name, y=outcome, data=df)
        ax.set_title(f'{outcome} by {var_name} -- pearson coef: {pc}')
        ax.set_xlabel(var_name)
        ax.set_ylabel(outcome)
        # plt.show()
        name = f'{outcome}_by_{var_name}'
        plt.savefig(base_wd + f'\\outputs\\plots\\{subfolder}\\' + name + '.png')


def bivariate_eda(df, num_vars, cate_vars_short, base_wd, subfolder):

    outcome = 'final_result'
    num_vars.append('final_result')
    cate_vars_short.append('final_result')
    num_df = df.loc[:, df.columns.isin(num_vars)]
    cate_df = df.loc[:, df.columns.isin(cate_vars_short)]

    draw_corrplots(num_df, base_wd, subfolder)
    draw_barplots(cate_df, cate_df.columns, base_wd, subfolder, group=True)
    draw_boxplots(num_df, num_df.columns, outcome, base_wd, subfolder)


def univariate_eda(df, base_wd, subfolder):

    num_vars = ['num_of_prev_attempts', 'studied_credits', 'total_n_days',
                'avg_total_sum_clicks']
    num_df = df.loc[:, df.columns.isin(num_vars)]
    draw_histograms(num_df, num_df.columns, base_wd, subfolder)

    cate_vars_short = ['code_module', 'code_presentation', 'gender', 'region', 'highest_education', 'imd_band',
                       'age_band', 'disability', 'final_result']
    cate_df = df.loc[:, df.columns.isin(cate_vars_short)]
    draw_barplots(cate_df, cate_df.columns, base_wd, subfolder)

    return num_vars, cate_vars_short


def eda(df):

    base_wd = os.path.normpath(os.getcwd())
    df = df.drop(columns=['id_student'])

    subfolder = 'RQ1_EDA'
    dir = 'outputs\\plots\\' + subfolder
    if not os.path.exists(dir):
        os.makedirs(dir)

    num_vars, cate_vars_short = univariate_eda(df, base_wd, subfolder)
    bivariate_eda(df, num_vars, cate_vars_short, base_wd, subfolder)
    vle_eda(df, base_wd)
