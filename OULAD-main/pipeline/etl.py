import logging
import os
import glob
import pandas as pd
import requests
from zipfile import ZipFile
from io import BytesIO

_LOG = logging.getLogger(__name__)


def prep_vle_data(df, rq):

    keys_act = ['code_module', 'code_presentation', 'id_student', 'activity_type']
    keys_assess = ['code_module', 'code_presentation', 'id_student', 'id_assessment', 'activity_type']

    if rq == 1:
        keys = keys_act
        keys_2 = keys_act[0:3]
    else:
        keys = keys_assess
        keys_2 = keys_assess[0:4]

    n_days = df.groupby(keys)['date'] \
        .nunique().to_frame().reset_index().rename(columns={'date': 'n_days'})
    avg_clicks = df.groupby(keys)['sum_click'] \
        .mean().to_frame().reset_index().rename(columns={'sum_click': 'avg_sum_clicks'})
    vle_df = n_days.merge(avg_clicks, on=keys, how='inner')

    if rq == 2:
        vle_df['id_assessment'] = vle_df['id_assessment'].astype(str)

    vle_df = vle_df.set_index(keys).unstack().reset_index()
    vle_df.columns = ['_'.join(c) for c in list(vle_df.columns)]
    vle_df.columns = vle_df.columns.str.rstrip('_')

    if rq == 2:
        vle_df['id_assessment'] = vle_df['id_assessment'].astype(int)

    n_tot_days = df.groupby(keys_2)['date'] \
        .nunique().to_frame().reset_index().rename(columns={'date': 'total_n_days'})
    avg_tot_clicks = df.groupby(keys_2)['sum_click'] \
        .mean().to_frame().reset_index().rename(columns={'sum_click': 'avg_total_sum_clicks'})
    tot_vle = n_tot_days.merge(avg_tot_clicks, on=keys_2, how='inner')

    vle_df = vle_df.merge(tot_vle, on=keys_2, how='inner')

    return vle_df


def get_master_df_rq1(data_dict):
    """
    Merge relevant dataframes and fields from raw data to create master dataframes

    :param data_dict: dictionary of raw dataframes
    :return: dataframes with student outcome (final score classification and final exam score), vle info,
        and other student info
    """
    df1 = data_dict['studentInfo']
    df2 = data_dict['studentVle']
    df3 = data_dict['studentAssessment']
    df4 = data_dict['studentRegistration']
    df5 = data_dict['assessments']
    df6 = data_dict['vle']
    keys = ['code_module', 'code_presentation', 'id_student', 'id_assessment', 'id_site']

    rq1_df = df1.merge(df4, on=keys[0:3], how='inner')

    # aggregate VLE student interactions per course
    student_vle = df2.merge(df6, on=['code_module', 'code_presentation', 'id_site'], how='inner')\
        .drop(['week_from', 'week_to'], axis=1)

    rq1_vle_final = prep_vle_data(student_vle, 1)

    # merge vle info into student dfs
    rq1_df = rq1_df.merge(rq1_vle_final, on=keys[0:3], how='inner')

    return rq1_df


def raw_data_eda(data_dict, base_wd):

    cols = []
    shape = []
    missing_rows = []
    tbl_names = []
    for name, df in data_dict.items():
        column_names = list(df.columns.values)
        cols.append(column_names)
        shape.append(df.shape)
        missing_rows.append(df.isnull().any(axis=1).sum())
        tbl_names.append(name)

    summary_df = pd.DataFrame({
        'Table': tbl_names,
        'Row X Cols': shape,
        'Missing Rows': missing_rows,
        'Column Names': cols
    })

    dir = 'outputs\\dataframes'
    if not os.path.exists(dir):
        os.makedirs(dir)

    summary_df.to_csv(base_wd + "\\outputs\\dataframes\\raw_tbl_eda.csv")
    print(summary_df)


def read_data(base_wd):
    """
    Function to read in all data files in 'data' directory

    :return: dictionary of raw dataframes from csv files
    """

    data_dict = {}

    for file in glob.glob(base_wd + '\\data\\' + '*.csv'):
        name = file.split('\\')[-1][:-len('.csv')]
        df = pd.read_csv(file)
        data_dict[name] = df

    return data_dict


def load_data(base_wd):
    '''
    Load zipped data from OULAD's website and save in to local 'data' directory
    '''

    # create data directory if doesn't exist
    dir = 'data'
    if not os.path.exists(dir):
        os.makedirs(dir)

    zip_url = 'https://analyse.kmi.open.ac.uk/open_dataset/download'

    r = requests.get(zip_url)
    z = ZipFile(BytesIO(r.content))
    save_path = base_wd + '\\data\\'
    z.extractall(save_path)


def run_etl():

    _LOG.info('Downloading data from OULAD.')
    base_wd = os.path.normpath(os.getcwd())
    load_data(base_wd)

    _LOG.info('Reading in raw data into dataframes.')
    data_dict = read_data(base_wd)

    _LOG.info('Running initial EDA on raw data.')
    raw_data_eda(data_dict, base_wd)

    _LOG.info('Creating master dataframes.')
    rq1_df = get_master_df_rq1(data_dict)
    rq1_df.to_csv(base_wd + "\\outputs\\dataframes\\etl_rq1_df.csv")
    _LOG.info('Master dataframes created and saved.\n')

    return rq1_df
