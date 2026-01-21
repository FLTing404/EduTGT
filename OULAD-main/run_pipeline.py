import logging
from datetime import datetime

from pipeline.etl import run_etl
from pipeline.clean_data import clean_data
from pipeline.eda import eda
from pipeline.modeling import run_model

_LOG = logging.getLogger(__name__)


def main():

    logging.basicConfig(level=logging.INFO)

    kickoff = datetime.now()

    _LOG.info('Beginning initial EDA and ETL script.')
    rq1_df = run_etl()
    rq1_df.name = 'rq1_df'
    _LOG.info('Initial EDA and ETL script complete.\n')

    _LOG.info('Beginning initial data exploration and data cleaning.')
    rq1_df_cleaned, rq1_df2 = clean_data(rq1_df)
    _LOG.info('Initial data exploration and data cleaning complete.\n')

    _LOG.info('Beginning EDA with cleaned data.')
    eda(rq1_df2)
    _LOG.info('EDA with cleaned data complete.\n')

    _LOG.info('Beginning modeling.')
    start_time = datetime.now()
    run_model(rq1_df_cleaned, 4)
    reg_time = datetime.now()
    _LOG.info(f'RQ1 model duration: {reg_time - start_time}')

    _LOG.info('Modeling complete.')


if __name__ == '__main__':
    main()
