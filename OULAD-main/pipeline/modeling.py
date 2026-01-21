""" Outcomes Notes:
RQ1 - final result (classification) - predict negative outcomes
"""

import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, r2_score, mean_squared_error, \
    confusion_matrix, roc_curve, RocCurveDisplay
import shap
from shap import Explanation

_LOG = logging.getLogger(__name__)


def get_x_y(df):

    drop_ivs = ['code_presentation', 'id_student', 'final_result', 'date_registration', 'date_unregistration',
                'id_assessment', 'date_submitted', 'score', 'dv', 'score']

    drop_VLE_ivs = ['n_days_dataplus', 'n_days_dualpane', 'n_days_externalquiz', 'n_days_folder', 'n_days_forumng',
                'n_days_glossary', 'n_days_homepage', 'n_days_htmlactivity', 'n_days_oucollaborate',
                'n_days_oucontent', 'n_days_ouelluminate', 'n_days_ouwiki', 'n_days_page', 'n_days_questionnaire',
                'n_days_quiz', 'n_days_repeatactivity', 'n_days_resource', 'n_days_sharedsubpage', 'n_days_subpage',
                'n_days_url', 'avg_sum_clicks_dataplus', 'avg_sum_clicks_dualpane', 'avg_sum_clicks_externalquiz',
                'avg_sum_clicks_folder', 'avg_sum_clicks_forumng', 'avg_sum_clicks_glossary', 'avg_sum_clicks_homepage',
                'avg_sum_clicks_htmlactivity', 'avg_sum_clicks_oucollaborate', 'avg_sum_clicks_oucontent',
                'avg_sum_clicks_ouelluminate', 'avg_sum_clicks_ouwiki', 'avg_sum_clicks_page',
                'avg_sum_clicks_questionnaire', 'avg_sum_clicks_quiz', 'avg_sum_clicks_repeatactivity',
                'avg_sum_clicks_resource', 'avg_sum_clicks_sharedsubpage', 'avg_sum_clicks_subpage',
                'avg_sum_clicks_url']

    drop_ivs += drop_VLE_ivs

    # withdraw = 3, fail = 1, pass = 2, distinction = 0
    df['dv'] = np.where((df['final_result'] == 3) | (df['final_result'] == 1), 1, 0)
    y = df['dv']

    cols = [x for x in df.columns if x not in drop_ivs]

    X = df[cols]

    return X, y


def grid_search(X, y):

    grid = {
        'n_estimators': [400, 500, 600],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [3, 5, 7],
        'min_samples_split': [6, 8, 10, 12],
        'bootstrap': [True, False],
        'random_state': [18]
    }

    metric = 'roc_auc'  # 'accuracy', 'f1'
    est = RandomForestClassifier()

    gs_rf = GridSearchCV(estimator=est,
                         param_grid=grid,
                         cv=5,
                         n_jobs=-1,
                         refit=True,
                         verbose=3,
                         scoring=metric)
    gs_rf.fit(X, y)
    params = gs_rf.best_params_

    return params


def run_model(df, outcome):

    base_wd = os.path.normpath(os.getcwd())
    dir = 'outputs\\models\\'
    if not os.path.exists(dir):
        os.makedirs(dir)

    X, y = get_x_y(df)

    # grid search
    params = grid_search(X, y)
    print(f'optimal parameters: {params}')

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=101)

    # train model with optimal parameters
    mod = RandomForestClassifier(n_estimators=list(params.values())[4],
                                 bootstrap=list(params.values())[0],
                                 max_features=list(params.values())[2],
                                 max_depth=list(params.values())[1],
                                 min_samples_split=list(params.values())[3],
                                 random_state=18)

    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)

    # Save to dataframe
    y_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}).join(X_test)
    y_df.to_csv(base_wd + f"\\outputs\\models\\rq1_y_predict.csv")

    results = {}

    # save feature importances
    feats = {}
    for feature, importance in zip(X_train.columns, mod.feature_importances_):
        feats[feature] = importance
    results['importances'] = feats

    results['f1_score'] = f1_score(y_test, y_pred)  # binary targets only
    results['roc_auc'] = roc_auc_score(y_test, y_pred)

    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    # NOTE - cm[0][0] = TP, cm[1][1] = TN, cm[0][1] = FP, cm[1][0] = FN

    _LOG.info(f'Model RQ1 results: {results}')
    results_df = pd.DataFrame.from_dict(results.items())
    results_df.to_csv(base_wd + f"\\outputs\\models\\rq1_results.csv")
    _LOG.info(f'Model metrics for RQ1 results saved.')

    # Shapley values
    _LOG.info(f'Calculating Shapley values for RQ1 model.')
    X_sub = shap.utils.sample(X, 1500)
    X_sub['year'] = X_sub['year'] .astype(float)
    explainer = shap.TreeExplainer(mod, X_sub)
    shap_values = explainer.shap_values(X_sub)

    # SHAP waterfall plot - local prediction
    plt.clf()
    shap.initjs()
    row = 1
    shap.waterfall_plot(shap.Explanation(values=shap_values[0][row],
                                         base_values=explainer.expected_value[0],  # 1
                                         data=X_sub.iloc[row],
                                         feature_names=X_sub.columns.tolist())
                        )
    waterfall = plt.gcf()
    plt.tight_layout()
    waterfall.savefig(base_wd + f"\\outputs\\models\\rq1_waterfall_row{row}.png")
    # Notes - f(x) is the model predict_proba value, E[f(x)] is the base value

    # SHAP beeswarm plot
    shap.initjs()
    shap.summary_plot(shap_values=shap_values[1],
                      features=X_sub,
                      feature_names=list(X_sub.columns),
                      max_display=11,
                      show=False
                      )
    beeswarm = plt.gcf()
    beeswarm.savefig(base_wd + f"\\outputs\\models\\rq1_beeswarm.png")
