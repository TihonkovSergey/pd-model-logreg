import click
import numpy as np
import pandas as pd
import scorecardpy as sc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from definitions import SEED
from definitions import TARGET_NAME, REPORT_DIR

from src.data.dataframes import get_test
from src.features.feature_extraction import woe_transform
from src.features.feature_selection import best_logreg_features
from src.models.parameter_selection import get_best_logreg_params
from src.models.threshold_tuning import get_optimal_threshold


@click.command()
def make_submission(test_path, train_fr_path, train_nofr_path):
    train_df_fr = pd.read_csv(train_fr_path)
    train_df_nofr = pd.read_csv(train_nofr_path)
    test = pd.read_csv(test_path)

    train_nofr, val_nofr = train_test_split(train_df_nofr, test_size=0.2, random_state=SEED)
    train_fr, val_fr = train_test_split(train_df_fr, test_size=0.2, random_state=SEED)

    train_nofr_woe, val_nofr_woe, bins_nofr = woe_transform(train_nofr, val_nofr)
    train_fr_woe, val_fr_woe, bins_fr = woe_transform(train_fr, val_fr)

    best_features_nofr = best_logreg_features(train_nofr_woe.drop(TARGET_NAME, axis=1),
                                              train_nofr_woe[TARGET_NAME],
                                              use_precalc=True,
                                              financial_report=False)
    X_train_nofr = train_nofr_woe.drop(TARGET_NAME, axis=1)[best_features_nofr]
    y_train_nofr = train_nofr_woe[TARGET_NAME]

    best_features_fr = best_logreg_features(train_fr_woe.drop(TARGET_NAME, axis=1),
                                            train_fr_woe[TARGET_NAME],
                                            use_precalc=True,
                                            financial_report=True)
    X_train_fr = train_fr_woe.drop(TARGET_NAME, axis=1)[best_features_fr]
    y_train_fr = train_fr_woe[TARGET_NAME]

    param_grid = {
        'C': np.arange(0.0, 5, 0.05),
        'class_weight': [None, 'balanced']
    }
    best_score_nofr, best_params_nofr = get_best_logreg_params(X_train_nofr,
                                                               y_train_nofr,
                                                               param_grid=param_grid,
                                                               use_precalc=True,
                                                               financial_report=False
                                                               )
    best_score_fr, best_params_fr = get_best_logreg_params(X_train_fr,
                                                           y_train_fr,
                                                           param_grid=param_grid,
                                                           use_precalc=True,
                                                           financial_report=True
                                                           )

    log_reg = LogisticRegression(random_state=SEED, **best_params_nofr)
    log_reg.fit(X_train_nofr[best_features_nofr], y_train_nofr)
    optimal_threshold_nofr = get_optimal_threshold(val_nofr_woe[best_features_nofr + [TARGET_NAME]],
                                                   log_reg,
                                                   financial_report=False)

    log_reg = LogisticRegression(random_state=SEED, **best_params_fr)
    log_reg.fit(X_train_fr[best_features_fr], y_train_fr)
    optimal_threshold_fr = get_optimal_threshold(val_fr_woe[best_features_fr + [TARGET_NAME]],
                                                 log_reg,
                                                 financial_report=True)

    # Train model on all train data
    data_nofr = pd.concat([train_nofr_woe, val_nofr_woe])
    X_train_nofr = data_nofr[best_features_nofr]
    y_train_nofr = data_nofr[TARGET_NAME]
    log_reg_nofr = LogisticRegression(random_state=SEED, **best_params_nofr)
    log_reg_nofr.fit(X_train_nofr, y_train_nofr)

    data_fr = pd.concat([train_fr_woe, val_fr_woe])
    X_train_fr = data_fr[best_features_fr]
    y_train_fr = data_fr[TARGET_NAME]
    log_reg_fr = LogisticRegression(random_state=SEED, **best_params_fr)
    log_reg_fr.fit(X_train_fr, y_train_fr)

    # Make submission
    test_df = get_test()
    test['id'] = test_df['record_id']

    test_fr = test[~test['ar_revenue'].isna()]
    test_fr = sc.woebin_ply(test_fr, bins_fr)

    test_nofr = test[test['ar_revenue'].isna()]
    test_nofr.dropna(axis=1, inplace=True)
    test_nofr = sc.woebin_ply(test_nofr, bins_nofr)

    pred_nofr = log_reg_nofr.predict_proba(test_nofr[best_features_nofr])[:, 1]
    pred_fr = log_reg_fr.predict_proba(test_fr[best_features_fr])[:, 1]

    pred_nofr = pred_nofr > optimal_threshold_nofr
    pred_nofr = pred_nofr.astype(int)

    pred_fr = pred_fr > optimal_threshold_fr
    pred_fr = pred_fr.astype(int)

    test_nofr['predict'] = pred_nofr
    test_fr['predict'] = pred_fr

    result_dict = dict(zip(test_nofr['id'], pred_nofr))
    result_dict.update(dict(zip(test_fr['id'], pred_fr)))

    result_df = test[['id']]
    result_df['target'] = result_df['id'].apply(lambda x: result_dict[x])
    result_df.to_csv(REPORT_DIR.joinpath('submit.csv'), index=False, sep=';')


if __name__ == '__main__':
    make_submission()
