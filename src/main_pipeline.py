from datetime import datetime

import numpy as np
import pandas as pd
import scorecardpy as sc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from definitions import SEED
from definitions import TARGET_NAME, REPORT_DIR
from definitions import LOGGER, USE_PRECALC

from src.data.download import download_data
from src.data.dataframes import get_train, get_test
from src.features.feature_extraction import prepare_dataset, woe_transform
from src.features.feature_selection import best_logreg_features
from src.models.parameter_selection import get_best_logreg_params
from src.models.parameter_selection import validate_logreg
from src.models.threshold_tuning import get_optimal_threshold

if __name__ == '__main__':
    start_time = datetime.now()

    # download data
    LOGGER.info("Download data...")
    download_data()

    # Getting original dataset
    LOGGER.info("Getting original dataset...")
    train_df = get_train()
    test_df = get_test()

    # Feature extraction
    # Build new features and split dataset for 2 parts: companies with and without financial report
    LOGGER.info(
        "Prepare data. Build new features and split dataset for 2 parts: companies with and without financial report")
    train_df_fr, train_df_nofr, test = prepare_dataset(train_df, test_df)

    train_nofr, val_nofr = train_test_split(train_df_nofr, test_size=0.2, random_state=SEED)
    train_fr, val_fr = train_test_split(train_df_fr, test_size=0.2, random_state=SEED)

    # WOE-transformation
    LOGGER.info("Calculate WOE-transformation...")
    train_nofr_woe, val_nofr_woe, bins_nofr = woe_transform(train_nofr, val_nofr)
    train_fr_woe, val_fr_woe, bins_fr = woe_transform(train_fr, val_fr)

    # Feature selection
    LOGGER.info("Calculate best features...")
    best_features_nofr = best_logreg_features(train_nofr_woe.drop(TARGET_NAME, axis=1),
                                              train_nofr_woe[TARGET_NAME],
                                              use_precalc=USE_PRECALC,
                                              financial_report=False)
    LOGGER.info(f"Best features nofr is: {best_features_nofr}")
    X_train_nofr = train_nofr_woe.drop(TARGET_NAME, axis=1)[best_features_nofr]
    y_train_nofr = train_nofr_woe[TARGET_NAME]
    X_val_nofr = val_nofr_woe.drop(TARGET_NAME, axis=1)[best_features_nofr]
    y_val_nofr = val_nofr_woe[TARGET_NAME]

    best_features_fr = best_logreg_features(train_fr_woe.drop(TARGET_NAME, axis=1),
                                            train_fr_woe[TARGET_NAME],
                                            use_precalc=USE_PRECALC,
                                            financial_report=True)
    LOGGER.info(f"Best features fr is: {best_features_fr}")
    X_train_fr = train_fr_woe.drop(TARGET_NAME, axis=1)[best_features_fr]
    y_train_fr = train_fr_woe[TARGET_NAME]
    X_val_fr = val_fr_woe.drop(TARGET_NAME, axis=1)[best_features_fr]
    y_val_fr = val_fr_woe[TARGET_NAME]

    # Logreg hyperparameters tuning
    LOGGER.info("Calculate best logreg parameters...")
    param_grid = {
        'C': np.arange(0.0, 5, 0.05),
        'class_weight': [None, 'balanced']
    }
    best_score_nofr, best_params_nofr = get_best_logreg_params(X_train_nofr,
                                                               y_train_nofr,
                                                               param_grid=param_grid,
                                                               use_precalc=USE_PRECALC,
                                                               financial_report=False
                                                               )
    best_score_fr, best_params_fr = get_best_logreg_params(X_train_fr,
                                                           y_train_fr,
                                                           param_grid=param_grid,
                                                           use_precalc=USE_PRECALC,
                                                           financial_report=True
                                                           )

    # Validation
    LOGGER.info("Validate best models")
    train_score_nofr, val_score_nofr = validate_logreg(X_train_nofr,
                                                       y_train_nofr,
                                                       X_val_nofr,
                                                       y_val_nofr,
                                                       params=best_params_nofr)

    LOGGER.info(f"""Logreg for data without financial report:
Train score: {train_score_nofr:.4f}, validation score: {val_score_nofr:.4f}""")

    train_score_fr, val_score_fr = validate_logreg(X_train_fr,
                                                   y_train_fr,
                                                   X_val_fr,
                                                   y_val_fr,
                                                   params=best_params_fr)

    LOGGER.info(f"""Logreg for data with financial report:
    Train score: {train_score_fr:.4f}, validation score: {val_score_fr:.4f}""")

    # Threshold tuning
    LOGGER.info("Calculate optimal thresholds.")
    log_reg = LogisticRegression(random_state=SEED, **best_params_nofr)
    log_reg.fit(X_train_nofr[best_features_nofr], y_train_nofr)
    optimal_threshold_nofr = get_optimal_threshold(val_nofr_woe[best_features_nofr + [TARGET_NAME]],
                                                   log_reg,
                                                   financial_report=False)
    LOGGER.info(f"Threshold for data without financial report: {optimal_threshold_nofr:.3f}")

    log_reg = LogisticRegression(random_state=SEED, **best_params_fr)
    log_reg.fit(X_train_fr[best_features_fr], y_train_fr)
    optimal_threshold_fr = get_optimal_threshold(val_fr_woe[best_features_fr + [TARGET_NAME]],
                                                 log_reg,
                                                 financial_report=True)
    LOGGER.info(f"Threshold for data with financial report: {optimal_threshold_fr:.3f}")

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
    LOGGER.info(f"Positive label ratio is {sum(pred_nofr) / len(pred_nofr):.4f} for data without financial report.")

    pred_fr = pred_fr > optimal_threshold_fr
    pred_fr = pred_fr.astype(int)
    LOGGER.info(f"Positive label ratio is {sum(pred_fr) / len(pred_fr):.4f} for data with financial report.")

    test_nofr['predict'] = pred_nofr
    test_fr['predict'] = pred_fr

    result_dict = dict(zip(test_nofr['id'], pred_nofr))
    result_dict.update(dict(zip(test_fr['id'], pred_fr)))

    result_df = test[['id']]
    result_df['target'] = result_df['id'].apply(lambda x: result_dict[x])
    result_df.to_csv(REPORT_DIR.joinpath('submit.csv'), index=False, sep=';')
    LOGGER.info(f"Submit dataframe saved into {REPORT_DIR} with shape {result_df.shape}")

    LOGGER.info(f"Execution time: {datetime.now() - start_time}")
