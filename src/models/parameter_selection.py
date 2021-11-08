import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from definitions import SEED
from definitions import REPORT_DIR


def get_best_logreg_params(X, y, param_grid, use_precalc=True, financial_report=False):
    if use_precalc:
        if financial_report:
            return None, {'C': 0.65, 'class_weight': None}
        else:
            return None, {'C': 0.8, 'class_weight': None}

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    log_reg = LogisticRegression(random_state=SEED)

    gs = GridSearchCV(
        estimator=log_reg,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=4,
        scoring='roc_auc'
    )
    gs.fit(X, y)

    gs_report = {
        'rocauc': gs.cv_results_['mean_test_score'],
        'std': gs.cv_results_['std_test_score'],
        'params': gs.cv_results_['params'],
    }
    filename = f"grid_search_report_data_with{'' if financial_report else 'out'}_finance.csv"
    pd.DataFrame(gs_report).sort_values('rocauc', ascending=False).to_csv(REPORT_DIR.joinpath(filename))

    return gs.best_score_, gs.best_params_


def validate_logreg(X_train, y_train, X_val, y_val, params, metric=roc_auc_score):
    log_reg = LogisticRegression(random_state=SEED, **params)
    log_reg.fit(X_train, y_train)

    y_pred_train = log_reg.predict_proba(X_train)[:, 1]
    y_pred_val = log_reg.predict_proba(X_val)[:, 1]

    train_score = metric(y_train, y_pred_train)
    val_score = metric(y_val, y_pred_val)
    return train_score, val_score
