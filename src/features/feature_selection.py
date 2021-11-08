from mlxtend.feature_selection import SequentialFeatureSelector as SFSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from definitions import SEED
from definitions import N_FEATURES_FR, N_FEATURES_NFR


def best_logreg_features(X, y, financial_report=False, use_precalc=True, n_splits=5):
    n_features = N_FEATURES_FR if financial_report else N_FEATURES_NFR
    if use_precalc:
        if financial_report:
            return ['f_12^2_woe', 'gen_19^2_woe', 'ar_taxes_woe', 'ul_capital_sum^2_woe', 'ul_founders_cnt_woe',
                    'gen_7_woe', 'gen_15_woe', 'ab_own_capital_woe', 'f_6_woe', 'f_9_woe', 'gen_6^2_woe', 'bus_age_woe',
                    'adr_actual_age_woe', 'ab_cash_and_securities_woe', 'ar_profit_before_tax_woe']
        return ['ul_founders_cnt_woe',
                'adr_actual_age_woe',
                'ogrn_age_woe',
                'ul_capital_sum_woe']

    log_reg = LogisticRegression(
        random_state=SEED,
        solver='liblinear',
        penalty='l1',
        max_iter=300,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    feature_selector = SFSelector(
        log_reg,
        k_features=n_features,
        forward=True,
        scoring='roc_auc',
        cv=cv,
        verbose=0,
    )

    feature_selector_report = feature_selector.fit(X, y)
    return list(feature_selector_report.subsets_[n_features]['feature_names'])
