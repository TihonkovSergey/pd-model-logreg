import os

import pandas as pd
import numpy as np

from definitions import TARGET_NAME, DATA_DIR
from src.features.feature_extraction import prepare_dataset, woe_transform
from src.data.dataframes import get_train, get_test

train_df_fr_columns = {
    'ab_other_current_assets', 'gen_21', 'ar_selling_expenses', 'f_2', 'ar_taxes^2', 'gen_15',
    'gen_3', 'ab_cash_and_securities', 'ar_profit_before_tax^2', 'f_11^2', 'ul_capital_sum',
    'ab_immobilized_assets^2', 'gen_2^2', 'gen_6_1^2', 'gen_16^2', 'gen_11', 'f_1', 'gen_3^2',
    'ab_accounts_receivable^2', 'gen_7^2', 'f_5', 'ab_other_current_assets^2', 'gen_8', 'gen_19',
    'ar_management_expenses^2', 'ar_revenue^2', 'gen_19^2', 'gen_0', 'gen_2', 'ar_sale_cost^2',
    'ab_mobile_current_assets^2', 'ar_other_profit_and_losses', 'gen_16', 'gen_6^2',
    'ab_borrowed_capital', 'gen_21^2', 'bus_age^2', 'gen_4^2', 'ogrn_age', 'adr_actual_age',
    'f_7^2', 'gen_10', 'f_6^2', 'ab_immobilized_assets', 'ar_taxes', 'gen_0^2', 'ar_sale_profit',
    'gen_17^2', 'f_1^2', 'adr_actual_age^2', 'f_0', 'gen_10^2', 'f_6', 'gen_11^2',
    'ab_short_term_borrowing', 'gen_15^2', 'ar_sale_profit^2', 'ab_own_capital^2', 'gen_18^2',
    'f_9', 'gen_5^2', 'f_11', 'gen_4', 'gen_14', 'f_8', 'f_2^2', 'gen_1', 'ar_management_expenses',
    'ar_revenue', 'f_12^2', 'gen_9', 'ar_total_expenses', 'gen_13', 'ab_long_term_liabilities',
    'ar_total_expenses^2', 'ul_capital_sum^2', 'gen_1^2', 'ab_losses', 'gen_14^2', 'f_12',
    'ul_branch_cnt', 'ul_staff_range', 'ab_other_borrowings^2', 'f_3^2', 'gen_13^2', 'f_4',
    'ar_profit_before_tax', 'ar_other_profit_and_losses^2', 'gen_6_1', 'gen_6', 'f_10^2',
    'gen_12^2', 'ab_borrowed_capital^2', 'ab_accounts_receivable', 'ab_accounts_payable^2',
    'gen_9^2', 'ab_inventory', 'ab_accounts_payable', 'ab_inventory^2', 'ab_other_borrowings',
    'f_9^2', 'f_7', 'gen_20^2', 'ul_founders_cnt^2', 'ar_balance_of_rvns_and_expns',
    'ab_mobile_current_assets', 'gen_12', 'default_12m', 'f_0^2', 'gen_8^2',
    'ar_balance_of_rvns_and_expns^2', 'bus_age', 'gen_7', 'f_10', 'ar_sale_cost', 'ul_founders_cnt',
    'ab_own_capital', 'ab_long_term_liabilities^2', 'gen_17', 'ab_cash_and_securities^2', 'f_3',
    'gen_20', 'ogrn_age^2', 'ar_net_profit', 'f_5^2', 'gen_18', 'gen_5', 'f_4^2', 'ab_losses^2',
    'f_8^2', 'ar_net_profit^2', 'ab_short_term_borrowing^2', 'ar_selling_expenses^2'
}
train_df_nofr_columns = {
    'ul_founders_cnt', 'bus_age^2', 'ul_capital_sum^2', 'adr_actual_age', 'ogrn_age', 'ul_branch_cnt',
    'default_12m', 'ul_staff_range', 'ogrn_age^2', 'ul_capital_sum', 'bus_age', 'ul_founders_cnt^2',
    'adr_actual_age^2'
}
test_df_columns = {
    'ab_other_current_assets', 'gen_21', 'ar_selling_expenses', 'f_2', 'ar_taxes^2', 'gen_15', 'gen_3',
    'ab_cash_and_securities', 'ar_profit_before_tax^2', 'f_11^2', 'ul_capital_sum', 'ab_immobilized_assets^2',
    'gen_2^2', 'gen_6_1^2', 'gen_16^2', 'gen_11', 'f_1', 'gen_3^2', 'ab_accounts_receivable^2', 'gen_7^2',
    'f_5', 'ab_other_current_assets^2', 'gen_8', 'gen_19', 'ar_management_expenses^2', 'ar_revenue^2',
    'gen_19^2', 'gen_0', 'gen_2', 'ar_sale_cost^2', 'ab_mobile_current_assets^2', 'ar_other_profit_and_losses',
    'gen_16', 'gen_6^2', 'ab_borrowed_capital', 'gen_21^2', 'bus_age^2', 'gen_4^2', 'ogrn_age',
    'adr_actual_age', 'f_7^2', 'gen_10', 'f_6^2', 'ab_immobilized_assets', 'ar_taxes', 'gen_0^2',
    'ar_sale_profit', 'gen_17^2', 'f_1^2', 'adr_actual_age^2', 'f_0', 'gen_10^2', 'f_6', 'gen_11^2',
    'ab_short_term_borrowing', 'gen_15^2', 'ar_sale_profit^2', 'ab_own_capital^2', 'gen_18^2', 'f_9', 'gen_5^2',
    'f_11', 'gen_4', 'gen_14', 'f_8', 'f_2^2', 'gen_1', 'ar_management_expenses', 'ar_revenue', 'f_12^2',
    'gen_9', 'ar_total_expenses', 'gen_13', 'ab_long_term_liabilities', 'ar_total_expenses^2',
    'ul_capital_sum^2', 'gen_1^2', 'ab_losses', 'gen_14^2', 'f_12', 'ul_branch_cnt', 'ul_staff_range',
    'ab_other_borrowings^2', 'f_3^2', 'gen_13^2', 'f_4', 'ar_profit_before_tax', 'ar_other_profit_and_losses^2',
    'gen_6_1', 'gen_6', 'f_10^2', 'gen_12^2', 'ab_borrowed_capital^2', 'ab_accounts_receivable',
    'ab_accounts_payable^2', 'gen_9^2', 'ab_inventory', 'ab_accounts_payable', 'ab_inventory^2',
    'ab_other_borrowings', 'f_9^2', 'f_7', 'gen_20^2', 'ul_founders_cnt^2', 'ar_balance_of_rvns_and_expns',
    'ab_mobile_current_assets', 'gen_12', 'default_12m', 'f_0^2', 'gen_8^2', 'ar_balance_of_rvns_and_expns^2',
    'bus_age', 'gen_7', 'f_10', 'ar_sale_cost', 'ul_founders_cnt', 'ab_own_capital',
    'ab_long_term_liabilities^2', 'gen_17', 'ab_cash_and_securities^2', 'f_3', 'gen_20', 'ogrn_age^2',
    'ar_net_profit', 'f_5^2', 'gen_18', 'gen_5', 'f_4^2', 'ab_losses^2', 'f_8^2', 'ar_net_profit^2',
    'ab_short_term_borrowing^2', 'ar_selling_expenses^2'
}


class TestFeatures:
    prepared_data_dir = DATA_DIR.joinpath("prepared")
    path_train_data_with_financial_report = prepared_data_dir.joinpath('prepared_train_with_financial_report.csv')
    path_train_data_without_financial_report = prepared_data_dir.joinpath(
        'prepared_train_without_financial_report.csv')
    path_test_data = prepared_data_dir.joinpath('prepared_test.csv')

    def test_prepare_dataset_load(self):
        if self.path_train_data_with_financial_report.exists():
            os.remove(self.path_train_data_with_financial_report)
        if self.path_train_data_without_financial_report.exists():
            os.remove(self.path_train_data_without_financial_report)
        if self.path_test_data.exists():
            os.remove(self.path_test_data)

        train = get_train()
        test = get_test()
        train_df_fr, train_df_nofr, test = prepare_dataset(train, test, load_from_disk=False)
        assert set(train_df_fr.columns) == train_df_fr_columns
        assert set(train_df_nofr.columns) == train_df_nofr_columns
        assert set(test.columns) == test_df_columns
        assert self.path_train_data_with_financial_report.exists()
        assert self.path_train_data_without_financial_report.exists()
        assert self.path_test_data.exists()

    def test_prepare_dataset(self):
        train_df_fr, train_df_nofr, test = prepare_dataset(None, None, load_from_disk=True)
        assert set(train_df_fr.columns) == train_df_fr_columns
        assert set(train_df_nofr.columns) == train_df_nofr_columns
        assert set(test.columns) == test_df_columns

    def test_woe_transform(self):
        train_columns = {
            'record_id', 'ar_revenue', 'ar_total_expenses', 'ar_sale_cost', 'ar_selling_expenses',
            'ar_management_expenses', 'ar_sale_profit', 'ar_balance_of_rvns_and_expns',
            'ar_profit_before_tax', 'ar_taxes', 'ar_other_profit_and_losses', 'ar_net_profit',
            'ab_immobilized_assets', 'ab_mobile_current_assets', 'ab_inventory', 'ab_accounts_receivable',
            'ab_other_current_assets', 'ab_cash_and_securities', 'ab_losses', 'ab_own_capital',
            'ab_borrowed_capital', 'ab_long_term_liabilities', 'ab_short_term_borrowing',
            'ab_accounts_payable', 'ab_other_borrowings', 'bus_age', 'ogrn_age', 'adr_actual_age',
            'head_actual_age', 'cap_actual_age', 'ul_staff_range', 'ul_capital_sum', 'ul_founders_cnt',
            'ul_branch_cnt', 'ul_strategic_flg', 'ul_systematizing_flg', 'default_12m'
        }
        size = 10
        train_df = pd.DataFrame(np.random.randint(0, 100, size=(size, len(train_columns))),
                                columns=train_columns)
        train_df[TARGET_NAME] = train_df[TARGET_NAME] > 50
        val_df = pd.DataFrame(np.random.randint(0, 100, size=(size, len(train_columns))),
                              columns=train_columns)
        val_df[TARGET_NAME] = val_df[TARGET_NAME] > 50

        train_woe, val_woe, bins = woe_transform(train_df, val_df)

        true_columns = {
            'ab_accounts_payable_woe',
            'ab_accounts_receivable_woe',
            'ab_borrowed_capital_woe',
            'ab_cash_and_securities_woe',
            'ab_immobilized_assets_woe',
            'ab_inventory_woe',
            'ab_long_term_liabilities_woe',
            'ab_losses_woe',
            'ab_mobile_current_assets_woe',
            'ab_other_borrowings_woe',
            'ab_other_current_assets_woe',
            'ab_own_capital_woe',
            'ab_short_term_borrowing_woe',
            'adr_actual_age_woe',
            'ar_balance_of_rvns_and_expns_woe',
            'ar_management_expenses_woe',
            'ar_net_profit_woe',
            'ar_other_profit_and_losses_woe',
            'ar_profit_before_tax_woe',
            'ar_revenue_woe',
            'ar_sale_cost_woe',
            'ar_sale_profit_woe',
            'ar_selling_expenses_woe',
            'ar_taxes_woe',
            'ar_total_expenses_woe',
            'bus_age_woe',
            'cap_actual_age_woe',
            'default_12m',
            'head_actual_age_woe',
            'ogrn_age_woe',
            'record_id_woe',
            'ul_branch_cnt_woe',
            'ul_capital_sum_woe',
            'ul_founders_cnt_woe',
            'ul_staff_range_woe',
            'ul_strategic_flg_woe',
            'ul_systematizing_flg_woe'
        }

        assert train_woe.shape == (size, len(true_columns))
        assert set(train_woe.columns) == true_columns
        assert val_woe.shape == (size, len(true_columns))
        assert set(val_woe.columns) == true_columns
        assert isinstance(bins, dict)
