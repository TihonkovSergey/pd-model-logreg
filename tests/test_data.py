import pytest
from pathlib import Path

from definitions import DATA_DIR
from src.data.download import download_data
from src.data.dataframes import get_train, get_test, get_data_description


class TestData:
    def test_download(self):
        download_data(rewrite=True)
        path_raw_data = Path(DATA_DIR).joinpath("raw")
        assert path_raw_data.joinpath('PD-data-train.csv').exists()
        assert path_raw_data.joinpath('PD-data-test.csv').exists()
        assert path_raw_data.joinpath('PD-data-desc.csv').exists()

    def test_get_train(self):
        df = get_train()
        true_columns = {
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
        assert set(df.columns) == true_columns

    def test_get_test(self):
        df = get_test()
        true_columns = {
            'ar_total_expenses', 'ul_systematizing_flg', 'ar_other_profit_and_losses', 'ar_sale_profit',
            'ab_losses',
            'ul_strategic_flg', 'ul_capital_sum', 'ar_profit_before_tax', 'ab_borrowed_capital',
            'head_actual_age',
            'cap_actual_age', 'ab_mobile_current_assets', 'ul_staff_range', 'record_id',
            'ab_immobilized_assets',
            'ab_own_capital', 'ul_branch_cnt', 'ar_management_expenses', 'ab_accounts_receivable',
            'ar_selling_expenses',
            'ar_taxes', 'ab_short_term_borrowing', 'adr_actual_age', 'ar_revenue', 'ab_inventory',
            'ab_other_borrowings',
            'ogrn_age', 'ul_founders_cnt', 'ar_sale_cost', 'bus_age', 'ab_other_current_assets',
            'ar_net_profit',
            'ab_accounts_payable', 'ab_cash_and_securities', 'ar_balance_of_rvns_and_expns',
            'ab_long_term_liabilities'
        }
        assert set(df.columns) == true_columns

    def test_get_data_description(self):
        df = get_data_description()
        true_columns = {'desc_eng', 'desc_rus', 'field'}
        assert set(df.columns) == true_columns
        assert df.shape == (35, 3)
