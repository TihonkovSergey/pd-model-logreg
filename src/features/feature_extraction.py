import pandas as pd
import scorecardpy as sc

from definitions import TARGET_NAME
from definitions import DATA_DIR
from definitions import LOGGER


def prepare_dataset(train_df, test_df, load_from_disk=True):
    prepared_data_dir = DATA_DIR.joinpath("prepared")
    path_train_data_with_financial_report = prepared_data_dir.joinpath('prepared_train_with_financial_report.csv')
    path_train_data_without_financial_report = prepared_data_dir.joinpath('prepared_train_without_financial_report.csv')
    path_test_data = prepared_data_dir.joinpath('prepared_test.csv')

    if load_from_disk \
            and path_train_data_with_financial_report.exists() \
            and path_train_data_without_financial_report.exists() \
            and path_test_data.exists():
        train_with_fr = pd.read_csv(path_train_data_with_financial_report)
        train_without_fr = pd.read_csv(path_train_data_without_financial_report)
        test = pd.read_csv(path_test_data)
        LOGGER.info("Data successfully loaded from disk.")
        return train_with_fr, train_without_fr, test

    # удалим уникальный для каждое строки record_id
    train_df_ = train_df.drop(['record_id'], axis=1)
    test_df_ = test_df.drop(['record_id'], axis=1)

    test_size = test_df_.shape[0]

    # в обучающем множестве имеются дубликаты записей с разными record_id. Удалим их.
    train_df_.drop_duplicates(inplace=True)

    # конкатенируем трейн и тест для удобной работы
    df = pd.concat([train_df_, test_df_])

    # выкидываем неинформативные признаки
    df.drop(['ul_strategic_flg', 'ul_systematizing_flg'], axis=1, inplace=True)

    # 3 фичи (head_actual_age, adr_actual_age и cap_actual_age) полностью дублируют друг друга. удаляем 2 из них.
    df.drop(['head_actual_age', 'cap_actual_age'], axis=1, inplace=True)

    # категориальную фичу "ul_staff_range" превращаем в числовую
    df.replace({'ul_staff_range': {
        '[1-100]': 1,
        '(100-500]': 2,
        '> 500': 3,
    }}, inplace=True)

    # Буду прибавлять константу чтобы не получить бесконечности
    # или неопределенности при делении во время генерации новых фичей
    eps = 1e-2

    # Сокращения:
    total_capital = df.ab_own_capital + df.ab_borrowed_capital + eps  # капитал компании
    total_liabilities = df.ab_long_term_liabilities + df.ab_other_borrowings + df.ab_short_term_borrowing + eps  # сколько денег заняли
    revenue = df.ar_revenue + eps  # доход
    cas = df.ab_cash_and_securities + eps
    inventory = df.ab_inventory + eps

    # Генерируем новые признаки. Помним, что они должны быть объяснимы.
    # насколько большие займы по сравнению с капиталом
    df['gen_0'] = total_liabilities / total_capital

    # отношение дохода к
    df['gen_1'] = revenue / (df.ab_accounts_receivable + eps)  # дебиторской задолженности
    df['gen_2'] = revenue / total_capital  # капиталу
    df['gen_3'] = revenue / total_liabilities  # долгам
    df['gen_4'] = revenue / (total_capital - total_liabilities + eps)  # собственным средствам
    df['gen_5'] = (df.ab_immobilized_assets + eps) / revenue  # внеоборотным средствам
    df['gen_6'] = revenue / (df.ab_mobile_current_assets + eps)  # оборотным активам
    df['gen_6_1'] = revenue / (df.ab_immobilized_assets + eps)  # внеоборотным активам
    df['gen_7'] = revenue / cas  # денежным средствам и денежным эквивалентам

    # сколько платят налогов
    df['gen_8'] = (df.ar_taxes + eps) / revenue
    # какую часть дохода составляют продажи
    df['gen_9'] = (df.ar_sale_profit + eps) / revenue

    # если я правильно понимаю, ab_cash_and_securities это в том числе посчитанное имущество компании
    # можно учесть сможет ли компания отдать долги, если будет продавать имущество:
    df['gen_10'] = cas / total_liabilities
    df['gen_11'] = cas / total_capital
    df['gen_12'] = cas / (df.ab_accounts_receivable + eps)
    df['gen_13'] = cas / (df.ab_accounts_payable + eps)
    df['gen_14'] = cas / (df.ab_borrowed_capital + eps)
    df['gen_15'] = cas / (df.ab_short_term_borrowing + eps)
    df['gen_16'] = (df.ab_other_borrowings + eps) / cas

    # не знаю что значит "Запасы", но хочется посчитать насколько они большие относительно
    df['gen_17'] = inventory / total_capital  # капитала компании
    df['gen_18'] = inventory / total_liabilities  # ее займов
    df['gen_19'] = inventory / (df.ab_accounts_payable + eps)  # кредиторской задолженности
    df['gen_20'] = inventory / (df.ab_accounts_receivable + eps)  # дебиторской задолженности
    df['gen_21'] = inventory / revenue  # дохода

    # далее идут формулы, которые удалось найти на тематических сайтах
    df['f_0'] = (cas + df.ab_other_current_assets) / (
            df.ab_accounts_payable + df.ab_other_borrowings + df.ab_short_term_borrowing + eps)
    df['f_1'] = (cas + df.ab_other_current_assets + df.ab_accounts_receivable) / (
            df.ab_accounts_payable + df.ab_other_borrowings + df.ab_short_term_borrowing + eps)
    df['f_2'] = (inventory + df.ab_accounts_receivable + df.ab_other_current_assets + cas) / (
            df.ab_accounts_payable + df.ab_other_borrowings + df.ab_short_term_borrowing + eps)
    df['f_3'] = (inventory + df.ab_accounts_receivable + df.ab_other_current_assets + cas) / (
            total_liabilities + df.ab_accounts_payable)
    df['f_4'] = (df.ab_long_term_liabilities + df.ab_other_borrowings + df.ab_accounts_payable + eps) / inventory
    df['f_5'] = (inventory - df.ab_immobilized_assets) / (
            inventory + df.ab_accounts_receivable + df.ab_other_current_assets + cas)
    df['f_6'] = inventory / (df.ar_balance_of_rvns_and_expns + eps)
    df['f_7'] = (inventory + df.ab_long_term_liabilities) / (df.ar_balance_of_rvns_and_expns + eps)

    df['f_8'] = (df.ar_sale_profit + eps) / (df.ab_short_term_borrowing + eps)
    df['f_9'] = (df.ab_own_capital + eps) / inventory
    df['f_10'] = (df.ar_other_profit_and_losses + df.ar_sale_profit + eps) / inventory
    df['f_11'] = (df.ar_other_profit_and_losses + df.ar_sale_profit + eps) / total_capital
    df['f_12'] = total_liabilities / (df.ar_other_profit_and_losses + df.ar_sale_profit + eps)

    # кроме ручной генерации применим автоматическую и добавим квадраты всех фичей
    # как попытка поймать нелинейную зависимость
    for col in df.columns:
        if col not in [TARGET_NAME, 'ul_staff_range', 'ul_branch_cnt']:
            df[f'{col}^2'] = df[col] ** 2

    # разделяем трейн и тест обратно
    train = df.iloc[:-test_size, :]
    test = df.iloc[-test_size:, :]

    # будем отдельно работать с компаниями, которые имеют полные и неполные данные
    train_with_fr = train[~train_df['ar_revenue'].isna()]

    train_without_fr = train[train_df['ar_revenue'].isna()]
    train_without_fr.dropna(axis=1, inplace=True)
    LOGGER.info("Data successfully prepared.")

    prepared_data_dir.mkdir(exist_ok=True)
    train_with_fr.to_csv(path_train_data_with_financial_report, index=False)
    train_without_fr.to_csv(path_train_data_without_financial_report, index=False)
    test.to_csv(path_test_data, index=False)
    LOGGER.info("Prepared data successfully saved.")

    return train_with_fr, train_without_fr, test


def woe_transform(train, val):
    breaks_adj = {
        'ul_staff_range': [1, 2, 3],
        'ul_branch_cnt': [1]
    }
    bins = sc.woebin(train, y=TARGET_NAME, breaks_list=breaks_adj)
    train_woe = sc.woebin_ply(train, bins)
    val_woe = sc.woebin_ply(val, bins)
    return train_woe, val_woe, bins
