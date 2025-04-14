import pandas as pd
import logging
from pathlib import Path
import argparse
from datasets import Dataset, DatasetDict
import numpy as np
from scipy import stats

from const import *

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
    level=logging.INFO,
    )
logger = logging.getLogger(__name__)


def get_datetime_feature(
    df, name_column_datetime, check_add_weekend=1, check_add_day_name=1, check_add_period_month=1,
    check_add_month_name=1, check_add_holiday=1, check_add_pre_3_holiday_days=1, check_add_pre_7_holiday_days=1, 
):
    # Flag выходного
    if check_add_weekend:
        logger.info('Add weekend')
        df['Weekend'] = (df[name_column_datetime].dt.day_of_week // 5).astype(float)
        df.loc[df[name_column_datetime].isin(pd.to_datetime(DATELIST_DAYSOFF, format='%d-%m-%Y', utc=True)), 'Weekend'] = 1.0
        df.loc[df[name_column_datetime].isin(pd.to_datetime(DATELIST_WORKING_DAYSOFF, format='%d-%m-%Y', utc=True)), 'Weekend'] = 0.0
    # Flag дня недели
    if check_add_day_name:
        logger.info('Add day_name')
        df = pd.concat([df, pd.get_dummies(df[name_column_datetime].dt.day_name(), dtype=float)], axis=1)
    # Flag 4-х периодов месяца
    if check_add_period_month:
        logger.info('Add period month')
        df['Month'] = (df[name_column_datetime].dt.day // (df[name_column_datetime].dt.days_in_month / 4)).astype(int)
        df.loc[df['Month']==4, 'Month'] = 3
        df = pd.concat([df, pd.get_dummies(df['Month'], dtype=float, prefix='Month')], axis=1)
        df.drop(['Month'], axis=1, inplace=True)
    # Flag 4 месяца
    if check_add_month_name:
        logger.info('Add month name')
        df = pd.concat([df, pd.get_dummies(df[name_column_datetime].dt.month_name(), dtype=float)], axis=1)
    # Flag прадников
    if check_add_holiday:
        logger.info('Add holiday')
        df['Holiday'] = df[name_column_datetime].apply(
            lambda x: 1.0 if x in pd.to_datetime(DATELIST_HOLIDAYS,format='%d-%m-%Y', utc=True) else 0.0
        )
        df.loc[df['Holiday']==1.0, 'Weekend'] = 1.0
    # Flag 3 дня до праздника
    if check_add_pre_3_holiday_days:
        logger.info('Add pre_3_holiday_days')
        df['Pre_3_Holiday_days'] = df[name_column_datetime].apply(
            lambda x: 1.0 if x in pd.to_datetime(DATELIST_PRE_3_HOLIDAYS_DAYS, format='%d-%m-%Y', utc=True) else 0.0
        )
    # Flag 7 дней до праздника
    if check_add_pre_7_holiday_days:
        logger.info('Add pre_7_holiday_days')
        df['Pre_7_Holiday_days'] = df[name_column_datetime].apply(
            lambda x: 1.0 if x in pd.to_datetime(DATELIST_PRE_7_HOLIDAYS_DAYS, format='%d-%m-%Y', utc=True) else 0.0
        )
    return df

def merge_static_feature(inn_df:pd.DataFrame, merge_data:pd.DataFrame, name_pay_type, name_type_date, name_column_groupby='trns_amount', name_column_inn='inn_id'):
    merge_data = merge_data.groupby([name_column_inn], as_index=False).agg( ['min', 'max', 'mean', 'median'])
    merge_data.columns = merge_data.columns.droplevel(0)
    merge_data.rename(columns={
        '': name_column_inn,
        'min': f'{name_pay_type}_{name_column_groupby}_{name_type_date}_min',
        'max': f'{name_pay_type}_{name_column_groupby}_{name_type_date}_max',
        'mean': f'{name_pay_type}_{name_column_groupby}_{name_type_date}_mean',
        'median': f'{name_pay_type}_{name_column_groupby}_{name_type_date}_median',
    }, inplace=True)
    inn_df = inn_df.merge(merge_data, how='left', on='inn_id')
    return inn_df

def get_groupby_static_feature(
    result:pd.DataFrame, df:pd.DataFrame, name_pay_type, list_columns_date_type, 
    name_column_groupby='trns_amount', name_column_inn='inn_id',
):
    logger.info(42 * '#')
    logger.info('Get consts data for inn with %s', name_pay_type)
    result = merge_static_feature(
        result, df[[name_column_inn, name_column_groupby]], 
        name_pay_type, f'All_{name_column_groupby}', name_column_inn='inn_id',
    )
    for name_type_date in list_columns_date_type:
        logger.info('Add %s consts data', name_type_date)
        result = merge_static_feature(
            result, df[df[name_type_date]==1][[name_column_inn, name_column_groupby]],
            name_pay_type, name_type_date, name_column_groupby, name_column_inn='inn_id',
        )
    return result

def get_groupby_dynamic_feature(result:pd.DataFrame, df:pd.DataFrame, name_feature, column_groupby_list=['date', 'inn_id']):
    logger.info(42 * '#')
    logger.info('Get dynamic data %s', name_feature)
    result = result.merge(df.groupby(column_groupby_list, as_index=False).sum(), how='left', on=column_groupby_list)
    result.rename(columns={'trns_amount': name_feature,}, inplace=True)
    return result

def load_calendar(calendar_path):
    logger.info(42 * '#')
    logger.info('Load calendar %s', calendar_path)
    calendar = pd.read_csv(calendar_path)
    calendar['date'] = pd.to_datetime(calendar['date'], utc=True)
    return calendar

def load_transactions(transactions_path_list):
    logger.info(42 * '#')
    logger.info('Load transactions')
    logger.info('Len transactions path list: %d', len(transactions_path_list))
    transactions = pd.concat((pd.read_parquet(filepath) for filepath in transactions_path_list), ignore_index=True)
    return transactions

def load_target_inn(target_path):
    logger.info(42 * '#')
    logger.info('Load target %s', target_path)
    target_inn = pd.read_parquet(target_path)[['inn_id']].drop_duplicates()
    target_inn.reset_index(drop=True, inplace=True)
    target_inn['target_inn'] = 1
    return target_inn

def get_payer_or_payee_data(
        data:pd.DataFrame, name_rename_column:str, target_inn:pd.DataFrame, 
        calendar:pd.DataFrame, flag_merge_calendar=True,
    ):
    data.reset_index(drop=True, inplace=True)
    data.rename(columns={name_rename_column:'inn_id'}, inplace=True)
    data = data.merge(target_inn, how='left', on='inn_id')
    data = data[data['target_inn']==1].drop(['target_inn'], axis=1)
    if flag_merge_calendar:
        data = data.merge(calendar.drop(['week', 'part'], axis=1), how='left', on='date').drop(['date'], axis=1)
    return data

def get_static_feature(static_data_path:Path, calendar_path:Path, transactions_path_list, target_path:Path):
    calendar = load_calendar(calendar_path)
    logger.info('Add type feature in calendar')
    calendar = get_datetime_feature(calendar, 'date')
    
    transactions = load_transactions(transactions_path_list)

    target_inn = load_target_inn(target_path)
    result = target_inn.copy()
    result = result.drop(['target_inn'], axis=1)

    logger.info(42 * '#')
    logger.info('Get payer data')
    payer_data = transactions[transactions['doc_payer_bank_name_flag']==1][['date', 'doc_payer_inn', 'trns_count', 'trns_amount']].copy()
    payer_data = get_payer_or_payee_data(payer_data, 'doc_payer_inn', target_inn, calendar)
    result = get_groupby_static_feature(result, payer_data, 'Payer', list(payer_data.columns)[3:], name_column_groupby='trns_amount')
    result = get_groupby_static_feature(result, payer_data, 'Payer', list(payer_data.columns)[3:], name_column_groupby='trns_count')
    temp_data = payer_data[['inn_id', 'trns_amount', 'trns_count']].groupby(['inn_id'], as_index=False).sum()
    temp_data['Payer_trns_amount_on_count'] = temp_data['trns_amount'] / temp_data['trns_count']
    result = result.merge(temp_data[['inn_id', 'Payer_trns_amount_on_count']], how='left', on='inn_id')

    logger.info(42 * '#')
    logger.info('Get payee data')
    payee_data = transactions[transactions['doc_payee_bank_name_flag']==1][['date', 'doc_payee_inn', 'trns_count', 'trns_amount']].copy()
    payee_data = get_payer_or_payee_data(payee_data, 'doc_payee_inn', target_inn, calendar)
    result = get_groupby_static_feature(result, payee_data, 'Payee', list(payee_data.columns)[3:], name_column_groupby='trns_amount')
    result = get_groupby_static_feature(result, payee_data, 'Payee', list(payee_data.columns)[3:], name_column_groupby='trns_count')
    temp_data = payee_data[['inn_id', 'trns_amount', 'trns_count']].groupby(['inn_id'], as_index=False).sum()
    temp_data['Payee_trns_amount_on_count'] = temp_data['trns_amount'] / temp_data['trns_count']
    result = result.merge(temp_data[['inn_id', 'Payee_trns_amount_on_count']], how='left', on='inn_id')

    result.fillna(value=0.0, inplace=True)
    result.to_parquet(static_data_path)
    logger.info('Complete')

def get_trns_class_encoded_feature(result:pd.DataFrame, df:pd.DataFrame, feat_name:str):
    df.rename(columns={f'doc_{feat_name}_inn':'inn_id', 'trns_class_encoded':f'trns_class_encoded_{feat_name}'}, inplace=True)
    feat_name = f'trns_class_encoded_{feat_name}'
    result = result.merge(df, how='left', on='inn_id')
    result.loc[result[feat_name].isna(), feat_name] = 0
    result = result.groupby(['inn_id'], as_index=False).agg({feat_name: lambda x: stats.mode(x)[0]})
    result[feat_name] = result[feat_name].astype(int)
    return result

def get_cat_feature(target_path:Path, profiles_path:Path, cat_feature_path:Path, transactions_path_list:Path):
    logger.info('Load target %s', target_path)
    result = load_target_inn(target_path)
    result = result.drop(['target_inn'], axis=1)
    logger.info('Load profiles %s', profiles_path)
    profiles = pd.read_parquet(profiles_path)
    result = result.merge(
        profiles[['inn_id', 'ipul', 'id_region', 'main_okved_group']].dropna().groupby(['inn_id'], as_index=False).first(),
        how='left',
        on='inn_id',
    )
    del profiles
    logger.info('Get cat feature')
    result.loc[result['ipul']=='ip', 'ipul'] = 1
    result.loc[result['ipul']=='ul', 'ipul'] = 2
    result.loc[result['ipul'].isna(), 'ipul'] = 0
    result.loc[result['id_region'].isna(), 'id_region'] = 0
    result.loc[result['main_okved_group'].isna(), 'main_okved_group'] = '00'
    result['id_region'] = result['id_region'].astype(int)
    result['main_okved_group'] = result['main_okved_group'].astype(int)

    transactions = load_transactions(transactions_path_list)
    payer_data = transactions[transactions['doc_payer_bank_name_flag']==1][['doc_payer_inn', 'trns_class_encoded']].copy()
    payer_data = get_trns_class_encoded_feature(result[['inn_id']].copy(), payer_data, 'payer')
    result['trns_class_encoded_payer'] = payer_data['trns_class_encoded_payer'] 
    payee_data = transactions[transactions['doc_payee_bank_name_flag']==1][['doc_payee_inn', 'trns_class_encoded']].copy()
    payee_data = get_trns_class_encoded_feature(result[['inn_id']].copy(), payee_data, 'payee')
    result['trns_class_encoded_payee'] = payee_data['trns_class_encoded_payee'] 

    logger.info('Save cat feature %s', cat_feature_path)
    result.to_parquet(cat_feature_path)
    logger.info('Complete')


def get_dynamic_feature_day(dynamic_data_day_path:Path, calendar_path:Path, transactions_path_list, target_path:Path):
    calendar = load_calendar(calendar_path)
    logger.info('Add type feature in calendar')
    calendar = get_datetime_feature(calendar, 'date')
    transactions = load_transactions(transactions_path_list)
    target_inn = load_target_inn(target_path)
    result = target_inn.copy()
    result = result.drop(['target_inn'], axis=1)
    result = calendar.loc[calendar.index.repeat(len(result))]
    result['inn_id'] = list(target_inn['inn_id'].values) * (calendar['week'].max()+1) * 7
    result.reset_index(drop=True, inplace=True)
    logger.info(42 * '#')
    logger.info('Get payer data')
    payer_data = transactions[transactions['doc_payer_bank_name_flag']==1][['date', 'doc_payer_inn', 'trns_amount']].copy()
    payer_data = get_payer_or_payee_data(payer_data, 'doc_payer_inn', target_inn, calendar, flag_merge_calendar=False)
    result = get_groupby_dynamic_feature(result, payer_data, 'target')
    result.fillna(value=0.0, inplace=True)
    result.to_parquet(dynamic_data_day_path)
    logger.info('Complete')

def get_dynamic_feature_week(dynamic_data_week_path:Path, dynamic_data_day_path:Path):
    logger.info(42 * '#')
    logger.info('Load dynamic data day %s', dynamic_data_day_path)
    result = pd.read_parquet(dynamic_data_day_path)
    logger.info('Groupby data')
    result = result.drop(DROP_FEATURE_WEEK_LIST, axis=1).groupby(['inn_id', 'week'], as_index=False).agg(
        Weekend=('Weekend', 'mean'), Month_0=('Month_0', 'max'), Month_1=('Month_1', 'max'),
        Month_2=('Month_2', 'max'), Month_3=('Month_3', 'max'), January=('January', 'max'),
        February=('February', 'max'), March=('March', 'max'), April=('April', 'max'),
        May=('May', 'max'), June=('June', 'max'), July=('July', 'max'),
        August=('August', 'max'), September=('September', 'max'), October=('October', 'max'),
        November=('November', 'max'), December=('December', 'max'), Holiday=('Holiday', 'mean'),
        Pre_3_Holiday_days=('Pre_3_Holiday_days', 'mean'), Pre_7_Holiday_days=('Pre_7_Holiday_days', 'mean'),
        target=('target', 'sum'),
    )
    logger.info('Save dynamic data week %s', dynamic_data_week_path)
    result.reset_index(drop=True, inplace=True)
    result.to_parquet(dynamic_data_week_path)
    logger.info('Complete')

def plot_sum_payer_and_target(dynamic_data_day_path, target_path, inn_id):
    data = pd.read_parquet(dynamic_data_day_path)
    data[['week', 'inn_id', 'target']][(data.week<118) & (data.inn_id==inn_id)][['week', 'target']].groupby(['week']).sum().plot()
    target = pd.read_parquet(target_path)
    target = target[target.inn_id==inn_id]
    target.reset_index(drop=True, inplace=True)
    target['target'].plot(linestyle='dashed')

def plot_dynamic_week_and_target(dynamic_data_week_path, target_path, inn_id):
    data = pd.read_parquet(dynamic_data_week_path)
    data = data[(data.week<118) & (data.inn_id==inn_id)]
    data.reset_index(drop=True, inplace=True)
    data['target'].plot()
    target = pd.read_parquet(target_path)
    target = target[target.inn_id==inn_id]
    target.reset_index(drop=True, inplace=True)
    target['target'].plot(linestyle='dashed')

def dataframe2dataset_day(df:pd.DataFrame, type_dataset, feat_static_real, feat_static_cat):
    logger.info('Get %s dataset', type_dataset)
    df.reset_index(drop=True, inplace=True)
    feat_dynamic_real = list(df[['inn_id', 'temp']].groupby(['inn_id'])['temp'].apply(list))
    feat_dynamic_real = np.array(feat_dynamic_real)
    feat_dynamic_real = [data.T.astype(np.float16) for data in feat_dynamic_real]
    return Dataset.from_dict({
        'start': ['2022-07-25'] * 51963,
        'target': list(df[['inn_id', 'target']].groupby(['inn_id'])['target'].apply(list)),
        'feat_dynamic_real': feat_dynamic_real,
        'feat_static_real': feat_static_real,
        'feat_static_cat': feat_static_cat,
        'inn_id':list(df['inn_id'].drop_duplicates()),
    })

def save_dataset_day(dataset_day_path, dynamic_data_day_path, static_data_path, cat_feature_path):
    logger.info('Load dynamic feature %s', dynamic_data_day_path)
    data_dynamic = pd.read_parquet(dynamic_data_day_path)
    for column in data_dynamic.columns[3:-2]:
        data_dynamic[column] = data_dynamic[column].round(4)
    data_dynamic['temp'] = data_dynamic[data_dynamic.columns[3:-2]].values.tolist()
    data_dynamic['target'] = np.log1p(data_dynamic['target'])
    logger.info('Load static data %s', static_data_path)
    data_static = pd.read_parquet(static_data_path)
    max_list = []
    for column in data_static.columns[1:]:
        data_static[column] = np.log1p(data_static[column])
        max_list.append(data_static[column].max())
    for column in data_static.columns[1:]:
        data_static[column] /= max(max_list)
    feat_static_real = data_static[data_static.columns[1:]].values.tolist()
    cat_feature = pd.read_parquet(cat_feature_path)
    cat_feature['id'] = list(range(len(cat_feature)))
    cat_feature['id_region'] = get_collection_feat(cat_feature, 'id_region')
    cat_feature['main_okved_group'] = get_collection_feat(cat_feature, 'main_okved_group')
    cat_feature['trns_class_encoded_payer'] = get_collection_feat(cat_feature, 'trns_class_encoded_payer')
    cat_feature['trns_class_encoded_payee'] = get_collection_feat(cat_feature, 'trns_class_encoded_payee')
    #feat_static_cat = cat_feature[['id', 'ipul', 'id_region', 'main_okved_group', 'trns_class_encoded_payer', 'trns_class_encoded_payee']].values.tolist()
    feat_static_cat = cat_feature[['ipul', 'id_region', 'main_okved_group', 'trns_class_encoded_payer', 'trns_class_encoded_payee']].values.tolist()
    dataset = DatasetDict({
        'train' : dataframe2dataset_day(data_dynamic[data_dynamic['week']<106], 'train', feat_static_real, feat_static_cat),
        'val' : dataframe2dataset_day(data_dynamic[data_dynamic['week']<118], 'val', feat_static_real, feat_static_cat),
        'predict' : dataframe2dataset_day(data_dynamic, 'predict', feat_static_real, feat_static_cat),
    })
    logger.info('Save dataset %s', dataset_day_path)
    dataset.save_to_disk(str(dataset_day_path))

def dataframe2dataset_week(df:pd.DataFrame, type_dataset, feat_static_real, feat_static_cat):
    logger.info('Get %s dataset', type_dataset)
    df.reset_index(drop=True, inplace=True)
    feat_dynamic_real = list(df[['inn_id', 'temp']].groupby(['inn_id'])['temp'].apply(list))
    feat_dynamic_real = np.array(feat_dynamic_real)
    feat_dynamic_real = [data.T.astype(np.float16) for data in feat_dynamic_real]
    return Dataset.from_dict({
        'start': ['2022-07-25'] * 51963,
        'target': list(df[['inn_id', 'target']].groupby(['inn_id'])['target'].apply(list)),
        'feat_dynamic_real': feat_dynamic_real,
        'feat_static_real': feat_static_real,
        'feat_static_cat': feat_static_cat,
        'inn_id':list(df['inn_id'].drop_duplicates()),
    })

def get_collection_feat(df, column):
    collection_class = {}
    index = 0
    for row in df[column].values:
        if row not in collection_class:
            collection_class[row] = index
            index += 1
    return df[column].apply(lambda x: collection_class[x])


def save_dataset_week(dataset_week_path, dynamic_data_week_path, static_data_path, cat_feature_path):
    logger.info('Load dynamic data %s', dynamic_data_week_path)
    data_dynamic = pd.read_parquet(dynamic_data_week_path)
    for column in DYNAMIC_FEATURE_WEEK_FEATURE_LIST:
        data_dynamic[column] = data_dynamic[column].round(4)
    data_dynamic['temp'] = data_dynamic[DYNAMIC_FEATURE_WEEK_FEATURE_LIST].values.tolist()
    data_dynamic['target'] = np.log1p(data_dynamic['target'])
    logger.info('Load static data %s', static_data_path)
    data_static = pd.read_parquet(static_data_path)
    for column in data_static.columns[1:]:
        if 'trns_amount' in column:
            data_static[column] = np.log1p(data_static[column])
    feat_static_real = data_static[data_static.columns[1:]].values.tolist()
    cat_feature = pd.read_parquet(cat_feature_path)
    cat_feature['id'] = list(range(len(cat_feature)))
    cat_feature['id_region'] = get_collection_feat(cat_feature, 'id_region')
    cat_feature['main_okved_group'] = get_collection_feat(cat_feature, 'main_okved_group')
    cat_feature['trns_class_encoded_payer'] = get_collection_feat(cat_feature, 'trns_class_encoded_payer')
    cat_feature['trns_class_encoded_payee'] = get_collection_feat(cat_feature, 'trns_class_encoded_payee')
    #feat_static_cat = cat_feature[['id', 'ipul', 'id_region', 'main_okved_group']].values.tolist()
    feat_static_cat = cat_feature[['ipul', 'id_region', 'main_okved_group', 'trns_class_encoded_payer', 'trns_class_encoded_payee']].values.tolist()

    dataset = DatasetDict({
        'train' : dataframe2dataset_week(data_dynamic[data_dynamic['week']<106], 'train', feat_static_real, feat_static_cat),
        'val' : dataframe2dataset_week(data_dynamic[data_dynamic['week']<118], 'val', feat_static_real, feat_static_cat),
        'predict' : dataframe2dataset_week(data_dynamic, 'predict', feat_static_real, feat_static_cat),
    })
    dataset.save_to_disk(str(dataset_week_path))

def run_pipeline(
    static_data_path:str, 
    dynamic_data_day_path:str, dynamic_data_week_path:str, 
    dataset_day_path:str, dataset_week_path:str, 
    calendar_path:str, transactions_path_list:list, target_path:str
):
    """Run pipeline from arg"""
    if static_data_path:
        get_static_feature(Path(static_data_path), Path(calendar_path), transactions_path_list, Path(target_path))
    if dynamic_data_day_path:
        get_dynamic_feature_day(Path(dynamic_data_day_path), Path(calendar_path), transactions_path_list, Path(target_path))
    if dynamic_data_week_path:
        get_dynamic_feature_week(Path(dynamic_data_week_path), Path(dynamic_data_day_path))
    if dataset_day_path and dynamic_data_day_path:
        save_dataset_day(Path(dataset_day_path), Path(dynamic_data_day_path))
    if dataset_week_path and dynamic_data_week_path:
        save_dataset_week(Path(dataset_week_path), Path(dynamic_data_week_path), Path(static_data_path))


def handle_args(args):
    """
    Extract command line args and call delegate function.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    return run_pipeline(
        args.static_data,
        args.dynamic_data_day,
        args.dynamic_data_week,
        args.save_dataset_day,
        args.save_dataset_week,
        args.calendar,
        args.transaction,
        args.target,
    )

def main(args=None):
    """
    Parse command line args and call handler when run as a script.
    Parameters
    ----------
    args : list
        Command line arguments as a list of strings [optional]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sd", "--static_data", type=str,
        help="Path .parquet file for save static feature",
    )
    parser.add_argument(
        "-ddp", "--dynamic_data_day", type=str,
        help="Path .parquet file for save dynamic feature day",
    )
    parser.add_argument(
        "-dwp", "--dynamic_data_week", type=str,
        help="Path .parquet file for save dynamic feature week",
    )
    parser.add_argument(
        "-sdd", "--save_dataset_day", type=str,
        help="Path file for save dataset day",
    )
    parser.add_argument(
        "-sdw", "--save_dataset_week", type=str,
        help="Path file for save dataset week",
    )
    parser.add_argument(
        "-cl", "--calendar", type=str, default=str(CALENDAR_PATH),
        help="Path .csv file calendar",
    )
    parser.add_argument(
        "-trns", "--transaction", type=list, default=TRANSACTIONS_PATH_LIST,
        help="List path .parquet file transaction",
    )
    parser.add_argument(
        "-tgt", "--target", type=str, default=str(TARGET_PATH),
        help="Path .parquet file target",
    )
    parser.set_defaults(func=handle_args)
    args = parser.parse_args()
    # Call args default handler
    args.func(args)

if __name__ == '__main__':
    main()

