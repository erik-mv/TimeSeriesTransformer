from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


BASE_DIR = Path(__file__).parent

OUTPATH = BASE_DIR / 'out'
DATAPATH = BASE_DIR / 'data'
MODELS_PATH = BASE_DIR / 'models'

CAT_FEATURE_PATH = DATAPATH / 'cat_feature.parquet'
STATIC_FEATURE_PATH = DATAPATH / 'static_feature.parquet'
DYNAMIC_FEATURE_DAY_PATH = DATAPATH / 'dynamic_feature_day.parquet'
DYNAMIC_FEATURE_WEEK_PATH = DATAPATH / 'dynamic_feature_week.parquet'

DATASET_DAY_PATH = DATAPATH / 'dataset_day'
DATASET_WEEK_PATH = DATAPATH / 'dataset_week'

CALENDAR_PATH = DATAPATH / 'calendar_extended.csv'
TARGET_PATH = DATAPATH / 'target_series_extended.parquet'
TRANSACTIONS_PATH_LIST = [DATAPATH / f'transactions_{index}.parquet' for index in range(1,6)]
PROFILES_PATH = DATAPATH / 'profiles_extended.parquet'
SAMPLE_SUBMIT = DATAPATH / 'sample_submit_extended.csv'
TRANSACTIONS_EXTRA_PATH_LIST = [DATAPATH / f'transactions_extra_{index}.parquet' for index in range(1,6)]

DATELIST_HOLIDAYS = [
    # 2022
    '01-01-2022', '02-01-2022', '03-01-2022', '04-01-2022', 
    '05-01-2022', '06-01-2022', '07-01-2022', '08-01-2022', 
    '23-02-2022', '08-03-2022', '01-05-2022', '09-05-2022',  
    '12-06-2022', '04-11-2022', 
    # 2023
    '01-01-2023', '02-01-2023', '03-01-2023', '04-01-2023', 
    '05-01-2023', '06-01-2023', '07-01-2023', '08-01-2023', 
    '23-02-2023', '08-03-2023', '01-05-2023', '09-05-2023',  
    '12-06-2023', '04-11-2023', 
    # 2024
    '01-01-2024', '02-01-2024', '03-01-2024', '04-01-2024', 
    '05-01-2024', '06-01-2024', '07-01-2024', '08-01-2024', 
    '23-02-2024', '08-03-2024', '01-05-2024', '09-05-2024',  
    '12-06-2024', '04-11-2024', 
    # 2025
    '01-01-2025', '02-01-2025', '03-01-2025', '04-01-2025', 
    '05-01-2025', '06-01-2025', '07-01-2025', '08-01-2025', 
    '23-02-2025', '08-03-2025', '01-05-2025', '09-05-2025',  
    '12-06-2025', '04-11-2025', 
]

DATELIST_PRE_3_HOLIDAYS_DAYS = []
for date in [datetime.strptime(datastr, '%d-%m-%Y') for datastr in DATELIST_HOLIDAYS]:
    for _ in range(3):
        date -= timedelta(days=1)
        if date.strftime('%d-%m-%Y') not in DATELIST_PRE_3_HOLIDAYS_DAYS:
            DATELIST_PRE_3_HOLIDAYS_DAYS.append(date.strftime('%d-%m-%Y'))

DATELIST_PRE_7_HOLIDAYS_DAYS = []
for date in [datetime.strptime(datastr, '%d-%m-%Y') for datastr in DATELIST_HOLIDAYS]:
    for _ in range(7):
        date -= timedelta(days=1)
        if date.strftime('%d-%m-%Y') not in DATELIST_PRE_7_HOLIDAYS_DAYS:
            DATELIST_PRE_7_HOLIDAYS_DAYS.append(date.strftime('%d-%m-%Y'))

DATELIST_DAYSOFF = [
    # 2022
    '07-03-2022', '02-05-2022', '03-05-2022', '10-05-2022', 
    '13-06-2022',
    # 2023
    '24-02-2023', '08-05-2023', '06-11-2023',
    # 2024
    '29-04-2024', '30-04-2024', '10-05-2024', '30-12-2024',
    '31-12-2024',
    # 2025
    '02-05-2025', '09-05-2025', '13-06-2025', '03-11-2025', 
    '31-12-2025', 
]

DATELIST_WORKING_DAYSOFF = [
    # 2022
    '05-03-2022',
    # 2024
    '27-04-2024', '02-11-2024', '28-12-2024',
    # 2025
    '01-11-2025',
]

DYNAMIC_FEATURE_WEEK_FEATURE_LIST = [
 'Weekend', 'Month_0', 'Month_1', 'Month_2', 'Month_3',
 'April', 'August', 'December', 'February', 'January', 'July',
 'June', 'March', 'May', 'November', 'October', 'September',
 'Holiday', 'Pre_3_Holiday_days', 'Pre_7_Holiday_days',
]

DROP_FEATURE_WEEK_LIST = [
    'date', 'Friday', 'Monday', 'Saturday',
    'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
]
