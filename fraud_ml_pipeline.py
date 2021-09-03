import numpy as np
import pandas as pd
from helpers.eda import *
from helpers.data_prep import *
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
import gc
import time
from contextlib import contextmanager
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    print(" ")



def load_fraud_datasets():
    data = pd.read_pickle("fraud_df.pkl")
    return data


#########################
# Feature  Engineering
##########################

def feature_engineering(df):
    # TransactionDT DATE FEATURES
    # Transaction day of week
    df['Transaction_day_of_week'] = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 7).astype(int)

    # Transaction_month
    df['Transaction_month'] = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 12).astype(int) + 1

    # Transaction_day_of_month
    df['Transaction_day_of_month'] = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 30).astype(int) + 1

    # Transaction_day_of_year
    df['Transaction_day_of_year'] = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 365).astype(int) + 1

    # Transforming TransactionDT to a start date to get datetime features
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    df["Date"] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

    df['_Weekdays'] = df['Date'].dt.dayofweek
    df['_Hours'] = df['Date'].dt.hour
    df['_Days'] = df['Date'].dt.day
    # Sum of C values
    c_cols = [col for col in df if col.startswith('C')]
    df["c_sum"] = df[c_cols].sum(axis=1)

    # Time series lag, roll mean, ewm_features
    df = lag_features(df, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 21, 23, 24])
    df = roll_mean_features(df, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 21, 23, 24])
    alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 21, 23, 24]
    df = ewm_features(df, alphas, lags)

    # Email & Device Aggregation
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()

    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]

    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'
    df['P_emaildomain'] = df['P_emaildomain'].fillna('unknown_email').str.lower()
    df['pemail_name'] = df['P_emaildomain'].str.split('/', expand=True)[0]
    df.loc[df['pemail_name'].str.contains('gmail', na=False), 'pemail_name'] = 'Google'
    df.loc[df['pemail_name'].str.contains('hotmail', na=False), 'pemail_name'] = 'Microsoft'
    df.loc[df['pemail_name'].str.contains('outlook', na=False), 'pemail_name'] = 'Microsoft'
    df.loc[df['pemail_name'].str.contains('msn', na=False), 'pemail_name'] = 'Microsoft'
    df.loc[df['pemail_name'].str.contains('live', na=False), 'pemail_name'] = 'Microsoft'
    df.loc[df['pemail_name'].str.contains('yahoo', na=False), 'pemail_name'] = 'Yahoo'
    df.loc[df['pemail_name'].str.contains('ymail', na=False), 'pemail_name'] = 'Yahoo'
    df.loc[df['pemail_name'].str.contains('att', na=False), 'pemail_name'] = 'Yahoo'
    df.loc[df['pemail_name'].str.contains('bellsouth', na=False), 'pemail_name'] = 'Yahoo'
    df.loc[df['pemail_name'].str.contains('rocketmail', na=False), 'pemail_name'] = 'Yahoo'
    df.loc[df['pemail_name'].str.contains('bellsouth.net', na=False), 'pemail_name'] = 'Yahoo'
    df.loc[df['pemail_name'].str.contains('icloud', na=False), 'pemail_name'] = 'Apple'
    df.loc[df['pemail_name'].str.contains('me', na=False), 'pemail_name'] = 'Apple'
    df.loc[df['pemail_name'].str.contains('mac', na=False), 'pemail_name'] = 'Apple'
    df.loc[df['pemail_name'].str.contains('aol', na=False), 'pemail_name'] = 'AOL'
    df.loc[df['pemail_name'].str.contains('aim', na=False), 'pemail_name'] = 'AOL'
    df.loc[df['pemail_name'].str.contains('embarqmail', na=False), 'pemail_name'] = 'CenturyLink'
    df.loc[df['pemail_name'].str.contains('centurylink', na=False), 'pemail_name'] = 'CenturyLink'
    df.loc[df['pemail_name'].str.contains('frontier', na=False), 'pemail_name'] = 'Frontier'
    df.loc[df['pemail_name'].str.contains('netzero', na=False), 'pemail_name'] = 'Netzero'
    df.loc[df['pemail_name'].str.contains('twc', na=False), 'pemail_name'] = 'Spectrum'
    df.loc[df['pemail_name'].str.contains('cfl', na=False), 'pemail_name'] = 'Spectrum'
    df.loc[df['pemail_name'].str.contains('sc.rr', na=False), 'pemail_name'] = 'Spectrum'
    df.loc[df['pemail_name'].str.contains('protonmail.com', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('ptd.net', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('servicios-ta.com', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('gmx.de', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('scranton.edu', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('prodigy.net.mx', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('cableone.net', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('suddenlink.net', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('q.com', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('web.de', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('windstream.net', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('juno.com', na=False), 'pemail_name'] = 'Others'
    df.loc[df['pemail_name'].str.contains('roadrunner.com', na=False), 'pemail_name'] = 'Others'
    df['R_emaildomain'] = df['R_emaildomain'].fillna('unknown_email').str.lower()
    df['remail_name'] = df['R_emaildomain'].str.split('/', expand=True)[0]
    df.loc[df['remail_name'].str.contains('gmail', na=False), 'remail_name'] = 'Google'
    df.loc[df['remail_name'].str.contains('hotmail', na=False), 'remail_name'] = 'Microsoft'
    df.loc[df['remail_name'].str.contains('outlook', na=False), 'remail_name'] = 'Microsoft'
    df.loc[df['remail_name'].str.contains('msn', na=False), 'remail_name'] = 'Microsoft'
    df.loc[df['remail_name'].str.contains('live', na=False), 'remail_name'] = 'Microsoft'
    df.loc[df['remail_name'].str.contains('yahoo', na=False), 'remail_name'] = 'Yahoo'
    df.loc[df['remail_name'].str.contains('ymail', na=False), 'remail_name'] = 'Yahoo'
    df.loc[df['remail_name'].str.contains('att', na=False), 'remail_name'] = 'Yahoo'
    df.loc[df['remail_name'].str.contains('bellsouth', na=False), 'remail_name'] = 'Yahoo'
    df.loc[df['remail_name'].str.contains('rocketmail', na=False), 'remail_name'] = 'Yahoo'
    df.loc[df['remail_name'].str.contains('bellsouth.net', na=False), 'remail_name'] = 'Yahoo'
    df.loc[df['remail_name'].str.contains('icloud', na=False), 'remail_name'] = 'Apple'
    df.loc[df['remail_name'].str.contains('me', na=False), 'remail_name'] = 'Apple'
    df.loc[df['remail_name'].str.contains('mac', na=False), 'remail_name'] = 'Apple'
    df.loc[df['remail_name'].str.contains('aol', na=False), 'remail_name'] = 'AOL'
    df.loc[df['remail_name'].str.contains('aim', na=False), 'remail_name'] = 'AOL'
    df.loc[df['remail_name'].str.contains('embarqmail', na=False), 'remail_name'] = 'CenturyLink'
    df.loc[df['remail_name'].str.contains('centurylink', na=False), 'remail_name'] = 'CenturyLink'
    df.loc[df['remail_name'].str.contains('frontier', na=False), 'remail_name'] = 'Frontier'
    df.loc[df['remail_name'].str.contains('netzero', na=False), 'remail_name'] = 'Netzero'
    df.loc[df['remail_name'].str.contains('twc', na=False), 'remail_name'] = 'Spectrum'
    df.loc[df['remail_name'].str.contains('cfl', na=False), 'remail_name'] = 'Spectrum'
    df.loc[df['remail_name'].str.contains('sc.rr', na=False), 'remail_name'] = 'Spectrum'
    df.loc[df['remail_name'].str.contains('protonmail.com', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('ptd.net', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('servicios-ta.com', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('gmx.de', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('scranton.edu', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('prodigy.net.mx', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('cableone.net', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('suddenlink.net', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('q.com', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('web.de', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('windstream.net', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('juno.com', na=False), 'remail_name'] = 'Others'
    df.loc[df['remail_name'].str.contains('roadrunner.com', na=False), 'remail_name'] = 'Others'

    # Aggregating Devices name
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]
    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('sm', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('samsung', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('ios', na=False), 'device_name'] = 'Apple'
    df.loc[df['device_name'].str.contains('mac', na=False), 'device_name'] = 'Apple'
    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    df.loc[df['device_name'].str.contains('lg', na=False), 'device_name'] = 'LG'
    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('huawei', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ale', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('cam', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('rne', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ane', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('pre', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('pra', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    df.loc[df['device_name'].str.contains('linux', na=False), 'device_name'] = 'Linux'
    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'
    df.loc[df.device_name.isin(
        df.device_name.value_counts()[df.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    #  Aggregating Browsers
    df['id_31'] = df['id_31'].fillna('unknown_browser').str.lower()
    df['browser_name'] = df['id_31'].str.split('/', expand=True)[0]
    df.loc[df['browser_name'].str.contains('safari', na=False), 'browser_name'] = 'Safari'
    df.loc[df['browser_name'].str.contains('mobile safari', na=False), 'browser_name'] = 'Safari'
    df.loc[df['browser_name'].str.contains('firefox', na=False), 'browser_name'] = 'Firefox'
    df.loc[df['browser_name'].str.contains('edge', na=False), 'browser_name'] = 'Microsoft'
    df.loc[df['browser_name'].str.contains('ie', na=False), 'browser_name'] = 'Microsoft'
    df.loc[df['browser_name'].str.contains('chrome', na=False), 'browser_name'] = 'Chrome'
    df.loc[df['browser_name'].str.contains('google', na=False), 'browser_name'] = 'Chrome'
    df.loc[df['browser_name'].str.contains('samsung', na=False), 'browser_name'] = 'Samsung'
    df.loc[df.browser_name.isin(
        df.browser_name.value_counts()[df.browser_name.value_counts() < 341].index), 'browser_name'] = "Others"

    # Creating unique id with Cards
    df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
    df['uid1'] = df['uid'].astype(str) + '_' + df['card3'].astype(str)
    df['uid2'] = df['uid'].astype(str) + '_' + df['card3'].astype(str) + '_' + df['card5'].astype(str)
    df['uid3'] = df['uid2'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
    df['uid4'] = df['card4'].astype(str) + '_' + df['card6'].astype(str)

    # Aggregating os name
    df['id_30'] = df['id_30'].fillna('unknown_30')
    df['os_name'] = df['id_30'].str.split('/', expand=True)
    df.loc[df['os_name'].str.contains('Mac', na=False), 'os_name'] = 'Mac'
    df.loc[df['os_name'].str.contains('Android', na=False), 'os_name'] = 'Android'
    df.loc[df['os_name'].str.contains('iOS', na=False), 'os_name'] = 'iOS'
    df.loc[df['os_name'].str.contains('Windows', na=False), 'os_name'] = 'Windows'
    df.loc[df['os_name'].str.contains('func', na=False), 'os_name'] = 'Others'
    df.loc[df['os_name'].str.contains('other', na=False), 'os_name'] = 'Others'

    # Most correlated v cols sum
    df["44_45_total_score"] = df["V44"] + df["V45"]
    df["86_87_total_score"] = df["V86"] + df["V87"]
    df["188_189_total_score"] = df["V188"] + df["V189"]
    df["200_201_total_score"] = df["V200"] + df["V201"]
    df["244_246_total_score"] = df["V244"] + df["V246"]
    df["257_258_total_score"] = df["V257"] + df["V258"]
    df["257_258_246_244_total_score"] = df["V200"] + df["V201"] + df["V244"] + df["V246"]

    # Dropping most correlated v_cols
    v_cols = [col for col in df if col.startswith('V')]
    print(v_cols)
    corr_droper(df,v_cols)
    return df


########################
# Data Preprocessing
#########################


def preprocessing_fraud(df):
    # Rare encoding
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    cat_cols = cat_cols + cat_but_car
    rare_encoder(df, 0.02, cat_cols)

    # One-hot encoding
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    cat_cols = cat_cols + cat_but_car
    cat_cols = [col for col in cat_cols if col not in ["isFraud"]]

    df = one_hot_encoder(df, cat_cols, drop_first=True)


    # Dropping useless cols
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    cat_cols = cat_cols + cat_but_car
    cat_cols = [col for col in cat_cols if col not in ["isFraud"]]
    useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.02).any(axis=None)]

    for col in useless_cols_new:
        df.drop(col, axis=1, inplace=True)

    # Renaming columns for json error
    df_prep = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    return df_prep


###########################
# Modeling
###########################
def fraud_data_prep():
    dataframe = load_fraud_datasets()
    dataframe = feature_engineering(dataframe)
    dataframe = preprocessing_fraud(dataframe)
    return dataframe




def train_test_split():
    df = fraud_data_prep()
    # Train Test split
    train_df = df[df['isFraud'].notnull()].drop(["Date"], axis=1)
    test_df = df[df['isFraud'].isnull()].drop(["isFraud", "Date"], axis=1)
    return train_df, test_df


def kfold_lightgbm(num_folds=10, stratified=True, debug=False):
    train_df, test_df = train_test_split()
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['isFraud', 'Date']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['isFraud'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['isFraud'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['isFraud'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=-1,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='auc', verbose=200,
                early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['isFraud'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        sub = pd.read_csv("datasets/fraud/sample_submission.csv")
        submission_file_name = "submission_fraud_kfold2.csv"
        test_df['isFraud'] = sub_preds
        test_df['TransactionID'] = sub["TransactionID"]
        test_df[['TransactionID', 'isFraud']].to_csv(submission_file_name, index=False)
    display_importances(feature_importance_df)
    return feature_importance_df

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
    plt.show()


def main(debug=False):
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(num_folds=2, stratified=True, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()
