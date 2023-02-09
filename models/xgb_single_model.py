"""_summary_."""
# Core
import sys

import numpy as np
import pandas as pd
# Machine Learning
import xgboost as xgb
# Error metrics
from sklearn.metrics import max_error as ME
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
# Optimization
# Time series split for cross validation
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     TimeSeriesSplit)
# Scalers
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import ML

sys.path.append('../')


target = 'DNI'
train_all = True

solar = pd.read_csv('../data/NASA_Dataset_Cleaned.csv',
                    index_col='DATETIME', parse_dates=['DATETIME', 'DATETIME'])

vars_to_drop = ['season', 'DNI', 'DHI', 'GHI']
vars_to_drop.remove(target)
solar.drop(labels=vars_to_drop, axis=1, inplace=True)


def add_lags(df):
    """_summary_.

    :param df: _description_
    :return: _description_
    """
    target_map = solar[target].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df


solar = add_lags(solar)

# not adding year variable here because the model will use the most recent lag energy consumption values
cols_to_transform = ['T2MDEW', 'DIFFUSE_ILLUMINANCE', 'DIRECT_ILLUMINANCE',
                     'GLOBAL_ILLUMINANCE', 'RH2M', 'QV2M', 'PS', 'T2M', 'SZA', 'WS2M']  # other columns are binary values
X_trainF, X_testF, y_trainF, y_testF = ML.train_test(
    data=solar,
    target=target,
    test_size=0.15,
    scale=True,
    cols_to_transform=cols_to_transform,
    plot=False)

if train_all:
    X_testF = pd.concat([X_trainF, X_testF])
    y_testF = pd.concat([y_trainF, y_testF])


# # Tuning the XGBoost model
# xgbtuned = xgb.XGBRegressor()

# param_grid = {
#         'objective': ['reg:squarederror'],
#         'max_depth': [8],
#         'learning_rate': [0.05],
#         'subsample': [0.3],
#         'colsample_bytree': [0.7],
#         'colsample_bylevel': [0.7],
#         'min_child_weight': [1.5],
#         'gamma': [0.2],
#         'n_estimators': [400]
#         }

# # Best score: -10.616212801508329

# tscv = TimeSeriesSplit(n_splits=5)

# xgbtunedreg = RandomizedSearchCV(xgbtuned, param_distributions=param_grid ,
#                                    scoring='neg_root_mean_squared_error', n_iter=20, n_jobs=-1,
#                                    cv=tscv, verbose=2, random_state=1)

# xgbtunedreg.fit(X_trainF, y_trainF)
# best_score = xgbtunedreg.best_score_
# best_params = xgbtunedreg.best_params_
# print("Best score: {}".format(best_score))
# print("Best params: ")
# for param_name in sorted(best_params.keys()):
#     print('%s: %r' % (param_name, best_params[param_name]))


# Prepare data as DMatrix objects
train = xgb.DMatrix(X_trainF, y_trainF)
test = xgb.DMatrix(X_testF, y_testF)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.3,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'min_child_weight': 1.5,
    'gamma': 0.2,
    # 'n_estimators': 400,
    'n_jobs': -1,
    'random_state': 1
}

xgbtunedreg = xgb.train(
    params,
    train, evals=[(train, 'train'), (test, 'validation')],
    num_boost_round=1000, early_stopping_rounds=50
)

xgbtunedreg.save_model(f'xgb_model_{target}.json')

preds_xgb_train = xgbtunedreg.predict(
    train, ntree_limit=xgbtunedreg.best_ntree_limit)
preds_xgb_test = xgbtunedreg.predict(
    test, ntree_limit=xgbtunedreg.best_ntree_limit)

dict_error = {}

# Test Set
dict_error = ML.error_metrics(
    y_pred=preds_xgb_train,
    y_truth=y_trainF,
    model_name='Tuned XGBoost on Train Set',
    test=False,
    dict_error=dict_error)
# RMSE: 5.99
# R2: 1.00
# MAE: 2.96

# Training Set
dict_error = ML.error_metrics(
    y_pred=preds_xgb_test,
    y_truth=y_testF,
    model_name='Tuned XGBoost on Test Set',
    test=True,
    dict_error=dict_error)
# RMSE: 15.56
# R2: 1.00
# MAE: 5.95

ML.plot_ts_pred(
    og_ts=y_testF,
    pred_ts=preds_xgb_test,
    model_name=f'Tuned XGBoost on Test Set')


ML.plot_predvstrue_reg(
    pred=preds_xgb_test,
    truth=y_testF,
    model_name='Tuned XGBoost on Test set')


# pd.plotting.register_matplotlib_converters()
xgb.plot_importance(xgbtunedreg)
plt.rcParams['figure.figsize'] = [15, 15]
