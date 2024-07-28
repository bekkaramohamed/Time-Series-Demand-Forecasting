import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

#train test split 
def plot_train_test_split(df: pd.DataFrame, split_date: str) -> None:
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date]

    fig, ax = plt.subplots(figsize=(15, 5))
    train.plot(ax=ax, label='Training Set', title='Data Train/Test Split', linewidth=1, markersize=5)
    test.plot(ax=ax, label='Test Set', linewidth=1, markersize=5)
    ax.axvline(pd.Timestamp(split_date), color='black', ls='--')
    ax.legend(['Training Set', 'Test Set'])
    plt.show()


# see the evolution of our target value in a specific period of time
def plot_week_of_data(df: pd.DataFrame, start_date: str, end_date: str) -> None:
    df.loc[(df.index > start_date) & (df.index < end_date)] \
        .plot(figsize=(15, 5), title='Week Of Data', linewidth=1, markersize=5)
    plt.show()


def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From DataTime column of the original DF create features for TIme Series

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with additional time series features.

    Raises:
        ValueError: If input DataFrame does not have a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")

    df = df.copy()

    features = {
        'dayofweek': df.index.dayofweek,
        'quarter': df.index.quarter,
        'month': df.index.month,
        'year': df.index.year,
        'dayofyear': df.index.dayofyear,
        'dayofmonth': df.index.day,
        'weekofyear': df.index.isocalendar().week
    }

    for feature_name, feature_values in features.items():
        df[feature_name] = feature_values

    return df


def train_xgb_regressor(X_train, y_train, X_test, y_test, use_gpu=False):
    """
    Train an XGBoost Regressor model using the provided training and test data.

    Args:
        X_train: Training feature set.
        y_train: Training target set.
        X_test: Test feature set.
        y_test: Test target set.
        use_gpu: Whether to use GPU for training. Default is False.

    Returns:
        xgb.XGBRegressor: Trained XGBoost Regressor model.
    """
    additional_params = {}
    if use_gpu:
        additional_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}

    xgb_regressor = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=3000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=6,
                           learning_rate=0.01,
                           min_child_weight=1,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           gamma=0,
                           reg_alpha=0,
                           reg_lambda=1,
                           **additional_params)
    xgb_regressor.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)


    return xgb_regressor


def plot_feature_importance(model):
    """
    Plot the feature importance of a trained XGBoost model.

    Args:
        model (xgb.XGBRegressor): Trained XGBoost Regressor model.

    Returns:
        None
    """
    feat_importances = pd.DataFrame(data=model.feature_importances_,
                      index=model.feature_names_in_,
                      columns=['importance'])
    feat_importances.sort_values('importance').plot(kind='barh', title='Feature Importance')
    plt.show()
