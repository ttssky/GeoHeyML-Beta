# -*-coding: utf-8-*-
# @Time: 2020-04-20
# @Author: lvyikai
# @Filename: model.py

import lightgbm as lgb
import numpy as np

from sklearn.model_selection import train_test_split

from .config import FIT_PARAMS


class GeoHeyMLRegressionModel(object):
    def __init__(self, model='lightgbm', **params):
        self.model = model

    
    def _inv_boxcox(self, y, fitted_lambda):
        if fitted_lambda == 0:
            return (np.exp(y) - 1)

        else:
            return (np.exp(np.log(fitted_lambda*y + 1)) / fitted_lambda - 1)

    def get_validation_set(self):
        
        return self.x_val, self.y_val

    def train(self, params, df_X, df_y, val_size=0.2):
        self.features = df_X.columns

        if self.model == 'lightgbm':
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(np.array(df_X), np.array(df_y), test_size=val_size)
            train_set = lgb.Dataset(np.array(self.x_train), label=np.array(self.y_train))
            val_set = lgb.Dataset(np.array(self.x_val), label=np.array(self.y_val))
            self.booster = lgb.train(
                params,
                train_set,
                valid_sets=[train_set,val_set],
                **FIT_PARAMS
            )

            return self.booster

        else:
            raise ValueError

    def predict(self, X):
        y_pred_box = self.booster.predict(X)
        return np.int16(10**y_pred_box - 1)

    def feature_importance(self, importance_type='gain'):
        return {'features': self.features, 'values': self.booster.feature_importance(importance_type=importance_type)}