# -*-coding: utf-8-*-
# @Time: 2020-04-20
# @Author: lvyikai
# @Filename: preprocess.py

import numpy as np
import pandas as pd

import lightgbm as lgb

# from scipy.stats import boxcox
# from scipy.special import inv_boxcox

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from .config import *



class GeoHeyMLPreProcess(object):

# 1. 自动计算目标变量的峰度和偏度，自适应的进行box-cox变换
# Example y, fitted_lambda = boxcox(y, lmbda=None)
# Reverse inv_y = inversebox(y, fitted_lambda)

        def _box_cox(self, df_y):
                # y, fitted_lambda = boxcox(df_y + 1, lmbda = None)
                y = np.log10(df_y+1)
                return y

# 2. 特征工程: 

     # 处理非数值型特征，对这些特征进行编码

        def _encode_categorical(self, df_x):
                for col in df_x.columns:
                        dtype = df_x[col].dtypes
                        if dtype == 'object':
                                le = LabelEncoder()
                                not_null = df_x[col][df_x[col].notnull()]
                                df_x[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
                return df_x

        def autoFE(self, df_x, df_y=None):
                if df_y is not None:
                        self.df_y = self._box_cox(df_y)
                        self.df_x = self._encode_categorical(df_x)
                        return self.df_x, self.df_y
                else:
                        return self._encode_categorical(df_x)


# 3. 设定参数搜索空间,形成超参调节的流水线
        def _rmse(self, y_true, y_pred):
                return np.sqrt(np.mean((y_true-y_pred)**2))

        def _tuning_params(self, cv_grid, metric, df_x, df_y):

                lgb_model = lgb.LGBMRegressor(**FIANL_PARAMS)
                gs = GridSearchCV(lgb_model, cv_grid, verbose=50,
                                refit=True, cv=5, scoring=make_scorer(metric, greater_is_better=False))
                gs.fit(np.array(df_x), np.array(df_y))
                FIANL_PARAMS.update(gs.best_params_)

        def autoTP(self):
                #step1. 对n_esimator调参
                self._tuning_params(CV_ESTI_GRID, self._rmse, 
                                self.df_x, self.df_y)
                #step2. 对num_leaves调参
                self._tuning_params(CV_NUM_LEAVES_GRID, self._rmse, 
                                self.df_x, self.df_y)
                #step3. 对num_leaves调参
                self._tuning_params(CV_LEARNING_RATE_GRID, 
                                self._rmse, self.df_x, self.df_y)
                
                return FIANL_PARAMS
