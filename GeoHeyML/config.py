import numpy as np

FIANL_PARAMS = {
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'objective': 'poisson'
}

FIT_PARAMS = {
    'num_boost_round': 1000,
    'early_stopping_rounds': 20,
    'verbose_eval': 1
}

CV_ESTI_GRID = {'n_estimators': np.linspace(20, 500, 10, dtype=int)}

CV_NUM_LEAVES_GRID = {'num_leaves': np.linspace(10, 50, 5, dtype=int)}

CV_LEARNING_RATE_GRID = {'learning_rate': [i / 100 for i in range(4,10)]}



