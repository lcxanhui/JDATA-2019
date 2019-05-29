# 感谢大佬分享的参数
ctb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.02,
    'random_seed': 4590,
    'reg_lambda': 0.08,
    'subsample': 0.7,
    'bootstrap_type': 'Bernoulli',
    'boosting_type': 'Plain',
    'one_hot_max_size': 10,
    'rsm': 0.5,
    'leaf_estimation_iterations': 5,
    'use_best_model': True,
    'max_depth': 6,
    'verbose': -1,
    'thread_count': 4
}
ctb_model = ctb.CatBoostRegressor(**ctb_params)
