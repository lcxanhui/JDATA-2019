import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
pd.set_option('display.max_columns', None)

df_train=pd.read_csv('../output/df_train.csv')
df_test=pd.read_csv('../output/df_test.csv')
df_user=pd.read_csv('../data/jdata_user.csv')
df_comment=pd.read_csv('../data/jdata_comment.csv')
df_shop=pd.read_csv('../data/jdata_shop.csv')

# 1）行为数据（jdata_action）
jdata_action = pd.read_csv('../data/jdata_action.csv')
# 3）商品数据（jdata_product）
jdata_product =  pd.read_csv('../data/jdata_product.csv')
jdata_data = jdata_action.merge(jdata_product,on=['sku_id'])

train_buy = jdata_data[(jdata_data['action_time']>='2018-04-09') \
                           & (jdata_data['action_time']<='2018-04-15') \
                           & (jdata_data['type']==2)][['user_id','cate','shop_id']].drop_duplicates()
train_buy['label'] = 1
# 候选集 时间 ： '2018-03-26'-'2018-04-08' 最近两周有行为的（用户，类目，店铺）
train_set = jdata_data[(jdata_data['action_time']>='2018-03-26') \
                           & (jdata_data['action_time']<='2018-04-08')][['user_id','cate','shop_id']].drop_duplicates()
train_set = train_set.merge(train_buy,on=['user_id','cate','shop_id'],how='left').fillna(0)
train_set = train_set.merge(df_train,on=['user_id','cate','shop_id'],how='left')
def mapper(x):
    if x is not np.nan:
        year=int(x[:4])
        return 2018-year

df_user['user_reg_tm']=df_user['user_reg_tm'].apply(lambda x:mapper(x))
df_shop['shop_reg_tm']=df_shop['shop_reg_tm'].apply(lambda x:mapper(x))
df_shop['shop_reg_tm']=df_shop['shop_reg_tm'].fillna(df_shop['shop_reg_tm'].mean())
df_user['age']=df_user['age'].fillna(df_user['age'].mean())
df_comment=pd.read_csv('../data/jdata_comment.csv')
df_comment=df_comment.groupby(['sku_id'],as_index=False).sum()
df_product=pd.read_csv('../data/jdata_product.csv')
df_product_comment=pd.merge(df_product,df_comment,on='sku_id',how='left')
df_product_comment=df_product_comment.fillna(0)
df_product_comment=df_product_comment.groupby(['shop_id'],as_index=False).sum()
df_product_comment=df_product_comment.drop(['sku_id','brand','cate'],axis=1)
df_shop_product_comment=pd.merge(df_shop,df_product_comment,how='left',on='shop_id')

train_set=pd.merge(train_set,df_user,how='left',on='user_id')
train_set=pd.merge(train_set,df_shop_product_comment,on='shop_id',how='left')
test_set = jdata_data[(jdata_data['action_time']>='2018-04-02') \
                           & (jdata_data['action_time']<='2018-04-15')][['user_id','cate','shop_id']].drop_duplicates()
test_set = test_set.merge(df_test,on=['user_id','cate','shop_id'],how='left')

del df_train
del df_test
test_set=pd.merge(test_set,df_user,how='left',on='user_id')
test_set=pd.merge(test_set,df_shop_product_comment,on='shop_id',how='left')
test_set=test_set.sort_values('user_id')
train_set=train_set.sort_values('user_id')
train_set.rename(columns={'cate_x':'cate'}, inplace = True)
test_set.rename(columns={'cate_x':'cate'}, inplace = True)

test_head=test_set[['user_id','cate','shop_id']]
train_head=train_set[['user_id','cate','shop_id']]
test_set=test_set.drop(['user_id','cate','shop_id'],axis=1)
train_set=train_set.drop(['user_id','cate','shop_id'],axis=1)

# 数据准备
X_train = train_set.drop(['label'],axis=1).values
y_train = train_set['label'].values
X_test = test_set.values
del test_set
del train_set

# 模型工具
class SBBTree():
    """Stacking,Bootstap,Bagging----SBBTree"""
    def __init__(self, params, stacking_num, bagging_num, bagging_test_size, num_boost_round, early_stopping_rounds):
        """
        Initializes the SBBTree.
        Args:
          params : lgb params.
          stacking_num : k_flod stacking.
          bagging_num : bootstrap num.
          bagging_test_size : bootstrap sample rate.
          num_boost_round : boost num.
          early_stopping_rounds : early_stopping_rounds.
        """
        self.params = params
        self.stacking_num = stacking_num
        self.bagging_num = bagging_num
        self.bagging_test_size = bagging_test_size
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.model = lgb
        self.stacking_model = []
        self.bagging_model = []

    def fit(self, X, y):
        """ fit model. """
        if self.stacking_num > 1:
            layer_train = np.zeros((X.shape[0], 2))
            self.SK = StratifiedKFold(n_splits=self.stacking_num, shuffle=True, random_state=1)
            for k,(train_index, test_index) in enumerate(self.SK.split(X, y)):
                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test = y[test_index]

                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

                gbm = lgb.train(self.params,
                            lgb_train,
                            num_boost_round=self.num_boost_round,
                            valid_sets=lgb_eval,
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose_eval=300)

                self.stacking_model.append(gbm)

                pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                layer_train[test_index, 1] = pred_y

            X = np.hstack((X, layer_train[:,1].reshape((-1,1)))) 
        else:
            pass
        for bn in range(self.bagging_num):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.bagging_test_size, random_state=bn)

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

            gbm = lgb.train(self.params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=200,
                        verbose_eval=300)

            self.bagging_model.append(gbm)

    def predict(self, X_pred):
        """ predict test data. """
        if self.stacking_num > 1:
            test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
            for sn,gbm in enumerate(self.stacking_model):
                pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
                test_pred[:, sn] = pred
            X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1,1))))  
        else:
            pass 
        for bn,gbm in enumerate(self.bagging_model):
            pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
            if bn == 0:
                pred_out=pred
            else:
                pred_out+=pred
        return pred_out/self.bagging_num

# 模型参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 2 ** 5 - 1,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': .7,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'scale_pos_weight': 25,
    'seed': 2018,
    'nthread': 16,
    'verbose': 0,
}

# 使用模型
model = SBBTree(params=params,\
                      stacking_num=5,\
                      bagging_num=5,\
                      bagging_test_size=0.33,\
                      num_boost_round=10000,\
                      early_stopping_rounds=200)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)

test_head['pred_prob'] = y_predict


test_head.to_csv('../output/EDA16-twoWeek.csv', index=False)

twoOld = test_head[test_head['pred_prob'] >= 0.5205][['user_id', 'cate', 'shop_id']]
twoOld.to_csv('../output/res_twoWeekOld5205.csv', index=False)




