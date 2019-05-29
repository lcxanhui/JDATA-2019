import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
pd.set_option('display.max_columns', None)


## 读取文件减少内存，参考鱼佬的腾讯赛
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

df_train = reduce_mem_usage(pd.read_csv('./df_train.csv'))
df_test = reduce_mem_usage(pd.read_csv('./df_test.csv'))
##这里可以选择 加载多个特征文件 进行merge 如果df_train变了 记得在输出文件名称加以备注 使用了什么特征文件
###设置特征标志位 如果 使用一周特征为1 加上两周特征为12 再加上三周特征 为123 只使用二周特征为2


df_user=reduce_mem_usage(pd.read_csv('./jdata_user.csv'))
df_comment=reduce_mem_usage(pd.read_csv('./jdata_comment.csv'))
df_shop=reduce_mem_usage(pd.read_csv('./jdata_shop.csv'))

# 1）行为数据（jdata_action）
jdata_action = reduce_mem_usage(pd.read_csv('./jdata_action.csv'))

# 3）商品数据（jdata_product）
jdata_product =  reduce_mem_usage(pd.read_csv('./jdata_product.csv'))

jdata_data = jdata_action.merge(jdata_product,on=['sku_id'])
time_1 = time.process_time()
print('<< 数据读取完成！用时', time_1 - time_0, 's')


label_flag = 2
train_buy = jdata_data[(jdata_data['action_time']>='2018-04-02')&(jdata_data['action_time']<'2018-04-09')                            & (jdata_data['type']==2)][['user_id','cate','shop_id']].drop_duplicates()
train_buy['label'] = 1
# 候选集 时间 ： '2018-03-26'-'2018-04-08' 最近两周有行为的（用户，类目，店铺）
win_size = 3#如果选择两周行为则为2 三周则为3
train_set = jdata_data[(jdata_data['action_time']>='2018-03-12')&(jdata_data['action_time']<'2018-04-02')][['user_id','cate','shop_id']].drop_duplicates()
train_set = train_set.merge(train_buy,on=['user_id','cate','shop_id'],how='left').fillna(0)

train_set = train_set.merge(df_train,on=['user_id','cate','shop_id'],how='left')



def mapper_year(x):
    if x is not np.nan:
        year = int(x[:4])
        return 2018 - year


def mapper_month(x):
    if x is not np.nan:
        year = int(x[:4])
        month = int(x[5:7])
        return (2018 - year) * 12 + month


def mapper_day(x):
    if x is not np.nan:
        year = int(x[:4])
        month = int(x[5:7])
        day = int(x[8:10])
        return (2018 - year) * 365 + month * 30 + day


df_user['user_reg_year'] = df_user['user_reg_tm'].apply(lambda x: mapper_year(x))
df_user['user_reg_month'] = df_user['user_reg_tm'].apply(lambda x: mapper_month(x))
df_user['user_reg_day'] = df_user['user_reg_tm'].apply(lambda x: mapper_day(x))

df_shop['shop_reg_year'] = df_shop['shop_reg_tm'].apply(lambda x: mapper_year(x))
df_shop['shop_reg_month'] = df_shop['shop_reg_tm'].apply(lambda x: mapper_month(x))
df_shop['shop_reg_day'] = df_shop['shop_reg_tm'].apply(lambda x: mapper_day(x))



df_shop['shop_reg_year'] = df_shop['shop_reg_year'].fillna(1)
df_shop['shop_reg_month'] = df_shop['shop_reg_month'].fillna(21)
df_shop['shop_reg_day'] = df_shop['shop_reg_day'].fillna(101)

df_user['age'] = df_user['age'].fillna(5)

df_comment = df_comment.groupby(['sku_id'], as_index=False).sum()
print('check point ...')
df_product_comment = pd.merge(jdata_product, df_comment, on='sku_id', how='left')

df_product_comment = df_product_comment.fillna(0)

df_product_comment = df_product_comment.groupby(['shop_id'], as_index=False).sum()

df_product_comment = df_product_comment.drop(['sku_id', 'brand', 'cate'], axis=1)

df_shop_product_comment = pd.merge(df_shop, df_product_comment, how='left', on='shop_id')

train_set = pd.merge(train_set, df_user, how='left', on='user_id')
train_set = pd.merge(train_set, df_shop_product_comment, on='shop_id', how='left')




train_set['vip_prob'] = train_set['vip_num']/train_set['fans_num']
train_set['goods_prob'] = train_set['good_comments']/train_set['comments']

train_set = train_set.drop(['comments','good_comments','bad_comments'],axis=1)



test_set = jdata_data[(jdata_data['action_time'] >= '2018-03-26') & (jdata_data['action_time'] < '2018-04-16')][['user_id', 'cate', 'shop_id']].drop_duplicates()

test_set = test_set.merge(df_test, on=['user_id', 'cate', 'shop_id'], how='left')

test_set = pd.merge(test_set, df_user, how='left', on='user_id')
test_set = pd.merge(test_set, df_shop_product_comment, on='shop_id', how='left')

train_set.drop(['user_reg_tm', 'shop_reg_tm'], axis=1, inplace=True)
test_set.drop(['user_reg_tm', 'shop_reg_tm'], axis=1, inplace=True)




test_set['vip_prob'] = test_set['vip_num']/test_set['fans_num']
test_set['goods_prob'] = test_set['good_comments']/test_set['comments']

test_set = test_set.drop(['comments','good_comments','bad_comments'],axis=1)



###取六周特征 特征为2.19-4.1
train_set = train_set.drop([
       '2018-04-02-2018-04-09-action_1', '2018-04-02-2018-04-09-action_2',
       '2018-04-02-2018-04-09-action_3', '2018-04-02-2018-04-09-action_4',
       '2018-02-12-2018-02-19-action_1', '2018-02-12-2018-02-19-action_2',
       '2018-02-12-2018-02-19-action_3', '2018-02-12-2018-02-19-action_4',
       '2018-02-05-2018-02-12-action_1', '2018-02-05-2018-02-12-action_2',
       '2018-02-05-2018-02-12-action_3', '2018-02-05-2018-02-12-action_4'],axis=1)


###取六周特征 特征为3.05-4.15
test_set = test_set.drop(['2018-02-26-2018-03-05-action_1',
       '2018-02-26-2018-03-05-action_2', '2018-02-26-2018-03-05-action_3',
       '2018-02-26-2018-03-05-action_4', '2018-02-19-2018-02-26-action_1',
       '2018-02-19-2018-02-26-action_2', '2018-02-19-2018-02-26-action_3',
       '2018-02-19-2018-02-26-action_4', '2018-02-12-2018-02-19-action_1',
       '2018-02-12-2018-02-19-action_2', '2018-02-12-2018-02-19-action_3',
       '2018-02-12-2018-02-19-action_4'],axis=1)



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

del train_set
del test_set

print('------------------start modelling----------------')
# 模型工具
class SBBTree():
    """Stacking,Bootstap,Bagging----SBBTree"""

    def __init__(self, params, stacking_num, bagging_num, bagging_test_size, num_boost_round, early_stopping_rounds):
        """
        Initializes the SBBTree.
        Args:
          params : xgb params.
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

        self.model = xgb
        self.stacking_model = []
        self.bagging_model = []

    def fit(self, X, y):
        """ fit model. """
        if self.stacking_num > 1:
            layer_train = np.zeros((X.shape[0], 2))
            self.SK = StratifiedKFold(n_splits=self.stacking_num, shuffle=True, random_state=1)
            for k, (train_index, test_index) in enumerate(self.SK.split(X, y)):
                print('fold_{}'.format(k))
                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test = y[test_index]

                xgb_train = xgb.DMatrix(X_train, y_train)
                xgb_eval = xgb.DMatrix(X_test, y_test)
                watchlist = [(xgb_train, 'train'), (xgb_eval, 'valid')]

                xgb_model = xgb.train(dtrain=xgb_train,
                                      num_boost_round=self.num_boost_round,
                                      evals=watchlist,
                                      early_stopping_rounds=self.early_stopping_rounds,
                                      verbose_eval=300,
                                      params=self.params)
                self.stacking_model.append(xgb_model)

                pred_y = xgb_model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_ntree_limit)
                layer_train[test_index, 1] = pred_y

            X = np.hstack((X, layer_train[:, 1].reshape((-1, 1))))
        else:
            pass
        for bn in range(self.bagging_num):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.bagging_test_size, random_state=bn)

            xgb_train = xgb.DMatrix(X_train, y_train)
            xgb_eval = xgb.DMatrix(X_test, y_test)
            watchlist = [(xgb_train, 'train'), (xgb_eval, 'valid')]

            xgb_model = xgb.train(dtrain=xgb_train,
                                  num_boost_round=10000,
                                  evals=watchlist,
                                  early_stopping_rounds=200,
                                  verbose_eval=300,
                                  params=self.params)

            self.bagging_model.append(xgb_model)

    def predict(self, X_pred):
        """ predict test data. """
        if self.stacking_num > 1:
            test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
            for sn, gbm in enumerate(self.stacking_model):
                pred = gbm.predict(xgb.DMatrix(X_pred), ntree_limit=gbm.best_ntree_limit)
                test_pred[:, sn] = pred
            X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1, 1))))
        else:
            pass
        for bn, gbm in enumerate(self.bagging_model):
            pred = gbm.predict(xgb.DMatrix(X_pred), ntree_limit=gbm.best_ntree_limit)
            if bn == 0:
                pred_out = pred
            else:
                pred_out += pred
        return pred_out / self.bagging_num


# 模型参数
params = {
    'booster': 'gbtree',
    'tree_method': 'exact',
    'eta': 0.01,
    'max_depth': 7,
    'gamma': 0.1,
    "min_child_weight": 1.1,  # 6 0.06339878
    'subsample': 0.7,
    'colsample_bytree': 0.7,  # 0.06349307
    'colsample_bylevel': 0.7,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': True,
    'lambda': 3,  # 0.06365710
    'nthread': 24,
    'seed': 42}

# 使用模型
model = SBBTree(params=params, \
                stacking_num=5, \
                bagging_num=5, \
                bagging_test_size=0.33, \
                num_boost_round=10000, \
                early_stopping_rounds=200)
model.fit(X_train, y_train)


print('train is ok')
y_predict = model.predict(X_test)
print('pred test is ok')
# y_train_predict = model.predict(X_train)



from tqdm import tqdm
test_head['pred_prob'] = y_predict
test_head.to_csv('feature/'+str(win_size)+'_xgb_get_'+str(label_flag)+'_test.csv',index=False)
