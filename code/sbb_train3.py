
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
pd.set_option('display.max_columns', None)


# In[2]:


df_train=pd.read_csv('../output/df_train.csv')
df_test=pd.read_csv('../output/df_test.csv')
##这里可以选择 加载多个特征文件 进行merge 如果df_train变了 记得在输出文件名称加以备注 使用了什么特征文件
###设置特征标志位 如果 使用一周特征为1 加上两周特征为12 再加上三周特征 为123 只使用二周特征为2

# df_train_two=pd.read_csv('../output/df_train_two.csv')
# df_test_two=pd.read_csv('../output/df_test_two.csv')
# df_train = df_train.merge(df_train_two,on=['user_id','cate','shop_id'],how='left')
# df_test = df_test.merge(df_test_two,on=['user_id','cate','shop_id'],how='left')


# In[15]:


df_user=pd.read_csv('../data/jdata_user.csv')
df_comment=pd.read_csv('../data/jdata_comment.csv')
df_shop=pd.read_csv('../data/jdata_shop.csv')

# 1）行为数据（jdata_action）
jdata_action = pd.read_csv('../data/jdata_action.csv')

# 3）商品数据（jdata_product）
jdata_product =  pd.read_csv('../data/jdata_product.csv')

jdata_data = jdata_action.merge(jdata_product,on=['sku_id'])
label_flag = 3
train_buy = jdata_data[(jdata_data['action_time']>='2018-03-26')                            & (jdata_data['action_time']<'2018-04-02')                            & (jdata_data['type']==2)][['user_id','cate','shop_id']].drop_duplicates()
train_buy['label'] = 1
# 候选集 时间 ： '2018-03-26'-'2018-04-08' 最近两周有行为的（用户，类目，店铺）
win_size = 3#如果选择两周行为则为2 三周则为3
train_set = jdata_data[(jdata_data['action_time']>='2018-03-05')                            & (jdata_data['action_time']<'2018-03-26')][['user_id','cate','shop_id']].drop_duplicates()
train_set = train_set.merge(train_buy,on=['user_id','cate','shop_id'],how='left').fillna(0)

train_set = train_set.merge(df_train,on=['user_id','cate','shop_id'],how='left')


# In[17]:


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


# In[25]:


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


# In[30]:


train_set['vip_prob'] = train_set['vip_num']/train_set['fans_num']
train_set['goods_prob'] = train_set['good_comments']/train_set['comments']

train_set = train_set.drop(['comments','good_comments','bad_comments'],axis=1)


# In[35]:


test_set = jdata_data[(jdata_data['action_time'] >= '2018-03-26') & (jdata_data['action_time'] < '2018-04-16')][
    ['user_id', 'cate', 'shop_id']].drop_duplicates()

test_set = test_set.merge(df_test, on=['user_id', 'cate', 'shop_id'], how='left')

test_set = pd.merge(test_set, df_user, how='left', on='user_id')
test_set = pd.merge(test_set, df_shop_product_comment, on='shop_id', how='left')

train_set.drop(['user_reg_tm', 'shop_reg_tm'], axis=1, inplace=True)
test_set.drop(['user_reg_tm', 'shop_reg_tm'], axis=1, inplace=True)


# In[36]:


test_set['vip_prob'] = test_set['vip_num']/test_set['fans_num']
test_set['goods_prob'] = test_set['good_comments']/test_set['comments']

test_set = test_set.drop(['comments','good_comments','bad_comments'],axis=1)


# In[40]:


###取六周特征 特征为2.26-4.9
train_set = train_set.drop(['2018-04-02-2018-04-09-action_1', '2018-04-02-2018-04-09-action_2',
       '2018-04-02-2018-04-09-action_3', '2018-04-02-2018-04-09-action_4',
       '2018-03-26-2018-04-02-action_1', '2018-03-26-2018-04-02-action_2',
       '2018-03-26-2018-04-02-action_3', '2018-03-26-2018-04-02-action_4',
       '2018-02-05-2018-02-12-action_1', '2018-02-05-2018-02-12-action_2',
       '2018-02-05-2018-02-12-action_3', '2018-02-05-2018-02-12-action_4'],axis=1)


# In[41]:


test_set = test_set.drop(['2018-02-26-2018-03-05-action_1',
       '2018-02-26-2018-03-05-action_2', '2018-02-26-2018-03-05-action_3',
       '2018-02-26-2018-03-05-action_4', '2018-02-19-2018-02-26-action_1',
       '2018-02-19-2018-02-26-action_2', '2018-02-19-2018-02-26-action_3',
       '2018-02-19-2018-02-26-action_4', '2018-02-12-2018-02-19-action_1',
       '2018-02-12-2018-02-19-action_2', '2018-02-12-2018-02-19-action_3',
       '2018-02-12-2018-02-19-action_4'],axis=1)


# In[44]:


train_set.rename(columns={'cate_x':'cate'}, inplace = True)
test_set.rename(columns={'cate_x':'cate'}, inplace = True)


# In[45]:


test_head=test_set[['user_id','cate','shop_id']]
train_head=train_set[['user_id','cate','shop_id']]
test_set=test_set.drop(['user_id','cate','shop_id'],axis=1)
train_set=train_set.drop(['user_id','cate','shop_id'],axis=1)
if(train_set.shape[1]-1==test_set.shape[1]):
    print('ok',train_set.shape[1])
else:
    exit()
# In[46]:



# 数据准备
X_train = train_set.drop(['label'],axis=1).values
y_train = train_set['label'].values
X_test = test_set.values


# In[ ]:


del train_set
del test_set


# In[48]:


import gc
gc.collect()


# In[50]:


# 模型工具
class SBBTree():
    """Stacking,Bootstap,Bagging----SBBTree"""
    """ author：Cookly 洪鹏飞 """
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
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'min_child_weight': 0,
    'scale_pos_weight': 25,
    'seed': 2019,
    'nthread': 4,
    'verbose': 0,
}

# 使用模型
model = SBBTree(params=params,                      stacking_num=5,                      bagging_num=5,                      bagging_test_size=0.33,                      num_boost_round=10000,                      early_stopping_rounds=200)
model.fit(X_train, y_train)
print('train is ok')
y_predict = model.predict(X_test)
print('pred test is ok')
# y_train_predict = model.predict(X_train)


# In[ ]:


from tqdm import tqdm
test_head['pred_prob'] = y_predict
test_head.to_csv('../feature/'+str(win_size)+'_sbb_get_'+str(label_flag)+'_test.csv',index=False)


