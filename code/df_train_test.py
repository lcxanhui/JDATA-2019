import pandas as pd
from sklearn.model_selection import  train_test_split
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from joblib import dump
import time

time_0 = time.process_time()
print('>> 开始读取数据')
df_action=pd.read_csv("./jdata_action.csv")
df_product=pd.read_csv("./jdata_product.csv")

df_action=pd.merge(df_action,df_product,how='left',on='sku_id')
df_action=df_action.groupby(['user_id','shop_id','cate'], as_index=False).sum()
time_1 = time.process_time()
print('<< 数据读取完成！用时', time_1 - time_0, 's')

df_action=df_action[['user_id','shop_id','cate']]
df_action_head=df_action.copy()

df_action=pd.read_csv("./jdata_action.csv")

def makeActionData(startDate,endDate):
    df=df_action[(df_action['action_time']>startDate)&(df_action['action_time']<endDate)]
    df_1= pd.get_dummies(df['type'], prefix='%s-%s-action' % (startDate, endDate))
    df= pd.concat([df, df_1], axis=1) # type: pd.DataFrame
    df = df.groupby(['user_id', 'sku_id'], as_index=False).sum()
    df=df.drop(['type'],axis=1)
    df=pd.merge(df,df_product,how='left',on='sku_id')
    df=df.groupby(['user_id','shop_id','cate'], as_index=False).sum()
    df=df.drop(['sku_id','module_id','brand'],axis=1)
    return df

df_train=df_action_head

print('----------------start merge--------------------------')
time_2 = time.process_time()

df9=makeActionData('2018-04-02','2018-04-09')
df_train=pd.merge(df_train,df9,how='left',on=['user_id','shop_id','cate'])
del df9

del df_action_head

df8=makeActionData('2018-03-26','2018-04-02')
df_train=pd.merge(df_train,df8,how='left',on=['user_id','shop_id','cate'])
del df8

df7=makeActionData('2018-03-19','2018-03-26')
df_train=pd.merge(df_train,df7,how='left',on=['user_id','shop_id','cate'])
del df7

df6=makeActionData('2018-03-12','2018-03-19')
df_train=pd.merge(df_train,df6,how='left',on=['user_id','shop_id','cate'])
del df6

df5=makeActionData('2018-03-05','2018-03-12')
df_train=pd.merge(df_train,df5,how='left',on=['user_id','shop_id','cate'])
del df5

df4=makeActionData('2018-02-26','2018-03-05')
df_train=pd.merge(df_train,df4,how='left',on=['user_id','shop_id','cate'])
del df4

df3=makeActionData('2018-02-19','2018-02-26')
df_train=pd.merge(df_train,df3,how='left',on=['user_id','shop_id','cate'])
del df3

df2=makeActionData('2018-02-12','2018-02-19')
df_train=pd.merge(df_train,df2,how='left',on=['user_id','shop_id','cate'])
del df2

df1=makeActionData('2018-02-05','2018-02-12')
df_train=pd.merge(df_train,df1,how='left',on=['user_id','shop_id','cate'])
del df1

df10=makeActionData('2018-04-09','2018-04-16')
df_train=pd.merge(df_train,df10,how='left',on=['user_id','shop_id','cate'])
del df10
time_3 = time.process_time()
print('----------------merge Ok, 用时', time_3 - time_2, 's')
      
df_train=df_train.fillna(0)

#df_train.to_csv('output/train.csv',index=False)

columns=['user_id', 'shop_id', 'cate', '2018-04-09-2018-04-16-action_1',
        '2018-04-09-2018-04-16-action_2',
       '2018-04-09-2018-04-16-action_3', '2018-04-09-2018-04-16-action_4',
       '2018-04-09-2018-04-16-action_5', '2018-04-02-2018-04-09-action_1',
       '2018-04-02-2018-04-09-action_2', '2018-04-02-2018-04-09-action_3',
       '2018-04-02-2018-04-09-action_4', '2018-04-02-2018-04-09-action_5',
       '2018-03-26-2018-04-02-action_1', '2018-03-26-2018-04-02-action_2',
       '2018-03-26-2018-04-02-action_3', '2018-03-26-2018-04-02-action_4',
       '2018-03-19-2018-03-26-action_1', '2018-03-19-2018-03-26-action_2',
       '2018-03-19-2018-03-26-action_3', '2018-03-19-2018-03-26-action_4',
       '2018-03-12-2018-03-19-action_1', '2018-03-12-2018-03-19-action_2',
       '2018-03-12-2018-03-19-action_3', '2018-03-12-2018-03-19-action_4',
       '2018-03-05-2018-03-12-action_1', '2018-03-05-2018-03-12-action_2',
       '2018-03-05-2018-03-12-action_3', '2018-03-05-2018-03-12-action_4',
       '2018-02-26-2018-03-05-action_1', '2018-02-26-2018-03-05-action_2',
       '2018-02-26-2018-03-05-action_3', '2018-02-26-2018-03-05-action_4',
       '2018-02-19-2018-02-26-action_1', '2018-02-19-2018-02-26-action_2',
       '2018-02-19-2018-02-26-action_3', '2018-02-19-2018-02-26-action_4',
       '2018-02-12-2018-02-19-action_1', '2018-02-12-2018-02-19-action_2',
       '2018-02-12-2018-02-19-action_3', '2018-02-12-2018-02-19-action_4',
       '2018-02-05-2018-02-12-action_1', '2018-02-05-2018-02-12-action_2',
       '2018-02-05-2018-02-12-action_3', '2018-02-05-2018-02-12-action_4',
       ]

df_train=df_train[columns]
df_train=df_train.drop(['2018-04-09-2018-04-16-action_5','2018-04-02-2018-04-09-action_5'],axis=1)

train=df_train.drop(['2018-04-09-2018-04-16-action_1',
        '2018-04-09-2018-04-16-action_2',
       '2018-04-09-2018-04-16-action_3', '2018-04-09-2018-04-16-action_4',],axis=1)

test=df_train.drop(['2018-02-05-2018-02-12-action_1',
       '2018-02-05-2018-02-12-action_2', '2018-02-05-2018-02-12-action_3',
       '2018-02-05-2018-02-12-action_4'],axis=1)
print('------------------------All done-------------,gen train and test------')
train.to_csv('./df_train.csv',index=False)
test.to_csv('./df_test.csv',index=False)
