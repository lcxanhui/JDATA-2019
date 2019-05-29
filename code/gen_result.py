import os
import pandas as pd
import numpy as np
from tqdm import tqdm

sbb4_3 = pd.read_csv('feature/4_sbb_get_3_test.csv')
sbb4_2 = pd.read_csv('feature/4_sbb_get_2_test.csv')
sbb4_1 = pd.read_csv('feature/4_sbb_get_1_test.csv')


best_u = 0.660
#设置阈值 计算行数
print('sbb4_1 best_len',len(sbb4_1[sbb4_1['pred_prob']>=best_u]))
sbb4_1[sbb4_1['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/sbb4_1.csv',index=False)

best_u = 0.495
#设置阈值 计算行数
print('sbb4_2 best_len',len(sbb4_2[sbb4_2['pred_prob']>=best_u]))
sbb4_2[sbb4_2['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/sbb4_2.csv',index=False)

best_u = 0.680
#设置阈值 计算行数
print('sbb4_3 best_len',len(sbb4_3[sbb4_3['pred_prob']>=best_u]))
sbb4_3[sbb4_3['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/sbb4_3.csv',index=False)


sbb3_3 = pd.read_csv('feature/3_sbb_get_3_test.csv')
sbb3_2 = pd.read_csv('feature/3_sbb_get_2_test.csv')
sbb3_1 = pd.read_csv('feature/3_sbb_get_1_test.csv')


best_u = 0.680
#设置阈值 计算行数
print('sbb3_3 best_len',len(sbb3_3[sbb3_3['pred_prob']>=best_u]))
sbb3_3[sbb3_3['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/sbb3_3.csv',index=False)


best_u = 0.600
#设置阈值 计算行数
print('sbb3_2 best_len',len(sbb3_2[sbb3_2['pred_prob']>=best_u]))
sbb3_2[sbb3_2['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/sbb3_2.csv',index=False)


best_u = 0.520
#设置阈值 计算行数
print('sbb3_1 best_len',len(sbb3_1[sbb3_1['pred_prob']>=best_u]))
sbb3_1[sbb3_1['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/sbb3_1.csv',index=False)


sbb2_3 = pd.read_csv('feature/2_sbb_get_3_test.csv')
sbb2_2 = pd.read_csv('feature/2_sbb_get_2_test.csv')
sbb2_1 = pd.read_csv('feature/2_sbb_get_1_test.csv')


best_u = 0.495
#设置阈值 计算行数
print('sbb2_1 best_len',len(sbb2_1[sbb2_1['pred_prob']>=best_u]))
sbb2_1[sbb2_1['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/sbb2_1.csv',index=False)


best_u = 0.310
#设置阈值 计算行数
print('sbb2_2 best_len',len(sbb2_2[sbb2_2['pred_prob']>=best_u]))
sbb2_2[sbb2_2['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/sbb2_2.csv',index=False)


best_u = 0.480
#设置阈值 计算行数
print('sbb2_3 best_len',len(sbb2_3[sbb2_3['pred_prob']>=best_u]))
sbb2_3[sbb2_3['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/sbb2_3.csv',index=False)



##同特征相交
##不同特征投票
sbb21 = pd.read_csv('result/sbb2_1.csv')
sbb31 = pd.read_csv('result/sbb3_1.csv')
sbb41 = pd.read_csv('result/sbb4_1.csv')
all_data = pd.concat([sbb21,sbb31,sbb41],axis=0).drop_duplicates()

sbb21['label2']=1
sbb31['label3']=1
sbb41['label4']=1

all_data = all_data.merge(sbb21,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb31,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb41,on=['user_id','cate','shop_id'],how='left')
all_data= all_data.fillna(0)
all_data['sum'] = all_data['label2']+all_data['label3']+all_data['label4']

all_data['sum'].value_counts()




all_data[all_data['sum']>=3][['user_id','cate','shop_id']].to_csv('result/sbb*1_u3.csv',index=False)




sbb22 = pd.read_csv('result/sbb2_2.csv')
sbb32 = pd.read_csv('result/sbb3_2.csv')
sbb42 = pd.read_csv('result/sbb4_2.csv')
all_data = pd.concat([sbb22,sbb32,sbb42],axis=0).drop_duplicates()

sbb22['label2']=1
sbb32['label3']=1
sbb42['label4']=1

all_data = all_data.merge(sbb22,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb32,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb42,on=['user_id','cate','shop_id'],how='left')
all_data= all_data.fillna(0)
all_data['sum'] = all_data['label2']+all_data['label3']+all_data['label4']

all_data['sum'].value_counts()
all_data[all_data['sum']>=3][['user_id','cate','shop_id']].to_csv('result/sbb*2_u3.csv',index=False)




sbb23 = pd.read_csv('result/sbb2_3.csv')
sbb33 = pd.read_csv('result/sbb3_3.csv')
sbb43 = pd.read_csv('result/sbb4_3.csv')
all_data = pd.concat([sbb23,sbb33,sbb43],axis=0).drop_duplicates()

sbb23['label2']=1
sbb33['label3']=1
sbb43['label4']=1

all_data = all_data.merge(sbb23,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb33,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb43,on=['user_id','cate','shop_id'],how='left')
all_data= all_data.fillna(0)
all_data['sum'] = all_data['label2']+all_data['label3']+all_data['label4']

all_data['sum'].value_counts()
all_data[all_data['sum']>=3][['user_id','cate','shop_id']].to_csv('result/sbb*3_u3.csv',index=False)




sbb1_vote = pd.read_csv('result/sbb*1_u3.csv')
sbb2_vote = pd.read_csv('result/sbb*2_u3.csv')
sbb3_vote = pd.read_csv('result/sbb*3_u3.csv')
all_data = pd.concat([sbb1_vote,sbb2_vote,sbb3_vote],axis=0).drop_duplicates()




sbb1_vote['label2']=1
sbb2_vote['label3']=1
sbb3_vote['label4']=1



all_data = all_data.merge(sbb1_vote,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb2_vote,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb3_vote,on=['user_id','cate','shop_id'],how='left')
all_data= all_data.fillna(0)
all_data['sum'] = all_data['label2']+all_data['label3']+all_data['label4']





all_data[all_data['sum']>=1][['user_id','cate','shop_id']].to_csv('result/sbb_final.csv',index=False)




sbb_final = pd.read_csv('result/sbb_final.csv')
sbb_pre = pd.read_csv('result/all_item_2.csv')





all_data = pd.concat([sbb_final,sbb_pre],axis=0).drop_duplicates()


all_data[['user_id','cate','shop_id']].to_csv('result/zsq_my_final.csv',index=False)



xgb3_3 = pd.read_csv('feature/3_xgb_get_3_test.csv')
xgb3_2 = pd.read_csv('feature/3_xgb_get_2_test.csv')
xgb3_1 = pd.read_csv('feature/3_xgb_get_1_test.csv')


best_u = 0.095
#设置阈值 计算行数
print('xgb3_3 best_len',len(xgb3_3[xgb3_3['pred_prob']>=best_u]))
xgb3_3[xgb3_3['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/xgb3_3.csv',index=False)


best_u = 0.155
#设置阈值 计算行数
print('xgb3_2 best_len',len(xgb3_2[xgb3_2['pred_prob']>=best_u]))
xgb3_2[xgb3_2['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/xgb3_2.csv',index=False)


best_u = 0.095
#设置阈值 计算行数
print('xgb3_1 best_len',len(xgb3_1[xgb3_1['pred_prob']>=best_u]))
xgb3_1[xgb3_1['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('result/xgb3_1.csv',index=False)



##同特征相交
##不同特征投票
sbb21 = pd.read_csv('result/sbb2_1.csv')
sbb31 = pd.read_csv('result/sbb3_1.csv')
sbb41 = pd.read_csv('result/sbb4_1.csv')
xgb31 = pd.read_csv('result/xgb3_1.csv')
all_data = pd.concat([sbb21,sbb31,sbb41,xgb31],axis=0).drop_duplicates()

sbb21['label2']=1
sbb31['label3']=1
sbb41['label4']=1
xgb31['label5']=1

all_data = all_data.merge(sbb21,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb31,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb41,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(xgb31,on=['user_id','cate','shop_id'],how='left')
all_data= all_data.fillna(0)
all_data['sum'] = all_data['label2']+all_data['label3']+all_data['label4']+all_data['label5']


all_data[all_data['sum']>=3][['user_id','cate','shop_id']].to_csv('result/*b*1_u3.csv',index=False)


sbb22 = pd.read_csv('result/sbb2_2.csv')
sbb32 = pd.read_csv('result/sbb3_2.csv')
sbb42 = pd.read_csv('result/sbb4_2.csv')
xgb32 = pd.read_csv('result/xgb3_2.csv')
all_data = pd.concat([sbb22,sbb32,sbb42,xgb32],axis=0).drop_duplicates()

sbb22['label2']=1
sbb32['label3']=1
sbb42['label4']=1
xgb32['label5']=1

all_data = all_data.merge(sbb22,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb32,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb42,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(xgb32,on=['user_id','cate','shop_id'],how='left')
all_data= all_data.fillna(0)
all_data['sum'] = all_data['label2']+all_data['label3']+all_data['label4']+all_data['label5']

all_data['sum'].value_counts()

all_data[all_data['sum']>=3][['user_id','cate','shop_id']].to_csv('result/*b*2_u3.csv',index=False)


sbb23 = pd.read_csv('result/sbb2_3.csv')
sbb33 = pd.read_csv('result/sbb3_3.csv')
sbb43 = pd.read_csv('result/sbb4_3.csv')
xgb33 = pd.read_csv('result/xgb3_3.csv')
all_data = pd.concat([sbb23,sbb33,sbb43,xgb33],axis=0).drop_duplicates()

sbb23['label2']=1
sbb33['label3']=1
sbb43['label4']=1
xgb33['label5']=1

all_data = all_data.merge(sbb23,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb33,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb43,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(xgb33,on=['user_id','cate','shop_id'],how='left')
all_data= all_data.fillna(0)
all_data['sum'] = all_data['label2']+all_data['label3']+all_data['label4']+all_data['label5']

all_data['sum'].value_counts()

all_data[all_data['sum']>=3][['user_id','cate','shop_id']].to_csv('result/*b*3_u3.csv',index=False)




n3 = pd.read_csv('result/n_3_593.csv')
n4 = pd.read_csv('result/n_4_590.csv')
o2 = pd.read_csv('result/o_2_573.csv')
o3 = pd.read_csv('result/o_3_583.csv')
o4 = pd.read_csv('result/o_4_578.csv')
all_data = pd.concat([n3,n4,o2,o3,o4],axis=0).drop_duplicates()

n3['label2']=1
n4['label3']=1
o2['label4']=1
o3['label5']=1
o4['label6']=1
all_data = all_data.merge(n3,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(n4,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(o2,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(o3,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(o4,on=['user_id','cate','shop_id'],how='left')
all_data= all_data.fillna(0)
all_data['sum'] = all_data['label2']+all_data['label3']+all_data['label4']+all_data['label5']+all_data['label6']


all_data[all_data['sum']>=4][['user_id','cate','shop_id']].to_csv('result/*b_all_u4.csv',index=False)


b1 = pd.read_csv('result/*b*1_u3.csv')
b2 = pd.read_csv('result/*b*2_u3.csv')
b3 = pd.read_csv('result/*b*3_u3.csv')
ball = pd.read_csv('result/*b_all_u4.csv')


all_data = pd.concat([b1,b2,b3,ball],axis=0).drop_duplicates()

b1['label2']=1
b2['label3']=1
b3['label4']=1
ball['label5']=1

all_data = all_data.merge(b1,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(b2,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(b3,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(ball,on=['user_id','cate','shop_id'],how='left')

all_data= all_data.fillna(0)
all_data['sum'] = all_data['label2']+all_data['label3']+all_data['label4']+all_data['label5']


all_data[all_data['sum']>=1][['user_id','cate','shop_id']].to_csv('result/5_26_final.csv',index=False)

