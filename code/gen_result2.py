import pandas as pd
import numpy as np
###同特征相交 不同特征投票
sbb4_3 = pd.read_csv('../feature/4_sbb_get_3_test.csv')
sbb4_2 = pd.read_csv('../feature/4_sbb_get_2_test.csv')
sbb4_1 = pd.read_csv('../feature/4_sbb_get_1_test.csv')

from tqdm import tqdm
# sbb4_1['pred_prob'] = y_predict
best_u = 0.662
#设置阈值 计算行数
# print('sbb4_1 best_len',len(sbb4_1[sbb4_1['pred_prob']>=best_u]))
sbb4_1[sbb4_1['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb4_1.csv',index=False)

from tqdm import tqdm
# sbb4_1['pred_prob'] = y_predict
best_u = 0.500
#设置阈值 计算行数
# print('sbb4_2 best_len',len(sbb4_2[sbb4_2['pred_prob']>=best_u]))
sbb4_2[sbb4_2['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb4_2.csv',index=False)

from tqdm import tqdm
# sbb4_3['pred_prob'] = y_predict
best_u = 0.685
#设置阈值 计算行数
# print('sbb4_3 best_len',len(sbb4_3[sbb4_3['pred_prob']>=best_u]))
sbb4_3[sbb4_3['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb4_3.csv',index=False)

sbb3_3 = pd.read_csv('../feature/3_sbb_get_3_test.csv')
sbb3_2 = pd.read_csv('../feature/3_sbb_get_2_test.csv')
sbb3_1 = pd.read_csv('../feature/3_sbb_get_1_test.csv')

from tqdm import tqdm
# sbb3_3['pred_prob'] = y_predict
best_u = 0.686
#设置阈值 计算行数
# print('sbb3_3 best_len',len(sbb3_3[sbb3_3['pred_prob']>=best_u]))
sbb3_3[sbb3_3['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb3_3.csv',index=False)

from tqdm import tqdm
# sbb3_2['pred_prob'] = y_predict
best_u = 0.602
#设置阈值 计算行数
# print('sbb3_2 best_len',len(sbb3_2[sbb3_2['pred_prob']>=best_u]))
sbb3_2[sbb3_2['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb3_2.csv',index=False)

from tqdm import tqdm
# sbb3_1['pred_prob'] = y_predict
best_u = 0.521
#设置阈值 计算行数
# print('sbb3_1 best_len',len(sbb3_1[sbb3_1['pred_prob']>=best_u]))
sbb3_1[sbb3_1['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb3_1.csv',index=False)

sbb2_3 = pd.read_csv('../feature/2_sbb_get_3_test.csv')
sbb2_2 = pd.read_csv('../feature/2_sbb_get_2_test.csv')
sbb2_1 = pd.read_csv('../feature/2_sbb_get_1_test.csv')

from tqdm import tqdm
# sbb2_1['pred_prob'] = y_predict
best_u = 0.495
#设置阈值 计算行数
# print('sbb2_1 best_len',len(sbb2_1[sbb2_1['pred_prob']>=best_u]))
sbb2_1[sbb2_1['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb2_1.csv',index=False)

from tqdm import tqdm
# sbb4_2['pred_prob'] = y_predict
best_u = 0.310
#设置阈值 计算行数
# print('sbb2_2 best_len',len(sbb2_2[sbb2_2['pred_prob']>=best_u]))
sbb2_2[sbb2_2['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb2_2.csv',index=False)

from tqdm import tqdm
# sbb4_2['pred_prob'] = y_predict
best_u = 0.480
#设置阈值 计算行数
# print('sbb2_3 best_len',len(sbb2_3[sbb2_3['pred_prob']>=best_u]))
sbb2_3[sbb2_3['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb2_3.csv',index=False)

##同特征相交
##不同特征投票
sbb21 = pd.read_csv('../output/sbb2_1.csv')
sbb31 = pd.read_csv('../output/sbb3_1.csv')
sbb41 = pd.read_csv('../output/sbb4_1.csv')
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

all_data[all_data['sum']>=3][['user_id','cate','shop_id']].to_csv('../output/sbb*1_u3.csv',index=False)

sbb22 = pd.read_csv('../output/sbb2_2.csv')
sbb32 = pd.read_csv('../output/sbb3_2.csv')
sbb42 = pd.read_csv('../output/sbb4_2.csv')
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
all_data[all_data['sum']>=3][['user_id','cate','shop_id']].to_csv('../output/sbb*2_u3.csv',index=False)

sbb23 = pd.read_csv('../output/sbb2_3.csv')
sbb33 = pd.read_csv('../output/sbb3_3.csv')
sbb43 = pd.read_csv('../output/sbb4_3.csv')
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
all_data[all_data['sum']>=3][['user_id','cate','shop_id']].to_csv('../output/sbb*3_u3.csv',index=False)

sbb1_vote = pd.read_csv('../output/sbb*1_u3.csv')
sbb2_vote = pd.read_csv('../output/sbb*2_u3.csv')
sbb3_vote = pd.read_csv('../output/sbb*3_u3.csv')
a_result = pd.read_csv('../submit/8_model_2.csv')
all_data = pd.concat([sbb1_vote,sbb2_vote,sbb3_vote,a_result],axis=0).drop_duplicates()


sbb1_vote['label2']=1
sbb2_vote['label3']=1
sbb3_vote['label4']=1
a_result['label5'] = 1

all_data = all_data.merge(sbb1_vote,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb2_vote,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(sbb3_vote,on=['user_id','cate','shop_id'],how='left')
all_data = all_data.merge(a_result,on=['user_id','cate','shop_id'],how='left')
all_data= all_data.fillna(0)
all_data['sum'] = all_data['label2']+all_data['label3']+all_data['label4']+all_data['label5']

all_data['sum'].value_counts()

all_data[all_data['sum']>=2][['user_id','cate','shop_id']].to_csv('../submit/b_final.csv',index=False)