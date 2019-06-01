import pandas as pd
import numpy as np
import os
from tqdm import tqdm

sbb3_3 = pd.read_csv('../feature/3_sbb_get_3_test.csv')
sbb3_2 = pd.read_csv('../feature/3_sbb_get_2_test.csv')
sbb3_1 = pd.read_csv('../feature/3_sbb_get_1_test.csv')


# sbb3_3['pred_prob'] = y_predict
best_u = 0.686
#设置阈值 计算行数
# print('sbb3_3 best_len',len(sbb3_3[sbb3_3['pred_prob']>=best_u]))
sbb3_3[sbb3_3['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb3_3.csv',index=False)

# sbb3_2['pred_prob'] = y_predict
best_u = 0.602
#设置阈值 计算行数
# print('sbb3_2 best_len',len(sbb3_2[sbb3_2['pred_prob']>=best_u]))
sbb3_2[sbb3_2['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb3_2.csv',index=False)
# sbb3_1['pred_prob'] = y_predict
best_u = 0.521
#设置阈值 计算行数
# print('sbb3_1 best_len',len(sbb3_1[sbb3_1['pred_prob']>=best_u]))
sbb3_1[sbb3_1['pred_prob']>=best_u][['user_id','cate','shop_id']].to_csv('../output/sbb3_1.csv',index=False)


n_3_593 = pd.read_csv('../output/res_threeWeekNew65.csv')
n_4_590 = pd.read_csv('../output/res_fourWeekNew675.csv')
o_2_573 = pd.read_csv('../output/res_twoWeekOld5205.csv')
o_3_583 = pd.read_csv('../output/res_threeWeekOld595.csv')
o_4_578 = pd.read_csv('../output/res_fourWeekOld60.csv')
sbb_1 = pd.read_csv('../output/sbb3_1.csv')
sbb_2 = pd.read_csv('../output/sbb3_2.csv')
sbb_3 = pd.read_csv('../output/sbb3_3.csv')



all_item = pd.concat([n_3_593,n_4_590,o_2_573,o_3_583,o_4_578,sbb_1,sbb_2,sbb_3],axis=0)
all_item = all_item.drop_duplicates()


n_3_593['label1'] = 1
n_4_590['label2'] = 1
o_2_573['label3'] = 1
o_3_583['label4'] = 1
o_4_578['label5'] = 1
sbb_1['label6'] = 1
sbb_2['label7'] = 1
sbb_3['label8'] = 1



all_item = all_item.merge(n_3_593,on=['user_id','cate','shop_id'],how='left')
all_item = all_item.merge(n_4_590,on=['user_id','cate','shop_id'],how='left')
all_item = all_item.merge(o_2_573,on=['user_id','cate','shop_id'],how='left')
all_item = all_item.merge(o_3_583,on=['user_id','cate','shop_id'],how='left')
all_item = all_item.merge(o_4_578,on=['user_id','cate','shop_id'],how='left')
all_item = all_item.merge(sbb_1,on=['user_id','cate','shop_id'],how='left')
all_item = all_item.merge(sbb_2,on=['user_id','cate','shop_id'],how='left')
all_item = all_item.merge(sbb_3,on=['user_id','cate','shop_id'],how='left')


all_item = all_item.fillna(0)


all_item['sum'] = all_item['label1']+all_item['label2']+all_item['label3']+all_item['label4']+all_item['label5']+all_item['label6']+all_item['label7']+all_item['label8']

all_item[all_item['sum']>=2][['user_id',
                              'cate','shop_id']].to_csv('../submit/8_model_2.csv',index=False)


