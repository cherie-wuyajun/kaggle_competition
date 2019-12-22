import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from scipy.stats import kurtosis
import time
import warnings
import lightgbm as lgb
warnings.filterwarnings('ignore')
import datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import mode

pd.set_option('display.max_columns', None)


path = './data/'

# 获得训练数据并编码时间，超时这还款金额N设为0，逾期标签为-1.
train_df = pd.read_csv(path+'train.csv', parse_dates=['auditing_date', 'due_date', 'repay_date'])
train_df['repay_date'] = train_df[['due_date', 'repay_date']].apply(lambda x: x['repay_date'] \
                         if x['repay_date'] != '\\N' else x['due_date']+datetime.timedelta(days=1), axis=1)
train_df['repay_amt'] = train_df['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')
train_df['label'] = (train_df['due_date'] - train_df['repay_date']).dt.days

# 设定还款日期为创建日期的标签为32
train_df.loc[train_df['auditing_date'] == train_df['repay_date'], 'label'] = 31

# 绘制占比图
plt.hist(train_df['label'].value_counts())
plt.show()

# train_df['auditing_date'].min()：Timestamp('2018-01-01 00:00:00') \
# ——train_df['auditing_date'].max()：Timestamp('2018-12-31 00:00:00') 训练数据为2018年一年创建的相应订单数据

clf_labels = train_df['label'].values
amt_labels = train_df['repay_amt'].values
# del train_df['label'], train_df['repay_amt'], train_df['repay_date']
train_due_amt_df = train_df['due_amt'].values
train_num = train_df.shape[0]


# 读取test数据集
test_df = pd.read_csv(path+'test.csv', parse_dates=['auditing_date', 'due_date'])
sub = test_df[['listing_id', 'auditing_date', 'due_amt']]

# 合并两数据集
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

## 获得截至日期的月，天属性
df['due_month'] = df['due_date'].dt.month
df['due_weekday'] = df['due_date'].apply(lambda x:x.isoweekday())

## 获得还款日期的天属性
df['repay_weekday'] = df['repay_date'].apply(lambda x:x.isoweekday())

# test_df['auditing_date'].min()：Timestamp('2019-02-01 00:00:00')——
#  test_df['auditing_date'].max()：Timestamp('2019-03-31 00:00:00') 测试数据为2月-3月两个月数据，！特别注意是2月为28天。

## 读取listing 表数据
listing_info_df = pd.read_csv(path+'listing_info.csv',parse_dates=['auditing_date'])

# Timestamp('2016-07-05 00:00:00')——Timestamp('2019-03-31 00:00:00') listing表包含大量历史订单数据。

listing_info_df['principal_per_term'] = listing_info_df['principal']/listing_info_df['term']


list_count=listing_info_df['user_id'].value_counts().reset_index()
list_count.columns = ['user_id','times']
list_count

list_pricipal = listing_info_df.groupby('user_id')['principal'].\
                                agg({'mean_principal':'mean','std_principal':'std'}).reset_index()


# repay_log 表
repay_log_df = pd.read_csv(path+'user_repay_logs.csv', parse_dates=['due_date', 'repay_date'])
# Timestamp('2017-07-05 00:00:00')——Timestamp('2020-03-18 00:00:00')

# 由于题目任务只预测第一期的还款情况，因此这里只保留第一期的历史记录。当然非第一期的记录也能提取很多特征。
repay_log_df = repay_log_df[repay_log_df['order_id'] == 1].reset_index(drop=True)
repay_log_df['repay'] = repay_log_df['repay_date'].astype('str').apply(lambda x: 1 if x != '2200-01-01' else 0)
repay_log_df['early_repay_days'] = (repay_log_df['due_date'] - repay_log_df['repay_date']).dt.days
repay_log_df['early_repay_days'] = repay_log_df['early_repay_days'].apply(lambda x: x if x >= 0 else -1)

# 合并 repay_log 和 listing_info
repay_log_df_listing = repay_log_df.merge(listing_info_df, on=['user_id','listing_id'],how='left')

repay_log_df['due_month'] = repay_log_df['due_date'].dt.month
repay_log_df['due_day'] = repay_log_df['due_date'].dt.day
repay_log_df['due_weekday'] = repay_log_df['due_date'].apply(lambda x:x.isoweekday())
repay_log_df['repay_weekday'] = repay_log_df['repay_date'].apply(lambda x:x.isoweekday())

repay_log_df['early_month'] = repay_log_df['due_day'].apply(lambda x:1 if x<10 else 0)
repay_log_df['median_month'] = repay_log_df['due_day'].apply(lambda x:1 if 10<=x<20 else 0)
repay_log_df['last_month'] = repay_log_df['due_day'].apply(lambda x:1 if 20<=x else 0)


def mode_0(x):
    return mode(x).mode[0]


# user_id:early_repay_days
repay_log_df_1 = repay_log_df.groupby('user_id')['early_repay_days'].agg({
    'u_early_repay_days_max': 'max', 
    'u_early_repay_days_mode': mode_0, 
    'u_early_repay_days_min': 'min',
    'u_early_repay_days_mean': 'mean', 
    'u_early_repay_days_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id'],how='left')

# user_id:due_weekday
repay_log_df_1 = repay_log_df.groupby('user_id')['due_weekday'].agg({
    'u_repay_due_weekday_mode': mode_0, 
    'u_repay_due_weekday_mean': 'mean', 
    'u_repay_due_weekday_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id'],how='left')

# user_id:repay_weekday
repay_log_df_1 = repay_log_df.groupby('user_id')['repay_weekday'].agg({
    'u_repay_weekday_mean': 'mean', 
    'u_repay_weekday_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id'],how='left')

# user_id:repay
repay_log_df_1 = repay_log_df.groupby('user_id')['repay'].agg({
    'u_repay_mean': 'mean', 
    'u_repay_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id'],how='left')

# user_id,due_month:repay
repay_log_df_1 = repay_log_df.groupby(['user_id','due_month'])['repay'].agg({
    'u_dm_repay_mean': 'mean', 
    'u_dm_repay_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id','due_month'],how='left')


# user_id,due_month:early_repay_days
repay_log_df_1 = repay_log_df.groupby(['user_id','due_month'])['early_repay_days'].agg({
    'u_dm_early_repay_days_mean': 'mean', 
    'u_dm_early_repay_days_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id','due_month'],how='left')


# user_id,due_weekday:repay
repay_log_df_1 = repay_log_df.groupby(['user_id','due_weekday'])['repay'].agg({
    'u_dw_repay_mean': 'mean', 
    'u_dw_repay_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id','due_weekday'],how='left')


# user_id,due_weekday:early_repay_days
repay_log_df_1 = repay_log_df.groupby(['user_id','due_weekday'])['early_repay_days'].agg({
    'u_dw_early_repay_days_mean': 'mean', 
    'u_dw_early_repay_days_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id','due_weekday'],how='left')


# user_id,rate:repay
repay_log_df_1 = repay_log_df.groupby(['user_id','rate'])['repay'].agg({
    'u_r_repay_mean': 'mean', 
    'u_r_repay_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id','rate'],how='left')

# user_id,rate:early_repay_days
repay_log_df_5 = repay_log_df.groupby(['user_id','rate'])['early_repay_days'].agg({
    'u_r_early_repay_days_mean': 'mean', 
    'u_r_early_repay_days_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id','rate'],how='left')

# user_id,repay_amt:repay
repay_log_df_1 = repay_log_df.groupby(['user_id','repay_amt'])['repay'].agg({
    'u_ra_repay_mean': 'mean', 
    'u_ra_repay_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id','repay_amt'],how='left')

# user_id,repay_amt:early_repay_days
repay_log_df_1 = repay_log_df.groupby(['user_id','repay_amt'])['early_repay_days'].agg({
    'u_ra_early_repay_days_mean': 'mean', 
    'u_ra_early_repay_days_std': 'std'
}).reset_index()
repay_log_df_1 = repay_log_df_1.fillna(0)
repay_log_df_1

df = df.merge(repay_log_df_1,on=['user_id','repay_amt'],how='left')


## 区分不同频次的用户
old_user_list = list(repay_log_df['user_id'])

user_id_freq = pd.DataFrame(repay_log_df['user_id'].value_counts()).reset_index()
user_id_freq.columns = ['user_id','freq']
user_id_freq


def percent_75(df):
    return np.percentile(df,75)

def percent_25(df):
    return np.percentile(df,25)

freq_75 = percent_75(user_id_freq['freq'].values)
freq_25 = percent_25(user_id_freq['freq'].values)

user_id_freq['most_freq'] = user_id_freq['freq'].apply(lambda x: 1 if x>=freq_75 else 0)
user_id_freq['median_freq'] = user_id_freq['freq'].apply(lambda x: 1 if freq_25<=x<freq_75 else 0 )
user_id_freq['lower_freq'] = user_id_freq['freq'].apply(lambda x: 1 if x<freq_25 else 0 )
user_id_freq

repay_log_df = repay_log_df.merge(user_id_freq,on=['user_id'],how='left')

df = df.merge(user_id_freq,on=['user_id'],how='left')

df['freq_early_repay_days_mean'] = 0
df['freq_early_repay_days_std'] = 0

# 不同频率层次打标
most_fre_mean = repay_log_df[repay_log_df['most_freq']==1]['early_repay_days'].mean()
most_fre_var = repay_log_df[repay_log_df['most_freq']==1]['early_repay_days'].var()

median_fre_mean = repay_log_df[repay_log_df['median_freq']==1]['early_repay_days'].mean()
median_fre_var = repay_log_df[repay_log_df['median_freq']==1]['early_repay_days'].var()

lower_fre_mean = repay_log_df[repay_log_df['lower_freq']==1]['early_repay_days'].mean()
lower_fre_var = repay_log_df[repay_log_df['lower_freq']==1]['early_repay_days'].var()


df['freq_early_repay_days_mean'].loc[df['most_freq']==1] = most_fre_mean
df['freq_early_repay_days_mean'].loc[df['median_freq']==1] = median_fre_mean
df['freq_early_repay_days_mean'].loc[df['lower_freq']==1] = lower_fre_mean

df['freq_early_repay_days_var'].loc[df['most_freq']==1] = most_fre_var
df['freq_early_repay_days_var'].loc[df['median_freq']==1] = median_fre_var
df['freq_early_repay_days_var'].loc[df['lower_freq']==1] = lower_fre_var

## 训练-测试的新用户打标
df['new_user'] = df['user_id'].apply(lambda x:0 if x in old_user_list else 1)

# # 不同类型用户的不同逾期频繁次
# repay_log_df['freq_early_repay_days_mean'] = 0
# repay_log_df['freq_early_repay_days_std'] = 0


# repay_log_df_1 = repay_log_df[repay_log_df['most_freq']==1].groupby(['repay_amt'])['early_repay_days'].agg({
#     'most_freq_early_repay_days_mean': 'mean', 
#     'most_freq_early_repay_days_std': 'std'
# }).reset_index()

# repay_log_df[repay_log_df['most_freq']==1]['freq_early_repay_days_mean'] = repay_log_df_1['most_freq_early_repay_days_mean']
# repay_log_df[repay_log_df['most_freq']==1]['freq_early_repay_days_std'] = repay_log_df_1['most_freq_early_repay_days_std']
# repay_log_df

# repay_log_df_1 = repay_log_df[repay_log_df['median_freq']==1].groupby(['repay_amt'])['early_repay_days'].agg({
#     'median_freq_early_repay_days_mean': 'mean', 
#     'median_freq_early_repay_days_std': 'std'
# }).reset_index()

# repay_log_df[repay_log_df['median_freq']==1]['freq_early_repay_days_mean'] = repay_log_df_1['median_freq_early_repay_days_mean']
# repay_log_df[repay_log_df['median_freq']==1]['freq_early_repay_days_std'] = repay_log_df_1['median_freq_early_repay_days_std']

# repay_log_df_1 = repay_log_df[repay_log_df['lower_freq']==1].groupby(['repay_amt'])['early_repay_days'].agg({
#     'lower_freq_early_repay_days_mean': 'mean', 
#     'lower_freq_early_repay_days_std': 'std'
# }).reset_index()
# repay_log_df_1
# repay_log_df['freq_early_repay_days_mean'].loc[repay_log_df['lower_freq']==1] = repay_log_df_1['lower_freq_early_repay_days_mean']
# repay_log_df['freq_early_repay_days_std'].loc[repay_log_df['lower_freq']==1] = repay_log_df_1['lower_freq_early_repay_days_std']

# ## user_id,
# repay_log_df_1 = repay_log_df.groupby(['user_id'])['freq_early_repay_days_mean'].agg({
#     'freq_early_repay_days': 'mean', 
# }).reset_index()
# repay_log_df_1 = repay_log_df_1.fillna(0)
# repay_log_df_1

# df = df.merge(repay_log_df_1,on=['user_id','repay_amt'],how='left')

