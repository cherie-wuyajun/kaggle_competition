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

# 获得训练数据并编码时间，超时这还款金额N设为0，逾期标签为32.
train_df = pd.read_csv(path+'train.csv', parse_dates=['auditing_date', 'due_date', 'repay_date'])
train_df['repay_date'] = train_df[['due_date', 'repay_date']].apply(lambda x: x['repay_date'] \
                         if x['repay_date'] != '\\N' else x['due_date']+datetime.timedelta(days=1), axis=1)
train_df['repay_amt'] = train_df['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')
train_df['label'] = (train_df['due_date'] - train_df['repay_date']).dt.days

# 设定还款日期为创建日期的标签为32
train_df.loc[train_df['auditing_date'] == train_df['repay_date'], 'label'] = 31

train_df.loc[train_df['repay_amt'] == 0, 'label'] = 32

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

list_count=listing_info_df['user_id'].value_counts().reset_index()
list_count.columns = ['user_id','times']
list_count

list_pricipal = listing_info_df.groupby('user_id')['principal'].\
                                agg({'mean_principal':'mean','std_principal':'std'}).reset_index()
list_pricipal = list_pricipal.fillna(0)

df = df.merge(list_pricipal,on=['user_id'],how='left')


listing_info_df['auditing_date_month'] = pd.to_datetime(listing_info_df['auditing_date']).dt.month
listing_info_df['auditing_date_days'] = listing_info_df['auditing_date_month'].copy()
# print(sub_example['due_date'])
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==1] = 31
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==3] = 31
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==5] = 31
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==7] = 31
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==8] = 31
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==10] = 31
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==12] = 31
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==4] = 30
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==6] = 30
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==9] = 30
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==11] = 30
listing_info_df['auditing_date_days'][listing_info_df['auditing_date_days']==2] = 28

listing_info_df['principal_per_term'] = listing_info_df['principal']/listing_info_df['term']
list_count=listing_info_df['user_id'].value_counts()
df_count_times=pd.DataFrame(list_count)
df_count_times.rename(columns={ df_count_times.columns[0]: "loan_times" },inplace=True)
df_count_times['user_id']=df_count_times.index
bin_size=(1,2,4,8)
def count_to_style(count):
    n=1
    num=1
    while count>n and num<5:
        n=n*2
        num=num+1
    return num
df_count_times['loan_times_type']=df_count_times['loan_times'].apply(count_to_style)
del df_count_times['loan_times']


list_principle=listing_info_df['principal'].value_counts()
principal_df=pd.DataFrame(list_principle)
principal_df.rename(columns={principal_df.columns[0]: "frequency"},inplace=True)
principal_df['principal']=principal_df.index
principal_df.reset_index(drop=True, inplace=True)
def ChangeToClass(principal):
    if principal<=1170:
        return 6
    elif 1170<principal<=2260:
        return 5
    elif 2260<principal<=3350:
        return 4
    elif 3350<principal<=3910:
        return 3
    elif 3910<principal<=5540:
        return 2
    else:
        return 1
principal_df['principal_class']=principal_df['principal'].apply(ChangeToClass)
del principal_df['frequency']

listing_info_df=listing_info_df.merge(principal_df, on="principal", how="left")

listing_info_df['TermRate']=listing_info_df['rate']/listing_info_df['term']

print(listing_info_df.head())
# del listing_info_df['user_id'], listing_info_df['auditing_date']
df = df.merge(listing_info_df, on=['user_id','listing_id','auditing_date'], how='left')

# 表中有少数user不止一条记录，因此按日期排序，去重，只保留最新的一条记录。
user_info = pd.read_csv(path+'user_info.csv', parse_dates=['reg_mon', 'insertdate'])

user_info.rename(columns={'insertdate': 'info_insert_date'}, inplace=True)

user_info = user_info.sort_values(by='info_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)

user_info['gender']=user_info['gender'].apply(lambda x:1 if x.strip()=='男' else 0)

user_info['id_city'] = user_info['id_city'].apply(lambda x: x.replace('\\N', 'c0'))
def compare(a,b):
    if a==b: return 1
    else: return 0
user_info['remote_boolean']=user_info.apply(lambda user_info:compare(user_info['cell_province'],user_info['id_province']),axis=1)

city_map=user_info.groupby('id_city').agg({'id_city':'count'})
city_map[city_map['id_city']>11000]
city_map.columns=['count']
city_map.reset_index(inplace=True)
x = city_map[city_map["count"] < 11000]; 
max_count = city_map["count"].max()
min_count = city_map["count"].min()

bin_size = int((max_count - min_count) / 5); 
def count_to_class(count): 
    return min(6, int((count - min_count) / bin_size))+1
city_map['class']=city_map['count'].apply(count_to_class)
city_map["class"][0]=0
city_map = city_map[["id_city", "class"]]
user_info=user_info.merge(city_map, on="id_city", how="left")

user_info=user_info.merge(df_count_times, on="user_id", how="left")  # 合并贷款次数到user_info里面

df = df.merge(user_info, on='user_id', how='left')

# 同上
user_tag_df = pd.read_csv(path+'user_taglist.csv', parse_dates=['insertdate'])
user_tag_df.rename(columns={'insertdate': 'tag_insert_date'}, inplace=True)
user_tag_df = user_tag_df.sort_values(by='tag_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)

df = df.merge(user_tag_df, on='user_id', how='left')



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
repay_log_df_1 = repay_log_df.groupby(['user_id','rate'])['early_repay_days'].agg({
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


## repay_all 特征
repay_log_df_all = pd.read_csv(path+'user_repay_logs.csv', parse_dates=['due_date', 'repay_date'])
repay_log_df_all['repay_all_order'] = repay_log_df_all['repay_date'].astype('str').apply(lambda x: 1 if x != '2200-01-01' else 0)
repay_log_df_all['early_repay_days_all_order'] = (repay_log_df_all['due_date'] - repay_log_df_all['repay_date']).dt.days
repay_log_df_all['early_repay_days_all_order'] = repay_log_df_all['early_repay_days_all_order'].apply(lambda x: x if x >= 0 else -1)
for f in ['listing_id', 'order_id', 'due_date', 'repay_date', 'repay_amt']:
    del repay_log_df_all[f]

group = repay_log_df_all.groupby('user_id', as_index=False)
repay_log_df_all = repay_log_df_all.merge(group['repay_all_order'].agg({'repay_mean_all_order': 'mean'}), on='user_id', how='left')

repay_log_df_all = repay_log_df_all.merge(
    group['early_repay_days_all_order'].agg({
        'early_repay_days_max_all_order': 'max', 'early_repay_days_median_all_order': 'median', 'early_repay_days_sum_all_order': 'sum',
        'early_repay_days_mean_all_order': 'mean', 'early_repay_days_std_all_order': 'std'
    }), on='user_id', how='left'
)

repay_log_df_all = repay_log_df_all.merge(
    group['due_amt'].agg({
        'due_amt_max_all_order': 'max', 'due_amt_min_all_order': 'min', 'due_amt_median_all_order': 'median',
        'due_amt_mean_all_order': 'mean', 'due_amt_sum_all_order': 'sum', 'due_amt_std_all_order': 'std',
        'due_amt_skew_all_order': 'skew', 'due_amt_kurt_all_order': kurtosis, 'due_amt_ptp_all_order': np.ptp
    }), on='user_id', how='left'
)
del repay_log_df_all['repay_all_order'], repay_log_df_all['early_repay_days_all_order'], repay_log_df_all['due_amt']
repay_log_df_all = repay_log_df_all.drop_duplicates('user_id').reset_index(drop=True)
df = df.merge(repay_log_df_all, on='user_id', how='left')


cate_cols = ['cell_province', 'id_province', 'id_city']
for f in cate_cols:
    df[f] = df[f].map(dict(zip(df[f].unique(), range(df[f].nunique())))).astype('int32')

df['due_amt_per_days'] = df['due_amt'] / (train_df['due_date'] - train_df['auditing_date']).dt.days
date_cols = ['auditing_date', 'due_date', 'reg_mon', 'info_insert_date', 'tag_insert_date']

for f in date_cols:
    if f in ['reg_mon', 'info_insert_date', 'tag_insert_date']:
        df[f + '_year'] = df[f].dt.year
    df[f + '_month'] = df[f].dt.month
    if f in ['auditing_date', 'due_date', 'info_insert_date', 'tag_insert_date']:
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek

df.drop(columns=date_cols, axis=1, inplace=True)

# df['taglist'] = df['taglist'].astype('str').apply(lambda x: x.strip().replace('|', ' ').strip())
# tag_cv = CountVectorizer(min_df=10, max_df=0.9).fit_transform(df['taglist'])

del df['user_id'], df['listing_id'], df['taglist']
df_count_times
df = pd.get_dummies(df, columns=cate_cols)
# df = sparse.hstack((df.values, tag_cv), format='csr', dtype='float32')
# train_values, test_values = df[:train_num].values, df[train_num:].values
# print(train_values.shape)




############
## Train
############

train_values, test_values = df[:train_num].values, df[train_num:].values
# 五折验证也可以改成一次验证，按时间划分训练集和验证集，以避免由于时序引起的数据穿越问题。
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'metric_freq': 5,
    'num_class': 33,
    'num_iterations': 600,
    'num_leaves': 31,
    'learning_rate': 0.05,
#     'device_type':'gpu'
}
amt_oof = np.zeros(train_num)
prob_oof = np.zeros((train_num, 33))
test_pred_prob = np.zeros((test_values.shape[0], 33))


for i, (trn_idx, val_idx) in enumerate(skf.split(train_values, clf_labels)):
    print(i, 'fold...')
    t = time.time()
    trn_x, trn_y = train_values[trn_idx], clf_labels[trn_idx]
    val_x, val_y = train_values[val_idx], clf_labels[val_idx]
    val_repay_amt = amt_labels[val_idx]
    val_due_amt = train_due_amt_df.iloc[val_idx]
   
    train = lgb.Dataset(trn_x, label=trn_y)
    val = lgb.Dataset(val_x, label=val_y, reference=train)
    gbm = lgb.train(params, train, valid_sets=val, early_stopping_rounds=10)

    # shepe = (-1, 33)
    val_pred_prob_everyday = gbm.predict(val_x, num_iteration=gbm.best_iteration)
    prob_oof[val_idx] = val_pred_prob_everyday
    val_pred_prob_today = [val_pred_prob_everyday[i][val_y[i]] for i in range(val_pred_prob_everyday.shape[0])]
    val_pred_repay_amt = val_due_amt['due_amt'].values * val_pred_prob_today
    print('val rmse:', np.sqrt(mean_squared_error(val_repay_amt, val_pred_repay_amt)))
    print('val mae:', mean_absolute_error(val_repay_amt, val_pred_repay_amt))
    amt_oof[val_idx] = val_pred_repay_amt
    test_pred_prob += gbm.predict(test_values, num_iteration=gbm.best_iteration) / skf.n_splits
    print('runtime: {}\n'.format(time.time() - t))

print('\ncv rmse:', np.sqrt(mean_squared_error(amt_labels, amt_oof)))
print('cv mae:', mean_absolute_error(amt_labels, amt_oof))
print('cv logloss:', log_loss(clf_labels, prob_oof))
print('cv acc:', accuracy_score(clf_labels, np.argmax(prob_oof, axis=1)))

prob_cols = ['prob_{}'.format(i) for i in range(33)]

for i, f in enumerate(prob_cols):
    sub[f] = test_pred_prob[:, i]

sub_example = pd.read_csv(path+'submission.csv', parse_dates=['repay_date'])

sub_example = sub_example.merge(sub, on='listing_id', how='left')
# sub_example['due_date'] = pd.to_datetime((sub_example['auditing_date'] + np.timedelta64(1, 'M') + np.timedelta64(1, 'D')).dt.date)
sub_example['due_date'] = sub_example['auditing_date'].copy()
# print(sub_example['due_date'])
sub_example['due_date'][sub_example['due_date'].dt.month == 4] = pd.to_datetime((sub_example['due_date'][sub_example['due_date'].dt.month == 4] + np.timedelta64(1, 'M')).dt.date)
sub_example['due_date'][sub_example['due_date'].dt.month == 3] = pd.to_datetime((sub_example['due_date'][sub_example['due_date'].dt.month == 3] + np.timedelta64(1, 'M') + np.timedelta64(1, 'D')).dt.date)
sub_example['due_date'][sub_example['due_date'].dt.month == 2] = pd.to_datetime((sub_example['due_date'][sub_example['due_date'].dt.month == 2] + np.timedelta64(1, 'M') - np.timedelta64(2, 'D')).dt.date)

sub_example['days'] = (sub_example['due_date'] - sub_example['repay_date']).dt.days

# shape = (-1, 33)
test_prob = sub_example[prob_cols].values
test_labels = sub_example['days'].values
test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]
sub_example['repay_amt'] = sub_example['due_amt'] * test_prob
sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('sub_cross_5_end.csv', index=False)
