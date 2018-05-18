#!/usr/bin/env python
# -*-coding:utf-8-*-

'''

'''

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from datetime import *
import matplotlib.pylab as pylab
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.preprocessing import *

# 拆分多维度拼接的字段
def splitMultiFea(df):
    tempDf = df.drop_duplicates(subset=['item_id'])[['item_id','item_category_list','item_property_list']]
    tempDf['item_category_list_str'] = tempDf['item_category_list'].values
    tempDf['item_property_list_str'] = tempDf['item_property_list'].values
    tempDf['item_category_list'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x.split(';'))
    tempDf['item_category0'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[0])
    tempDf['item_category1'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[1] if len(x)>1 else np.nan)
    tempDf['item_category2'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[2] if len(x)>2 else np.nan)
    tempDf['item_property_list'] = tempDf[tempDf.item_property_list.notnull()]['item_property_list'].map(lambda x: x.split(';'))
    df = df.drop(['item_category_list','item_property_list'], axis=1).merge(tempDf, how='left', on='item_id')
    df['item_prop_num'] = df['item_property_list'].dropna().map(lambda x: len(x))
    df['predict_category_property_str'] = df['predict_category_property'].values
    df['predict_category_property'] = df[df.predict_category_property.notnull()]['predict_category_property'].map(
        lambda x: {kv.split(':')[0]:((kv.split(':')[1].split(',') if kv.split(':')[1]!='-1' else []) if len(kv.split(':')) >= 2 else []) for kv in x.split(';')})
    return df

# 添加广告商品与查询词的相关性特征
def addContextFea(df):
    df['predict_category'] = df['predict_category_property'].dropna().map(lambda x: list(x.keys()))
    df['predict_cate_num'] = df['predict_category'].dropna().map(lambda x: len(x))
    idx = df[df.predict_category_property.notnull()].index
    df.loc[idx,'cate_intersect_num'] = list(map(lambda x: len(np.intersect1d(x[0],x[1])), df.loc[idx, ['item_category_list','predict_category']].values))
    df['predict_property'] = [set() for i in range(len(df))]
    idx = df[(df.item_category2.notnull())&(df.predict_category_property.notnull())].index
    df.loc[idx,'predict_property'] = list(map(lambda x: x[2]|set(x[1][x[0]]) if (x[0] in x[1].keys()) else x[2], df.loc[idx,['item_category2','predict_category_property','predict_property']].values))
    idx = df[(df.item_category1.notnull())&(df.predict_category_property.notnull())].index
    df.loc[idx,'predict_property'] = list(map(lambda x: x[2]|set(x[1][x[0]]) if (x[0] in x[1].keys()) else x[2], df.loc[idx,['item_category1','predict_category_property','predict_property']].values))
    df['predict_property'] = df['predict_property'].map(lambda x: np.nan if len(x)==0 else list(x))
    df['predict_prop_num'] = df[df.predict_property.notnull()]['predict_property'].map(lambda x: len(x))
    idx = df[(df.predict_property.notnull())&(df.item_property_list.notnull())].index
    df.loc[idx, 'prop_intersect_num'] = list(map(lambda x: len(np.intersect1d(x[0],x[1])), df.loc[idx, ['item_property_list','predict_property']].values))
    df.loc[idx,'prop_union_num'] = list(map(lambda x: len(np.union1d(x[0],x[1])), df.loc[idx, ['item_property_list','predict_property']].values))
    df['prop_jaccard'] = df['prop_intersect_num'] / df['prop_union_num']
    df['prop_predict_ratio'] = df['prop_intersect_num'] / df['predict_prop_num']
    df['prop_item_ratio'] = df['prop_intersect_num'] / df['item_prop_num']
    df.fillna({k:-1 for k in ['predict_prop_num','prop_intersect_num','prop_union_num','prop_jaccard','prop_predict_ratio','prop_item_ratio']}, inplace=True)
    return df

def addSet(setList):
    allSet = []
    for member in setList:
        allSet = allSet + member
    allSet = set(allSet)
    return allSet

#处理跟商品类目和属性相关的特征
def getCategoryFuture(df):
    df['predict_category_list'] = df['predict_category_property'].map(lambda x: [] if x == np.nan else list(kv.split(':')[0] for kv in str(x).split(';')))
    df['predict_category_set'] = df['predict_category_list'].map(lambda x: set(x))
    df['real_item_category_list'] = df['item_category_list'].map(lambda x: set(x))

    df['predict_property_list'] = df['predict_category_property'].map(lambda x: [] if x == np.nan else ((kv.split(':')[1].split(',') if kv.split(':')[1]!='-1' else []) if len(kv.split(':')) >= 2 else [] for kv in str(x).split(';')))
    df['predict_property_list'] = df['predict_property_list'].map(lambda x: addSet(x))
    df['item_property_list'] = df['item_property_list'].map(lambda x: set(x))
    return df

def getMatchProportion(df):
    match_category_proportion = []
    match_property_proportion = []
    for x,y,m,n in df[['real_item_category_list', 'predict_category_set', 'item_property_list', 'predict_property_list']].values:
        match_category = x & y
        match_property = m & n
        if len(y) > 0:
            category_proportion = len(match_category) / len(y)
            match_category_proportion.append(category_proportion)
        else:
            match_category_proportion.append(0)
        if len(n) > 0:
            property_proportion = len(match_property) / len(n)
            match_property_proportion.append(property_proportion)
        else:
            match_property_proportion.append(0)
    df['match_category_proportion'] = match_category_proportion
    df['match_property_proportion'] = match_property_proportion
    return df

#构造跟预测数目相关的特征
def getPredictNumber(df):
    df['predict_category_number'] = df['predict_category_set'].map(lambda x: len(x))
    df['predict_property_number'] = df['predict_property_list'].map(lambda x: len(x))
    return df

#构造跟类目预测精确性相关的特征
def getPredictAccuracy(df):
    isFirstCategoryIn = []
    isLastCategoryIn = []
    for x,y in df[['real_item_category_list', 'predict_category_list']].values:
        if y[0] in x:
            isFirstCategoryIn.append(1)
        else:
            isFirstCategoryIn.append(0)
        if y[len(y)-1] in x:
            isLastCategoryIn.append(1)
        else:
            isLastCategoryIn.append(0)
    df['isFirstCategoryIn'] = isFirstCategoryIn
    df['isLastCategoryIn'] = isLastCategoryIn
    return df

#添加商品属性个数以及类目个数特征
def getCPNumber(df):
    df['category_number'] = df['item_category_list'].map(lambda x: len(x))
    df['property_number'] = df['item_property_list'].map(lambda x: len(x))
    return df

def getBayesSmoothParam(origion_rate):
    origion_rate_mean = origion_rate.mean()
    origion_rate_var = origion_rate.var()
    alpha = origion_rate_mean / origion_rate_var * (origion_rate_mean * (1 - origion_rate_mean) - origion_rate_var)
    beta = (1 - origion_rate_mean) / origion_rate_var * (origion_rate_mean * (1 - origion_rate_mean) - origion_rate_var)
    print('origion_rate_mean : ', origion_rate_mean)
    print('origion_rate_var : ', origion_rate_var)
    print('alpha : ', alpha)
    print('beta : ', beta)
    return alpha, beta

# 缩放字段至0-1
def scalerFea(df, cols):
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[[cols]].values)
    return df,scaler

def getHistoryInfoByCol(train_df_normal, train_df_1, train_df_2, train_df_3, test_df, colName):
    train_df_pivot_table_all = pd.pivot_table(train_df_normal[['instance_id', colName]], index=[colName], values=['instance_id'], aggfunc=len)
    train_df_pivot_table_all.reset_index(inplace=True)
    train_df_pivot_table_all.rename(columns={'instance_id' : 'all_' + colName + '_click_number'}, inplace=True)
#     print(train_df_pivot_table_all.head(10))

    train_df_pivot_table_buy = pd.pivot_table(train_df_normal[['instance_id', colName]][train_df_normal.is_trade == 1], index=[colName], values=['instance_id'], aggfunc=len)
    train_df_pivot_table_buy.reset_index(inplace=True)
    train_df_pivot_table_buy.rename(columns={'instance_id' : 'all_' + colName + '_buy_number'}, inplace=True)
#     print(train_df_pivot_table_buy.head(10))

    train_df_pivot_table = pd.merge(train_df_pivot_table_all, train_df_pivot_table_buy, on=[colName], how='left')
    train_df_pivot_table['all_' + colName + '_buy_number'] = train_df_pivot_table['all_' + colName + '_buy_number'].fillna(0)
    train_df_pivot_table['all_' + colName + '_buy_number'][train_df_pivot_table[colName] == -1] = train_df_pivot_table['all_' + colName + '_buy_number'][train_df_pivot_table[colName] == -1] / len(train_df_pivot_table)
    train_df_pivot_table['all_' + colName + '_click_number'][train_df_pivot_table[colName] == -1] = train_df_pivot_table['all_' + colName + '_click_number'][train_df_pivot_table[colName] == -1] / len(train_df_pivot_table)
#     print(train_df_pivot_table['all_' + colName + '_click_number'][train_df_pivot_table.item_brand_id == -1])
    train_df_pivot_table['history_' + colName + '_rate'] = train_df_pivot_table['all_' + colName + '_buy_number'] / train_df_pivot_table['all_' + colName + '_click_number']
    alpha, beta = getBayesSmoothParam(train_df_pivot_table['history_' + colName + '_rate'])
    train_df_pivot_table['history_' + colName + '_smooth_rate'] = (train_df_pivot_table['all_' + colName + '_buy_number'] + alpha) / (train_df_pivot_table['all_' + colName + '_click_number'] + alpha + beta)

    train_df_pivot_table, all_buy_number_scaler = scalerFea(train_df_pivot_table, 'all_' + colName + '_buy_number')
    train_df_pivot_table, all_click_number_scaler = scalerFea(train_df_pivot_table, 'all_' + colName + '_click_number')
#     print(train_df_pivot_table.head(10))
#     print(train_df_pivot_table.columns.values)

    train_df_1 = pd.merge(train_df_1, train_df_pivot_table, on=[colName], how='left')
    train_df_2 = pd.merge(train_df_2, train_df_pivot_table, on=[colName], how='left')
    train_df_3 = pd.merge(train_df_3, train_df_pivot_table, on=[colName], how='left')
    train_df_3['all_' + colName + '_click_number'] = train_df_3['all_' + colName + '_click_number'].fillna(0)
    train_df_3['all_' + colName + '_buy_number'] = train_df_3['all_' + colName + '_buy_number'].fillna(0)
    train_df_3['history_' + colName + '_smooth_rate'] = train_df_3['history_' + colName + '_smooth_rate'].fillna((alpha / (alpha + beta)))

    test_df = pd.merge(test_df, train_df_pivot_table, on=[colName], how='left')
    test_df['all_' + colName + '_click_number'] = test_df['all_' + colName + '_click_number'].fillna(0)
    test_df['all_' + colName + '_buy_number'] = test_df['all_' + colName + '_buy_number'].fillna(0)
    test_df['history_' + colName + '_smooth_rate'] = test_df['history_' + colName + '_smooth_rate'].fillna((alpha / (alpha + beta)))

    return train_df_1, train_df_2, train_df_3, test_df

# 统计过去一个小时某用户点击某个相同商品的次数
def getOneHourSameItemCount(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'item_id', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['item_id_str'] = train_df_1_copy['item_id'].map(lambda x: str(x))
    train_df_1_copy['user_item_id'] = train_df_1_copy['user_id_str'] + train_df_1_copy['item_id_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['item_id_str'] = train_df_2_copy['item_id'].map(lambda x: str(x))
    train_df_2_copy['user_item_id'] = train_df_2_copy['user_id_str'] + train_df_2_copy['item_id_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['item_id_str'] = train_df_3_copy['item_id'].map(lambda x: str(x))
    train_df_3_copy['user_item_id'] = train_df_3_copy['user_id_str'] + train_df_3_copy['item_id_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['item_id_str'] = test_df_copy['item_id'].map(lambda x: str(x))
    test_df_copy['user_item_id'] = test_df_copy['user_id_str'] + test_df_copy['item_id_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_item_id','date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_item'] = tempDf['user_item_id'].shift(1)
    tempDf['last_user_item'] = tempDf['last_user_item']==tempDf['user_item_id']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_item, 'last_show_time'] = np.nan
    tempDf['ui_last_show_timedelta'] = tempDf['date'] - tempDf['last_show_time']
    tempDf['ui_last_show_timedelta'] = tempDf['ui_last_show_timedelta'].dt.seconds
    tempDf['ui_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['last_user_item','date','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['lastOneHour_sameItem_count'] = hourShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_item_id', 'date', 'lastOneHour_sameItem_count']], how='left', on=['user_item_id', 'date'])
    train_df_1['lastOneHour_sameItem_count'] = train_df_1_copy['lastOneHour_sameItem_count']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_item_id', 'date', 'lastOneHour_sameItem_count']], how='left', on=['user_item_id', 'date'])
    train_df_2['lastOneHour_sameItem_count'] = train_df_2_copy['lastOneHour_sameItem_count']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_item_id', 'date', 'lastOneHour_sameItem_count']], how='left', on=['user_item_id', 'date'])
    train_df_3['lastOneHour_sameItem_count'] = train_df_3_copy['lastOneHour_sameItem_count']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_item_id', 'date', 'lastOneHour_sameItem_count']], on = ['user_item_id', 'date'], how='left')
    test_df['lastOneHour_sameItem_count'] = test_df_copy['lastOneHour_sameItem_count']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计过去一个小时某用户点击同种根类目商品的次数
def getOneHourSameFirstCategoryCount(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'real_first_category', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'real_first_category', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'real_first_category', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'real_first_category', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['real_first_category_str'] = train_df_1_copy['real_first_category'].map(lambda x: str(x))
    train_df_1_copy['user_real_first_category'] = train_df_1_copy['user_id_str'] + train_df_1_copy['real_first_category_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['real_first_category_str'] = train_df_2_copy['real_first_category'].map(lambda x: str(x))
    train_df_2_copy['user_real_first_category'] = train_df_2_copy['user_id_str'] + train_df_2_copy['real_first_category_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['real_first_category_str'] = train_df_3_copy['real_first_category'].map(lambda x: str(x))
    train_df_3_copy['user_real_first_category'] = train_df_3_copy['user_id_str'] + train_df_3_copy['real_first_category_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['real_first_category_str'] = test_df_copy['real_first_category'].map(lambda x: str(x))
    test_df_copy['user_real_first_category'] = test_df_copy['user_id_str'] + test_df_copy['real_first_category_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_real_first_category','date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_real_first_category'] = tempDf['user_real_first_category'].shift(1)
    tempDf['last_user_real_first_category'] = tempDf['last_user_real_first_category']==tempDf['user_real_first_category']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_real_first_category, 'last_show_time'] = np.nan
    tempDf['ui_last_show_timedelta'] = tempDf['date'] - tempDf['last_show_time']
    tempDf['ui_last_show_timedelta'] = tempDf['ui_last_show_timedelta'].dt.seconds
    tempDf['ui_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['last_user_real_first_category','date','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['lastOneHour_sameFirstCategory_count'] = hourShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_real_first_category', 'date', 'lastOneHour_sameFirstCategory_count']], how='left', on=['user_real_first_category', 'date'])
    train_df_1['lastOneHour_sameFirstCategory_count'] = train_df_1_copy['lastOneHour_sameFirstCategory_count']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_real_first_category', 'date', 'lastOneHour_sameFirstCategory_count']], how='left', on=['user_real_first_category', 'date'])
    train_df_2['lastOneHour_sameFirstCategory_count'] = train_df_2_copy['lastOneHour_sameFirstCategory_count']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_real_first_category', 'date', 'lastOneHour_sameFirstCategory_count']], how='left', on=['user_real_first_category', 'date'])
    train_df_3['lastOneHour_sameFirstCategory_count'] = train_df_3_copy['lastOneHour_sameFirstCategory_count']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_real_first_category', 'date', 'lastOneHour_sameFirstCategory_count']], on = ['user_real_first_category', 'date'], how='left')
    test_df['lastOneHour_sameFirstCategory_count'] = test_df_copy['lastOneHour_sameFirstCategory_count']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计过去一个小时某用户点击同种叶子类目商品的次数
def getOneHourSameLastCategoryCount(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'real_last_category', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['real_last_category_str'] = train_df_1_copy['real_last_category'].map(lambda x: str(x))
    train_df_1_copy['user_real_last_category'] = train_df_1_copy['user_id_str'] + train_df_1_copy['real_last_category_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['real_last_category_str'] = train_df_2_copy['real_last_category'].map(lambda x: str(x))
    train_df_2_copy['user_real_last_category'] = train_df_2_copy['user_id_str'] + train_df_2_copy['real_last_category_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['real_last_category_str'] = train_df_3_copy['real_last_category'].map(lambda x: str(x))
    train_df_3_copy['user_real_last_category'] = train_df_3_copy['user_id_str'] + train_df_3_copy['real_last_category_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['real_last_category_str'] = test_df_copy['real_last_category'].map(lambda x: str(x))
    test_df_copy['user_real_last_category'] = test_df_copy['user_id_str'] + test_df_copy['real_last_category_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_real_last_category','date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_real_last_category'] = tempDf['user_real_last_category'].shift(1)
    tempDf['last_user_real_last_category'] = tempDf['last_user_real_last_category']==tempDf['user_real_last_category']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_real_last_category, 'last_show_time'] = np.nan
    tempDf['ui_last_show_timedelta'] = tempDf['date'] - tempDf['last_show_time']
    tempDf['ui_last_show_timedelta'] = tempDf['ui_last_show_timedelta'].dt.seconds
    tempDf['ui_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['last_user_real_last_category','date','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['lastOneHour_sameLastCategory_count'] = hourShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_real_last_category', 'date', 'lastOneHour_sameLastCategory_count']], how='left', on=['user_real_last_category', 'date'])
    train_df_1['lastOneHour_sameLastCategory_count'] = train_df_1_copy['lastOneHour_sameLastCategory_count']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_real_last_category', 'date', 'lastOneHour_sameLastCategory_count']], how='left', on=['user_real_last_category', 'date'])
    train_df_2['lastOneHour_sameLastCategory_count'] = train_df_2_copy['lastOneHour_sameLastCategory_count']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_real_last_category', 'date', 'lastOneHour_sameLastCategory_count']], how='left', on=['user_real_last_category', 'date'])
    train_df_3['lastOneHour_sameLastCategory_count'] = train_df_3_copy['lastOneHour_sameLastCategory_count']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_real_last_category', 'date', 'lastOneHour_sameLastCategory_count']], on = ['user_real_last_category', 'date'], how='left')
    test_df['lastOneHour_sameLastCategory_count'] = test_df_copy['lastOneHour_sameLastCategory_count']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计过去一个小时某用户点击同种品牌商品的次数
def getOneHourSameBrandCount(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'item_brand_id', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'item_brand_id', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'item_brand_id', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'item_brand_id', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['brand_id_str'] = train_df_1_copy['item_brand_id'].map(lambda x: str(x))
    train_df_1_copy['user_brand_id'] = train_df_1_copy['user_id_str'] + train_df_1_copy['brand_id_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['brand_id_str'] = train_df_2_copy['item_brand_id'].map(lambda x: str(x))
    train_df_2_copy['user_brand_id'] = train_df_2_copy['user_id_str'] + train_df_2_copy['brand_id_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['brand_id_str'] = train_df_3_copy['item_brand_id'].map(lambda x: str(x))
    train_df_3_copy['user_brand_id'] = train_df_3_copy['user_id_str'] + train_df_3_copy['brand_id_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['brand_id_str'] = test_df_copy['item_brand_id'].map(lambda x: str(x))
    test_df_copy['user_brand_id'] = test_df_copy['user_id_str'] + test_df_copy['brand_id_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_brand_id','date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_brand_id'] = tempDf['user_brand_id'].shift(1)
    tempDf['last_user_brand_id'] = tempDf['last_user_brand_id']==tempDf['user_brand_id']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_brand_id, 'last_show_time'] = np.nan
    tempDf['ui_last_show_timedelta'] = tempDf['date'] - tempDf['last_show_time']
    tempDf['ui_last_show_timedelta'] = tempDf['ui_last_show_timedelta'].dt.seconds
    tempDf['ui_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['last_user_brand_id','date','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['lastOneHour_sameBrand_count'] = hourShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_brand_id', 'date', 'lastOneHour_sameBrand_count']], how='left', on=['user_brand_id', 'date'])
    train_df_1['lastOneHour_sameBrand_count'] = train_df_1_copy['lastOneHour_sameBrand_count']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_brand_id', 'date', 'lastOneHour_sameBrand_count']], how='left', on=['user_brand_id', 'date'])
    train_df_2['lastOneHour_sameBrand_count'] = train_df_2_copy['lastOneHour_sameBrand_count']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_brand_id', 'date', 'lastOneHour_sameBrand_count']], how='left', on=['user_brand_id', 'date'])
    train_df_3['lastOneHour_sameBrand_count'] = train_df_3_copy['lastOneHour_sameBrand_count']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_brand_id', 'date', 'lastOneHour_sameBrand_count']], on = ['user_brand_id', 'date'], how='left')
    test_df['lastOneHour_sameBrand_count'] = test_df_copy['lastOneHour_sameBrand_count']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计过去一个小时某用户点击同种店铺商品的次数
def getOneHourSameShopCount(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'shop_id', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'shop_id', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'shop_id', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'shop_id', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['item_shop_id_str'] = train_df_1_copy['shop_id'].map(lambda x: str(x))
    train_df_1_copy['user_item_shop_id'] = train_df_1_copy['user_id_str'] + train_df_1_copy['item_shop_id_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['item_shop_id_str'] = train_df_2_copy['shop_id'].map(lambda x: str(x))
    train_df_2_copy['user_item_shop_id'] = train_df_2_copy['user_id_str'] + train_df_2_copy['item_shop_id_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['item_shop_id_str'] = train_df_3_copy['shop_id'].map(lambda x: str(x))
    train_df_3_copy['user_item_shop_id'] = train_df_3_copy['user_id_str'] + train_df_3_copy['item_shop_id_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['item_shop_id_str'] = test_df_copy['shop_id'].map(lambda x: str(x))
    test_df_copy['user_item_shop_id'] = test_df_copy['user_id_str'] + test_df_copy['item_shop_id_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_item_shop_id','date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_item_shop_id'] = tempDf['user_item_shop_id'].shift(1)
    tempDf['last_user_item_shop_id'] = tempDf['last_user_item_shop_id']==tempDf['user_item_shop_id']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_item_shop_id, 'last_show_time'] = np.nan
    tempDf['ui_last_show_timedelta'] = tempDf['date'] - tempDf['last_show_time']
    tempDf['ui_last_show_timedelta'] = tempDf['ui_last_show_timedelta'].dt.seconds
    tempDf['ui_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['last_user_item_shop_id','date','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['lastOneHour_sameShop_count'] = hourShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_item_shop_id', 'date', 'lastOneHour_sameShop_count']], how='left', on=['user_item_shop_id', 'date'])
    train_df_1['lastOneHour_sameShop_count'] = train_df_1_copy['lastOneHour_sameShop_count']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_item_shop_id', 'date', 'lastOneHour_sameShop_count']], how='left', on=['user_item_shop_id', 'date'])
    train_df_2['lastOneHour_sameShop_count'] = train_df_2_copy['lastOneHour_sameShop_count']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_item_shop_id', 'date', 'lastOneHour_sameShop_count']], how='left', on=['user_item_shop_id', 'date'])
    train_df_3['lastOneHour_sameShop_count'] = train_df_3_copy['lastOneHour_sameShop_count']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_item_shop_id', 'date', 'lastOneHour_sameShop_count']], on = ['user_item_shop_id', 'date'], how='left')
    test_df['lastOneHour_sameShop_count'] = test_df_copy['lastOneHour_sameShop_count']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 获取是否是该用户在这1个小时内第一次点击这个商品的特征
def getIsOneHourFirstClickItem(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'item_id', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['item_id_str'] = train_df_1_copy['item_id'].map(lambda x: str(x))
    train_df_1_copy['user_item_id'] = train_df_1_copy['user_id_str'] + train_df_1_copy['item_id_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['item_id_str'] = train_df_2_copy['item_id'].map(lambda x: str(x))
    train_df_2_copy['user_item_id'] = train_df_2_copy['user_id_str'] + train_df_2_copy['item_id_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['item_id_str'] = train_df_3_copy['item_id'].map(lambda x: str(x))
    train_df_3_copy['user_item_id'] = train_df_3_copy['user_id_str'] + train_df_3_copy['item_id_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['item_id_str'] = test_df_copy['item_id'].map(lambda x: str(x))
    test_df_copy['user_item_id'] = test_df_copy['user_id_str'] + test_df_copy['item_id_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_item_id','date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_item_id'] = tempDf['user_item_id'].shift(1)
    tempDf['last_user_item_id'] = tempDf['last_user_item_id']==tempDf['user_item_id']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_item_id, 'last_show_time'] = np.nan
    tempDf['ui_last_show_timedelta'] = tempDf['date'] - tempDf['last_show_time']
    tempDf['ui_last_show_timedelta'] = tempDf['ui_last_show_timedelta'].dt.seconds
    tempDf['ui_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['last_user_item_id','date','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            if len(hourShowTemp) > 0:
                hourShowList.append(0)
            else:
                hourShowList.append(1)
            hourShowTemp[dt] = show
        else:
            hourShowList.append(1)
            hourShowTemp = {dt:show}
    tempDf['isLastOneHour_firstClickItem'] = hourShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_item_id', 'date', 'isLastOneHour_firstClickItem']], how='left', on=['user_item_id', 'date'])
    train_df_1['isLastOneHour_firstClickItem'] = train_df_1_copy['isLastOneHour_firstClickItem']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_item_id', 'date', 'isLastOneHour_firstClickItem']], how='left', on=['user_item_id', 'date'])
    train_df_2['isLastOneHour_firstClickItem'] = train_df_2_copy['isLastOneHour_firstClickItem']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_item_id', 'date', 'isLastOneHour_firstClickItem']], how='left', on=['user_item_id', 'date'])
    train_df_3['isLastOneHour_firstClickItem'] = train_df_3_copy['isLastOneHour_firstClickItem']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_item_id', 'date', 'isLastOneHour_firstClickItem']], on = ['user_item_id', 'date'], how='left')
    test_df['isLastOneHour_firstClickItem'] = test_df_copy['isLastOneHour_firstClickItem']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计某用户距离上次点击相同商品的时间
def getUserItemLastClickDeltaTime(train_df_1, train_df_2, train_df_3, test_df):

    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'item_id', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['item_id_str'] = train_df_1_copy['item_id'].map(lambda x: str(x))
    train_df_1_copy['user_item_id'] = train_df_1_copy['user_id_str'] + train_df_1_copy['item_id_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['item_id_str'] = train_df_2_copy['item_id'].map(lambda x: str(x))
    train_df_2_copy['user_item_id'] = train_df_2_copy['user_id_str'] + train_df_2_copy['item_id_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['item_id_str'] = train_df_3_copy['item_id'].map(lambda x: str(x))
    train_df_3_copy['user_item_id'] = train_df_3_copy['user_id_str'] + train_df_3_copy['item_id_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['item_id_str'] = test_df_copy['item_id'].map(lambda x: str(x))
    test_df_copy['user_item_id'] = test_df_copy['user_id_str'] + test_df_copy['item_id_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    temp_df['user_id_str'] = temp_df['user_id'].map(lambda x: str(x))
    temp_df['item_id_str'] = temp_df['item_id'].map(lambda x: str(x))
    temp_df['user_item_id'] = temp_df['user_id_str'] + temp_df['item_id_str']
    tempDf = pd.pivot_table(temp_df, index=['user_item_id', 'date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_item_id'] = tempDf['user_item_id'].shift(1)
    tempDf['last_user_item_id'] = tempDf['last_user_item_id']==tempDf['user_item_id']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_item_id, 'last_show_time'] = np.nan
    historyShowList = []
    deltaTime = 99999999
    for same, dt, lastShowTime in tempDf[['last_user_item_id', 'date', 'last_show_time']].values:
        if same:
            deltaTime = (dt - lastShowTime) / np.timedelta64(1, 's')
            historyShowList.append(deltaTime)
            deltaTime = 99999999
        else:
            historyShowList.append(deltaTime)
    tempDf['userItem_lastClickDeltaTime'] = historyShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_item_id', 'date', 'userItem_lastClickDeltaTime']], how='left', on=['user_item_id', 'date'])
    train_df_1['userItem_lastClickDeltaTime'] = train_df_1_copy['userItem_lastClickDeltaTime']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_item_id', 'date', 'userItem_lastClickDeltaTime']], how='left', on=['user_item_id', 'date'])
    train_df_2['userItem_lastClickDeltaTime'] = train_df_2_copy['userItem_lastClickDeltaTime']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_item_id', 'date', 'userItem_lastClickDeltaTime']], how='left', on=['user_item_id', 'date'])
    train_df_3['userItem_lastClickDeltaTime'] = train_df_3_copy['userItem_lastClickDeltaTime']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_item_id', 'date', 'userItem_lastClickDeltaTime']], on = ['user_item_id', 'date'], how='left')
    test_df['userItem_lastClickDeltaTime'] = test_df_copy['userItem_lastClickDeltaTime']
    print(len(train_df_1))

    return train_df_1, train_df_2, train_df_3, test_df

# 统计某用户距离上次点击相同品牌商品的时间
def getUserBrandLastClickDeltaTime(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'item_brand_id', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'item_brand_id', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'item_brand_id', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'item_brand_id', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['brand_id_str'] = train_df_1_copy['item_brand_id'].map(lambda x: str(x))
    train_df_1_copy['user_brand_id'] = train_df_1_copy['user_id_str'] + train_df_1_copy['brand_id_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['brand_id_str'] = train_df_2_copy['item_brand_id'].map(lambda x: str(x))
    train_df_2_copy['user_brand_id'] = train_df_2_copy['user_id_str'] + train_df_2_copy['brand_id_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['brand_id_str'] = train_df_3_copy['item_brand_id'].map(lambda x: str(x))
    train_df_3_copy['user_brand_id'] = train_df_3_copy['user_id_str'] + train_df_3_copy['brand_id_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['brand_id_str'] = test_df_copy['item_brand_id'].map(lambda x: str(x))
    test_df_copy['user_brand_id'] = test_df_copy['user_id_str'] + test_df_copy['brand_id_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_brand_id', 'date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_item_brand_id'] = tempDf['user_brand_id'].shift(1)
    tempDf['last_user_item_brand_id'] = tempDf['last_user_item_brand_id']==tempDf['user_brand_id']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_item_brand_id, 'last_show_time'] = np.nan
    historyShowList = []
    deltaTime = 99999999
    for same, dt, lastShowTime in tempDf[['last_user_item_brand_id', 'date', 'last_show_time']].values:
        if same:
            deltaTime = (dt - lastShowTime) / np.timedelta64(1, 's')
            historyShowList.append(deltaTime)
            deltaTime = 99999999
        else:
            historyShowList.append(deltaTime)
    tempDf['userBrand_lastClickDeltaTime'] = historyShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_brand_id', 'date', 'userBrand_lastClickDeltaTime']], how='left', on=['user_brand_id', 'date'])
    train_df_1['userBrand_lastClickDeltaTime'] = train_df_1_copy['userBrand_lastClickDeltaTime']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_brand_id', 'date', 'userBrand_lastClickDeltaTime']], how='left', on=['user_brand_id', 'date'])
    train_df_2['userBrand_lastClickDeltaTime'] = train_df_2_copy['userBrand_lastClickDeltaTime']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_brand_id', 'date', 'userBrand_lastClickDeltaTime']], how='left', on=['user_brand_id', 'date'])
    train_df_3['userBrand_lastClickDeltaTime'] = train_df_3_copy['userBrand_lastClickDeltaTime']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_brand_id', 'date', 'userBrand_lastClickDeltaTime']], on = ['user_brand_id', 'date'], how='left')
    test_df['userBrand_lastClickDeltaTime'] = test_df_copy['userBrand_lastClickDeltaTime']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计某用户距离上次点击相同店铺的时间
def getUserShopLastClickDeltaTime(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'shop_id', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'shop_id', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'shop_id', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'shop_id', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['item_shop_id_str'] = train_df_1_copy['shop_id'].map(lambda x: str(x))
    train_df_1_copy['user_item_shop_id'] = train_df_1_copy['user_id_str'] + train_df_1_copy['item_shop_id_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['item_shop_id_str'] = train_df_2_copy['shop_id'].map(lambda x: str(x))
    train_df_2_copy['user_item_shop_id'] = train_df_2_copy['user_id_str'] + train_df_2_copy['item_shop_id_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['item_shop_id_str'] = train_df_3_copy['shop_id'].map(lambda x: str(x))
    train_df_3_copy['user_item_shop_id'] = train_df_3_copy['user_id_str'] + train_df_3_copy['item_shop_id_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['item_shop_id_str'] = test_df_copy['shop_id'].map(lambda x: str(x))
    test_df_copy['user_item_shop_id'] = test_df_copy['user_id_str'] + test_df_copy['item_shop_id_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_item_shop_id', 'date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_shop_id'] = tempDf['user_item_shop_id'].shift(1)
    tempDf['last_user_shop_id'] = tempDf['last_user_shop_id']==tempDf['user_item_shop_id']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_shop_id, 'last_show_time'] = np.nan
    historyShowList = []
    deltaTime = 99999999
    for same, dt, lastShowTime in tempDf[['last_user_shop_id', 'date', 'last_show_time']].values:
        if same:
            deltaTime = (dt - lastShowTime) / np.timedelta64(1, 's')
            historyShowList.append(deltaTime)
            deltaTime = 99999999
        else:
            historyShowList.append(deltaTime)
    tempDf['userShop_lastClickDeltaTime'] = historyShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_item_shop_id', 'date', 'userShop_lastClickDeltaTime']], how='left', on=['user_item_shop_id', 'date'])
    train_df_1['userShop_lastClickDeltaTime'] = train_df_1_copy['userShop_lastClickDeltaTime']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_item_shop_id', 'date', 'userShop_lastClickDeltaTime']], how='left', on=['user_item_shop_id', 'date'])
    train_df_2['userShop_lastClickDeltaTime'] = train_df_2_copy['userShop_lastClickDeltaTime']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_item_shop_id', 'date', 'userShop_lastClickDeltaTime']], how='left', on=['user_item_shop_id', 'date'])
    train_df_3['userShop_lastClickDeltaTime'] = train_df_3_copy['userShop_lastClickDeltaTime']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_item_shop_id', 'date', 'userShop_lastClickDeltaTime']], on = ['user_item_shop_id', 'date'], how='left')
    test_df['userShop_lastClickDeltaTime'] = test_df_copy['userShop_lastClickDeltaTime']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计某用户距离上次点击相同根类目的时间
def getUserFirstCategoryLastClickDeltaTime(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'real_first_category', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'real_first_category', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'real_first_category', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'real_first_category', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['real_first_category_str'] = train_df_1_copy['real_first_category'].map(lambda x: str(x))
    train_df_1_copy['user_real_first_category'] = train_df_1_copy['user_id_str'] + train_df_1_copy['real_first_category_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['real_first_category_str'] = train_df_2_copy['real_first_category'].map(lambda x: str(x))
    train_df_2_copy['user_real_first_category'] = train_df_2_copy['user_id_str'] + train_df_2_copy['real_first_category_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['real_first_category_str'] = train_df_3_copy['real_first_category'].map(lambda x: str(x))
    train_df_3_copy['user_real_first_category'] = train_df_3_copy['user_id_str'] + train_df_3_copy['real_first_category_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['real_first_category_str'] = test_df_copy['real_first_category'].map(lambda x: str(x))
    test_df_copy['user_real_first_category'] = test_df_copy['user_id_str'] + test_df_copy['real_first_category_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_real_first_category', 'date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_real_first_category'] = tempDf['user_real_first_category'].shift(1)
    tempDf['last_user_real_first_category'] = tempDf['last_user_real_first_category']==tempDf['user_real_first_category']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_real_first_category, 'last_show_time'] = np.nan
    historyShowList = []
    deltaTime = 99999999
    for same, dt, lastShowTime in tempDf[['last_user_real_first_category', 'date', 'last_show_time']].values:
        if same:
            deltaTime = (dt - lastShowTime) / np.timedelta64(1, 's')
            historyShowList.append(deltaTime)
            deltaTime = 99999999
        else:
            historyShowList.append(deltaTime)
    tempDf['userFirstCategory_lastClickDeltaTime'] = historyShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_real_first_category', 'date', 'userFirstCategory_lastClickDeltaTime']], how='left', on=['user_real_first_category', 'date'])
    train_df_1['userFirstCategory_lastClickDeltaTime'] = train_df_1_copy['userFirstCategory_lastClickDeltaTime']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_real_first_category', 'date', 'userFirstCategory_lastClickDeltaTime']], how='left', on=['user_real_first_category', 'date'])
    train_df_2['userFirstCategory_lastClickDeltaTime'] = train_df_2_copy['userFirstCategory_lastClickDeltaTime']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_real_first_category', 'date', 'userFirstCategory_lastClickDeltaTime']], how='left', on=['user_real_first_category', 'date'])
    train_df_3['userFirstCategory_lastClickDeltaTime'] = train_df_3_copy['userFirstCategory_lastClickDeltaTime']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_real_first_category', 'date', 'userFirstCategory_lastClickDeltaTime']], on = ['user_real_first_category', 'date'], how='left')
    test_df['userFirstCategory_lastClickDeltaTime'] = test_df_copy['userFirstCategory_lastClickDeltaTime']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计某用户距离上次点击相同叶子类目的时间
def getUserLastCategoryLastClickDeltaTime(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'real_last_category', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['real_last_category_str'] = train_df_1_copy['real_last_category'].map(lambda x: str(x))
    train_df_1_copy['user_real_last_category'] = train_df_1_copy['user_id_str'] + train_df_1_copy['real_last_category_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['real_last_category_str'] = train_df_2_copy['real_last_category'].map(lambda x: str(x))
    train_df_2_copy['user_real_last_category'] = train_df_2_copy['user_id_str'] + train_df_2_copy['real_last_category_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['real_last_category_str'] = train_df_3_copy['real_last_category'].map(lambda x: str(x))
    train_df_3_copy['user_real_last_category'] = train_df_3_copy['user_id_str'] + train_df_3_copy['real_last_category_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['real_last_category_str'] = test_df_copy['real_last_category'].map(lambda x: str(x))
    test_df_copy['user_real_last_category'] = test_df_copy['user_id_str'] + test_df_copy['real_last_category_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_real_last_category', 'date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_real_last_category'] = tempDf['user_real_last_category'].shift(1)
    tempDf['last_user_real_last_category'] = tempDf['last_user_real_last_category']==tempDf['user_real_last_category']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_real_last_category, 'last_show_time'] = np.nan
    historyShowList = []
    deltaTime = 99999999
    for same, dt, lastShowTime in tempDf[['last_user_real_last_category', 'date', 'last_show_time']].values:
        if same:
            deltaTime = (dt - lastShowTime) / np.timedelta64(1, 's')
            historyShowList.append(deltaTime)
            deltaTime = 99999999
        else:
            historyShowList.append(deltaTime)
    tempDf['userLastCategory_lastClickDeltaTime'] = historyShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_real_last_category', 'date', 'userLastCategory_lastClickDeltaTime']], how='left', on=['user_real_last_category', 'date'])
    train_df_1['userLastCategory_lastClickDeltaTime'] = train_df_1_copy['userLastCategory_lastClickDeltaTime']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_real_last_category', 'date', 'userLastCategory_lastClickDeltaTime']], how='left', on=['user_real_last_category', 'date'])
    train_df_2['userLastCategory_lastClickDeltaTime'] = train_df_2_copy['userLastCategory_lastClickDeltaTime']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_real_last_category', 'date', 'userLastCategory_lastClickDeltaTime']], how='left', on=['user_real_last_category', 'date'])
    train_df_3['userLastCategory_lastClickDeltaTime'] = train_df_3_copy['userLastCategory_lastClickDeltaTime']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_real_last_category', 'date', 'userLastCategory_lastClickDeltaTime']], on = ['user_real_last_category', 'date'], how='left')
    test_df['userLastCategory_lastClickDeltaTime'] = test_df_copy['userLastCategory_lastClickDeltaTime']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

#定义添加每个小时转化率特征，test数据集采用预测方法填充
def getHourTradeRate(train_df_1, train_df_2, train_df_3, test_df):
    train_df_all = pd.concat([train_df_1, train_df_2, train_df_3])
    train_df_hour_pivot_table_all = pd.pivot_table(train_df_all[['day', 'hour', 'instance_id']], index=['day', 'hour'], values=['instance_id'], aggfunc=len)
    train_df_hour_pivot_table_all.reset_index(inplace=True)
    train_df_hour_pivot_table_all.rename(columns={'instance_id' : 'all_click_number'}, inplace=True)

    train_df_hour_pivot_table_buy = pd.pivot_table(train_df_all[['day', 'hour', 'instance_id']][train_df_all.is_trade == 1], index=['day', 'hour'], values=['instance_id'], aggfunc=len)
    train_df_hour_pivot_table_buy.reset_index(inplace=True)
    train_df_hour_pivot_table_buy.rename(columns={'instance_id' : 'all_buy_number'}, inplace=True)

    train_df_hour_pivot_table_all = pd.merge(train_df_hour_pivot_table_all, train_df_hour_pivot_table_buy, on=['day', 'hour'], how='left')
    train_df_hour_pivot_table_all['rate'] = train_df_hour_pivot_table_all['all_buy_number'] / train_df_hour_pivot_table_all['all_click_number']

    print(train_df_hour_pivot_table_all[train_df_hour_pivot_table_all.day == 7])

    hour_11_normal_mean = train_df_hour_pivot_table_all['rate'][train_df_hour_pivot_table_all.hour == 11].mean()
    hour_11_diff = train_df_hour_pivot_table_all['rate'][(train_df_hour_pivot_table_all.hour == 11) & (train_df_hour_pivot_table_all.day == 7)] - hour_11_normal_mean
    print(hour_11_diff)

    hour_nextHalf_normal_mean = pd.pivot_table(train_df_hour_pivot_table_all[['hour', 'rate']][(train_df_hour_pivot_table_all.hour > 11) & ((train_df_hour_pivot_table_all.day == 31) | (train_df_hour_pivot_table_all.day <= 4))], index=['hour'], values=['rate'], aggfunc=mean)
    hour_nextHalf_normal_mean.reset_index(inplace=True)
    hour_nextHalf_normal_mean['special_rate'] = hour_nextHalf_normal_mean['rate'] + hour_11_diff.values
    print(hour_nextHalf_normal_mean.head)
    print(hour_nextHalf_normal_mean['special_rate'].mean())

    train_df_hour_pivot_table_all.rename(columns={'rate' : 'hour_rate'}, inplace=True)
    hour_nextHalf_normal_mean.rename(columns={'special_rate' : 'hour_rate'}, inplace=True)

    train_df_1 = pd.merge(train_df_1, train_df_hour_pivot_table_all[['day', 'hour', 'hour_rate']], on=['day', 'hour'], how='left')
    train_df_2 = pd.merge(train_df_2, train_df_hour_pivot_table_all[['day', 'hour', 'hour_rate']], on=['day', 'hour'], how='left')
    train_df_3 = pd.merge(train_df_3, train_df_hour_pivot_table_all[['day', 'hour', 'hour_rate']], on=['day', 'hour'], how='left')
    test_df = pd.merge(test_df, hour_nextHalf_normal_mean[['hour', 'hour_rate']], on=['hour'], how='left')
    print(train_df_3[['day', 'hour', 'hour_rate']].head(10))
    print(test_df[['day', 'hour', 'hour_rate']].head(10))
    return train_df_1, train_df_2, train_df_3, test_df

# 定义获取某种店铺，品牌，城市对应商品种类，用户数的函数
def getCorrespondNumber(train_df_1, train_df_2, train_df_3, test_df, colName1, colName2, newColName):

    df = pd.concat([train_df_1[[colName1, colName2, 'instance_id']], train_df_2[[colName1, colName2, 'instance_id']], train_df_3[[colName1, colName2, 'instance_id']], test_df[[colName1, colName2, 'instance_id']]])
    temp_df = df[[colName1, colName2, 'instance_id']]
    tempDf = temp_df.sort_values(by=colName1, ascending=False)
    tempDf['last_' + colName1] = tempDf[colName1].shift(1)
    tempDf['same'] = tempDf['last_' + colName1]==tempDf[colName1]
#     print(tempDf.head(10))
    colName1List = []
    countList = []
    colName2Set = set()
    for same, col2, last_col1 in tempDf[['same', colName2, 'last_' + colName1]].values:
        if same:
            colName2Set.add(col2)
        else:
            colName1List.append(last_col1)
            countList.append(len(colName2Set))
            colName2Set = {col2}
    #处理最后一行数据
    last_col1 = tempDf.iloc[-1][colName1]
    last_count = len(colName2Set)
    colName1List.append(last_col1)
    countList.append(last_count)

    #将结果组合到tempDf中
    result_df = {colName1: colName1List, newColName: countList}
    result_df = DataFrame(result_df)
    result_df = result_df[1:]

    tempDf = tempDf.drop_duplicates([colName1])
    tempDf[newColName] = result_df[newColName].values

    print(len(train_df_1))
    train_df_1 = pd.merge(train_df_1, tempDf[[colName1, newColName]], on=[colName1], how='left')
    print(len(train_df_1))
    train_df_2 = pd.merge(train_df_2, tempDf[[colName1, newColName]], on=[colName1], how='left')
    train_df_3 = pd.merge(train_df_3, tempDf[[colName1, newColName]], on=[colName1], how='left')
    test_df = pd.merge(test_df, tempDf[[colName1, newColName]], on=[colName1], how='left')

    return train_df_1, train_df_2, train_df_3, test_df

# 定义获取用户浏览过商品的平均值，众数，中位数，最大值，最小值
def getUserItemStatFuture(train_df_1, train_df_2, train_df_3, test_df, colName):
    df = pd.concat([train_df_1, train_df_2, train_df_3, test_df])
    df_user_item_pivot_table = pd.pivot_table(df, index=['user_id'], values=[colName], aggfunc=[np.mean, np.max, np.min, np.median])
    df_user_item_pivot_table.reset_index(inplace=True)
    df_user_item_pivot_table.columns = ['user_id', colName + '_mean', colName + '_max', colName + '_min', colName + '_median']

    df_mode_pivot_table = pd.pivot_table(df[['user_id', colName, 'instance_id']], index=['user_id', colName], values=['instance_id'], aggfunc=len)
    df_mode_pivot_table.reset_index(inplace=True)
    df_mode_pivot_table = df_mode_pivot_table.sort_values(by=['user_id', 'instance_id'], ascending=False)
    df_mode_pivot_table = df_mode_pivot_table.drop_duplicates(['user_id'])
    df_mode_pivot_table.rename(columns={colName:colName + '_mode'}, inplace=True)

    train_df_1 = pd.merge(train_df_1, df_user_item_pivot_table, on=['user_id'], how='left')
    train_df_1 = pd.merge(train_df_1, df_mode_pivot_table[['user_id', colName + '_mode']], on=['user_id'], how='left')
    train_df_2 = pd.merge(train_df_2, df_user_item_pivot_table, on=['user_id'], how='left')
    train_df_2 = pd.merge(train_df_2, df_mode_pivot_table[['user_id', colName + '_mode']], on=['user_id'], how='left')
    train_df_3 = pd.merge(train_df_3, df_user_item_pivot_table, on=['user_id'], how='left')
    train_df_3 = pd.merge(train_df_3, df_mode_pivot_table[['user_id', colName + '_mode']], on=['user_id'], how='left')
    test_df = pd.merge(test_df, df_user_item_pivot_table, on=['user_id'], how='left')
    test_df = pd.merge(test_df, df_mode_pivot_table[['user_id', colName + '_mode']], on=['user_id'], how='left')
    return train_df_1, train_df_2, train_df_3, test_df

# 统计历史记录中某用户后面是否有点击相同商品
def getIsClickSameItemLater(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'item_id', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['item_id_str'] = train_df_1_copy['item_id'].map(lambda x: str(x))
    train_df_1_copy['user_item_id'] = train_df_1_copy['user_id_str'] + train_df_1_copy['item_id_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['item_id_str'] = train_df_2_copy['item_id'].map(lambda x: str(x))
    train_df_2_copy['user_item_id'] = train_df_2_copy['user_id_str'] + train_df_2_copy['item_id_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['item_id_str'] = train_df_3_copy['item_id'].map(lambda x: str(x))
    train_df_3_copy['user_item_id'] = train_df_3_copy['user_id_str'] + train_df_3_copy['item_id_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['item_id_str'] = test_df_copy['item_id'].map(lambda x: str(x))
    test_df_copy['user_item_id'] = test_df_copy['user_id_str'] + test_df_copy['item_id_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_item_id', 'date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf = tempDf.sort_values(by=['user_item_id', 'date'], ascending=False)
    tempDf['last_user_item_id'] = tempDf['user_item_id'].shift(1)
    tempDf['last_user_item_id'] = tempDf['last_user_item_id']==tempDf['user_item_id']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_item_id, 'last_show_time'] = np.nan
    historyShowList = []
    for same, dt in tempDf[['last_user_item_id','date']].values:
        if same:
            historyShowList.append(1)
        else:
            historyShowList.append(0)
    tempDf['is_later_clickSameItem'] = historyShowList
    print(tempDf[['last_user_item_id', 'date', 'is_later_clickSameItem']].head(20))
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_item_id', 'date', 'is_later_clickSameItem']], how='left', on=['user_item_id', 'date'])
    train_df_1['is_later_clickSameItem'] = train_df_1_copy['is_later_clickSameItem']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_item_id', 'date', 'is_later_clickSameItem']], how='left', on=['user_item_id', 'date'])
    train_df_2['is_later_clickSameItem'] = train_df_2_copy['is_later_clickSameItem']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_item_id', 'date', 'is_later_clickSameItem']], how='left', on=['user_item_id', 'date'])
    train_df_3['is_later_clickSameItem'] = train_df_3_copy['is_later_clickSameItem']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_item_id', 'date', 'is_later_clickSameItem']], on = ['user_item_id', 'date'], how='left')
    test_df['is_later_clickSameItem'] = test_df_copy['is_later_clickSameItem']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计历史记录中某用户后面是否有点击相同叶子类目商品
def getIsClickSameLastCategoryLater(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'real_last_category', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['real_last_category_str'] = train_df_1_copy['real_last_category'].map(lambda x: str(x))
    train_df_1_copy['user_real_last_category'] = train_df_1_copy['user_id_str'] + train_df_1_copy['real_last_category_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['real_last_category_str'] = train_df_2_copy['real_last_category'].map(lambda x: str(x))
    train_df_2_copy['user_real_last_category'] = train_df_2_copy['user_id_str'] + train_df_2_copy['real_last_category_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['real_last_category_str'] = train_df_3_copy['real_last_category'].map(lambda x: str(x))
    train_df_3_copy['user_real_last_category'] = train_df_3_copy['user_id_str'] + train_df_3_copy['real_last_category_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['real_last_category_str'] = test_df_copy['real_last_category'].map(lambda x: str(x))
    test_df_copy['user_real_last_category'] = test_df_copy['user_id_str'] + test_df_copy['real_last_category_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_real_last_category', 'date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf = tempDf.sort_values(by=['user_real_last_category', 'date'], ascending=False)
    tempDf['last_user_real_last_category'] = tempDf['user_real_last_category'].shift(1)
    tempDf['last_user_real_last_category'] = tempDf['last_user_real_last_category']==tempDf['user_real_last_category']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_real_last_category, 'last_show_time'] = np.nan
    historyShowList = []
    for same, dt in tempDf[['last_user_real_last_category','date']].values:
        if same:
            historyShowList.append(1)
        else:
            historyShowList.append(0)
    tempDf['is_later_clickSameLastCategory'] = historyShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_real_last_category', 'date', 'is_later_clickSameLastCategory']], how='left', on=['user_real_last_category', 'date'])
    train_df_1['is_later_clickSameLastCategory'] = train_df_1_copy['is_later_clickSameLastCategory']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_real_last_category', 'date', 'is_later_clickSameLastCategory']], how='left', on=['user_real_last_category', 'date'])
    train_df_2['is_later_clickSameLastCategory'] = train_df_2_copy['is_later_clickSameLastCategory']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_real_last_category', 'date', 'is_later_clickSameLastCategory']], how='left', on=['user_real_last_category', 'date'])
    train_df_3['is_later_clickSameLastCategory'] = train_df_3_copy['is_later_clickSameLastCategory']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_real_last_category', 'date', 'is_later_clickSameLastCategory']], on = ['user_real_last_category', 'date'], how='left')
    test_df['is_later_clickSameLastCategory'] = test_df_copy['is_later_clickSameLastCategory']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计历史记录中某用户后面点击相同商品的个数
def getClickSameItemLaterNumber(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'item_id', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['item_id_str'] = train_df_1_copy['item_id'].map(lambda x: str(x))
    train_df_1_copy['user_item_id'] = train_df_1_copy['user_id_str'] + train_df_1_copy['item_id_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['item_id_str'] = train_df_2_copy['item_id'].map(lambda x: str(x))
    train_df_2_copy['user_item_id'] = train_df_2_copy['user_id_str'] + train_df_2_copy['item_id_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['item_id_str'] = train_df_3_copy['item_id'].map(lambda x: str(x))
    train_df_3_copy['user_item_id'] = train_df_3_copy['user_id_str'] + train_df_3_copy['item_id_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['item_id_str'] = test_df_copy['item_id'].map(lambda x: str(x))
    test_df_copy['user_item_id'] = test_df_copy['user_id_str'] + test_df_copy['item_id_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_item_id', 'date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf = tempDf.sort_values(by=['user_item_id', 'date'], ascending=False)
    tempDf['last_user_item_id'] = tempDf['user_item_id'].shift(1)
    tempDf['last_user_item_id'] = tempDf['last_user_item_id']==tempDf['user_item_id']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_item_id, 'last_show_time'] = np.nan
    historyShowList = []
    historyShowTemp = {}
    for same, dt in tempDf[['last_user_item_id','date']].values:
        if same:
            if len(historyShowTemp) > 0:
                historyShowList.append(len(historyShowTemp))
                historyShowTemp[dt] = same
            else:
                historyShowList.append(0)
                historyShowTemp[dt] = same
        else:
            historyShowList.append(0)
            historyShowTemp = {dt:same}
    tempDf['later_clickSameItem_count'] = historyShowList
    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_item_id', 'date', 'later_clickSameItem_count']], how='left', on=['user_item_id', 'date'])
    train_df_1['later_clickSameItem_count'] = train_df_1_copy['later_clickSameItem_count']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_item_id', 'date', 'later_clickSameItem_count']], how='left', on=['user_item_id', 'date'])
    train_df_2['later_clickSameItem_count'] = train_df_2_copy['later_clickSameItem_count']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_item_id', 'date', 'later_clickSameItem_count']], how='left', on=['user_item_id', 'date'])
    train_df_3['later_clickSameItem_count'] = train_df_3_copy['later_clickSameItem_count']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_item_id', 'date', 'later_clickSameItem_count']], on = ['user_item_id', 'date'], how='left')
    test_df['later_clickSameItem_count'] = test_df_copy['later_clickSameItem_count']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计历史记录中某用户后面点击相同叶子类目商品的个数
def getClickSameLastCategoryLaterNumber(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'real_last_category', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['real_last_category_str'] = train_df_1_copy['real_last_category'].map(lambda x: str(x))
    train_df_1_copy['user_real_last_category'] = train_df_1_copy['user_id_str'] + train_df_1_copy['real_last_category_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['real_last_category_str'] = train_df_2_copy['real_last_category'].map(lambda x: str(x))
    train_df_2_copy['user_real_last_category'] = train_df_2_copy['user_id_str'] + train_df_2_copy['real_last_category_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['real_last_category_str'] = train_df_3_copy['real_last_category'].map(lambda x: str(x))
    train_df_3_copy['user_real_last_category'] = train_df_3_copy['user_id_str'] + train_df_3_copy['real_last_category_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['real_last_category_str'] = test_df_copy['real_last_category'].map(lambda x: str(x))
    test_df_copy['user_real_last_category'] = test_df_copy['user_id_str'] + test_df_copy['real_last_category_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])

    tempDf = pd.pivot_table(temp_df, index=['user_real_last_category', 'date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf = tempDf.sort_values(by=['user_real_last_category', 'date'], ascending=False)
    tempDf['last_user_real_last_category'] = tempDf['user_real_last_category'].shift(1)
    tempDf['last_user_real_last_category'] = tempDf['last_user_real_last_category']==tempDf['user_real_last_category']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_real_last_category, 'last_show_time'] = np.nan
    historyShowList = []
    historyShowTemp = {}
    for same, dt in tempDf[['last_user_real_last_category','date']].values:
        if same:
            if len(historyShowTemp) > 0:
                historyShowList.append(len(historyShowTemp))
                historyShowTemp[dt] = same
            else:
                historyShowList.append(0)
                historyShowTemp[dt] = same
        else:
            historyShowList.append(0)
            historyShowTemp = {dt:same}
    tempDf['later_clickSameLastCategory_count'] = historyShowList

    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_real_last_category', 'date', 'later_clickSameLastCategory_count']], how='left', on=['user_real_last_category', 'date'])
    train_df_1['later_clickSameLastCategory_count'] = train_df_1_copy['later_clickSameLastCategory_count']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_real_last_category', 'date', 'later_clickSameLastCategory_count']], how='left', on=['user_real_last_category', 'date'])
    train_df_2['later_clickSameLastCategory_count'] = train_df_2_copy['later_clickSameLastCategory_count']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_real_last_category', 'date', 'later_clickSameLastCategory_count']], how='left', on=['user_real_last_category', 'date'])
    train_df_3['later_clickSameLastCategory_count'] = train_df_3_copy['later_clickSameLastCategory_count']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_real_last_category', 'date', 'later_clickSameLastCategory_count']], on = ['user_real_last_category', 'date'], how='left')
    test_df['later_clickSameLastCategory_count'] = test_df_copy['later_clickSameLastCategory_count']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计历史记录中某用户后面点击相同商品的时间间隔
def getClickSameItemLaterDeltaTime(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'item_id', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'item_id', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['item_id_str'] = train_df_1_copy['item_id'].map(lambda x: str(x))
    train_df_1_copy['user_item_id'] = train_df_1_copy['user_id_str'] + train_df_1_copy['item_id_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['item_id_str'] = train_df_2_copy['item_id'].map(lambda x: str(x))
    train_df_2_copy['user_item_id'] = train_df_2_copy['user_id_str'] + train_df_2_copy['item_id_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['item_id_str'] = train_df_3_copy['item_id'].map(lambda x: str(x))
    train_df_3_copy['user_item_id'] = train_df_3_copy['user_id_str'] + train_df_3_copy['item_id_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['item_id_str'] = test_df_copy['item_id'].map(lambda x: str(x))
    test_df_copy['user_item_id'] = test_df_copy['user_id_str'] + test_df_copy['item_id_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])
    tempDf = pd.pivot_table(temp_df, index=['user_item_id', 'date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf = tempDf.sort_values(by=['user_item_id', 'date'], ascending=False)
    tempDf['last_user_item_id'] = tempDf['user_item_id'].shift(1)
    tempDf['last_user_item_id'] = tempDf['last_user_item_id']==tempDf['user_item_id']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_item_id, 'last_show_time'] = np.nan
    historyShowList = []
    deltaTime = 99999999
    for same, dt, lastShowTime in tempDf[['last_user_item_id', 'date', 'last_show_time']].values:
        if same:
            deltaTime = (dt - lastShowTime) / np.timedelta64(1, 's')
            historyShowList.append(deltaTime)
            deltaTime = 99999999
        else:
            historyShowList.append(deltaTime)
    tempDf['later_clickSameItem_deltaTime'] = historyShowList

    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_item_id', 'date', 'later_clickSameItem_deltaTime']], how='left', on=['user_item_id', 'date'])
    train_df_1['later_clickSameItem_deltaTime'] = train_df_1_copy['later_clickSameItem_deltaTime']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_item_id', 'date', 'later_clickSameItem_deltaTime']], how='left', on=['user_item_id', 'date'])
    train_df_2['later_clickSameItem_deltaTime'] = train_df_2_copy['later_clickSameItem_deltaTime']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_item_id', 'date', 'later_clickSameItem_deltaTime']], how='left', on=['user_item_id', 'date'])
    train_df_3['later_clickSameItem_deltaTime'] = train_df_3_copy['later_clickSameItem_deltaTime']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_item_id', 'date', 'later_clickSameItem_deltaTime']], on = ['user_item_id', 'date'], how='left')
    test_df['later_clickSameItem_deltaTime'] = test_df_copy['later_clickSameItem_deltaTime']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 统计历史记录中某用户后面点击相同叶子类目商品的时间间隔
def getClickSameLastCategoryLaterDeltaTime(train_df_1, train_df_2, train_df_3, test_df):
    print(len(train_df_1))
    train_df_1_copy = train_df_1[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_2_copy = train_df_2[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_3_copy = train_df_3[['user_id', 'real_last_category', 'date', 'instance_id']]
    test_df_copy = test_df[['user_id', 'real_last_category', 'date', 'instance_id']]
    train_df_1_copy['user_id_str'] = train_df_1_copy['user_id'].map(lambda x: str(x))
    train_df_1_copy['real_last_category_str'] = train_df_1_copy['real_last_category'].map(lambda x: str(x))
    train_df_1_copy['user_real_last_category'] = train_df_1_copy['user_id_str'] + train_df_1_copy['real_last_category_str']
    train_df_2_copy['user_id_str'] = train_df_2_copy['user_id'].map(lambda x: str(x))
    train_df_2_copy['real_last_category_str'] = train_df_2_copy['real_last_category'].map(lambda x: str(x))
    train_df_2_copy['user_real_last_category'] = train_df_2_copy['user_id_str'] + train_df_2_copy['real_last_category_str']
    train_df_3_copy['user_id_str'] = train_df_3_copy['user_id'].map(lambda x: str(x))
    train_df_3_copy['real_last_category_str'] = train_df_3_copy['real_last_category'].map(lambda x: str(x))
    train_df_3_copy['user_real_last_category'] = train_df_3_copy['user_id_str'] + train_df_3_copy['real_last_category_str']
    test_df_copy['user_id_str'] = test_df_copy['user_id'].map(lambda x: str(x))
    test_df_copy['real_last_category_str'] = test_df_copy['real_last_category'].map(lambda x: str(x))
    test_df_copy['user_real_last_category'] = test_df_copy['user_id_str'] + test_df_copy['real_last_category_str']
    temp_df = pd.concat([train_df_1_copy, train_df_2_copy, train_df_3_copy, test_df_copy])

    tempDf = pd.pivot_table(temp_df, index=['user_real_last_category', 'date'], values=['instance_id'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf = tempDf.sort_values(by=['user_real_last_category', 'date'], ascending=False)
    tempDf['last_user_real_last_category'] = tempDf['user_real_last_category'].shift(1)
    tempDf['last_user_real_last_category'] = tempDf['last_user_real_last_category']==tempDf['user_real_last_category']
    tempDf['last_show_time'] = tempDf['date'].shift(1)
    tempDf.loc[~tempDf.last_user_real_last_category, 'last_show_time'] = np.nan

    historyShowList = []
    deltaTime = 99999999
    for same, dt, lastShowTime in tempDf[['last_user_real_last_category', 'date', 'last_show_time']].values:
        if same:
            deltaTime = (dt - lastShowTime) / np.timedelta64(1, 's')
            historyShowList.append(deltaTime)
            deltaTime = 99999999
        else:
            historyShowList.append(deltaTime)
    tempDf['later_clickSameLastCategory_deltaTime'] = historyShowList

    train_df_1_copy = train_df_1_copy.merge(tempDf[['user_real_last_category', 'date', 'later_clickSameLastCategory_deltaTime']], how='left', on=['user_real_last_category', 'date'])
    train_df_1['later_clickSameLastCategory_deltaTime'] = train_df_1_copy['later_clickSameLastCategory_deltaTime']
    train_df_2_copy = train_df_2_copy.merge(tempDf[['user_real_last_category', 'date', 'later_clickSameLastCategory_deltaTime']], how='left', on=['user_real_last_category', 'date'])
    train_df_2['later_clickSameLastCategory_deltaTime'] = train_df_2_copy['later_clickSameLastCategory_deltaTime']
    train_df_3_copy = train_df_3_copy.merge(tempDf[['user_real_last_category', 'date', 'later_clickSameLastCategory_deltaTime']], how='left', on=['user_real_last_category', 'date'])
    train_df_3['later_clickSameLastCategory_deltaTime'] = train_df_3_copy['later_clickSameLastCategory_deltaTime']
    test_df_copy = pd.merge(test_df_copy, tempDf[['user_real_last_category', 'date', 'later_clickSameLastCategory_deltaTime']], on = ['user_real_last_category', 'date'], how='left')
    test_df['later_clickSameLastCategory_deltaTime'] = test_df_copy['later_clickSameLastCategory_deltaTime']
    print(len(train_df_1))
    return train_df_1, train_df_2, train_df_3, test_df

# 导出训练集划分结果
def exportDataset(df, fileName):
    df.to_csv('~/kengkeng/alimama/data/%s.csv' % fileName, header=True, index=False)

def getBaseConversionRate(future_df, test_df, colName):
    t = future_df[[colName]]
    t[colName + '_total_number'] = 1
    t = t.groupby(colName).agg('sum').reset_index()

    t_buy = future_df[[colName]][future_df.is_trade == 1]
    t_buy[colName + '_buy_number'] = 1
    t_buy = t_buy.groupby(colName).agg('sum').reset_index()

    t = pd.merge(t, t_buy, on=colName, how='left')
    t[colName + '_buy_number'] = t[colName + '_buy_number'].map(lambda x: 0 if math.isnan(x) else x)
    t['buy_origion_rate'] = t[colName + '_buy_number'] / t[colName + '_total_number']
    alpha, beta = getBayesSmoothParam(t['buy_origion_rate'])
    t[colName + '_converse_smooth_rate'] = (t[colName + '_buy_number'] + alpha) / (t[colName + '_total_number'] + alpha + beta)
#     train_df = pd.merge(train_df, t[[colName, colName + '_converse_smooth_rate']], on=colName, how='left')
#     train_df[colName + '_converse_smooth_rate'] = train_df[colName + '_converse_smooth_rate'].map(lambda x: (alpha / (alpha + beta)) if math.isnan(x) else x)

    test_df = pd.merge(test_df, t[[colName, colName + '_converse_smooth_rate', colName + '_total_number', colName + '_buy_number']], on=colName, how='left')
    test_df[colName + '_converse_smooth_rate'] = test_df[colName + '_converse_smooth_rate'].map(lambda x: (alpha / (alpha + beta)) if math.isnan(x) else x)

    test_df[colName + '_total_number'] = test_df[colName + '_total_number'].fillna(0)
    test_df, total_number_scaler = scalerFea(test_df, colName + '_total_number')

    test_df[colName + '_buy_number'] = test_df[colName + '_buy_number'].fillna(0)
    test_df, buy_number_scaler = scalerFea(test_df, colName + '_buy_number')

    return test_df

#定义对每个窗口进行操作的函数
def dealHuaChuangDataset(future_df, dataset):
    dataset = getBaseConversionRate(future_df, dataset, 'user_id')
    dataset = getBaseConversionRate(future_df, dataset, 'item_brand_id')
    dataset = getBaseConversionRate(future_df, dataset, 'item_id')
    dataset = getBaseConversionRate(future_df, dataset, 'shop_id')
    dataset = getBaseConversionRate(future_df, dataset, 'real_last_category')
    return dataset


if __name__ == '__main__':
    # 特征处理
    startTime = datetime.datetime.now()

    print('~~~~~~~~~~~~~~开始导入数据~~~~~~~~~~~~~~~~~~~')

    train_df_3 = pd.read_csv('~/kengkeng/alimama/data/train_df_3.csv')
    train_df_2 = pd.read_csv('~/kengkeng/alimama/data/train_df_2.csv')
    train_df_1 = pd.read_csv('~/kengkeng/alimama/data/train_df_1.csv')

    print('~~~~~~~~~~~~~~导入数据完毕~~~~~~~~~~~~~~~~~~~')

    train_df_3 = splitMultiFea(train_df_3)
    train_df_2 = splitMultiFea(train_df_2)
    train_df_1 = splitMultiFea(train_df_1)

    train_df_3 = addContextFea(train_df_3)
    train_df_2 = addContextFea(train_df_2)
    train_df_1 = addContextFea(train_df_1)

    train_df_3 = getCategoryFuture(train_df_3)
    train_df_2 = getCategoryFuture(train_df_2)
    train_df_1 = getCategoryFuture(train_df_1)

    train_df_3 = getMatchProportion(train_df_3)
    train_df_2 = getMatchProportion(train_df_2)
    train_df_1 = getMatchProportion(train_df_1)

    train_df_3 = getPredictNumber(train_df_3)
    train_df_2 = getPredictNumber(train_df_2)
    train_df_1 = getPredictNumber(train_df_1)

    train_df_3 = getPredictAccuracy(train_df_3)
    train_df_2 = getPredictAccuracy(train_df_2)
    train_df_1 = getPredictAccuracy(train_df_1)

    train_df_3 = getCPNumber(train_df_3)
    train_df_2 = getCPNumber(train_df_2)
    train_df_1 = getCPNumber(train_df_1)

    print('~~~~~~~~~~~~~~开始导入测试集数据~~~~~~~~~~~~~~~~~~~')

    #导入测试集进行数据处理
    test_df_a = pd.read_csv('~/yuna/alimama/data/round2_ijcai_18_test_a_20180425.txt', sep=' ')
    test_df_b = pd.read_csv('~/yuna/alimama/data/round2_ijcai_18_test_b_20180510.txt', sep=' ')
    test_df = pd.concat([test_df_a, test_df_b])
    test_df['date'] = test_df.context_timestamp.map(lambda x: datetime.datetime.fromtimestamp(x))
    test_df['weekday'] = test_df['date'].map(lambda x: x.weekday())
    test_df['day'] = test_df['date'].map(lambda x: x.day)
    test_df['hour'] = test_df['date'].map(lambda x: x.hour)
    test_df = splitMultiFea(test_df)
    test_df = addContextFea(test_df)
    test_df = getCategoryFuture(test_df)
    test_df = getMatchProportion(test_df)
    test_df = getPredictNumber(test_df)
    test_df = getPredictAccuracy(test_df)
    test_df = getCPNumber(test_df)

    #构造跟商品实际的根类目和叶子类目特征
    train_df_1['real_first_category'] = train_df_1['item_category_list'].map(lambda x: np.nan if len(x) < 1 else x[0])
    train_df_1['real_last_category'] = train_df_1['item_category_list'].map(lambda x: np.nan if len(x) < 1 else x[len(x) -1])
    train_df_2['real_first_category'] = train_df_2['item_category_list'].map(lambda x: np.nan if len(x) < 1 else x[0])
    train_df_2['real_last_category'] = train_df_2['item_category_list'].map(lambda x: np.nan if len(x) < 1 else x[len(x) -1])
    train_df_3['real_first_category'] = train_df_3['item_category_list'].map(lambda x: np.nan if len(x) < 1 else x[0])
    train_df_3['real_last_category'] = train_df_3['item_category_list'].map(lambda x: np.nan if len(x) < 1 else x[len(x) -1])
    test_df['real_first_category'] = test_df['item_category_list'].map(lambda x: np.nan if len(x) < 1 else x[0])
    test_df['real_last_category'] = test_df['item_category_list'].map(lambda x: np.nan if len(x) < 1 else x[len(x) -1])

    #删除某些不必要的列
    drop_fea = ['item_category_list', 'item_property_list',
                'item_category_list_str', 'item_property_list_str', 'item_category0',
                'item_category1', 'item_category2', 'item_prop_num',
                'predict_category_property_str', 'predict_category', 'predict_cate_num',
                'cate_intersect_num', 'predict_property', 'predict_prop_num',
                'prop_intersect_num', 'prop_union_num', 'predict_category_list', 'predict_category_set',
                'real_item_category_list', 'predict_property_list']
    train_df_1.drop(drop_fea, axis=1, inplace=True)
    train_df_2.drop(drop_fea, axis=1, inplace=True)
    train_df_3.drop(drop_fea, axis=1, inplace=True)
    test_df.drop(drop_fea, axis=1, inplace=True)

    print('~~~~~~~~~~~~~~测试集数据导入完毕~~~~~~~~~~~~~~~~~~~')

    train_df_normal = pd.concat([train_df_1, train_df_2])
    train_df_1, train_df_2, train_df_3, test_df = getHistoryInfoByCol(train_df_normal, train_df_1, train_df_2, train_df_3, test_df, 'item_brand_id')
    train_df_1, train_df_2, train_df_3, test_df = getHistoryInfoByCol(train_df_normal, train_df_1, train_df_2, train_df_3, test_df, 'shop_id')
    train_df_1, train_df_2, train_df_3, test_df = getHistoryInfoByCol(train_df_normal, train_df_1, train_df_2, train_df_3, test_df, 'item_id')

    print('~~~~~~~~~~~~~~历史数据统计完毕~~~~~~~~~~~~~~~~~~~')

    train_df_1['date'] = pd.to_datetime(train_df_1['date'])
    train_df_2['date'] = pd.to_datetime(train_df_2['date'])
    train_df_3['date'] = pd.to_datetime(train_df_3['date'])
    train_df_1, train_df_2, train_df_3, test_df = getOneHourSameItemCount(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getOneHourSameFirstCategoryCount(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getOneHourSameLastCategoryCount(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getOneHourSameBrandCount(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getOneHourSameShopCount(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getIsOneHourFirstClickItem(train_df_1, train_df_2, train_df_3, test_df)

    print('~~~~~~~~~~~~~~一个小时统计特征构造完毕~~~~~~~~~~~~~~~~~~~')

    train_df_1, train_df_2, train_df_3, test_df = getUserItemLastClickDeltaTime(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getUserBrandLastClickDeltaTime(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getUserShopLastClickDeltaTime(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getUserFirstCategoryLastClickDeltaTime(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getUserLastCategoryLastClickDeltaTime(train_df_1, train_df_2, train_df_3, test_df)

    print('~~~~~~~~~~~~~~用户上次统计点击时间特征构造完毕~~~~~~~~~~~~~~~~~~~')

    train_df_1, train_df_2, train_df_3, test_df = getHourTradeRate(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getUserItemStatFuture(train_df_1, train_df_2, train_df_3, test_df, 'item_price_level')
    train_df_1, train_df_2, train_df_3, test_df = getUserItemStatFuture(train_df_1, train_df_2, train_df_3, test_df, 'item_sales_level')

    train_df_1, train_df_2, train_df_3, test_df = getCorrespondNumber(train_df_1, train_df_2, train_df_3, test_df, 'shop_id', 'item_id', 'shop_item_classNumber')
    train_df_1, train_df_2, train_df_3, test_df = getCorrespondNumber(train_df_1, train_df_2, train_df_3, test_df, 'item_brand_id', 'item_id', 'brand_item_classNumber')
    train_df_1, train_df_2, train_df_3, test_df = getCorrespondNumber(train_df_1, train_df_2, train_df_3, test_df, 'item_city_id', 'item_id', 'city_item_classNumber')
    train_df_1, train_df_2, train_df_3, test_df = getCorrespondNumber(train_df_1, train_df_2, train_df_3, test_df, 'shop_id', 'user_id', 'shop_user_classNumber')
    train_df_1, train_df_2, train_df_3, test_df = getCorrespondNumber(train_df_1, train_df_2, train_df_3, test_df, 'item_brand_id', 'user_id', 'brand_user_classNumber')
    train_df_1, train_df_2, train_df_3, test_df = getCorrespondNumber(train_df_1, train_df_2, train_df_3, test_df, 'item_city_id', 'user_id', 'city_user_classNumber')

    print('~~~~~~~~~~~~~~~~~开始统计穿越特征~~~~~~~~~~~~~~~~~~~~~~')

    train_df_1, train_df_2, train_df_3, test_df = getIsClickSameItemLater(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getIsClickSameLastCategoryLater(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getClickSameItemLaterNumber(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getClickSameLastCategoryLaterNumber(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getClickSameItemLaterDeltaTime(train_df_1, train_df_2, train_df_3, test_df)

    train_df_1, train_df_2, train_df_3, test_df = getClickSameLastCategoryLaterDeltaTime(train_df_1, train_df_2, train_df_3, test_df)

    print('~~~~~~~~~~~~~~穿越特征构造完毕~~~~~~~~~~~~~~~~~~~')

    train_df_all = pd.concat([train_df_1, train_df_2, train_df_3])
    exportDataset(train_df_all, 'fusai_b_train_df_all')
    exportDataset(test_df, 'fusai_b_test_df_all')

    #尝试统计两天前滑窗结果，包括商品，店铺，品牌，叶子类目的转化率，点击次数和购买次数
    #首先划分数据集
    future_dataset1 = train_df_1[(train_df_1.day == 31) | (train_df_1.day == 1)]
    train_df_huachuang_1 = train_df_1[train_df_1.day == 2]
    future_dataset2 = train_df_1[(train_df_1.day == 1) | (train_df_1.day == 2)]
    train_df_huachuang_2 = train_df_1[train_df_1.day == 3]
    future_dataset3 = train_df_1[(train_df_1.day == 2) | (train_df_1.day == 3)]
    train_df_huachuang_3 = train_df_1[train_df_1.day == 4]
    future_dataset4 = train_df_2
    train_df_huachuang_4 = train_df_3

    train_df_huachuang_1 = dealHuaChuangDataset(future_dataset1, train_df_huachuang_1)
    train_df_huachuang_2 = dealHuaChuangDataset(future_dataset2, train_df_huachuang_2)
    train_df_huachuang_3 = dealHuaChuangDataset(future_dataset3, train_df_huachuang_3)
    train_df_huachuang_4 = dealHuaChuangDataset(future_dataset4, train_df_huachuang_4)
    test_df = dealHuaChuangDataset(future_dataset4, test_df)

    #将2,3,4号数据抽样15%与7号当天数据结合作为训练接，拟合线上分布
    train_df_234 = pd.concat([train_df_huachuang_1, train_df_huachuang_2, train_df_huachuang_3])
    train_df_234 = train_df_234.sample(frac = 0.15, replace = True)

    train_df = pd.concat([train_df_huachuang_4, train_df_234])

    print('train_df.columns.values : ', train_df.columns.values)

    exportDataset(train_df, 'fusai_b_train_df_weilai')
    exportDataset(test_df, 'fusai_b_test_df_weilai')

    print('test_df.columns.values : ', test_df.columns.values)

    print('pretreatment time:', datetime.datetime.now()-startTime)
