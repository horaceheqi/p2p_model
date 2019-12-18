import os
import re
import sys
import time
import math
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class MF(object):
    def __init__(self, data, features, bins):
        self.Train_data = data
        self.Save_features = features
        self.Save_bins = bins
        self.Time = datetime.datetime.strptime('201906', '%Y%m')
        self.bin = dict()
        self.woe = dict()
        self.iv = dict()
        self.filterIv = list()

    # 模型评估
    @staticmethod
    def evaluate(label, predict_prob, predict_label):
        conf_matrix = metrics.confusion_matrix(label, predict_label)
        print('混淆矩阵：\n' + str(conf_matrix))
        acc = metrics.accuracy_score(label, predict_label)
        print('accuracy:' + str(round(acc, 4)))
        pre = metrics.precision_score(label, predict_label)
        print('recall:' + str(round(pre, 4)))
        rec = metrics.recall_score(label, predict_label)
        print('recall:' + str(round(rec, 4)))
        f1score = (2 * pre * rec) / (pre + rec)
        print('f1score:' + str(round(f1score, 4)))
        fpr, tpr, thresholds = metrics.roc_curve(label, predict_prob)
        auc = metrics.auc(fpr, tpr)
        print('auc:' + str(round(auc, 4)))
        ks = max(tpr - fpr)
        print('ks:' + str(round(ks, 4)))

    # 计算模型psi
    @staticmethod
    def get_psi(expected, actual, temp_list):
        expected_len = len(expected)
        actual_len = len(actual)
        psi_sum = 0
        for j in range(len(temp_list) - 1):
            actual_pct = len([x for x in actual if temp_list[j] <= x < temp_list[j+1]]) / actual_len
            expected_pct = len([x for x in expected if temp_list[j] <= x < temp_list[j+1]]) / expected_len
            if (actual_pct > 0) and (expected_pct > 0):
                psi_ = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
                psi_sum += psi_
        return np.round(psi_sum, 5)

    # 自定义sum函数
    @staticmethod
    def _sum(var):
        if var.count() is 0:
            return np.nan
        else:
            return var.sum()

    # 变量衍生方法
    @staticmethod
    def var_normal(data, var, times):
        print('%s: var_Normal...' % times)
        df = data.groupby(0)[var]
        df_mean = df.mean().round(3).reset_index(drop=True).add_prefix(str(times) + '_avg_')
        df_max = df.max().round(3).reset_index(drop=True).add_prefix(str(times) + '_max_')
        df_min = df.min().round(3).reset_index(drop=True).add_prefix(str(times) + '_min_')
        df_sum = df.agg(lambda x: MF._sum(x)).reset_index(drop=True).add_prefix(str(times) + '_sum_')
        return df_mean, df_max, df_min, df_sum

    @staticmethod
    def var_max(data, var, times):
        print('%s: var_Max...' % times)
        df = data.groupby(0)[var]
        df_mean = df.mean().round(3).reset_index(drop=True).add_prefix(str(times) + '_avg_')
        df_max = df.max().round(3).reset_index(drop=True).add_prefix(str(times) + '_max_')
        df_min = df.min().round(3).reset_index(drop=True).add_prefix(str(times) + '_min_')
        return df_mean, df_max, df_min

    @staticmethod
    def var_overdue(data, var, times):
        print('%s: var_Overdue...' % times)
        df = data.groupby(0)[var]
        df_yn = df.max().reset_index(drop=True).add_prefix(str(times) + '_YN_')
        return df_yn

    @staticmethod
    def var_once(data, var, times):
        print('%s: var_Once...' % times)
        df = data.groupby(0)[var]
        df_last = df.last().reset_index(drop=True).add_prefix(str(times) + '_last_')
        df_first = df.first().reset_index(drop=True).add_prefix(str(times) + '_first_')
        return df_last, df_first

    # 采用best-ks进行分箱
    @staticmethod
    def ks_bin(df_, rate_):
        f = df_.iloc[:, 1].value_counts()[0]
        t = df_.iloc[:, 1].value_counts()[1]
        df_cro = pd.crosstab(df_.iloc[:, 0], df_.iloc[:, 1])
        df_cro[0] = df_cro[0] / f
        df_cro[1] = df_cro[1] / t
        df_cro_cum = df_cro.cumsum()
        ks_list = abs(df_cro_cum[1] - df_cro_cum[0])
        ks_list_index = ks_list.nlargest(len(ks_list)).index.tolist()
        temp_num = 0
        for i in ks_list_index:
            temp_num = i
            df_1 = df_[df_.iloc[:, 0] <= i]
            df_2 = df_[df_.iloc[:, 0] > i]
            if len(df_1) >= rate_ and len(df_2) >= rate_:
                break
        return temp_num

    @staticmethod
    def ks_zone(df_, list_):
        df_.columns = [0, 1]
        list_zone = list()
        list_.sort()
        n = 0
        for i in list_:
            m = sum(df_[df_[0] <= i]) - n
            n = sum(df_[df_[0] <= i])
            list_zone.append(m)
        list_zone.append(50000 - sum(list_zone))
        max_index = list_zone.index(max(list_zone))
        if max_index == 0:
            rst = [df_.iloc[:, 0].unique().min(), list_[0]]
        elif max_index == len(list_):
            rst = [list_[-1], df_.iloc[:, 0].unique().max()]
        else:
            rst = [list_[max_index - 1], list_[max_index]]
        return rst

    @staticmethod
    def get_bin(df, var_name, bin_num):
        res_bin, filter_list = list(), list()
        df = df[[var_name, 'label']]
        temp_df = df.copy()
        temp_df.fillna('nan', inplace=True)
        var_list = pd.crosstab(temp_df[var_name], temp_df['label']).index.tolist()
        for i in var_list:
            if i == 'nan':
                res_bin.append(np.nan)
            elif i < 0:
                res_bin.append(i)
            else:
                filter_list.append(i)
        rate = pd.notnull(df[var_name]).sum() / 10
        df.dropna(axis=0, inplace=True)
        df = df[df[var_name] >= 0]
        df_ = df.copy()
        bins = list()
        for i in range(bin_num - 1):
            ks_ = MF.ks_bin(df_, rate)
            bins.append(ks_)
            new_bins = MF.ks_zone(df, bins)
            df_ = df[(df.iloc[:, 0] > new_bins[0]) & (df.iloc[:, 0] <= new_bins[1])]
            if df_.iloc[:, 1].value_counts().shape[0] < 2:
                break
        real_bins = res_bin + bins
        bins.append(max(filter_list))
        bins.append(min(filter_list))
        temp_bins = res_bin + sorted(list(set(bins)))
        return real_bins, temp_bins

