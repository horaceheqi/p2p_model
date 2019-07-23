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
        self.Time = datetime.datetime.strptime('20190722', '%Y%m%d')
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
    def simple_extract(df, data_):
        print('extract...')
        df_ = df.copy()
        for key, value in data_:
            temp_numerator = df[key]
            temp_denominator = df[value]
            for var in temp_numerator.columns:
                names = var + '/' + value
                df_[names] = df[var] / temp_denominator
        return df_

    @staticmethod
    def var_once(data, var, times):
        print('%s: var_Once...' % times)
        df = data.groupby(0)[var]
        df_last = df.last().add_prefix(str(times) + '_last_')
        df_first = df.first().add_prefix(str(times) + '_first_')
        return df_last, df_first

    @staticmethod
    def var_normal(data, var, times):
        print('%s: var_Normal...' % times)
        df = data.groupby(0)[var]
        df_mean = df.mean().round(3).add_prefix(str(times) + '_avg_')
        df_max = df.max().round(3).add_prefix(str(times) + '_max_')
        df_min = df.min().round(3).add_prefix(str(times) + '_min_')
        df_sum = df.agg(lambda x: MF._sum(x)).add_prefix(str(times) + '_sum_')
        return df_mean, df_max, df_min, df_sum

    @staticmethod
    def var_max(data, var, times):
        print('%s: var_Max...' % times)
        df = data.groupby(0)[var]
        df_mean = df.mean().round(3).add_prefix(str(times) + '_avg_')
        df_max = df.max().round(3).add_prefix(str(times) + '_max_')
        df_min = df.min().round(3).add_prefix(str(times) + '_min_')
        return df_mean, df_max, df_min

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

    # 计算iv值
    @staticmethod
    def cal_iv(df_, var_name, bins_, flag):
        res_iv, res_bins, starts = 0, [], 0
        total_1 = df_['label'].sum()  # label=1的样本
        total_0 = df_.shape[0] - total_1  # label=0的样本
        # 判断flag,0代表离散变量,1代表连续变量
        if flag:
            temp_bins, temp_bins2, temp1_arr_1, temp1_arr_0, temp2_arr_1, temp2_arr_0 = list(), list(), list(), list(), list(), list()
            for i in bins_:
                if np.isnan(i):
                    starts = 1
                else:
                    temp_bins.append(i)
            if starts:
                try:
                    temp1_arr_1.append(df_[df_[var_name].isnull()]['label'].value_counts()[1])
                except KeyError:
                    temp1_arr_1.append(0.001)
                try:
                    temp1_arr_0.append(df_[df_[var_name].isnull()]['label'].value_counts()[0])
                except KeyError:
                    temp1_arr_0.append(0.001)

            z = [(temp_bins[i], temp_bins[i+1]) for i in range(len(temp_bins)-1)]
            for x in z:
                try:
                    temp2_arr_1.append(df_[(df_[var_name] >= x[0]) & (df_[var_name] < x[1])]['label'].value_counts()[1])
                except KeyError:
                    temp2_arr_1.append(0.001)
                try:
                    temp2_arr_0.append(df_[(df_[var_name] >= x[0]) & (df_[var_name] < x[1])]['label'].value_counts()[0])
                except KeyError:
                    temp2_arr_0.append(0.001)
            arr_1, arr_0 = temp1_arr_1 + temp2_arr_1, temp1_arr_0 + temp2_arr_0

        else:
            df_[var_name].fillna('nan', inplace=True)
            df_cro = pd.crosstab(df_[var_name], df_['label'])
            temp_bins, arr_1, arr_0 = [], [], []
            for index, row in df_cro.iterrows():
                temp_bins.append(index)
                if row[1] == 0:
                    arr_1.append(0.001)
                else:
                    arr_1.append(row[1])
                if row[0] == 0:
                    arr_0.append(0.001)
                else:
                    arr_0.append(row[0])
            bins_ = [np.nan if i == 'nan' else i for i in temp_bins]

        arr_1_p = [x / total_1 for x in list(map(float, arr_1))]
        arr_0_p = [x / total_0 for x in list(map(float, arr_0))]
        woe = [math.log(x, math.e) for x in [a/b for a, b in zip(arr_1_p, arr_0_p)]]
        res_iv = sum([a*b for a, b in zip(woe, [x-y for x, y in zip(arr_1_p, arr_0_p)])])

        if flag:
            return woe, res_iv
        else:
            return bins_, woe, res_iv

    # 特征提取
    @staticmethod
    def get_var_v0(self, df_):
        print('Load data to extract v0 var...')
        print('df.shape:', df_.shape)
        # 标准多投变量衍生
        print('Process data...')
        time_features = ['m', 156, 157, 158, 159, 163, 164]
        for var in time_features:
            df_[var].fillna('201907', inplace=True)
            df_[var] = df_[var].map(int)
            df_[var] = df_[var].map(str)
            df_[var] = df_[var].apply(lambda x: (self.Time - datetime.datetime.strptime(x, '%Y%m')).days)

        ct_amount_var = [1] + list(range(2, 74, 2)) + list(range(97, 135, 2)) + [142, 144, 145]
        ct_count_var = list(range(3, 75, 2)) + list(range(98, 136, 2)) + [143]
        ct_max_var = list(range(74, 97)) + list(range(135, 142))
        once_var = list(range(147, 169)) + list(range(175, 181)) + list(range(186, 191))

        ct_var_all = ct_amount_var + ct_count_var + list(range(169, 175)) + list(range(181, 186))
        df_group = df_.groupby(df_['info'])
        rct_mth_12 = df_group.head(12)
        rct_mth_6 = df_group.head(6)
        rct_mth_3 = df_group.head(3)
        rct_mth_1 = df_group.head(1)
        df_1 = df_group.head(1).drop(['label'], axis=1).add_prefix(str(1) + '_')

        times = time.time()
        print('Extract 12 features...')
        df_la_on_12, df_ea_on_12 = MF.var_once(rct_mth_12, once_var, 12)
        df_me_no_12, df_ma_no_12, df_mi_no_12, df_su_no_12 = MF.var_normal(rct_mth_12, ct_var_all, 12)
        df_me_ma_12, df_ma_ma_12, df_mi_ma_12 = MF.var_max(rct_mth_12, ct_max_var, 12)

        print('Extract 6 features...')
        df_la_on_6, df_ea_on_6 = MF.var_once(rct_mth_6, once_var, 6)
        df_me_no_6, df_ma_no_6, df_mi_no_6, df_su_no_6 = MF.var_normal(rct_mth_6, ct_var_all, 6)
        df_me_ma_6, df_ma_ma_6, df_mi_ma_6 = MF.var_max(rct_mth_6, ct_max_var, 6)

        print('Extract 3 features...')
        df_la_on_3, df_ea_on_3 = MF.var_once(rct_mth_3, once_var, 3)
        df_me_no_3, df_ma_no_3, df_mi_no_3, df_su_no_3 = MF.var_normal(rct_mth_3, ct_var_all, 3)
        df_me_ma_3, df_ma_ma_3, df_mi_ma_3 = MF.var_max(rct_mth_3, ct_max_var, 3)
        print('Extract features finished...', time.time() - times)

        fin_data = pd.concat([df_la_on_12, df_ea_on_12,
                              df_me_no_12, df_ma_no_12, df_mi_no_12, df_su_no_12,
                              df_me_ma_12, df_ma_ma_12, df_mi_ma_12,
                              df_la_on_6, df_ea_on_6,
                              df_me_no_6, df_ma_no_6, df_mi_no_6, df_su_no_6,
                              df_me_ma_6, df_ma_ma_6, df_mi_ma_6,
                              df_la_on_3, df_ea_on_3,
                              df_me_no_3, df_ma_no_3, df_mi_no_3, df_su_no_3,
                              df_me_ma_3, df_ma_ma_3, df_mi_ma_3, df_1], axis=1)
        fin_data['info'] = rct_mth_1[0]
        print(fin_data.shape)
        fin_data['label'] = rct_mth_1['label']

    @staticmethod
    def get_var_v1(self, df):
        print('Load data to extract v1 var...')
        print('df.shape:', df.shape)

    @staticmethod
    def get_var_v2(self, df):
        print('Load data to extract v2 var...')
        print('df.shape:', df.shape)

    @staticmethod
    def get_var_v3(self, df):
        print('Load data to extract v3 var...')
        print('df.shape:', df.shape)

    def get_var_v4(self, df):
        print('Load data to extract v4 var...')
        df = pd.read_csv(self.Train_data)
        df.columns = ['info'] + ['m'] + list(range(1, df.shape[1]-2)) + ['label'] + ['is_train']
        print('df.shape:', df.shape)

        print('Process data...')
        time_features = ['month', 10, 11]
        for var in time_features:
            df[var].fillna('20190722', inplace=True)
            df[var] = df[var].map(int)
            df[var] = df[var].map(str)
            df[var] = df[var].apply(lambda x: (self.Time - datetime.datetime.strptime(x, '%Y%m%d')).days)

        simple_extract_numerator = [[3, 4, 5, 6, 7, 8, 9], [20, 21, 22, 23, 24, 25, 26], [33, 34, 35, 36, 37, 38, 39],
                                    [18, 31], [19, 32], [30, 43], [20, 33], [21, 34], [22, 35], [23, 36]]
        simple_extract_denominator = [2, 19, 32, 1, 2, 17, 3, 4, 5, 6]

        simple_extract_df = zip(simple_extract_numerator, simple_extract_denominator)

        times = time.time()
        fin_data = MF.simple_extract(df, simple_extract_df)
        print('fin_data.shape', fin_data.shape)
        print('Extract features finished...', time.time() - times)
        fin_data.to_csv(self.Save_features, index=False)

    def get_var_v5(self, df):
        print('Load data to extract v5 var...')
        df = pd.read_csv(self.Train_data)
        df.columns = ['info'] + ['m'] + list(range(1, df.shape[1]-2)) + ['label'] + ['is_train']
        print('df.shape:', df.shape)

    # 加载数据(未切分样本、标签)
    def load_data(self, df):
        print('load features...')
        df = pd.read_csv(self.Save_features)
        print(df.shape)
        df['label'] = df['label'].map(int)
        df_t = df[df['label'] == 0]  # 好人
        df_f = df[df['label'] == 1]  # 坏人
        print(df_t.shape, df_f.shape)
        df_t = df_t.sample(df_f.shape[0])
        df_f = df_f.sample(df_f.shape[0])
        df_ = df_t.append(df_f)
        print(df_.shape)

        train_label = df_['label']
        train_columns = [x for x in df_.columns if x not in ['info', 'm', 'label']]
        train_data = df_[train_columns]
        df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(train_data, train_label, test_size=0.2, random_state=0)
        return df_train_x, df_test_x, df_train_y, df_test_y

    # 开始分箱
    def start_bin(self, trains_x, trains_y):
        train_ = pd.concat([trains_x, trains_y], axis=1)
        print('start bin...')
        var_list = [var for var in train_.columns if var not in ['info', 'm', 'label']]
        total_bwi = dict()
        # 标准、逾期、多投变量中离散、连续定义
        overdue_feature = list(range(197, 201)) + list(range(206, 210)) + list(range(211, 215))
        overdue_features = [str(x) for x in overdue_feature]
        for var in var_list:
            train_[var] = train_[var].map(float)
            # 标准、多投、逾期、画像用以下代码
            flag = 0 if var.split('_')[-1] in overdue_features else 1  # 0是离散变量,1是连续变量
            # 非标准、多投、逾期用以下代码
            # flag = 0 if pd.crosstab(train_[var], train_['label']).shape[0] <= 5 else 1

            temp_list = list()
            if flag:
                res_bins = MF.get_bin(train_, var, 4)
                res_iv = MF.cal_iv(train_, var, res_bins[1], flag)
                temp_list.append(res_bins[1])
                temp_list.append(res_iv[0])
                temp_list.append(res_iv[1])
                temp_list.append(flag)
                total_bwi[var] = temp_list
            else:
                res_iv = MF.cal_iv(train_, var, [], flag)
                temp_list.append(res_iv[0])
                temp_list.append(res_iv[1])
                temp_list.append(res_iv[2])
                temp_list.append(flag)
                total_bwi[var] = temp_list
        print('save result...')
        df_bin = pd.DataFrame.from_dict(total_bwi, orient='index').T
        df_bin.to_csv(self.Save_bins, index=False)

    # 替换woe
    def replace(self, df_, var, column_dict, ths, flag):
        temp = df_[var].values.tolist()
        temp = ['NAN' if np.isnan(x) else x for x in temp]
        pattern = re.compile('\[(.*)\]')
        a = list()
        bin_dict = dict()
        line = column_dict[var].values.tolist()
        bin_list = [float(i) for i in pattern.findall(line[0])[0].split(',')]
        bin_list = ['NAN' if np.isnan(x) else x for x in bin_list]
        temp_list = [i for i in pattern.findall(line[1])[0].split(',')]
        woe_list = [0 if i == '' else float(i) for i in temp_list]
        iv = float(line[2])
        tag = float(line[3])
        sta = 0
        try:
            # 筛选大于设定iv阈值的变量
            if (iv >= ths) and (flag == 1) and len(bin_list) < 10 and iv < 2:
                self.iv[var] = iv
                self.bin[var] = bin_list
                self.woe[var] = woe_list
                self.filterIv.append(var)
            # 替换woe值
            if tag == 0:
                # 离散型分箱的字典
                for i in range(len(bin_list)):
                    bin_dict[bin_list[i]] = woe_list[i]
            else:
                # 连续型分箱的字典
                for num in bin_list:
                    if num == 'NAN' or num < 0:
                        bin_dict[num] = woe_list[sta]
                        sta += 1

            # 分箱只有一个的
            if len(bin_list) == 1:
                a = list(np.zeros(len(temp)))
                return a
            elif len(bin_list) >= 2 and len(woe_list) <= 1:
                flags = 0
                for i in bin_list:
                    if np.isnan(i):
                        flags = 1
                if flags:
                    a = list(np.zeros(len(temp)))
                    return a
            for x in temp:
                if x in bin_dict.keys():
                    a.append(bin_dict[x])
                elif sta > len(bin_list) - 1:
                    a.append(woe_list[-1])
                else:
                    for i in range(sta, len(bin_list) - 1):
                        if (x >= bin_list[i]) and (x < bin_list[i + 1]):
                            a.append(woe_list[i])
                            break
                        else:
                            a.append(woe_list[-1])
                            break
            return a
        except TypeError:
            return list(np.zeros(len(temp)))

    # 变量筛选(计算相似性矩阵)
    def filter_var(self, trains_x, tests_x, trains_y, tests_y, ths_):
        ths_ = float(ths_)
        print('设定IV筛选阈值为:', ths_)
        # 读取分箱
        df_bwi = pd.read_csv(self.Save_bins)
        print(df_bwi.shape)
        # 用分箱中的woe值进行替换
        temp_list = [x for x in df_bwi.columns]
        train_ = pd.concat([trains_x, trains_y], axis=1)
        test_ = pd.concat([tests_x, tests_y], axis=1)
        label_ = train_[['label']]
        for i in temp_list:
            try:
                train_[i] = self.replace(train_, i, df_bwi, ths_, 1)
            except ValueError:
                train_.drop([i], axis=1, inplace=True)
                test_.drop([i], axis=1, inplace=True)
            try:
                test_[i] = self.replace(test_, i, df_bwi, ths_, 0)
            except ValueError:
                train_.drop([i], axis=1, inplace=True)
                test_.drop([i], axis=1, inplace=True)

        # iv值筛选, 选取iv值大于阈值的特征
        print("iv值大于%s的变量数量:%d" % (ths_, len(self.filterIv)))
        # 对满足iv条件的变量计算相似性矩阵,选取线性无关的变量
        df_var = self.select_var(train_[self.filterIv])
        train_res = train_[df_var]
        df_res = pd.concat([label_, train_res], axis=1)
        return df_res, test_

    # 根据相似性筛选变量
    def select_var(self, df_):
        print('相似性阈值：', 0.2)
        p, q = [], []
        col = df_.columns
        df_ = df_.corr()
        arr = df_.as_matrix()
        arr_ = np.argwhere(np.tril(arr, k=0) > 0.2)
        for i in arr_:
            for j in range(0, len(i)):
                if j == 0:
                    p.append(i[j])
                elif j == 1:
                    q.append(i[j])
        h = [col[i] for i in p]
        g = [col[i] for i in q]
        fin_var = {}
        for i in list(zip(h, g)):
            if i[0] == i[1]:
                continue
            else:
                fin_var.setdefault(i[0], []).append(i[1])

        fin_list = []
        for key, value in fin_var.items():
            temp1 = list()
            temp1.append(key)
            temp2 = [j for j in value]
            fin_list.append(temp1 + temp2)

        b = list(set([j for i in fin_list for j in i]))
        b_ = [i for i in col if i not in b]
        a = list()
        for item in fin_list:
            temp_flag = 1
            for var_item in a:
                if var_item in item:
                    temp_flag = 0
            if temp_flag:
                temp_dict = dict()
                for value in item:
                    temp_dict[value] = self.iv[value]
                a.append(max(temp_dict, key=temp_dict.get))
        for var in a:
            b_.append(var)
        b_ = list(set(b_))
        print('筛选后变量个数:', len(b_))
        # 保存筛选后进入模型的变量名
        write_f = [line + "\n" for line in b_]
        with open('train_var.txt', 'w')as f:
            f.writelines(write_f)
        return b_

    # LR算法
    @staticmethod
    def lr_train(df_trains, df_tests):
        train_ = df_trains.reset_index(drop=True)
        test_ = df_tests.reset_index(drop=True)
        label_train = train_['label']
        label_test = test_['label']
        columns = [x for x in train_.columns if x not in ['info', 'month', 'label']]
        train_ = train_[columns]
        test_ = test_[columns]
        '''
        # 自定义网格搜索方法
        param_grid = {'param_C': [1e-3, 0.01, 0.1, 1, 10, 100, 1e3], 'param_penalty': ['l1', 'l2']}
        best_scores = 0.0
        best_param = dict()
        # 网格搜索最优参数训练模型
        for c in param_grid['param_C']:
            for penalty in param_grid['param_penalty']:
                lr_temp = LogisticRegression(C=c, penalty=penalty)
                scores = cross_val_score(lr_temp, train_, label_train, cv=5, scoring='roc_auc')
                score = scores.mean()  # 取平均得分
                if score > best_scores:
                    best_scores = score
                    best_param = {"C": c, "penalty": penalty}
        print("best_param:{}".format(best_param))
        lr = LogisticRegression(**best_param)
        lr.fit(train_, label_train)

        print('训练集效果...')
        pre_train = lr.predict(train_)
        prob_train = np.array(lr.predict_proba(train_))
        pre_list_train = list()
        for i in range(prob_train.shape[0]):
            pre_list_train.append(prob_train[i][1])
        MF.evaluate(label_train, pre_list_train, pre_train)

        print('验证集效果...')
        pre_test = lr.predict(test_)
        prob_test = np.array(lr.predict_proba(test_))
        pre_list_test = list()
        for i in range(prob_test.shape[0]):
            pre_list_test.append(prob_test[i][1])
        MF.evaluate(label_test, pre_list_test, pre_test)

        groups_count = 10
        temp_list = list(np.linspace(0, 1, groups_count+1))
        psi_tot = MF.get_psi(pre_list_train, pre_list_test, temp_list)
        print('PSI:', psi_tot)
        '''
        # 采用网格搜索框架
        lr = LogisticRegression()
        param_grid = {'C': [1e-3, 0.01, 0.1, 1, 10, 100, 1e3], 'penalty': ['l1', 'l2']}
        grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='roc_auc', cv=5)
        grid_search.fit(train_, label_train)
        print('best score: %f' % grid_search.best_score_)

        print('训练集效果...')
        pre_train = grid_search.predict(train_)
        prob_train = np.array(grid_search.predict_proba(train_))
        pre_list_train = list()
        for i in range(prob_train.shape[0]):
            pre_list_train.append(prob_train[i][1])
        MF.evaluate(label_train, pre_list_train, pre_train)

        print('验证集效果...')
        pre_test = grid_search.predict(test_)
        prob_test = np.array(grid_search.predict_proba(test_))
        pre_list_test = list()
        for i in range(prob_test.shape[0]):
            pre_list_test.append(prob_test[i][1])
        MF.evaluate(label_test, pre_list_test, pre_test)

        groups_count = 10
        temp_list = list(np.linspace(0, 1, groups_count+1))
        psi_tot = MF.get_psi(pre_list_train, pre_list_test, temp_list)
        print('PSI:', psi_tot)

        # 保存模型
        # joblib.dump(lr, "xy_New.m")

    # XGBoost算法算法
    @staticmethod
    def xgbt_train(trains_x, tests_x, trains_y, tests_y):
        dtrain = xgb.DMatrix(trains_x, label=trains_y)
        dtrain2 = xgb.DMatrix(trains_x)
        dtest = xgb.DMatrix(tests_x)
        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'learning_rate': 0.06,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'eta': 0.1,
                  'max_depth': 4,
                  'min_child_weight': 1,
                  'gamma': 0.0,
                  'silent': 1,
                  'seed': 0,
                  'eval_metric': 'auc',
                  'njob': 4}
        bst = xgb.train(params, dtrain, 10)
        print('训练集效果...')
        prob_train = bst.predict(dtrain2)
        pre_train = [round(value) for value in prob_train]
        label_train = trains_y.tolist()
        prob_train = prob_train.tolist()
        MF.evaluate(label_train, prob_train, pre_train)

        print('验证集效果...')
        prob_test = bst.predict(dtest)
        pre_test = [round(value) for value in prob_test]
        label_test = tests_y.tolist()
        prob_test = prob_test.tolist()
        MF.evaluate(label_test, prob_test, pre_test)

        groups_count = 10
        temp_list = list(np.linspace(0, 1, groups_count+1))
        psi_tot = MF.get_psi(pre_train, pre_test, temp_list)
        print('psi:', psi_tot)

        print('lr入模变量再xgbt上的效果')
        fin_var = list()
        with open('train_var.txt', 'r')as f:
            for line in f:
                fin_var.append(line.strip('\n').split(',')[0])
        dtrain3 = xgb.DMatrix(trains_x[fin_var], label=trains_y)
        dtrain4 = xgb.DMatrix(trains_x[fin_var])
        dtest3 = xgb.DMatrix(tests_x[fin_var])
        bst2 = xgb.train(params, dtrain3, 10)
        print('训练集效果...')
        prob_train2 = bst2.predict(dtrain4)
        pre_train2 = [round(value) for value in prob_train2]
        label_train2 = trains_y.tolist()
        prob_train2 = prob_train2.tolist()
        MF.evaluate(label_train2, prob_train2, pre_train2)

        print('验证集效果...')
        prob_test2 = bst2.predict(dtest3)
        pre_test2 = [round(value) for value in prob_test2]
        label_test2 = tests_y.tolist()
        prob_test2 = prob_test2.tolist()
        MF.evaluate(label_test2, prob_test2, pre_test2)

        psi_tot2 = MF.get_psi(pre_train2, pre_test2, temp_list)
        print('psi:', psi_tot2)

    # LR流程
    def start_lr(self):
        # 判断当前路径下是否存在该文件
        if not os.path.exists(self.Train_data):
            print('检查原始数据文件')
        # 判断是否有特征衍生数据
        if not os.path.exists(self.Save_features):
            demo.get_var()
        # 切分训练集和验证集
        train_x, test_x, train_y, test_y = demo.load_data()
        # 判断是否有分箱数据
        if not os.path.exists(self.Save_bins):
            demo.start_bin(train_x, train_y)
        # 根据IV值,相似性系数筛选入模变量
        df_train, df_test = demo.filter_var(train_x, test_x, train_y, test_y, 0.03)
        demo.lr_train(df_train, df_test)

    # XGBoost算法
    def start_xgbt(self):
        # 判断当前路径下是否存在该文件
        if not os.path.exists(self.Train_data):
            print('请确认数据路径...')
            sys.exit(0)

        df = pd.read_csv(self.Train_data)
        df.columns = ['info'] + ['m'] + list(range(1, df.shape[1] - 2)) + ['label'] + ['is_train']
        df = df.sort_values(['info', 'm'], ascending=[1, 0])
        cols = ['info'] + ['m'] + list(range(1, df.shape[1] - 2))
        if not os.path.exists(self.Save_features):
            version = int(input("请输入需要衍生的版本...(0:汇总、1:老逾期、2:新逾期、3:重构V1、4:重构V2、5:重构V3)"))
            if version == 0:
                demo.get_var_v0(df[cols])
            elif version == 1:
                demo.get_var_v1(df[cols])
            elif version == 2:
                demo.get_var_v2(df[cols])
            elif version == 3:
                demo.get_var_v3(df[cols])
            elif version == 4:
                demo.get_var_v4(df[cols])
            elif version == 5:
                demo.get_var_v5(df[cols])
            else:
                print("输入有误...程序终止")
                sys.exit(0)

        train_x, test_x, train_y, test_y = demo.load_data()
        demo.xgbt_train(train_x, test_x, train_y, test_y)


if __name__ == '__main__':
    data_path = sys.argv[1]
    feature_path = sys.argv[2]
    bin_path = sys.argv[3]
    start = time.clock()
    demo = MF(data_path, feature_path, bin_path)
    # demo.start_lr()
    demo.start_xgbt()
    end = time.clock()
    print("消耗时间：%f s" % (end - start))
