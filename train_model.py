from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, confusion_matrix, accuracy_score, recall_score, precision_score
import re
import sys
import time
import math
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn import preprocessing
import scipy.cluster.hierarchy as sch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


class Model(object):
    def __init__(self):
        # Data Path
        self.TRAINDATA = './data/mxf_fea_HZ.csv'
        self.fea_savePath = './data/mg_data.txt'
        self.bin_savePath = './data/mg_data_bin.csv'
        self.TIME = datetime.datetime.strptime('201905', '%Y%m')
        self.bin = dict()
        self.woe = dict()
        self.iv = dict()
        self.filterIv = list()

    # 评估模型
    def _evaluate(self, label, predict_p, predict_label):
        conf_matrix = confusion_matrix(label, predict_label)
        print('混淆矩阵：\n' + str(conf_matrix))
        acc = accuracy_score(label, predict_label)
        print('accuracy：' + str(round(acc, 4)))
        pre = precision_score(label, predict_label)
        print('precision：' + str(round(pre, 4)))
        rec = recall_score(label, predict_label)
        print('recall：' + str(round(rec, 4)))
        F1_score = (2 * pre * rec) / (pre + rec)
        print('F1_score：' + str(round(F1_score, 4)))
        fpr, tpr, thresholds = roc_curve(label, predict_p)
        AUC = auc(fpr, tpr)
        print('AUC：' + str(round(AUC, 4)))
        Ks = max(tpr - fpr)
        print('Ks: ' + str(round(Ks, 4)))

    # 计算模型psi指标
    def calc_psi(self, expected, actual, qlist):
        expected_len = len(expected)
        actual_len = len(actual)
        psi_tot = 0
        for j in range(len(qlist)-1):
            actual_pct = len([x for x in actual if x >= qlist[j] and x < qlist[j+1]]) / actual_len
            expected_pct = len([x for x in expected if x >= qlist[j] and x < qlist[j+1]]) / expected_len
            print(qlist[j], qlist[j+1])
            # print(expected_pct, actual_pct)
            if (actual_pct > 0) and (expected_pct > 0):
                # print(actual_pct - expected_pct, np.log(actual_pct/expected_pct))
                psi_cut = (actual_pct - expected_pct)*np.log(actual_pct/expected_pct)
                print(psi_cut)
                psi_tot += psi_cut
        return np.round(psi_tot,7)

    # 自定义求和函数
    def _sum(self, var):
        if var.count() is 0:
            return np.nan
        else:
            return var.sum()

    # 变量衍生方法
    def _varCntiNormal(self, data, var, times):
        print('%s：varCntiNormal...' % times)
        df = data.groupby(0)[var]
        df_mean = df.mean().round(3).reset_index(drop=True).add_prefix(str(times) + '_avg_')
        df_max = df.max().reset_index(drop=True).add_prefix(str(times) + '_max_')
        df_min = df.min().reset_index(drop=True).add_prefix(str(times) + '_min_')
        df_sum = df.agg(lambda x: self._sum(x)).reset_index(drop=True).add_prefix(str(times) + '_sum_')
        return df_mean, df_max, df_min, df_sum

    def _varCntiMax(self, data, var, times):
        print('%s：varCntiMax...' % times)
        df = data.groupby(0)[var]
        df_mean = df.mean().round(3).reset_index(drop=True).add_prefix(str(times) + '_avg_')
        df_max = df.max().reset_index(drop=True).add_prefix(str(times) + '_max_')
        df_min = df.min().reset_index(drop=True).add_prefix(str(times) + '_min_')
        return df_mean, df_max, df_min

    def _varCntiOverdue(self, data, var, times):
        print('%s：varCntiOverdue...' % times)
        df = data.groupby(0)[var]
        df_YN = df.max().reset_index(drop=True).add_prefix(str(times) + '_YN_')
        return df_YN

    def _varOnce(self, data, var, times):
        print('%s：varOnce...' % times)
        df = data.groupby(0)[var]
        df_last = df.last().reset_index(drop=True).add_prefix(str(times) + '_last_')
        df_first = df.first().reset_index(drop=True).add_prefix(str(times) + '_first_')
        return df_last, df_first

    # 特征提取
    def extractFeature(self):
        print('read data...')
        df = pd.read_csv(self.TRAINDATA)
        df_columns = [str(x) for x in list(range(0, 215))] + ['label']
        online_df = df[df_columns].sort_values(['0', '1'], ascending=[1, 0])
        online_df.columns = list(range(0, 215)) + ['label']

        print('process data...')
        time_features = [1, 157, 158, 159, 160, 164, 165]
        for var in time_features:
            online_df[var].fillna('201905', inplace=True)
            online_df[var] = online_df[var].map(int)
            online_df[var] = online_df[var].map(str)
            online_df[var] = online_df[var].apply(lambda x: (self.TIME - datetime.datetime.strptime(x, '%Y%m')).days)

        cnti_amount_features = [2] + list(range(3, 75, 2)) + list(range(98, 136, 2)) + [143, 145, 146]
        cnti_count_features = list(range(4, 76, 2)) + list(range(99, 137, 2)) + [144]
        cnti_max_features = list(range(75, 98)) + list(range(136, 143))
        once_features = list(range(148, 170)) + list(range(176, 182)) + list(range(187, 192)) + list(range(201, 211))
        overdue_features = list(range(197, 201)) + list(range(211, 215))

        map_overdue = {np.nan: np.nan, 0: 0, "M1": 1, "M2": 2, "M3": 3, "M3+": 4}
        online_df[195] = online_df[195].map(map_overdue)
        online_df[204] = online_df[204].map(map_overdue)

        var_cnti = cnti_amount_features + cnti_count_features + list(range(170, 176)) + list(range(182, 187)) + list(range(192, 197))
        df_group = online_df.groupby(online_df[0])
        rctMth_12 = df_group.head(12)
        rctMth_6 = df_group.head(6)
        rctMth_3 = df_group.head(3)
        rctMth_1 = df_group.head(1).reset_index()
        df_1 = df_group.head(1).reset_index(drop=True).drop(['label'], axis=1).add_prefix(str(1) + '_')

        times = time.time()
        print('Extract 12 features...')
        dfLaOn12, dfEaOn12 = self._varOnce(rctMth_12, once_features, 12)
        dfMeNo12, dfMaNo12, dfMiNo12, dfSuNo12 = self._varCntiNormal(rctMth_12, var_cnti, 12)
        dfMeMa12, dfMaMa12, dfMiMa12 = self._varCntiMax(rctMth_12, cnti_max_features, 12)
        dfYn12 = self._varCntiOverdue(rctMth_12, overdue_features, 12)

        print('Extract 6 features...')
        dfLaOn6, dfEaOn6 = self._varOnce(rctMth_6, once_features, 6)
        dfMeNo6, dfMaNo6, dfMiNo6, dfSuNo6 = self._varCntiNormal(rctMth_6, var_cnti, 6)
        dfMeMa6, dfMaMa6, dfMiMa6 = self._varCntiMax(rctMth_6, cnti_max_features, 6)
        dfYn6 = self._varCntiOverdue(rctMth_6, overdue_features, 6)

        print('Extract 3 features...')
        dfLaOn3, dfEaOn3 = self._varOnce(rctMth_3, once_features, 3)
        dfMeNo3, dfMaNo3, dfMiNo3, dfSuNo3 = self._varCntiNormal(rctMth_3, var_cnti, 3)
        dfMeMa3, dfMaMa3, dfMiMa3 = self._varCntiMax(rctMth_3, cnti_max_features, 3)
        dfYn3 = self._varCntiOverdue(rctMth_3, overdue_features, 3)
        print('Extract features finished...', time.time() - times)

        fin_data = pd.concat([dfLaOn12, dfEaOn12, dfYn12,
                              dfMeNo12, dfMaNo12, dfMiNo12, dfSuNo12,
                              dfMeMa12, dfMaMa12, dfMiMa12,
                              dfLaOn6, dfEaOn6, dfYn6,
                              dfMeNo6, dfMaNo6, dfMiNo6, dfSuNo6,
                              dfMeMa6, dfMaMa6, dfMiMa6,
                              dfLaOn3, dfEaOn3, dfYn3,
                              dfMeNo3, dfMaNo3, dfMiNo3, dfSuNo3,
                              dfMeMa3, dfMaMa3, dfMiMa3, df_1], axis=1)
        fin_data['info'] = rctMth_1[0]
        print(fin_data)
        fin_data['label'] = rctMth_1['label']
        print(fin_data['info'].head(5))
        fin_data.to_csv(self.fea_savePath, index=False)

    # 采用best-ks进行分箱
    def _getBin(self, df, var_name, bin_num):
        res_bin, filter_list = list(),list()
        df = df[[var_name, 'label']]
        temp_df = df.copy()
        temp_df.fillna('nan', inplace=True)
        var_list = pd.crosstab(temp_df[var_name],temp_df['label']).index.tolist()
        # var_list = list(np.unique(df[var_name]))
        for i in var_list:
            if i == 'nan':
                res_bin.append(np.nan)
            # 加入对3000变量的分箱
            elif i < 0:
                res_bin.append(i)
            else:
                filter_list.append(i)
        # rate = df.shape[0] / 20
        rate = pd.notnull(df[var_name]).sum() / 10
        df.dropna(axis=0, inplace=True)
        df = df[df[var_name] >= 0]
        df_ = df.copy()
        def ks_bin(df_, rate_):
            F = df_.iloc[:,1].value_counts()[0]  # label=1
            T = df_.iloc[:,1].value_counts()[1]  # label=0
            df_cro = pd.crosstab(df_.iloc[:,0], df_.iloc[:,1])
            df_cro[0] = df_cro[0] / F
            df_cro[1] = df_cro[1] / T
            df_cro_cum = df_cro.cumsum()
            ks_list = abs(df_cro_cum[1] - df_cro_cum[0])
            ks_list_index = ks_list.nlargest(len(ks_list)).index.tolist()
            for i in ks_list_index:
                df_1 = df_[df_.iloc[:, 0] <= i]
                df_2 = df_[df_.iloc[:, 0] > i]
                if len(df_1) >= rate_ and len(df_2) >= rate_:
                    break
            return i

        def ks_zone(df_, list_):
            list_zone = list()
            list_.sort()
            n = 0
            for i in list_:
                m = sum(df_.iloc[:, 0] <= i) - n
                n = sum(df_.iloc[:, 0] <= i)
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

        bins = list()
        for i in range(bin_num - 1):
            ks_ = ks_bin(df_, rate)
            bins.append(ks_)
            new_bins = ks_zone(df, bins)
            df_ = df[(df.iloc[:, 0] > new_bins[0]) & (df.iloc[:, 0] <= new_bins[1])]
            if df_.iloc[:, 1].value_counts().shape[0] < 2:
                break
        real_bins = res_bin + bins
        bins.append(max(filter_list))
        bins.append(min(filter_list))
        temp_bins = res_bin + sorted(list(set(bins)))
        return real_bins, temp_bins

    # 计算Iv值
    def _cal_iv(self, df_, var_name, bins_, flag):
        res_iv, res_bins, start = 0,[],0
        total_1 = df_['label'].sum()  # label=1的样本
        total_0 = df_.shape[0] - total_1  # label=0的样本
        # 判断flag,0代表离散变量,1代表连续变量
        if flag:
            temp_bins, temp_bins2, temp1_arr_1, temp1_arr_0, temp2_arr_1, temp2_arr_0 = list(),list(),list(),list(),list(),list()
            # 3千变量的iv值计算
            '''
            for i in bins_:
                if (np.isnan(i)) or (i < 0):
                    temp_bins2.append(i)
                else:
                    temp_bins.append(i)
            # 处理小于0 和 nan值的woe及iv
            # print(df_[df_[var_name].isnull()]['label'].value_counts())
            temp_df = df_[df_[var_name].isin(temp_bins2)]
            temp_df.fillna('nan', inplace=True)
            temp_df_cro = pd.crosstab(temp_df[var_name],temp_df['label'])
            temp_dict = dict()
            for index,data in temp_df_cro.iterrows():
                temp_dict[index] = data
            temp_bins2_ = ['nan' if np.isnan(i) else i for i in temp_bins2]
            # print(temp_df_cro)
            # print(temp_bins2_)
            # print(temp_dict)
            for i in temp_bins2_:
                try:
                    temp1_arr_1.append(temp_dict[i][1])
                except KeyError:
                    temp1_arr_1.append(0.001)
                try:
                    temp1_arr_0.append(temp_dict[i][0])
                except KeyError:
                    temp1_arr_0.append(0.001)
            '''
            # 汇总层变量的iv值计算
            for i in bins_:
                if np.isnan(i):
                    start = 1
                else:
                    temp_bins.append(i)
            if start:
                try:
                    temp1_arr_1.append(df_[df_[var_name].isnull()]['label'].value_counts()[1])
                except KeyError:
                    temp1_arr_1.append(0.001)
                try:
                    temp1_arr_0.append(df_[df_[var_name].isnull()]['label'].value_counts()[0])
                except KeyError:
                    temp1_arr_0.append(0.001)

            z = [(temp_bins[i],temp_bins[i+1]) for i in range(len(temp_bins)-1)]
            for x in z:
                # print(df_[(df_[var_name]>=x[0])&(df_[var_name]<x[1])]['label'].value_counts())
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
            # print(var_name)
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
            for i in bins_:
                if np.isnan(i):
                    nan_flag = 1
        # print(bins_)
        # print(arr_1,arr_0)
        arr_1_p = list(map(float, arr_1)) / total_1
        arr_0_p = list(map(float, arr_0)) / total_0
        woe = [math.log(x, math.e) for x in arr_1_p/arr_0_p]
        # if sorted(woe_, reverse=False)==list(woe_) and sorted(arr_0_p_, reverse=True)==list(arr_0_p_):
        #     res_iv = sum(woe*(arr_1_p-arr_0_p))
        # elif sorted(woe_, reverse=True)==list(woe_) and sorted(arr_0_p_, reverse=False)==list(arr_0_p_):
        #     res_iv = sum(woe*(arr_1_p-arr_0_p))
        res_iv = sum(woe*(arr_1_p-arr_0_p))

        if flag:
            return woe, res_iv
        else:
            return bins_, woe, res_iv

    # 替换woe
    def _replace(self, df_, var, column_dict, ths, flag):
        try:
            pattern = re.compile('\[(.*)\]')
            temp_list2 = len(pd.unique(df_[var]))
            temp = df_[var].values.tolist()
            a = list()
            bin_dict = dict()
            line = column_dict[var].values.tolist()
            bin_list = [float(i) for i in pattern.findall(line[0])[0].split(',')]
            temp_list = [i for i in pattern.findall(line[1])[0].split(',')]
            filter_list = []
            woe_list = [0 if i == '' else float(i) for i in temp_list]
            iv = float(line[2])
            # print(bin_list, woe_list, iv)
            # print(pd.unique(df_[var]))
            if (iv >= ths) and (flag == 1) and len(bin_list)<10 and iv < 2:
                self.iv[var] = iv
                self.bin[var] = bin_list
                self.woe[var] = woe_list
                self.filterIv.append(var)
            sta = 0
            if temp_list2 <= len(bin_list):
                for num in bin_list:
                    bin_dict[num] = woe_list[sta]
            else:
                for num in bin_list:
                    if np.isnan(num) or num < 0:
                        bin_dict[num] = woe_list[sta]
                        sta += 1
                    else:
                        filter_list.append(num)
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
            # print(bin_dict.keys())
            for x in temp:
                if x in bin_dict.keys():
                    a.append(bin_dict[x])
                elif sta > len(bin_list)-1:
                    a.append(woe_list[-1])
                else:
                    # print(x)
                    for i in range(sta, len(bin_list)-1):
                        if(x >= bin_list[i]) and (x < bin_list[i+1]):
                            a.append(woe_list[i])
                            break
                        else:
                            a.append(woe_list[-1])
                            break
            # print(var, len(df_[var]), len(a))
            return a
        except TypeError:
            return list(np.zeros(len(temp)))

    def _selectFeatures(self, df_):
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
                var = fin_var.setdefault(i[0], []).append(i[1])
                # var.append(i[1])
                fin_var[i[0]] = var

        fin_list = []
        for key, value in fin_var.items():
            temp_list = []
            temp_list.append(key)
            for j in value:
                temp_list.append(j)
            fin_list.append(temp_list)

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
        print('筛选后变量个数：', len(b_))
        return b_

    # 变量筛选(计算相似性矩阵)
    def filterVar(self, train_X, test_X, train_y, test_y, ths_):
        ths_ = float(ths_)
        print('设定IV筛选阈值为:', ths_)
        # 读取分箱
        df_bwi = pd.read_csv(self.bin_savePath)
        print(df_bwi.shape)
        # 用分箱中的woe值进行替换
        temp_list = [x for x in df_bwi.columns]
        df_train = pd.concat([train_X, train_y],axis=1)
        df_test = pd.concat([test_X, test_y],axis=1)
        df_label = df_train[['label']]
        # temp_list = ['1_127']
        for i in temp_list:
            try:
                df_train[i] = self._replace(df_train, i, df_bwi, ths_, 1)
            except ValueError:
                df_train.drop([i], axis=1, inplace=True)
                df_test.drop([i], axis=1, inplace=True)
            try:
                df_test[i] = self._replace(df_test, i, df_bwi, ths_, 0)
            except ValueError:
                df_train.drop([i], axis=1, inplace=True)
                df_test.drop([i], axis=1, inplace=True)

        # iv值筛选,选取iv值大于阈值的特征
        print("iv值大于%s的变量数量:%d" % (ths_, len(self.filterIv)))
        # 对满足IV条件的变量计算相似性矩阵，挑选线性无关变量
        df_Var = self._selectFeatures(df_train[self.filterIv])
        for i in df_Var:
            print(i)
            print(self.bin[i])
            print(self.woe[i])
            print(self.iv[i])
        df_train_ = df_train[df_Var]
        df_data = pd.concat([df_label, df_train_], axis=1)
        return df_data, df_test

    # 开始分箱
    def startBin(self, train_X, test_X, train_y, test_y):
        df_train = pd.concat([train_X, train_y], axis=1)
        df_test = pd.concat([test_X, test_y], axis=1)
        print('start bin...')
        overdue_feature = list(range(197, 201)) + list(range(206, 210)) + list(range(211, 215))
        overdue_features = [str(x) for x in overdue_feature]
        # df = df[['info', '12_sum_143', 'label']]
        var_list = [var for var in df_train.columns if var not in ['info', 'month', 'label', '0', '1']]
        # temp_feature = [str(x) for x in list(range(192, 215))]
        # temp_list = []
        # for var in var_list:
        #     if var.split('_')[-1] in temp_feature:
        #         temp_list.append(var)
        total_bwi = dict()
        for var in var_list:
            print(var)
            # if var.split('_')[-1] in ['217', '218', '238']:
            #     continue
            df_train[var] = df_train[var].map(float)
            # 标准、逾期、多投、画像用以下flag
            # flag = 0 if var.split('_')[-1] in overdue_features else 1  # 0是离散变量,1是连续变量
            # 三千变量用以下flag
            flag = 0 if pd.crosstab(df_train[var], df_train['label']).shape[0] <= 5 else 1

            temp_list = list()
            if flag:
                res_bins = self._getBin(df_train, var, 3)
                res_iv = self._cal_iv(df_train, var, res_bins[1], flag)
                temp_list.append(res_bins[1])
                temp_list.append(res_iv[0])
                temp_list.append(res_iv[1])
                total_bwi[var] = temp_list
            else:
                res_iv = self._cal_iv(df_train, var, [], flag)
                temp_list.append(res_iv[0])
                temp_list.append(res_iv[1])
                temp_list.append(res_iv[2])
                total_bwi[var] = temp_list
            print(temp_list)
        print('save result...')
        df_bin = pd.DataFrame.from_dict(total_bwi, orient='index').T
        df_bin.to_csv(self.bin_savePath, index=False)

    def train(self, df_train, df_test):
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        label_train = df_train['label']
        label_test = df_test['label']
        columns = [x for x in df_train.columns if x not in ['info', 'month', 'label', '0', '1']]
        train_ = df_train[columns]
        test_ = df_test[columns]
        model_cls = LogisticRegression()
        model_cls.fit(train_, label_train)
        # print(model_cls.fit(train_x, train_y).coef_)
        # print(model_cls.fit(train_x, train_y).intercept_)
        print('训练集效果...')
        pred_y_train = model_cls.predict(train_)
        pred_proba_train = model_cls.predict_proba(train_)
        pred_proba_train = np.array(pred_proba_train)
        # print(pred_proba.shape)
        pred_list_train = list()
        for i in range(pred_proba_train.shape[0]):
            pred_list_train.append(pred_proba_train[i][1])
        self._evaluate(label_train, pred_list_train, pred_y_train)

        print('验证集效果...')
        pred_y_test = model_cls.predict(test_)
        pred_proba_test = model_cls.predict_proba(test_)
        pred_proba_test = np.array(pred_proba_test)
        # print(pred_proba.shape)
        pred_list_test = list()
        for i in range(pred_proba_test.shape[0]):
            pred_list_test.append(pred_proba_test[i][1])
        self._evaluate(label_test, pred_list_test, pred_y_test)
        # 有问题的PSI计算
        cut_ = list(np.linspace(0,1,11))
        cut_train = pd.cut(pred_list_train, cut_)
        cut_test = pd.cut(pred_list_test, cut_)
        temp_train = pd.value_counts(cut_train)
        temp_test = pd.value_counts(cut_test)
        temp_train_ = sorted(temp_train.index, reverse=True)
        temp_test_ = sorted(temp_test.index, reverse=True)
        temp_train = temp_train[temp_train_]*1.0 / len(pred_list_train)
        temp_test = temp_test[temp_test_]*1.0 / len(pred_list_test)
        para_1 = temp_test - temp_train
        para_2 = temp_test/temp_train
        para_2 = para_2.map(lambda x: np.log(x))
        para_res = para_1 * para_2
        para_res.replace(np.nan, 0, inplace=True)
        psi = sum(para_res)
        print('psi:', np.round(psi,5))

        groups_count = 10
        qlist = list(np.linspace(0,1,groups_count+1))
        psi_tot = self.calc_psi(pred_list_train, pred_list_test, qlist)
        print('psi:', psi_tot)

        # 保存模型
        # joblib.dump(model_cls, "xy_New.m")

    # 加载模型数据(未切分数据)
    def loadData(self):
        print('load total data...')
        # df = pd.read_table(self.fea_savePath, sep='\t', header=None)  # 3000变量的读取
        # df.columns = ['info','month'] + list(range(0, df.shape[1]-2))
        # df = pd.read_table(self.fea_savePath, sep='\t')
        # df.columns = ['info', 'banknum', 'IDCard', 'label']

        df_label = pd.read_csv('./data/14000_label.txt', sep=',')
        df_var = pd.read_csv('./data/14000_var.txt', sep=',')
        df_label.columns = ['info', 'month', 'label']
        df_label.drop(['month'], axis=1, inplace=True)
        # dict_label = {'RESULT-PASS':0,'RESULT-REFUSE':1,'RESULT-PHONE':2}
        # dict_var = {'false':0, 'true':1, np.nan:2}
        # df_label['label'] = df_label['label'].map(dict_label)
        # df_var.columns = ['info', 'name', 'IDCard', 'phone_gray_score', 'searched_org_cnt', 'blacklist_name_with_idcard', 'blacklist_name_with_phone', 'etl_date']
        df_var.columns = ['m', 'info'] + list(range(0, df_var.shape[1]-2))
        df = pd.merge(df_var, df_label, on=['info'])
        print(df.shape)
        # df['blacklist_name_with_idcard'] = df['blacklist_name_with_idcard'].map(dict_var)
        # df['blacklist_name_with_phone'] = df['blacklist_name_with_phone'].map(dict_var)
        df['label'] = df['label'].map(int)
        df_t = df[df['label'] == 0]  # 无逾期 22804
        df_f = df[df['label'] == 1]  # 逾期>2 3315
        print(df_t.shape,df_f.shape)
        df_t = df_t.sample(n=10000)
        # df_t = df_t.iloc[7000:10000,]
        df_f = df_f.sample(n=10000)
        # df_f = df_f.iloc[:3000,]
        print(df_t.shape,df_f.shape)
        df_ = df_t.append(df_f)
        print(df_.shape)

        # df_['IDCard'].fillna('0000002019X', inplace=True)
        # df_['age'] = df_['IDCard'].map(lambda x: str(x)[6:10])
        # df_['age'] = df_['age'].map(lambda x: 2019-int(x))
        # df_.drop(['IDCard'], axis=1, inplace=True)

        train_label = df_['label']
        train_columns = [x for x in df_.columns if x not in ['info', 'month', 'label', '0', '1', 'etl_date', 'name', 'm']]
        train_data = df_[train_columns]
        train_x, test_x, train_y, test_y = train_test_split(train_data, train_label, test_size=0.2, random_state=0)

        return train_x, test_x, train_y, test_y

    # 加模型数据(切分好的数据)
    def loadData2(self):
        print('load data...')
        df_train = pd.read_csv('ori_train.csv')
        df_t = df_train[df_train['label'] == 0]
        # df_t['label'] = 1
        df_f = df_train[df_train['label'] == 1]
        # df_f['label'] = 0
        df_t = df_t.iloc[:2700]
        df_f = df_f.iloc[:2700]
        df_train = df_t.append(df_f)
        df_test = pd.read_csv('ori_test.csv')
        train_X = df_train.drop(['label'], axis=1)
        train_y = df_train['label']
        test_X = df_test.drop(['label'], axis=1)
        test_y = df_test['label']
        return train_X, test_X, train_y, test_y


if __name__ == '__main__':
    start = time.clock()
    demo = Model()
    # demo.extractFeature()
    train_x, test_x, train_y, test_y = demo.loadData()
    demo.startBin(train_x, test_x, train_y, test_y)
    df_train, df_test = demo.filterVar(train_x, test_x, train_y, test_y, 0.03)
    demo.train(df_train, df_test)
    end = time.clock()
    print("消耗时间：%f s" % (end - start))