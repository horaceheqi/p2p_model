import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics


class MF(object):
    def __init__(self, data):
        self.Train_data = data
        self.Time = datetime.datetime.strptime('20190722', '%Y%m%d')

    # 模型评估
    @staticmethod
    def evaluate(label, predict_prob, predict_label):
        conf_matrix = metrics.confusion_matrix(label, predict_label)
        print('混淆矩阵：\n' + str(conf_matrix))
        acc = metrics.accuracy_score(label, predict_label)
        print('accuracy:' + str(round(acc, 4)))
        pre = metrics.precision_score(label, predict_label)
        print('precission:' + str(round(pre, 4)))
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
        df = data.groupby(['mobile', 'm'])[var]
        df_mean = df.mean().round(3).reset_index(drop=True).add_prefix(str(times) + '_avg_')
        df_max = df.max().round(3).reset_index(drop=True).add_prefix(str(times) + '_max_')
        df_min = df.min().round(3).reset_index(drop=True).add_prefix(str(times) + '_min_')
        df_sum = df.agg(lambda x: MF._sum(x)).reset_index(drop=True).add_prefix(str(times) + '_sum_')
        return df_mean, df_max, df_min, df_sum

    @staticmethod
    def var_max(data, var, times):
        print('%s: var_Max...' % times)
        df = data.groupby(['mobile', 'm'])[var]
        df_mean = df.mean().round(3).reset_index(drop=True).add_prefix(str(times) + '_avg_')
        df_max = df.max().round(3).reset_index(drop=True).add_prefix(str(times) + '_max_')
        df_min = df.min().round(3).reset_index(drop=True).add_prefix(str(times) + '_min_')
        return df_mean, df_max, df_min

    @staticmethod
    def var_overdue(data, var, times):
        print('%s: var_Overdue...' % times)
        df = data.groupby(['mobile', 'm'])[var]
        df_yn = df.max().reset_index(drop=True).add_prefix(str(times) + '_YN_')
        return df_yn

    @staticmethod
    def var_once(data, var, times):
        print('%s: var_Once...' % times)
        df = data.groupby(['mobile', 'm'])[var]
        df_last = df.last().reset_index(drop=True).add_prefix(str(times) + '_last_')
        df_first = df.first().reset_index(drop=True).add_prefix(str(times) + '_first_')
        return df_last, df_first

    # 加载数据(未切分样本、标签)
    @staticmethod
    def load_data():
        print('load features...')
        df = pd.read_csv('./Data/MergeData_X_feature.csv')
        df['label'] = df['label'].map(int)
        df_t = df[df['label'] == 0]  # 好人
        df_f = df[df['label'] > 1]  # 坏人
        df_f['label'] = 1
        print(df_t.shape, df_f.shape)
        df_t = df_t.sample(df_f.shape[0])
        df_f = df_f.sample(df_f.shape[0])
        df_ = df_t.append(df_f)
        print(df_.shape)

        train_label = df_['label']
        train_columns = [x for x in df_.columns if x not in ['mobile', 'label', 'month', 'm']]
        train_data = df_[train_columns]
        df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(train_data, train_label, test_size=0.2, random_state=0)
        return df_train_x, df_test_x, df_train_y, df_test_y

    @staticmethod
    # 特征提取_主要是针对重构逾期变量
    def get_var(df_):
        cols = df_.columns
        print('Load data to get more var...')
        newfeature_cols = []
        # 历史逾期分量/历史逾期总量
        his_var = list(range(3, 10))
        for col in his_var:
            df_["his_" + str(col)] = df_.iloc[:, col] / df_.loc[:, 2]  # 分/总
            newfeature_cols.append("his_" + str(col))

        # 近一年逾期分量/近一年逾期总量
        one_var = list(range(20, 27))
        for col in one_var:
            df_["one_" + str(col)] = df_.iloc[:, col] / df_.loc[:, 19]  # 分/总
            newfeature_cols.append("one_" + str(col))

        # 近半年的逾期分量/近半年的逾期总量
        half_var = list(range(33, 40))
        for col in half_var:
            df_["half_" + str(col)] = df_.iloc[:, col] / df_.loc[:, 32]  # 分/总
            newfeature_cols.append("half_" + str(col))

        # 近一年逾期笔数/历史逾期笔数
        df_["one_his_bishu"] = df_.loc[:, 19] / df_.loc[:, 2]
        newfeature_cols.append("one_his_bishu")
        # 近半年逾期笔数/历史逾期笔数
        df_["half_his_bishu"] = df_.loc[:, 32] / df_.loc[:, 2]
        newfeature_cols.append("half_his_bishu")
        # 近一年存在逾期最大卡数量/历史逾期最大卡数量
        df_["one_his_card"] = df_.loc[:, 30] / df_.loc[:, 17]
        newfeature_cols.append("one_his_card")
        # 近半年存在逾期最大卡数量/历史逾期最大卡数量
        df_["half_his_card"] = df_.loc[:, 43] / df_.loc[:, 17]
        newfeature_cols.append("half_his_card")
        # 近一年M1逾期笔数/历史M1逾期笔数
        df_["one_his_M1"] = df_.loc[:, 20] / df_.loc[:, 3]
        newfeature_cols.append("one_his_M1")
        # 近半年M1逾期笔数/历史M1逾期笔数
        df_["half_his_M1"] = df_.loc[:, 33] / df_.loc[:, 3]
        newfeature_cols.append("half_his_M1")
        # 近一年M2逾期笔数/历史M2逾期笔数
        df_["one_his_M2"] = df_.loc[:, 21] / df_.loc[:, 4]
        newfeature_cols.append("one_his_M2")
        # 近半年M2逾期笔数/历史M2逾期笔数
        df_["half_his_M2"] = df_.loc[:, 34] / df_.loc[:, 4]
        newfeature_cols.append("half_his_M2")
        # 近一年M3逾期笔数/历史M3逾期笔数
        df_["one_his_M3"] = df_.loc[:, 22] / df_.loc[:, 5]
        newfeature_cols.append("one_his_M3")
        # 近半年M3逾期笔数/历史M3逾期笔数
        df_["half_his_M3"] = df_.loc[:, 35] / df_.loc[:, 5]
        newfeature_cols.append("half_his_M3")
        # 近一年M4逾期笔数/历史M4逾期笔数
        df_["one_his_M4"] = df_.loc[:, 23] / df_.loc[:, 6]
        newfeature_cols.append("one_his_M4")
        # 近半年M4逾期笔数/历史M4逾期笔数
        df_["half_his_M4"] = df_.loc[:, 36] / df_.loc[:, 6]
        newfeature_cols.append("half_his_M4")

        newfeature_cols += list(cols)
        fin_data = df_[newfeature_cols]
        print(fin_data.shape)
        return fin_data

    @staticmethod
    # 特征提取_主要是针对汇总层变量
    def get_var2(df_):
        print('Process data...')
        time_features = ['month', 156, 157, 158, 159, 163, 164]
        for var in time_features:
            df_[var].fillna('201908', inplace=True)
            df_[var] = df_[var].map(int)
            df_[var] = df_[var].map(str)
            df_[var] = df_[var].apply(lambda x: (datetime.datetime.strptime('201908', '%Y%m') - datetime.datetime.strptime(x, '%Y%m')).days)

        ct_amount_var = [1] + list(range(2, 74, 2)) + list(range(97, 135, 2)) + [142, 144, 145]
        ct_count_var = list(range(3, 75, 2)) + list(range(98, 136, 2)) + [143]
        ct_max_var = list(range(74, 97)) + list(range(135, 142))
        once_var = list(range(147, 169)) + list(range(175, 181)) + list(range(186, 191)) + list(range(200, 210))
        overdue_var = list(range(196, 200)) + list(range(210, 214))

        map_overdue = {np.nan: np.nan, 0: 0, 'M1': 1, 'M2': 2, 'M3': 3, 'M3+': 4}
        df_[194] = df_[194].map(map_overdue)
        df_[203] = df_[203].map(map_overdue)

        ct_var_all = ct_amount_var + ct_count_var + list(range(169, 175)) + list(range(181, 186)) + list(range(191, 196))
        df_group = df_.groupby(['mobile', 'm'])
        rct_mth_12 = df_group.head(12)
        rct_mth_6 = df_group.head(6)
        rct_mth_3 = df_group.head(3)
        rct_mth_1 = df_group.head(1).reset_index()
        df_1 = df_group.head(1).reset_index(drop=True).drop(['label', 'mobile', 'month', 'm'], axis=1).add_prefix(str(1) + '_')

        times = time.time()
        print('Extract 12 features...')
        df_la_on_12, df_ea_on_12 = MF.var_once(rct_mth_12, once_var, 12)
        df_me_no_12, df_ma_no_12, df_mi_no_12, df_su_no_12 = MF.var_normal(rct_mth_12, ct_var_all, 12)
        df_me_ma_12, df_ma_ma_12, df_mi_ma_12 = MF.var_max(rct_mth_12, ct_max_var, 12)
        df_yn_12 = MF.var_overdue(rct_mth_12, overdue_var, 12)

        print('Extract 6 features...')
        df_la_on_6, df_ea_on_6 = MF.var_once(rct_mth_6, once_var, 6)
        df_me_no_6, df_ma_no_6, df_mi_no_6, df_su_no_6 = MF.var_normal(rct_mth_6, ct_var_all, 6)
        df_me_ma_6, df_ma_ma_6, df_mi_ma_6 = MF.var_max(rct_mth_6, ct_max_var, 6)
        df_yn_6 = MF.var_overdue(rct_mth_6, overdue_var, 6)

        print('Extract 3 features...')
        df_la_on_3, df_ea_on_3 = MF.var_once(rct_mth_3, once_var, 3)
        df_me_no_3, df_ma_no_3, df_mi_no_3, df_su_no_3 = MF.var_normal(rct_mth_3, ct_var_all, 3)
        df_me_ma_3, df_ma_ma_3, df_mi_ma_3 = MF.var_max(rct_mth_3, ct_max_var, 3)
        df_yn_3 = MF.var_overdue(rct_mth_3, overdue_var, 3)
        print('Extract features finished...', time.time() - times)

        fin_data = pd.concat([df_la_on_12, df_ea_on_12, df_yn_12,
                              df_me_no_12, df_ma_no_12, df_mi_no_12, df_su_no_12,
                              df_me_ma_12, df_ma_ma_12, df_mi_ma_12,
                              df_la_on_6, df_ea_on_6, df_yn_6,
                              df_me_no_6, df_ma_no_6, df_mi_no_6, df_su_no_6,
                              df_me_ma_6, df_ma_ma_6, df_mi_ma_6,
                              df_la_on_3, df_ea_on_3, df_yn_3,
                              df_me_no_3, df_ma_no_3, df_mi_no_3, df_su_no_3,
                              df_me_ma_3, df_ma_ma_3, df_mi_ma_3, df_1], axis=1)
        fin_data['mobile'] = rct_mth_1['mobile']
        fin_data['mobile', 'm'] = rct_mth_1['m']
        print(fin_data.shape)
        fin_data['label'] = rct_mth_1['label']
        fin_data.to_csv('./Data/MergeData_X_feature.csv', index=None)
        return fin_data

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

    # lightGBM算法算法
    @staticmethod
    def lgb_train(trains_x, tests_x, trains_y, tests_y):
        lgb_train = lgb.Dataset(trains_x, trains_y)
        lgb_eval = lgb.Dataset(tests_x, tests_y, reference=lgb_train)
        # dtrain2 = xgb.DMatrix(trains_x)
        # dtest = xgb.DMatrix(tests_x)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',    # 设置提升类型
            'objective': 'regression',  # 目标函数
            'metric': {'l2', 'auc'},    # 评估函数
            'num_leaves': 31,           # 叶子节点数
            'learning_rate': 0.05,      # 学习速率
            'feature_fraction': 0.9,    # 建树的特征选择比例
            'bagging_fraction': 0.8,    # 建树的样本采样比例
            'bagging_freq': 5,          # k 意味着每 k 次迭代执行bagging
            'verbose': 1                # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
            }
        gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
        # bst = xgb.train(params, dtrain, 10)
        print('训练集效果...')
        y_pred = gbm.predict(trains_x, num_iteration=gbm.best_iteration)
        # prob_train = bst.predict(dtrain2)
        pre_train = [round(value) for value in y_pred]
        label_train = trains_y.tolist()
        prob_train = trains_x.tolist()
        MF.evaluate(label_train, prob_train, pre_train)
        print('验证集效果...')
        y_pred2 = gbm.predict(tests_x, num_iteration=gbm.best_iteration)
        # prob_test = bst.predict(dtest)
        pre_test = [round(value) for value in y_pred2]
        label_test = tests_y.tolist()
        prob_test = y_pred2.tolist()
        MF.evaluate(label_test, prob_test, pre_test)
        groups_count = 10
        temp_list = list(np.linspace(0, 1, groups_count + 1))
        psi_tot = MF.get_psi(pre_train, pre_test, temp_list)
        print('psi:', psi_tot)

    # XGBoost算法
    def start_xgbt(self):
        # 判断当前路径下是否存在该文件
        if not os.path.exists(self.Train_data):
            print('请确认数据路径...')
            sys.exit(0)
        print('XGBoost...')
        df = pd.read_csv(self.Train_data)
        df.columns = ['mobile', 'm', 'month', 'label'] + list(range(1, 214))
        df = df.sort_values(['mobile', 'm', 'month'], ascending=[1, 1, 0])
        cols = ['mobile', 'm', 'month', 'label'] + list(range(1, 214))
        # df_var = demo.get_var(df[cols])
        # df_var = demo.get_var2(df[cols])
        # train_x, test_x, train_y, test_y = demo.load_data(df_var)
        train_x, test_x, train_y, test_y = demo.load_data()
        demo.xgbt_train(train_x, test_x, train_y, test_y)

    # lightGBM算法
    def start_lgb(self):
        # 判断当前路径下是否存在该文件
        if not os.path.exists(self.Train_data):
            print('请确认数据路径...')
            sys.exit(0)
        print('lightGBM...')
        df = pd.read_table(self.Train_data, header=None, sep='\t')
        df.columns = ['mobile', 'month', 'label'] + list(range(1, df.shape[1]-2))
        df = df.sort_values(['mobile', 'month'], ascending=[1, 0])
        cols = ['mobile', 'month', 'label'] + list(range(1, df.shape[1]-2))
        train_x, test_x, train_y, test_y = demo.load_data(df[cols])
        demo.lgb_train(train_x, test_x, train_y, test_y)


if __name__ == '__main__':
    data_path = sys.argv[1]
    start = time.clock()
    demo = MF(data_path)
    demo.start_xgbt()
    # demo.start_lgb()
    end = time.clock()
    print("消耗时间：%f s" % (end - start))
