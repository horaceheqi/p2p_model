import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
import sys
import time
import V1_Credit_Card_Market_Evaluate as EM
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, confusion_matrix


def model_Fit(params, dtrain, max_round=300, cv_folds=5, n_stop_round=50):
    '''
    Args:
        params: dict, xgb 模型参数
    Returns:
        n_round: 最优迭代次数
        mean_auc: 最优结果
    '''
    cv_result = xgb.cv(params, dtrain, max_round, nfold=cv_folds, metrics='auc', early_stopping_rounds=n_stop_round, show_stdv=False)
    n_round = cv_result.shape[0]
    mean_auc = cv_result['test-auc-mean'].values[-1]
    return n_round, mean_auc


def xgb_GridSearch(params, key, search_params, dtrain, max_round=300, cv_folds=5, n_stop_round=50, return_best_model=True, verbose=True):
    '''
    Args:
        params: dict, xgb 模型参数
        key: 待搜索的参数
        search_params: list 待搜索参数集合
        dtrain: 训练数据
        max_round: 最多迭代次数
        cv_folds: 交叉验证的折数
        early_stopping_rounds: 迭代多少次没有提高则停止
        return_best_model: if True 使用最优的参数训练模型
        verbose: if True 打印训练过程
    Returns:
        cv_results: dict 所有参数组交叉验证的结果
        mean_aucs: 每组参数对应的结果
        n_round； 每组参数最优迭代论数
        list_params: 搜寻的每一组参数
        best_mean_auc: 最优的结果
        best_round: 最优迭代次数
        best_params: 最优的一组参数
    '''
    mean_aucs = list()
    n_rounds = list()
    list_params = list()
    print('Searching parameters: %s %s' % (key, str(search_params)))
    tic = time.time()
    for var in search_params:
        params[key] = var
        list_params.append(params.copy())
        n_round, mean_auc = model_Fit(params, dtrain, max_round, cv_folds, n_stop_round)
        if verbose:
            print('%s=%s: n_round=%d, mean_auc=%g. Time cost %gs' % (key, str(var), n_round, mean_auc, time.time() - tic))
        mean_aucs.append(mean_auc)
        n_rounds.append(n_round)
    best_mean_auc = max(mean_aucs)
    best_index = mean_aucs.index(best_mean_auc)
    best_round = n_rounds[best_index]
    best_params = list_params[best_index]

    cv_result = {'mean_aucs':mean_aucs, 'n_rounds':n_rounds, 'list_params':list_params, 'best_mean_auc':best_mean_auc, 'best_round':best_round, 'best_params':best_params}
    if return_best_model:
        best_model = xgb.train(best_params, dtrain, num_boost_round=best_round)
    else:
        best_model = None

    if verbose:
        print('best_mean_auc =%g' % best_mean_auc)
        print('best_round =%d' % best_round)
        print('best_params =%s' % best_params)
    return cv_result, best_model


if __name__ == '__main__':
    # READ DATA
    print('read data...')
    with open('../../../../SVN_Model/Data/Credit_Card_Market/PF_Bank/PF_19w_110_190328_MarketRisk_Feature_HQ/PF_19w_110_190328_FEA_Columns.txt') as f:
        line = f.readline()
        columns = [i for i in line.split(',')]
    print('Load Columns OK...')
    HeKa_Info = pd.read_table('../../../../SVN_Model/Data/Credit_Card_Market/PF_Bank/PF_19w_110_190328_MarketRisk_Feature_HQ/PF_2w_HeKa_Mobile.txt', names=['info'])
    BeiJu_Info = pd.read_table('../../../../SVN_Model/Data/Credit_Card_Market/PF_Bank/PF_19w_110_190328_MarketRisk_Feature_HQ/PF_2w_BeiJu_Mobile.txt', names=['info'])
    WuZhuangTai_Info = pd.read_table('../../../../SVN_Model/Data/Credit_Card_Market/PF_Bank/PF_19w_110_190328_MarketRisk_Feature_HQ/PF_15w_WuZhuangTai_Mobile.txt', names=[
'info'])
    print('Load Info OK ...')
    print(HeKa_Info.columns, BeiJu_Info.shape, WuZhuangTai_Info.shape)
    df = pd.read_csv('../../../../SVN_Model/Data/Credit_Card_Market/PF_Bank/PF_19w_110_190328_MarketRisk_Feature_HQ/PF_19w_110_190328_MarRis_FEA_HQ.txt', names=columns)
    print('Load Feature OK ...')
    print(df.shape)

    bhd = [(pd.notnull(df.iloc[:, j])).sum() / len(df) for j in range(0, len(df.columns))]
    pos = np.where(pd.Series(bhd) >= 0.1)[0]
    df = df.iloc[:,pos]
    print(df.shape)
    features = ['mobile']
    df = df[features]
    # Tag Labels
    df_HeKa = df[df['mobile'].isin(HeKa_Info['info'])]
    df_HeKa['label'] = 1
    print(df_HeKa.shape)
    df_BeiJu = df[df['mobile'].isin(BeiJu_Info['info'])]
    df_BeiJu['label'] = 1
    df_WuZhuangTai = df[df['mobile'].isin(WuZhuangTai_Info['info'])]
    df_WuZhuangTai['label'] = 0
    df_HeKa = df_HeKa.iloc[:5000,:]
    df_BeiJu = df_BeiJu.iloc[:5000,:]
    df_WuZhuangTai = df_WuZhuangTai.iloc[:100000,:]
    print(df_HeKa.shape, df_BeiJu.shape, df_WuZhuangTai.shape)
    df_data = df_HeKa.append([df_BeiJu, df_WuZhuangTai])
    print('Tag Labels OK ...')
    print(df_data.shape)
    # PROCESS DATA
    df_data.drop(['mobile'],axis=1,inplace=True)
    print(df_data['label'].value_counts())
    labels = df_data['label']
    features = df_data.drop(['label'],axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    print(X_train.shape, X_test.shape)

    dtrain = xgb.DMatrix(X_train, Y_train)
    dtest = xgb.DMatrix(X_test, Y_test)

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
              'njob':4}
    key = 'max_depth'
    values = [6]
    cv_results, best_model = xgb_GridSearch(params,key,values,dtrain)

    if best_model:
        pred_y = best_model.predict(dtest)
        # print(pred_y)
        predict = [round(value) for value in pred_y]
        test_y = Y_test.tolist()
        pred_y = pred_y.tolist()
        EM.evaluate(test_y, pred_y, predict)
        EM.showScore2(pred_y, Y_test, 'xgboost')
        best_model.save_model('Xgbt_Market_self.model')

    # Filter Features
    importance = best_model.get_fscore()
    var, value = list(), list()
    for k in importance:
        var.append(k)
        value.append(importance[k])
    print(pd.DataFrame({
          'column': var[:35],
          'importance': value[:35],
          }).sort_values(by=['importance'],ascending=0))