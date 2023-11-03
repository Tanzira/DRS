#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:01:41 2023

@author: tanzira
"""

#%% All Imports
import pandas as pd
import numpy as np
from scipy.io import loadmat
import pickle
from sklearn import linear_model
import os
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
import scipy
import time
import sys

#%%Helper functions
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

'''Lasso regression'''
def do_lasso(train_x, tf_locs, lambda_val):
    coefficients = []
    intercepts = []
    for loc in range(train_x.shape[1]):
        temp = train_x.copy()
        gene_exp = temp[:, loc].copy()#gene expression
        if loc in tf_locs:
            temp[:, loc] = 0
        tf_exp = temp[:, tf_locs].copy()#TFs expression
        reg = linear_model.Lasso(alpha = lambda_val, max_iter = 10000, random_state = 42)#Lasso regression model
        reg.fit(tf_exp, gene_exp)
        coefficients.append(reg.coef_)
        intercepts.append(reg.intercept_)
    return coefficients, intercepts

#%% Load Data Before model building
LAMBDA_VAL = 0.06
lambdas = [0.13, 0.06, 0.03]

'''AECS data'''
aces_raw  = loadmat( 'Dataset/ACES_Data/ACESExpr.mat')['data']
aces_p_type = loadmat('Dataset/ACES_Data/ACESLabel.mat')['label']
aces_entrez_id = loadmat('Dataset/ACES_Data/ACES_EntrezIds.mat')['entrez_ids']
aces_data = pd.DataFrame(aces_raw)
aces_data.columns = aces_entrez_id.reshape(-1)


''' Seperating the studies for leave one study out cross validation. '''

cv_train_idx_file = 'Dataset/ACES_Data/CVIndTrain200.txt'
train_cv_idx = pd.read_csv(cv_train_idx_file, header = None, sep = ' ')
d_map = pd.DataFrame(0, 
                     index = range(train_cv_idx.shape[0]),
                     columns = range(train_cv_idx.shape[1]))

for col in train_cv_idx.columns:
    idx_other = train_cv_idx[col][train_cv_idx[col] > 0]-1
    idx = np.setdiff1d(range(train_cv_idx.shape[0]), idx_other)
    d_map.loc[idx, col] = 1


'''Reading NKI data'''
nki_raw = loadmat('Dataset/NKI_Data/vijver.mat')['vijver']
nki_p_type = loadmat('Dataset/NKI_Data/VijverLabel.mat')['label']
nki_entrez_id = loadmat('Dataset/NKI_Data/vijver_gene_list.mat')['vijver_gene_list']
nki_data = pd.DataFrame(nki_raw)
nki_data.columns = nki_entrez_id.reshape(-1)

'''Reading TF file'''
tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[(human_tfs['Is TF?'] =='Yes') & (human_tfs['EntrezGene ID'] != 'None')]
human_tfs.set_index('EntrezGene ID', inplace = True)
human_tfs.index = human_tfs.index.astype(int)

'''read r2 values for each gene file'''
r2 = pd.read_csv('R2CVScores/cv_r2_score_lambda_{0}.txt'.format(LAMBDA_VAL), header=None)
r2.index = aces_data.columns
#%% ACES 10 fold statified CV model building

common_tf = np.intersect1d(aces_data.columns, human_tfs.index)
tf_locs = [aces_data.columns.get_loc(c) for c in common_tf]
#Here the model is built using 3 lambdas. In sklearn alpha means lambda really.
lambdas = [0.13, 0.06, 0.03]
n_fold = 10

skf = StratifiedKFold(n_splits = n_fold, random_state = 42, shuffle = True)

data = aces_data.to_numpy()
p_type = aces_p_type.ravel()

cv_indices = list(skf.split(data, p_type))
can_type = 1 #type of patient. 1 means metastatic and 0 means non metastatic patients
# print("cancer type: ", can_type)

for cv_idx in range(n_fold):
    start_time = time.time()
    print("CV_idx: ", cv_idx)
    train_index, test_index = cv_indices[cv_idx]
    data_train, c_train = data[train_index], p_type[train_index]
    
    data_train = data_train[c_train == can_type, :]
    
    print(can_type, data_train.shape)
    models = []
    intercepts = []
    #Doing lasso for 3 different lambdas
    for lambda_val in lambdas:
        coef, intercept = do_lasso(data_train, tf_locs, lambda_val)
        models.append(coef)
        intercepts.append(intercept)
    #Combining the models from different lambda values
    models = np.swapaxes(np.stack(models), 1, 2)
    intercepts = np.stack(intercepts)
    end_time = time.time()
    print(format_time(end_time - start_time))
    dir_name = './Models/ACESStratifiedKFold/'
    try:
        os.makedirs(dir_name, exist_ok=True)
    except OSError:
        print("Error occured")
    if can_type == 1:
        f1 = dir_name + '/meta_models_with_ensemble_lambda_cv_{}.pkl'.format(cv_idx)
        f11 = dir_name + '/meta_intercepts_with_ensemble_lambda_cv_{}.txt'.format(cv_idx)
    else:
        f1 = dir_name + '/nmeta_models_with_ensemble_lambda_cv_{}.pkl'.format(cv_idx)
        f11 = dir_name + '/nmeta_intercepts_with_ensemble_lambda_cv_{}.txt'.format(cv_idx)
    out = open(f1, 'wb')
    pickle.dump(models, out)
    out.close()
    np.savetxt(f11, intercepts)
    
#%% ACES leave-one-study-out model building
common_tf = np.intersect1d(aces_data.columns, human_tfs.index)
tf_locs = [aces_data.columns.get_loc(c) for c in common_tf]
#Here the model is built using 3 lambdas. In sklearn alpha means lambda really.
lambdas = [0.13, 0.06, 0.03]


#Here the model is built using 3 lambdas. In sklearn alpha means lambda really.
lambdas = [0.13, 0.06, 0.03]
# cv_idx = int(sys.argv[1])
# can_type = int(sys.argv[2])


'''
    Set cancer type and which cross validation index you want to run.
'''
# can_type = 0
n_studies = 12 #12 study
data = aces_data.to_numpy()
p_type = aces_p_type.ravel()
can_type = 1 #Cancer type metastatic(1) and Non metastatic 0

for cv_idx in range(n_studies):
    train_index, test_index = d_map[d_map[cv_idx] == 0].index, d_map[d_map[cv_idx] == 1].index
    data_train, c_train,  = data[train_index], p_type[train_index]
    data_test, c_test = data[test_index], p_type[test_index]
    #creating model only with one type of patient
    Xtrain = data_train[c_train.T[0] == can_type, :]
    print('CVIDX: ', cv_idx, can_type, Xtrain.shape)
    models, intercepts = [], []
    for lambda_val in lambdas:
        coef, intercept = do_lasso(Xtrain, tf_locs, lambda_val)
        models.append(coef)
        intercepts.append(intercept)
    #Combining the models from different lambda values
    models = np.swapaxes(np.stack(models), 1, 2)
    intercepts = np.stack(intercepts)
    
    dir_name = './Models/ACESLeaveOneStudyOut'
    try:
        os.makedirs(dir_name, exist_ok=True)
    except OSError:
        print("Error occured")
    if can_type == 1:
        f1 = dir_name + '/meta_models_with_ensemble_lambda_cv_{}.pkl'.format(cv_idx)
        f11 = dir_name + '/meta_intercepts_with_ensemble_lambda_cv_{}.txt'.format(cv_idx)
    else:
        f1 = dir_name + '/nmeta_models_with_ensemble_lambda_cv_{}.pkl'.format(cv_idx)
        f11 = dir_name + '/nmeta_intercepts_with_ensemble_lambda_cv_{}.txt'.format(cv_idx)
    out = open(f1, 'wb')
    pickle.dump(models, out)
    out.close()
    np.savetxt(f11, intercepts)
    
#%% NKI 10 fold statified CV model building

common_tf = np.intersect1d(nki_data.columns, human_tfs.index)
tf_locs = [nki_data.columns.get_loc(c) for c in common_tf]
print(len(tf_locs))

#Here the model is built using 3 lambdas. In sklearn alpha means lambda really.
lambdas = [0.13, 0.06, 0.03]

n_fold = 10
skf = StratifiedKFold(n_splits = n_fold, random_state = 42, shuffle = True)

data = nki_data.to_numpy()
p_type = nki_p_type.ravel()

cv_indices = list(skf.split(data, p_type))

can_type = 1 #type of patient. 1 means metastatic and 0 means non-metastatic patients
print("cancer type: ", can_type)

for cv_idx in range(n_fold):
    train_index, test_index = cv_indices[cv_idx]
    data_train, c_train = data[train_index], p_type[train_index]
    Xtrain = data_train[c_train == can_type, :]
    print(cv_idx, can_type, Xtrain.shape)
    models, intercepts = [], []
    #Doing lasso for 3 different lambdas
    for lambda_val in lambdas:
        print("Lambda: ", lambda_val)
        start_time = time.time()
        coef, intercept = do_lasso(Xtrain, tf_locs, lambda_val)
        models.append(coef)
        intercepts.append(intercept)
        end_time = time.time()
        print(format_time(end_time - start_time))
    #Combining the models from different lambda values
    models = np.swapaxes(np.stack(models), 1, 2)
    intercepts = np.stack(intercepts)
    
    dir_name = './Models/NKIStratifiedKFold'
    try:
        os.makedirs(dir_name, exist_ok=True)
    except OSError:
        print("Error occured")
    if can_type == 1:
        f1 = dir_name + '/meta_models_with_ensemble_lambda_cv_{}.pkl'.format(cv_idx)
        f11 = dir_name + '/meta_intercepts_with_ensemble_lambda_cv_{}.txt'.format(cv_idx)
    else:
        f1 = dir_name + '/nmeta_models_with_ensemble_lambda_cv_{}.pkl'.format(cv_idx)
        f11 = dir_name + '/nmeta_intercepts_with_ensemble_lambda_cv_{}.txt'.format(cv_idx)
    out = open(f1, 'wb')
    pickle.dump(models, out)
    out.close()
    np.savetxt(f11, intercepts)
#%% NKI data validation model generation from ACES data
'''
    Getting common number of TFs and genes between NKI and ACES data
'''
common_genes = np.intersect1d(aces_data.columns, nki_data.columns)
common_tfs = np.intersect1d(common_genes, human_tfs.index)

aces_filtered = aces_data.loc[:, common_genes]
nki_filtered = nki_data.loc[:, common_genes]

print(aces_filtered.shape, nki_filtered.shape)
tf_locs = [aces_filtered.columns.get_loc(c) for c in common_tfs]

#Here the model is built using 3 lambdas only that were optimal
lambdas = [0.13, 0.06, 0.03]

'''Using aces data as training'''
data = aces_filtered.to_numpy()
p_type = aces_p_type.ravel()

can_type = 0 #change cancer type according to the type of the model.
Xtrain = data[p_type == can_type, :]
print(can_type, Xtrain.shape)

models, intercepts = [], []
#Doing lasso for 3 different lambdas
for lambda_val in lambdas:
    start_time = time.time()
    coef, intercept = do_lasso(Xtrain, tf_locs, lambda_val)
    models.append(coef)
    intercepts.append(intercept)
    end_time = time.time()
    print(format_time(end_time - start_time))
#Combining the models from different lambda values
models = np.swapaxes(np.stack(models), 1, 2)
intercepts = np.stack(intercepts)

dir_name = './Models/NKIDataValidationModels'
try:
    os.makedirs(dir_name, exist_ok=True)
except OSError:
    print("Error occured")

if can_type == 1:
    f1 = dir_name + '/meta_models_with_ensemble_lambdas.pkl'
    f11 = dir_name + '/meta_intercepts_with_ensemble_lambdas.txt'
else:
    f1 = dir_name + '/nmeta_models_with_ensemble_lambdas.pkl'
    f11 = dir_name + '/nmeta_intercepts_with_ensemble_lambdas.txt'
out = open(f1, 'wb')
pickle.dump(models, out)
out.close()
np.savetxt(f11, intercepts)

#%% Helper functions for normalization and distance calculation
def qtnorm_with_ref_columnwise(Xref, X):
    '''
    Column-by-column normalization
    '''
    # both datasets should have same number of features
    assert Xref.shape[1] == X.shape[1]
    X_new = np.quantile(Xref, np.linspace(0, 1, X.shape[0]), axis = 0)
    rank = pd.DataFrame(X).rank(axis = 0, method = 'min').astype(int).values - 1
    X_new = np.take_along_axis(X_new, rank, axis = 0)
    return X_new

def get_distance(pred0, pred1, actual, distance = 'euclidean'):
    n = len(actual) #total number of patients
    if distance == 'pearson':
        s0 = np.corrcoef(actual, pred0)[range(n), range(n, n*2)]
        s1 = np.corrcoef(actual, pred1)[range(n), range(n, n*2)]
        s1, s0 = 1 - np.abs(s1), 1 - np.abs(s0) #changing correlation into distance
    elif distance == 'spearman':
        s0 = scipy.stats.spearmanr(actual, pred0, axis = 1)[0][range(n), range(n, n*2)]
        s1 = scipy.stats.spearmanr(actual, pred1, axis = 1)[0][range(n), range(n, n*2)]
        s1, s0 = 1 - np.abs(s1), 1 - np.abs(s0) #changing correlation into distance
        
    elif distance == 'logdistance':
        s0 = np.log(np.abs((pred0 / actual))).sum(axis = 1)
        s1 = np.log(np.abs((pred1 / actual))).sum(axis = 1)
    else:
        if distance == 'minkowski':
            norm_val = 1
        else:
            norm_val = 2
        s0 = np.linalg.norm((pred0 - actual), ord = norm_val, axis = 1)
        s1 = np.linalg.norm((pred1 - actual), ord = norm_val, axis = 1)
    return s0, s1

def get_pred_expr(Mp, Mn, bp, bn, Xtest, tf_locs, ALPHA_CUTOFF = 0.01):
    Mp[np.abs(Mp) < ALPHA_CUTOFF] = 0
    Mn[np.abs(Mn) < ALPHA_CUTOFF] = 0
    n_iter = Mp.shape[0]
    tfs_expression = Xtest[:, tf_locs] # expression levels of TFs
    m_bias = np.swapaxes(np.tile(bp, [Xtest.shape[0], 1, 1]), 0, 1)
    n_bias = np.swapaxes(np.tile(bn, [Xtest.shape[0], 1, 1]), 0, 1)
    exp_pred_p = np.tile(tfs_expression, [n_iter, 1, 1]) @ Mp # predicted expression levels of all genes using class = metastatic models
    exp_pred_n = np.tile(tfs_expression, [n_iter, 1, 1]) @ Mn # predicted expression levels of all genes using class = non-metastatic models
    exp_pred_p = exp_pred_p + m_bias
    exp_pred_n = exp_pred_n + n_bias
    avg_pred_p = exp_pred_p.mean(axis =0)
    avg_pred_n = exp_pred_n.mean(axis =0)
    return avg_pred_n, avg_pred_p

def get_filtered_expr(expr1, expr2, human_tfs):
    '''
        Getting common number of TFs and genes between NKI and ACES data
    '''
    common_genes = np.intersect1d(expr1.columns, expr2.columns)
    common_tfs = np.intersect1d(common_genes, human_tfs.index)
    #filtering ACES and NKI data based on common number of genes both has.
    expr1_filtered = expr1.loc[:, common_genes]
    expr2_filtered = expr2.loc[:, common_genes]
    return expr1_filtered, expr2_filtered, common_tfs

def get_different_model_scores(Xtrain, c_train, Xtest, c_test):
    
    m1 = RandomForestClassifier(random_state = 42)
    m2 = LogisticRegression(solver='liblinear', random_state=42)
    m3 = DecisionTreeClassifier(random_state = 42)
    m4 = KNeighborsClassifier()#No random state in parameter
    m5 = MLPClassifier(random_state=42)#Neural network classifier
    m6 = svm.SVC(kernel = 'linear', random_state=42, probability=True)
    
    m1.fit(Xtrain, c_train)
    m2.fit(Xtrain, c_train)
    m3.fit(Xtrain, c_train)
    m4.fit(Xtrain, c_train)
    m5.fit(Xtrain, c_train)
    m6.fit(Xtrain, c_train)
    
    rf = m1.predict_proba(Xtest)[:, 1]
    logr = m2.predict_proba(Xtest)[:, 1]
    dtc = m3.predict_proba(Xtest)[:, 1]
    knnc = m4.predict_proba(Xtest)[:, 1]
    mlpc = m5.predict_proba(Xtest)[:, 1]
    svcl = m6.predict_proba(Xtest)[:, 1]
    return rf, mlpc, logr, dtc, knnc, svcl
#%% Load Data before score calculation
LAMBDA_VAL = 0.06
# ALPHA_CUTOFF = 0.02
lambdas = [0.13, 0.06, 0.03]

'''AECS data'''
aces_raw  = loadmat( 'Dataset/ACES_Data/ACESExpr.mat')['data']
aces_p_type = loadmat('Dataset/ACES_Data/ACESLabel.mat')['label']
aces_entrez_id = loadmat('Dataset/ACES_Data/ACES_EntrezIds.mat')['entrez_ids']
aces_data = pd.DataFrame(aces_raw)
aces_data.columns = aces_entrez_id.reshape(-1)


''' Seperating the studies for leave one study out cross validation. '''

cv_train_idx_file = 'Dataset/ACES_Data/CVIndTrain200.txt'
train_cv_idx = pd.read_csv(cv_train_idx_file, header = None, sep = ' ')
d_map = pd.DataFrame(0, 
                     index = range(train_cv_idx.shape[0]),
                     columns = range(train_cv_idx.shape[1]))

for col in train_cv_idx.columns:
    idx_other = train_cv_idx[col][train_cv_idx[col] > 0]-1
    idx = np.setdiff1d(range(train_cv_idx.shape[0]), idx_other)
    d_map.loc[idx, col] = 1


'''Reading NKI data'''
nki_raw = loadmat('Dataset/NKI_Data/vijver.mat')['vijver']
nki_p_type = loadmat('Dataset/NKI_Data/VijverLabel.mat')['label']
nki_entrez_id = loadmat('Dataset/NKI_Data/vijver_gene_list.mat')['vijver_gene_list']
nki_data = pd.DataFrame(nki_raw)
nki_data.columns = nki_entrez_id.reshape(-1)

'''Reading TF file'''
tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[(human_tfs['Is TF?'] =='Yes') & (human_tfs['EntrezGene ID'] != 'None')]
human_tfs.set_index('EntrezGene ID', inplace = True)
human_tfs.index = human_tfs.index.astype(int)

'''read r2 values for each gene file'''
r2 = pd.read_csv('R2CVScores/cv_r2_score_lambda_{0}.txt'.format(LAMBDA_VAL), header=None)
r2.index = aces_data.columns
#%%ACES 10 fold cross validation score calculation
r2_threshold = 0.1
# r2_threshold = r2.values.min()-1 # if you want to use all the genes then use this threshold
ALPHA_CUTOFF = 0.02
K1, K2 = 1, 2 #K1, K2 are the indices for different lambdas. 0-1 for 0.13, 1-2, 0.06 and 2-3 for 0.03

print("lambda: {0}\nr2_th:{1}\nALPHA_CUTOFF: {2}\nK1: {3} \t K2: {4} ".format(LAMBDA_VAL, r2_threshold, 
                                                                  ALPHA_CUTOFF, K1, K2))
common_tf = np.intersect1d(aces_data.columns, human_tfs.index)
tf_locs = [aces_data.columns.get_loc(c) for c in common_tf]

#This value can be changed. Our result is based on all genes so r2_threshold should be r2.values.min()-1
good_genes = (r2.values >= r2_threshold)
good_genes_loc =[aces_data.columns.get_loc(c) for c in aces_data.loc[:, good_genes]]

data = aces_data.to_numpy()
p_type = aces_p_type.ravel()

'''
Here using the same configuration as we did for model genetation.
'''
n_fold = 10
cv = StratifiedKFold(n_splits = n_fold, random_state=42, shuffle=True)
scores = {}
dir_name = './Models/ACESStratifiedKFold'
cv_idx = -1
for train_index, test_index in cv.split(data, p_type):
    cv_idx += 1
    # print('CV_IDX: ', cv_idx)
    Xtrain, c_train,  = data[train_index], p_type[train_index]
    Xtest, c_test=  data[test_index], p_type[test_index] 

    f1 = dir_name + '/meta_models_with_ensemble_lambda_cv_{0}.pkl'.format(cv_idx)
    f2 = dir_name + '/nmeta_models_with_ensemble_lambda_cv_{0}.pkl'.format(cv_idx)
    
    f1_intercept = dir_name + '/meta_intercepts_with_ensemble_lambda_cv_{0}.txt'.format(cv_idx)
    f2_intercept = dir_name + '/nmeta_intercepts_with_ensemble_lambda_cv_{0}.txt'.format(cv_idx)
    
    with open(f1,'rb') as f:
        Mp = pickle.load(f)
    with open(f2,'rb') as f11:
        Mn = pickle.load(f11)
        
    #intercepts for both models
    bp = np.loadtxt(f1_intercept) 
    bn = np.loadtxt(f2_intercept)
    #get predicted expressions
    pred_n, pred_p = get_pred_expr(Mp[K1:K2], Mn[K1:K2], bp[K1:K2], bn[K1:K2], Xtest, tf_locs, ALPHA_CUTOFF)
    #filtering genes based on good gene locations
    Xtest_filtered = Xtest[:, good_genes_loc]
    pred_p = pred_p[:, good_genes_loc]
    pred_n = pred_n[:, good_genes_loc]
    print(pred_p.shape, pred_n.shape)
    dn, dp = get_distance(pred_n, pred_p, Xtest_filtered, 'pearson') #Using pearson distance
    pred_lasso = dn - dp
    rf, mlpc, logr, dtc, knnc, svcl = get_different_model_scores(Xtrain, c_train,
                                                                  Xtest, c_test)
    pred_all = pd.DataFrame([pred_lasso, rf, mlpc, logr, dtc, knnc, svcl, c_test],
                            index = ['DRS','RF', 'MLP', 'LogR', 'DecTree', 'KNN', 'SVC', 'real_class']).T
    scores[cv_idx] = pred_all

scores = pd.concat(scores, axis = 0)
scores.index.names = ['fold', 'patient']

print("ACES DRS 10 fold AUC: ",roc_auc_score(scores.real_class, scores.DRS))
print("ACES RF 10 fold AUC: ",roc_auc_score(scores.real_class, scores.RF))
print("ACES MLP 10 fold AUC: ",roc_auc_score(scores.real_class, scores.MLP))
print("ACES LogR 10 fold AUC: ",roc_auc_score(scores.real_class, scores.LogR))
print("ACES DecTree 10 fold AUC: ",roc_auc_score(scores.real_class, scores.DecTree))
print("ACES KNN 10 fold AUC: ",roc_auc_score(scores.real_class, scores.KNN))
print("ACES SVC 10 fold AUC: ",roc_auc_score(scores.real_class, scores.SVC))

sfile = 'aces_ten_fold_scores_lambda_{0}_r2_{1}_alpha_{2}.csv'.format(LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF)
scores.to_csv(sfile)

#%%ACES leave one study out cross validation score caluclation

r2_threshold = 0.1
# r2_threshold = r2.values.min()-1 # if you want to use all the genes then use this threshold
ALPHA_CUTOFF = 0.02
K1, K2 = 1, 2 #K1, K2 are the indices for different lambdas. 0-1 for 0.13, 1-2, 0.06 and 2-3 for 0.03

print("lambda: {0}\nr2_th:{1}\nALPHA_CUTOFF: {2}\nK1: {3} \t K2: {4} ".format(LAMBDA_VAL, r2_threshold, 
                                                                  ALPHA_CUTOFF, K1, K2))    
common_tf = np.intersect1d(aces_data.columns, human_tfs.index)
tf_locs = [aces_data.columns.get_loc(c) for c in common_tf]

good_genes = (r2.values >= r2_threshold)
good_genes_loc =[aces_data.columns.get_loc(c) for c in aces_data.loc[:, good_genes]]

data = aces_data.to_numpy()
p_type = aces_p_type.ravel()
#These are the lambdas we have in our model
n_studies = 12

'''
Here using the same configuration as we did for model genetation.
'''
columns=['Desmedt', 'Hatzis', 'Ivshina', 'Loi', 'Miller', 'Minn',
         'Pawitan', 'Schmidt', 'Symmans', 'WangY', 'WangYE', 'Zhang'] #Study names
scores = {}
dir_name = './Models/ACESLeaveOneStudyOut'
for cv_idx in range(n_studies):
    
    train_index, test_index = d_map[d_map[cv_idx] == 0].index, d_map[d_map[cv_idx] == 1].index
    Xtrain, c_train,  = data[train_index], p_type[train_index]
    Xtest, c_test = data[test_index], p_type[test_index]
    
    f1 = dir_name + '/meta_models_with_ensemble_lambda_cv_{0}.pkl'.format(cv_idx)
    f2 = dir_name + '/nmeta_models_with_ensemble_lambda_cv_{0}.pkl'.format(cv_idx)
    
    f1_intercept = dir_name + '/meta_intercepts_with_ensemble_lambda_cv_{0}.txt'.format(cv_idx)
    f2_intercept = dir_name + '/nmeta_intercepts_with_ensemble_lambda_cv_{0}.txt'.format(cv_idx)
    
    with open(f1,'rb') as f:
        Mp = pickle.load(f)
    with open(f2,'rb') as f11:
        Mn = pickle.load(f11)
        
    #intercepts for both models
    bp = np.loadtxt(f1_intercept) 
    bn = np.loadtxt(f2_intercept)
    #get predicted expressions
    pred_n, pred_p = get_pred_expr(Mp[K1:K2], Mn[K1:K2], bp[K1:K2], bn[K1:K2], Xtest, tf_locs, ALPHA_CUTOFF)
    #filtering genes based on good gene locations
    Xtest_filtered = Xtest[:, good_genes_loc]
    pred_p = pred_p[:, good_genes_loc]
    pred_n = pred_n[:, good_genes_loc]
    print(pred_p.shape, pred_n.shape)
    dn, dp = get_distance(pred_n, pred_p, Xtest_filtered, 'pearson')
    pred_lasso = dn - dp
    rf, mlpc, logr, dtc, knnc, svcl = get_different_model_scores(Xtrain, c_train,
                                                                  Xtest, c_test)
    pred_all = pd.DataFrame([pred_lasso, rf, mlpc, logr, dtc, knnc, svcl, c_test],
                            index = ['DRS','RF', 'MLP', 'LogR', 'DecTree', 'KNN', 'SVC', 'real_class']).T
    scores[columns[cv_idx]] = pred_all

scores = pd.concat(scores, axis = 0)
scores.index.names = ['study', 'patient']

print("DRS LOSO AUC: ",roc_auc_score(scores.real_class, scores.DRS))
print("RF LOSO AUC: ",roc_auc_score(scores.real_class, scores.RF))
print("MLP LOSO AUC: ",roc_auc_score(scores.real_class, scores.MLP))
print("LogR 1LOSO AUC: ",roc_auc_score(scores.real_class, scores.LogR))
print("DecTree LOSO AUC: ",roc_auc_score(scores.real_class, scores.DecTree))
print("KNN LOSO AUC: ",roc_auc_score(scores.real_class, scores.KNN))
print("SVC LOSO AUC: ",roc_auc_score(scores.real_class, scores.SVC))

sfile = 'aces_loso_scores_lambda_{0}_r2_{1}_alpha_{2}.csv'.format(LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF)
scores.to_csv(sfile)

#%%NKI 10 fold crosss validation score calculation

#Not using any alpha or r2 cutoff here
ALPHA_CUTOFF = 0.0
K1, K2 = 1, 2 #K1, K2 are the indices for different lambdas. 0-1 for 0.13, 1-2, 0.06 and 2-3 for 0.03

print("lambda: {0}\nALPHA_CUTOFF: {1}\nK1: {2} \t K2: {3} ".format(LAMBDA_VAL, 
                                                                  ALPHA_CUTOFF, K1, K2))    

common_tf = np.intersect1d(nki_data.columns, human_tfs.index)
tf_locs = [nki_data.columns.get_loc(c) for c in common_tf]

data = nki_data.to_numpy()
p_type = nki_p_type.ravel()

'''
Here using the same configuration as we did for model genetation.
'''
n_fold = 10
cv = StratifiedKFold(n_splits = n_fold, random_state = 42, shuffle = True)
scores = {}
dir_name = './Models/NKIStratifiedKFold'
cv_idx = -1
mdist, nmdist = [], []
for train_index, test_index in cv.split(data, p_type):
    cv_idx += 1
    # print('CV_IDX: ', cv_idx)
    Xtrain, c_train,  = data[train_index], p_type[train_index]
    Xtest, c_test=  data[test_index], p_type[test_index] 

    f1 = dir_name + '/meta_models_with_ensemble_lambda_cv_{0}.pkl'.format(cv_idx)
    f2 = dir_name + '/nmeta_models_with_ensemble_lambda_cv_{0}.pkl'.format(cv_idx)
    
    f1_intercept = dir_name + '/meta_intercepts_with_ensemble_lambda_cv_{0}.txt'.format(cv_idx)
    f2_intercept = dir_name + '/nmeta_intercepts_with_ensemble_lambda_cv_{0}.txt'.format(cv_idx)
    
    with open(f1,'rb') as f:
        Mp = pickle.load(f)
    with open(f2,'rb') as f11:
        Mn = pickle.load(f11)
    #intercepts for both models
    bp = np.loadtxt(f1_intercept) 
    bn = np.loadtxt(f2_intercept)
    #get predicted expressions
    pred_n, pred_p = get_pred_expr(Mp[K1:K2], Mn[K1:K2], bp[K1:K2], bn[K1:K2], Xtest, tf_locs, ALPHA_CUTOFF)
    #filtering genes based on good gene locations
    print(pred_p.shape, pred_n.shape)
    dn, dp = get_distance(pred_n, pred_p, Xtest, 'pearson') #minkowski 0.729
    pred_lasso = dn - dp
    rf, mlpc, logr, dtc, knnc, svcl = get_different_model_scores(Xtrain, c_train,
                                                                  Xtest, c_test)
    pred_all = pd.DataFrame([pred_lasso, rf, mlpc, logr, dtc, knnc, svcl, c_test],
                            index = ['DRS','RF', 'MLP', 'LogR', 'DecTree', 'KNN', 'SVC', 'real_class']).T
    scores[cv_idx] = pred_all
scores = pd.concat(scores, axis = 0)
scores.index.names = ['fold', 'patient']

print("10 fold AUC for NKI")
print("RF 10 fold AUC: ",roc_auc_score(scores.real_class, scores.DRS))
print("RF 10 fold AUC: ",roc_auc_score(scores.real_class, scores.RF))
print("MLP 10 fold AUC: ",roc_auc_score(scores.real_class, scores.MLP))
print("LogR 10 fold AUC: ",roc_auc_score(scores.real_class, scores.LogR))
print("DecTree 10 fold AUC: ",roc_auc_score(scores.real_class, scores.DecTree))
print("KNN 10 fold AUC: ",roc_auc_score(scores.real_class, scores.KNN))
print("SVC 10 fold AUC: ",roc_auc_score(scores.real_class, scores.SVC))

sfile = 'nki_ten_fold_scores_lambda_{0}_alpha_{1}.csv'.format(LAMBDA_VAL, ALPHA_CUTOFF)
scores.to_csv(sfile)
#%% NKI validation on model trained with ACES data
r2_threshold = 0.1
# r2_threshold = r2.values.min()-1 # if you want to use all the genes then use this threshold
ALPHA_CUTOFF = 0.02
K1, K2 = 1, 2 #K1, K2 are the indices for different lambdas. 0-1 for 0.13, 1-2, 0.06 and 2-3 for 0.03

print("lambda: {0}\nr2_th:{1}\nALPHA_CUTOFF: {2}\nK1: {3} \t K2: {4} ".format(LAMBDA_VAL, r2_threshold, 
                                                                  ALPHA_CUTOFF, K1, K2))

aces_filtered , nki_filtered, common_tfs = get_filtered_expr(aces_data, nki_data, human_tfs)
tf_locs = [aces_filtered.columns.get_loc(c) for c in common_tfs]

r2_filtered = r2.loc[aces_filtered.columns]
#This value can be changed. Our result is based on all genes so r2_threshold should be r2.values.min()-1

good_genes = (r2_filtered.values >= r2_threshold)
good_genes_loc =[aces_filtered.columns.get_loc(c) for c in aces_filtered.loc[:, good_genes]]

#training dataset for the models which is ACES
Xtrain, c_train = aces_filtered.to_numpy(), aces_p_type.ravel()
#test dataset for the models which is NKI
Xtest, c_test = nki_filtered.to_numpy(), nki_p_type.ravel()
#Normalizing the data using quantile normalization
Xtest = qtnorm_with_ref_columnwise(Xtrain, Xtest)
# data_test = qtnorm_with_ref_flat(data_train, data_test)
dir_name = './Models/NKIDataValidationModels'

f1 = dir_name + '/meta_models_with_ensemble_lambdas.pkl'
f2 = dir_name + '/nmeta_models_with_ensemble_lambdas.pkl'

f1_intercept = dir_name + '/meta_intercepts_with_ensemble_lambdas.txt'
f2_intercept = dir_name + '/nmeta_intercepts_with_ensemble_lambdas.txt'

with open(f1,'rb') as f:
    Mp = pickle.load(f)
with open(f2,'rb') as f11:
    Mn = pickle.load(f11)
    
#intercepts for both models
bp = np.loadtxt(f1_intercept) 
bn = np.loadtxt(f2_intercept)
#get predicted expressions
pred_n, pred_p = get_pred_expr(Mp[K1:K2], Mn[K1:K2], bp[K1:K2], bn[K1:K2], Xtest, tf_locs, ALPHA_CUTOFF)
#filtering genes based on good gene locations
Xtest_filtered = Xtest[:, good_genes_loc]
pred_p = pred_p[:, good_genes_loc]
pred_n = pred_n[:, good_genes_loc]
print(pred_p.shape, pred_n.shape)
dn, dp = get_distance(pred_n, pred_p, Xtest_filtered, 'pearson') #minkowski 0.729
pred_lasso = dn - dp
rf, mlpc, logr, dtc, knnc, svcl = get_different_model_scores(Xtrain, c_train,
                                                              Xtest, c_test)
pred_all = pd.DataFrame([pred_lasso, rf, mlpc, logr, dtc, knnc, svcl, c_test],
                        index = ['DRS','RF', 'MLP', 'LogR', 'DecTree', 'KNN', 'SVC', 'real_class']).T
scores = pred_all
print("DRS NKI AUC: ",roc_auc_score(scores.real_class, scores.DRS))
print("RF NKI AUC: ",roc_auc_score(scores.real_class, scores.RF))
print("MLP NKI AUC: ",roc_auc_score(scores.real_class, scores.MLP))
print("LogR NKI AUC: ",roc_auc_score(scores.real_class, scores.LogR))
print("DecTree NKI AUC: ",roc_auc_score(scores.real_class, scores.DecTree))
print("KNN NKI AUC: ",roc_auc_score(scores.real_class, scores.KNN))
print("SVC NKI AUC: ",roc_auc_score(scores.real_class, scores.SVC))

sfile = 'nki_validation_scores_lambda_{0}_r2_{1}_alpha_{2}.csv'.format(LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF)
scores.to_csv(sfile)