#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:01:41 2023

@author: tanzira, sakhawat
"""

#%% All Imports
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
import time

from drs import DRS
from utils import format_time, get_filtered_expr, qtnorm_with_ref_columnwise

RANDOM_SEED = 42
USE_SAVED_MODELS = False

probability_scores_dir = 'probability_scores'
lambdas = [0.13, 0.06, 0.03]
lambda_map = {k: v for v, k in enumerate(lambdas)}


#%% Helper functions

def get_different_model_scores(Xtrain, c_train, Xtest, c_test, random_state = None):
    
    models = [
        RandomForestClassifier(random_state = random_state),
        LogisticRegression(solver='liblinear', random_state=random_state),
        DecisionTreeClassifier(random_state = random_state),
        KNeighborsClassifier(),
        MLPClassifier(random_state=random_state),
        svm.SVC(kernel = 'linear', random_state=random_state, probability=True)
    ]
    
    predicted_proba = []
    for model in models:
        model.fit(Xtrain, c_train)
        predicted_proba.append(model.predict_proba(Xtest)[:, 1])
    return predicted_proba

def evaluate(X, y,
             train_test_indices,
             drs_dirname,
             alpha_cutoff,
             lambda_indices,
             good_genes_loc,
             use_saved_models = True):
    scores = {}
    for fold, (train_index, test_index) in enumerate(train_test_indices):
        start_time = time.time()
        if len(train_test_indices) == 1:
            # only one train/test split, do not use subdirectory for folds
            dirname_fold = dirname
        else:
            dirname_fold = dirname + f'/fold_{fold}'
        
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        if use_saved_models:
            try:
                drs_obj = DRS.load_from_file(dirname_fold)
            except:
                print('Could not load saved model')
                drs_obj = DRS.train(X_train, y_train, tf_locs, lambdas)
                drs_obj.save_to_file(dirname_fold)
        else:
            print('Training from scratch')
            drs_obj = DRS.train(X_train, y_train, tf_locs, lambdas)
            drs_obj.save_to_file(dirname_fold)
            
        pred_lasso = drs_obj.predict_proba(X_test, tf_locs, good_genes_loc, 
                                           lambda_indices, alpha_cutoff)
        rf, mlpc, logr, dtc, knnc, svcl = get_different_model_scores(X_train, y_train,
                                                                     X_test, y_test)
        pred_all = pd.DataFrame([pred_lasso, rf, mlpc, logr, dtc, knnc, svcl, y_test],
                                index = ['DRS','RF', 'MLP', 'LogR', 'DecTree', 'KNN', 'SVC', 'real_class']).T
        scores[fold] = pred_all
        print(dirname_fold, 'total time: ', format_time(time.time() - start_time))
    
    scores = pd.concat(scores, axis = 0)
    scores.index.names = ['fold', 'patient']
    return scores
        
    

#%% Load Data Before model building

LAMBDA_VAL = 0.06
lambdas = [0.13, 0.06, 0.03]
lambda_map = {k: v for v, k in enumerate(lambdas)}

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
# d_map is a binary n_samples x n_studies dataframe that indicates the study
# each sample belongs to


'''Reading NKI data'''
nki_raw = loadmat('Dataset/NKI_Data/vijver.mat')['vijver']
nki_p_type = loadmat('Dataset/NKI_Data/VijverLabel.mat')['label']
nki_entrez_id = loadmat('Dataset/NKI_Data/vijver_gene_list.mat')['vijver_gene_list']
nki_data = pd.DataFrame(nki_raw)
nki_data.columns = nki_entrez_id.reshape(-1)

'''Reading TF file'''
#tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
tf_file = 'Dataset/DatabaseExtract_v_1.01_human_TFs.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[(human_tfs['Is TF?'] =='Yes') & (human_tfs['EntrezGene ID'] != 'None')]
human_tfs.set_index('EntrezGene ID', inplace = True)
human_tfs = human_tfs.loc[~np.isnan(human_tfs.index.astype(float)), :]
human_tfs.index = human_tfs.index.astype(int)

'''read r2 values for each gene file'''
r2 = pd.read_csv('R2CVScores/cv_r2_score_lambda_{0}.txt'.format(LAMBDA_VAL), header=None)
r2.index = aces_data.columns

label_names = {0: 'nmeta', 1: 'meta'}

        
#%% ACES 10 fold statified CV training and evaluation

#-----------------------------------------
dirname = './models/ACESStratifiedKFold'
lambdas = [0.13, 0.06, 0.03] # all lambda values will be used for model training
n_fold = 10
r2_threshold = 0.1
# r2_threshold = r2.values.min()-1 # if you want to use all the genes then use this threshold
ALPHA_CUTOFF = 0.02
LAMBDA_VAL = 0.06
lambda_indices = [1] # which lambda to use for evaluation
#-----------------------------------------

# Training DRS

print(dirname.split('/')[-1], '\n---------------')

common_tf = np.intersect1d(aces_data.columns, human_tfs.index)
tf_locs = [aces_data.columns.get_loc(c) for c in common_tf]

good_genes = (r2.values >= r2_threshold)
good_genes_loc = [aces_data.columns.get_loc(c) for c in aces_data.loc[:, good_genes]]

X = aces_data.to_numpy()
y = aces_p_type.ravel()

skf = StratifiedKFold(n_splits = n_fold, random_state = RANDOM_SEED, shuffle = True)
   
# performance evaluation

scores = evaluate(X, y, skf.split(X, y), dirname, 
                  ALPHA_CUTOFF, lambda_indices, good_genes_loc,
                  USE_SAVED_MODELS)
print('AUC:\n------------------')
for colname in scores.columns:
    print('{:10s}:\t{:.3f}'.format(colname, roc_auc_score(scores['real_class'], scores[colname])))

sfile = probability_scores_dir + \
    '/aces_ten_fold_scores_lambda_{0}_r2_{1}_alpha_{2}.csv'.format(LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF)
scores.to_csv(sfile)
    
#%% ACES leave-one-study-out model building

#-----------------------------------------
dirname = './models/ACESLeaveOneStudyOut'
lambdas = [0.13, 0.06, 0.03]
r2_threshold = 0.1
# r2_threshold = r2.values.min()-1 # if you want to use all the genes then use this threshold
ALPHA_CUTOFF = 0.02
LAMBDA_VAL = 0.06
lambda_indices = [1]
#-----------------------------------------

print(dirname.split('/')[-1], '\n---------------')

columns=['Desmedt', 'Hatzis', 'Ivshina', 'Loi', 'Miller', 'Minn',
         'Pawitan', 'Schmidt', 'Symmans', 'WangY', 'WangYE', 'Zhang'] #Study names

common_tf = np.intersect1d(aces_data.columns, human_tfs.index)
tf_locs = [aces_data.columns.get_loc(c) for c in common_tf]

good_genes = (r2.values >= r2_threshold)
good_genes_loc =[aces_data.columns.get_loc(c) for c in aces_data.loc[:, good_genes]]

n_studies = d_map.shape[1] #12 study
X = aces_data.to_numpy()
y = aces_p_type.ravel()

indices = []

# Training DRS

for study_index in range(n_studies):
    start_time = time.time()
    
    train_index = d_map[d_map[study_index] == 0].index
    test_index = d_map[d_map[study_index] == 1].index
    indices.append([train_index, test_index])

# performance evaluation

scores = evaluate(X, y, indices, dirname, ALPHA_CUTOFF, lambda_indices, good_genes_loc, USE_SAVED_MODELS)

print('AUC:\n------------------')
for colname in scores.columns:
    print('{:10s}:\t{:.3f}'.format(colname, roc_auc_score(scores['real_class'], scores[colname])))
    
sfile = probability_scores_dir + \
    '/aces_loso_scores_lambda_{0}_r2_{1}_alpha_{2}.csv'.format(LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF)
scores.to_csv(sfile)

#%% NKI 10 fold statified CV model building

#-----------------------------------------
dirname = './models/NKIStratifiedKFold'
lambdas = [0.13, 0.06, 0.03]
#Not using any alpha or r2 cutoff here
ALPHA_CUTOFF = 0.0
n_fold = 10
#-----------------------------------------

lambda_indices = [1]

print(dirname.split('/')[-1], '\n---------------')

common_tf = np.intersect1d(nki_data.columns, human_tfs.index)
tf_locs = [nki_data.columns.get_loc(c) for c in common_tf]
X = nki_data.to_numpy()
y = nki_p_type.ravel()
good_genes_loc = np.arange(X.shape[1])

skf = StratifiedKFold(n_splits = n_fold, random_state = RANDOM_SEED, shuffle = True)

scores = evaluate(X, y, skf.split(X, y), dirname, ALPHA_CUTOFF, lambda_indices, good_genes_loc, USE_SAVED_MODELS)

print('AUC:\n------------------')
for colname in scores.columns:
    print('{:10s}:\t{:.3f}'.format(colname, roc_auc_score(scores['real_class'], scores[colname])))
    
sfile = probability_scores_dir + \
    '/nki_ten_fold_scores_lambda_{0}_alpha_{1}.csv'.format(LAMBDA_VAL, ALPHA_CUTOFF)
scores.to_csv(sfile)

#%% NKI data validation model generation from ACES data

#-----------------------------------------
lambdas = [0.13, 0.06, 0.03]
dirname = './models/NKIDataValidationModels'
r2_threshold = 0.1
# r2_threshold = r2.values.min()-1 # if you want to use all the genes then use this threshold
ALPHA_CUTOFF = 0.02
lambda_indices = [1]
#-----------------------------------------

print(dirname.split('/')[-1], '\n---------------')

aces_filtered , nki_filtered, common_tfs = get_filtered_expr(aces_data, nki_data, human_tfs)
tf_locs = [aces_filtered.columns.get_loc(c) for c in common_tfs]

print(aces_filtered.shape, nki_filtered.shape)

r2_filtered = r2.loc[aces_filtered.columns]
#This value can be changed. Our result is based on all genes so r2_threshold should be r2.values.min()-1

good_genes = (r2_filtered.values >= r2_threshold)
good_genes_loc = [aces_filtered.columns.get_loc(c) for c in aces_filtered.loc[:, good_genes]]

'''Using aces data as training'''
X_train = aces_filtered.to_numpy()
y_train = aces_p_type.ravel()
X_test = nki_filtered.to_numpy()
y_test = nki_p_type.ravel()
X_test = qtnorm_with_ref_columnwise(X_train, X_test)
X = np.vstack([X_train, X_test])
y = np.hstack([y_train, y_test])
indices = [[np.arange(len(X_train)), np.arange(len(X_test)) + len(X_train)]]

scores = evaluate(X, y, indices, dirname, ALPHA_CUTOFF, lambda_indices, good_genes_loc, USE_SAVED_MODELS)

print('AUC:\n------------------')
for colname in scores.columns:
    print('{:10s}:\t{:.3f}'.format(colname, roc_auc_score(scores['real_class'], scores[colname])))

sfile = probability_scores_dir + \
    '/nki_validation_scores_lambda_{0}_r2_{1}_alpha_{2}.csv'.format(LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF)
scores.to_csv(sfile)

#%% Sanity check

LAMBDA_VAL = 0.06
classifier_names = ['DRS', 'RF', 'MLP', 'LogR', 'DecTree', 'KNN', 'SVC']
score_files = [
        'aces_ten_fold_scores_lambda_{0}_r2_{1}_alpha_{2}.csv'.format(LAMBDA_VAL, 0.1, 0.02),
        'aces_loso_scores_lambda_{0}_r2_{1}_alpha_{2}.csv'.format(LAMBDA_VAL, 0.1, 0.02),
        'nki_ten_fold_scores_lambda_{0}_alpha_{1}.csv'.format(LAMBDA_VAL, 0.0),
        'nki_validation_scores_lambda_{0}_r2_{1}_alpha_{2}.csv'.format(LAMBDA_VAL, 0.1, 0.02)
    ]

for score_file in score_files:
    print(score_file.split('_lambda')[0])
    s_old = pd.read_csv(probability_scores_dir + '_old/' + score_file)
    s_new = pd.read_csv(probability_scores_dir + '/' + score_file)
    p_old = [roc_auc_score(s_old['real_class'].astype(int), s_old[col].values) for col in classifier_names]
    p_new = [roc_auc_score(s_new['real_class'].astype(int), s_new[col]) for col in classifier_names]
    comp = pd.DataFrame([p_old, p_new], columns = classifier_names, index = ['old', 'new']).T
    print(comp.round(3), '\n')