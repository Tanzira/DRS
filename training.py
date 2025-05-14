#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:01:41 2023

@author: tanzira, sakhawat
"""

#%% All Imports
import os
import sys
import pandas as pd
import numpy as np
import scipy
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
import warnings

from joblib import Parallel, delayed
import itertools as it

from drs import DRS
from utils import format_time, get_filtered_expr, qtnorm_with_ref_columnwise

RANDOM_SEED = 42
USE_SAVED_MODELS = True
DATASET_DIRECTORY = './dataset'
MODEL_DIRECTORY = './models'
SCORES_DIRECTORY = './probability_scores'

lambdas = [0.13, 0.06, 0.03]
lambda_map = {lambda_val: i for i, lambda_val in enumerate(lambdas)}

try:
    os.makedirs(SCORES_DIRECTORY, exist_ok=True)
except:
    print(f'Could not create directory: "{SCORES_DIRECTORY}"')
    
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses


#%% Helper functions

def get_different_model_scores(Xtrain, c_train, Xtest, c_test, good_genes_loc = None, random_state = None):
    models = [
        RandomForestClassifier(random_state = random_state),
        LogisticRegression(solver='liblinear', random_state=random_state),
        DecisionTreeClassifier(random_state = random_state),
        KNeighborsClassifier(),
        MLPClassifier(random_state=random_state),
        svm.SVC(kernel = 'linear', random_state=random_state, probability=True)
    ]
    
    predicted_proba = []
    if good_genes_loc is not None:
        Xtrain = Xtrain.copy()[:, good_genes_loc]
        Xtest = Xtest.copy()[:, good_genes_loc]
    for model in models:
        model.fit(Xtrain, c_train)
        predicted_proba.append(model.predict_proba(Xtest)[:, 1])
    return predicted_proba

def evaluate(X, y,
             train_test_indices,
             drs_dirname,
             alpha_cutoff,
             lambda_indices,
             tf_locs,
             good_genes_loc,
             use_saved_models = True,
             random_state = None):
    scores = {}
    indices_list = list(train_test_indices)
    for fold, (train_index, test_index) in enumerate(indices_list):
        start_time = time.time()
        if len(indices_list) <= 1:
            # only one train/test split, do not use subdirectory for folds
            print('Single split')
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
                drs_obj = DRS.train(X_train, y_train, tf_locs, 
                                    lambdas, random_state = random_state)
                drs_obj.save_to_file(dirname_fold)
        else:
            print('Training from scratch')
            drs_obj = DRS.train(X_train, y_train, tf_locs, 
                                lambdas, random_state = random_state)
            drs_obj.save_to_file(dirname_fold)
            
        pred_lasso = drs_obj.predict_proba(X_test, tf_locs, good_genes_loc, 
                                           lambda_indices, alpha_cutoff)
        print('Training baseline models')
        rf, mlpc, logr, dtc, knnc, svcl = get_different_model_scores(X_train, y_train,
                                                                     X_test, y_test, random_state = random_state)
        pred_all = pd.DataFrame([pred_lasso, rf, mlpc, logr, dtc, knnc, svcl, y_test],
                                index = ['DRS','RF', 'MLP', 'LogR', 'DecTree', 'KNN', 'SVC', 'real_class']).T
        scores[fold] = pred_all
        print(dirname_fold, 'total time: ', format_time(time.time() - start_time))
    scores = pd.concat(scores, axis = 0)
    scores.index.names = ['fold', 'patient']
    return scores
        
    

#%% Load Datasets

LAMBDA_VAL = 0.06
lambdas = [0.13, 0.06, 0.03] # all lambda values will be used for model training
lambda_map = {k: v for v, k in enumerate(lambdas)}

'''Reading TF file'''
#tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
tf_file = f'{DATASET_DIRECTORY}/DatabaseExtract_v_1.01_human_TFs.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[(human_tfs['Is TF?'] =='Yes') & (human_tfs['EntrezGene ID'] != 'None')]
human_tfs.set_index('EntrezGene ID', inplace = True)
human_tfs = human_tfs.loc[~np.isnan(human_tfs.index.astype(float)), :]
human_tfs.index = human_tfs.index.astype(int)


'''AECS data'''
aces_dirname = f'{DATASET_DIRECTORY}/ACES'
aces_raw  = loadmat(f'{aces_dirname}/ACESExpr.mat')['data']
aces_p_type = loadmat(f'{aces_dirname}/ACESLabel.mat')['label']
aces_entrez_id = loadmat(f'{aces_dirname}/ACES_EntrezIds.mat')['entrez_ids']
aces_data = pd.DataFrame(aces_raw)
aces_data.columns = aces_entrez_id.reshape(-1)


'''read r2 values for each gene file'''
r2 = pd.read_csv(f'{DATASET_DIRECTORY}/R2_scores/cv_r2_score_lambda_{LAMBDA_VAL}.txt', header=None)
r2.index = aces_data.columns


''' Seperating the studies for leave one study out cross validation. '''

cv_train_idx_file = f'{aces_dirname}/CVIndTrain200.txt'
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
nki_dirname = f'{DATASET_DIRECTORY}/NKI'
nki_raw = loadmat(f'{nki_dirname}/vijver.mat')['vijver']
nki_p_type = loadmat(f'{nki_dirname}/VijverLabel.mat')['label']
nki_entrez_id = loadmat(f'{nki_dirname}/vijver_gene_list.mat')['vijver_gene_list']
nki_data = pd.DataFrame(nki_raw)
nki_data.columns = nki_entrez_id.reshape(-1)



'''Reading METABRIC data'''
metabric_dirname = f'{DATASET_DIRECTORY}/METABRIC'
metabric_raw = pd.read_csv(f'{metabric_dirname}/data_mrna_illumina_microarray.txt', sep = '\t')
metabric_raw = metabric_raw.drop(['Hugo_Symbol'], axis = 1)
# Some entrez IDs are duplicated, use the average for those genes
metabric_data = metabric_raw.groupby('Entrez_Gene_Id').mean().T

metabric_p_type = pd.read_csv(f'{metabric_dirname}/data_clinical_patient.txt', sep = '\t', skiprows = 4, index_col = 0)
metabric_p_type = metabric_p_type[['OS_MONTHS', 'OS_STATUS', 'RFS_MONTHS', 'RFS_STATUS']]
metabric_p_type['label'] = 2
metabric_p_type.loc[metabric_p_type['RFS_MONTHS'] >= 60, 'label'] = 0
metabric_p_type.loc[(metabric_p_type['RFS_MONTHS'] < 60) & (metabric_p_type['RFS_STATUS'] == '1:Recurred'), 'label'] = 1
metabric_p_type = metabric_p_type[metabric_p_type['label'] < 2]['label']
metabric_p_type = metabric_p_type.loc[np.intersect1d(metabric_p_type.index, metabric_data.index)]
metabric_data = metabric_data.loc[metabric_p_type.index, :]
metabric_p_type = metabric_p_type.astype(int).values
# drop genes with nan values
metabric_data = metabric_data.loc[:, (np.isnan(metabric_data).sum(axis = 0) == 0).values]

#%% ACES 10 fold statified CV training and evaluation

#-----------------------------------------
dirname = f'{MODEL_DIRECTORY}/ACESStratifiedKFold'
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
                  ALPHA_CUTOFF, lambda_indices, 
                  tf_locs, good_genes_loc,
                  USE_SAVED_MODELS, RANDOM_SEED)
print('AUC:\n------------------')
for colname in scores.columns:
    print('{:10s}:\t{:.3f}'.format(colname, roc_auc_score(scores['real_class'], scores[colname])))

sfile = SCORES_DIRECTORY + \
    '/{0}_scores_lambda_{1}_r2_{2}_alpha_{3}.csv'.format(dirname.split('/')[-1],
                                                         LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF)
scores.to_csv(sfile)
    
#%% ACES leave-one-study-out model building

#-----------------------------------------
dirname = f'{MODEL_DIRECTORY}/ACESLeaveOneStudyOut'
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

n_studies = d_map.shape[1] #12 studies
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

scores = evaluate(X, y, indices, dirname,
                  ALPHA_CUTOFF, lambda_indices, 
                  tf_locs, good_genes_loc,
                  USE_SAVED_MODELS, RANDOM_SEED)

print('AUC:\n------------------')
for colname in scores.columns:
    print('{:10s}:\t{:.3f}'.format(colname, roc_auc_score(scores['real_class'], scores[colname])))
    
sfile = SCORES_DIRECTORY + \
    '/{0}_scores_lambda_{1}_r2_{2}_alpha_{3}.csv'.format(dirname.split('/')[-1],
                                                         LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF)
scores.to_csv(sfile)

#%% NKI 10 fold statified CV model building

#-----------------------------------------
dirname = f'{MODEL_DIRECTORY}/NKIStratifiedKFold'
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

scores = evaluate(X, y, skf.split(X, y), dirname,
                  ALPHA_CUTOFF, lambda_indices, 
                  tf_locs, good_genes_loc,
                  USE_SAVED_MODELS, RANDOM_SEED)

print('AUC:\n------------------')
for colname in scores.columns:
    print('{:10s}:\t{:.3f}'.format(colname, roc_auc_score(scores['real_class'], scores[colname])))
    
sfile = SCORES_DIRECTORY + \
    '/{0}_scores_lambda_{1}_alpha_{2}.csv'.format(dirname.split('/')[-1],
                                                  LAMBDA_VAL, ALPHA_CUTOFF)
scores.to_csv(sfile)

#%% NKI data validation model generation from ACES data

#-----------------------------------------
dirname = f'{MODEL_DIRECTORY}/NKIValidation'
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

scores = evaluate(X, y, indices, dirname,
                  ALPHA_CUTOFF, lambda_indices, 
                  tf_locs, good_genes_loc,
                  USE_SAVED_MODELS, RANDOM_SEED)

print('AUC:\n------------------')
for colname in scores.columns:
    print('{:10s}:\t{:.3f}'.format(colname, roc_auc_score(scores['real_class'], scores[colname])))

sfile = SCORES_DIRECTORY + \
    '/{0}_scores_lambda_{1}_r2_{2}_alpha_{3}.csv'.format(dirname.split('/')[-1],
                                                         LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF)
scores.to_csv(sfile)


#%% METABRIC 10 fold statified CV model building

#-----------------------------------------
dirname = f'{MODEL_DIRECTORY}/METABRICStratifiedKFold'
#Not using any alpha or r2 cutoff here
ALPHA_CUTOFF = 0.0
n_fold = 10
#-----------------------------------------

lambda_indices = [1]

print(dirname.split('/')[-1], '\n---------------')

common_tf = np.intersect1d(metabric_data.columns, human_tfs.index)
tf_locs = [metabric_data.columns.get_loc(c) for c in common_tf]
X = metabric_data.to_numpy()
y = metabric_p_type.ravel()
good_genes_loc = np.arange(X.shape[1])

skf = StratifiedKFold(n_splits = n_fold, random_state = RANDOM_SEED, shuffle = True)

scores = evaluate(X, y, skf.split(X, y), dirname,
                  ALPHA_CUTOFF, lambda_indices,
                  tf_locs, good_genes_loc,
                  USE_SAVED_MODELS, RANDOM_SEED)

print('AUC:\n------------------')
for colname in scores.columns:
    print('{:10s}:\t{:.3f}'.format(colname, roc_auc_score(scores['real_class'], scores[colname])))
    
sfile = SCORES_DIRECTORY + \
    '/{0}_scores_lambda_{1}_alpha_{2}.csv'.format(dirname.split('/')[-1],
                                                  LAMBDA_VAL, ALPHA_CUTOFF)
scores.to_csv(sfile)

#%% METABRIC data validation model generation from ACES data

#-----------------------------------------
dirname = f'{MODEL_DIRECTORY}/METABRICValidation'
r2_threshold = 0.1
# r2_threshold = r2.values.min()-1 # if you want to use all the genes then use this threshold
ALPHA_CUTOFF = 0.02
lambda_indices = [1]
#-----------------------------------------

print(dirname.split('/')[-1], '\n---------------')

aces_filtered , metabric_filtered, common_tfs = get_filtered_expr(aces_data, metabric_data, human_tfs)
tf_locs = [aces_filtered.columns.get_loc(c) for c in common_tfs]

print(aces_filtered.shape, metabric_filtered.shape)

r2_filtered = r2.loc[aces_filtered.columns]
#This value can be changed. Our result is based on all genes so r2_threshold should be r2.values.min()-1

good_genes = (r2_filtered.values >= r2_threshold)
good_genes_loc = [aces_filtered.columns.get_loc(c) for c in aces_filtered.loc[:, good_genes]]

'''Using aces data as training'''
X_train = aces_filtered.to_numpy()
y_train = aces_p_type.ravel()
X_test = metabric_filtered.to_numpy()
y_test = metabric_p_type.ravel()
X_test = qtnorm_with_ref_columnwise(X_train, X_test)
X = np.vstack([X_train, X_test])
y = np.hstack([y_train, y_test])
indices = [[np.arange(len(X_train)), np.arange(len(X_test)) + len(X_train)]]

scores = evaluate(X, y, indices, dirname,
                  ALPHA_CUTOFF, lambda_indices,
                  tf_locs, good_genes_loc,
                  USE_SAVED_MODELS, RANDOM_SEED)

print('AUC:\n------------------')
for colname in scores.columns:
    print('{:10s}:\t{:.3f}'.format(colname, roc_auc_score(scores['real_class'], scores[colname])))

sfile = SCORES_DIRECTORY + \
    '/{0}_scores_lambda_{1}_r2_{2}_alpha_{3}.csv'.format(dirname.split('/')[-1],
                                                         LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF)
scores.to_csv(sfile)

#%% ACES DRS hyperparam grid search

#-----------------------------------------
dirnames = ['ACESStratifiedKFold', 'ACESLeaveOneStudyOut']
score_filenames = ['ACESGridSearch10FCV', 'ACESGridSearchLOSOCV']
n_fold = 10
param_r2 = ['None'] + list(np.arange(-0.1, 0.3, 0.1).round(3))
param_alpha = np.array([0.0] + list(0.02*np.geomspace(1, 8, 4))).round(3)
param_lambdas = [0.13, 0.06, 0.03, 'All']
#-----------------------------------------

from itertools import product

common_tf = np.intersect1d(aces_data.columns, human_tfs.index)
tf_locs = [aces_data.columns.get_loc(c) for c in common_tf]

X = aces_data.to_numpy()
y = aces_p_type.ravel()

skf = StratifiedKFold(n_splits = n_fold, random_state = RANDOM_SEED, shuffle = True)
indices_10f = list(skf.split(X, y))
indices_loso = []
for study_index in range(n_studies):
    train_index = d_map[d_map[study_index] == 0].index
    test_index = d_map[d_map[study_index] == 1].index
    indices_loso.append([train_index, test_index])
    


for indices, dirname, filename in zip([indices_10f, indices_loso],
                                       dirnames, score_filenames):
    print(dirname)
    scores = {}
    for fold, (train_index, test_index) in enumerate(indices):
        X_test, y_test = X[test_index], y[test_index]
        scores_fold = {'real_class': y_test}
        dirname_fold = f'{MODEL_DIRECTORY}/{dirname}/fold_{fold}'
        drs_obj = DRS.load_from_file(dirname_fold)
        for r2_, alpha_, lambda_ in product(param_r2, param_alpha, param_lambdas):
            lambda_idx = None
            if lambda_ != 'All':
                lambda_idx = [lambda_map[lambda_]]
            good_genes = (~r2.isna()).values
            if r2_ != 'None':
                good_genes = (r2.values >= r2_)
            good_genes_loc = [aces_data.columns.get_loc(c) for c in aces_data.loc[:, good_genes]]
            n_good_genes = len(good_genes_loc)
            y_pred = drs_obj.predict_proba(X_test, tf_locs, good_genes_loc,
                                           lambda_indices = lambda_idx, alpha_cutoff = alpha_)
            scores_fold[f'DRS (r2={r2_}), alpha={alpha_}, lambda={lambda_}, n_genes={n_good_genes})'] = y_pred
        scores[fold] = pd.DataFrame(scores_fold)
    scores = pd.concat(scores, axis = 0)
    scores.index.names = ['fold', 'patient']
    sfile = f'{SCORES_DIRECTORY}/{filename}_scores.csv'
    scores.to_csv(sfile)

#%% ACES Baseline models with gene filtering

#-----------------------------------------
dirname = f'{MODEL_DIRECTORY}/ACESBaselinesWithGeneFiltering'
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

scores = {}
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    scores_fold = {}
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    
    rf, mlpc, logr, dtc, knnc, svcl = get_different_model_scores(
        X_train, y_train, X_test, y_test, 
        random_state = RANDOM_SEED)
    

    rf_filt, mlpc_filt, logr_filt, dtc_filt, knnc_filt, svcl_filt = get_different_model_scores(
        X_train, y_train, X_test, y_test, 
        good_genes_loc = good_genes_loc,
        random_state = RANDOM_SEED)

    scores_fold = pd.DataFrame([rf, mlpc, logr, dtc, knnc, svcl,
                                rf_filt, mlpc_filt, logr_filt, dtc_filt, knnc_filt, svcl_filt,
                                y_test], 
                               index = ['RF', 'MLP', 'LogR', 'DecTree', 'KNN', 'SVC',
                                        'RF (filtered)', 'MLP (filtered)', 'LogR (filtered)',
                                        'DecTree (filtered)', 'KNN (filtered)', 'SVC (filtered)',
                                        'real_class']).T

    scores[fold] = scores_fold

scores = pd.concat(scores, axis = 0)
scores.index.names = ['fold', 'patient']

print('AUC:\n------------------')
for colname in np.sort(scores.columns):
    print('{:20s}:\t{:.3f}'.format(colname, roc_auc_score(scores['real_class'], scores[colname])))

sfile = SCORES_DIRECTORY + \
    '/{0}_scores.csv'.format(dirname.split('/')[-1])
scores.to_csv(sfile)
    
#%% ACES External GRN 10-fold stratified cross-validation scores

# External GRNs have to be pre-trained using "training_external_grns.py"

#-----------------------------------------
dirname = f'{MODEL_DIRECTORY}/ACESStratifiedKFold'
grn_builders = ['BayesianRidgeScore', 'TIGRESS']
grn_weight_selection = ['all', 'above_mean']
n_fold = 10
r2_threshold = 0.1
# r2_threshold = r2.values.min()-1 # if you want to use all the genes then use this threshold
ALPHA_CUTOFF = 0.02
use_saved_models = True
LAMBDA_VAL = 0.06
lambda_indices = [1] # which lambda to use for evaluation
label_names = {0: 'nmeta', 1: 'meta'}
#-----------------------------------------

# Training DRS

print(dirname.split('/')[-1], '\n---------------')

common_tf = np.intersect1d(aces_data.columns, human_tfs.index)
tf_locs = [aces_data.columns.get_loc(c) for c in common_tf]

good_genes = (r2.values >= r2_threshold)
good_genes_loc = [aces_data.columns.get_loc(c) for c in aces_data.loc[:, good_genes]]

X = aces_data.copy().values
y = aces_p_type.copy().ravel()

skf = StratifiedKFold(n_splits = n_fold, random_state = RANDOM_SEED, shuffle = True)
indices_list = list(skf.split(X, y))
# performance evaluation

scores = {}
for predictor_name in grn_builders:
    dirname_predictor = f'{dirname}/with_{predictor_name}'
    scores_predictor = {}
    print(predictor_name, '\n-----------------')
    scores_fold = {}
    for fold, (train_index, test_index) in enumerate(indices_list):
        dirname_fold = f'{dirname_predictor}/fold_{fold}'
       
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]       
        # perform z-normalization using the mean and std of the train split
        mean, std = X_train.mean(axis = 0), X_train.std(axis = 0)
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        external_grn = {}
        for label_name in label_names.values():
            filename = f'{dirname_fold}/{predictor_name}_{label_name}.npz'
            grn = np.array(scipy.sparse.load_npz(filename).todense())
            external_grn[label_name] = grn
            
        # threshold application to GRN weights
        scores_weight_selection = {}
        for weight_selection in grn_weight_selection:
            dirname_drs = dirname_fold + f'/DRS_{weight_selection}'
            print(dirname_drs)
            if weight_selection == 'above_mean':
                for label_name in label_names.values():
                    grn = external_grn[label_name]
                    # apply the mean of TFs with non-zero weights as cutoff
                    # only use TF if above mean
                    cutoff = grn.sum(axis = 0) / (grn > 0).sum(axis = 0)
                    grn[grn < cutoff] = 0
                    external_grn[label_name] = grn
            if use_saved_models:
                try:
                    drs_obj = DRS.load_from_file(dirname_drs)
                except:
                    print('Failed to load saved model at', dirname_drs)
                    continue
                    drs_obj = DRS.train(X_train, y_train, tf_locs, 
                                        lambdas, 
                                        external_grn = external_grn,
                                        random_state = RANDOM_SEED)
                    drs_obj.save_to_file(dirname_drs)
            else:
                print('Training from scratch')
                drs_obj = DRS.train(X_train, y_train, tf_locs, 
                                    lambdas, 
                                    external_grn = external_grn,
                                    random_state = RANDOM_SEED)
                drs_obj.save_to_file(dirname_drs)
            pred_lasso = drs_obj.predict_proba(X_test, tf_locs, good_genes_loc, 
                                               lambda_indices, ALPHA_CUTOFF)
            scores_weight_selection[f'DRS_with_{predictor_name}_{weight_selection}'] =\
                pred_lasso
        scores_weight_selection = pd.DataFrame(scores_weight_selection)
        scores_fold[fold] = scores_weight_selection
    scores_fold = pd.concat(scores_fold)
    scores[predictor_name] = scores_fold
scores = pd.concat(scores, axis = 1).droplevel(0, axis = 1)
scores['real_class'] = np.hstack([y[idx] for _, idx in indices_list])
scores.index.names = ['fold', 'patient']
sfile = SCORES_DIRECTORY + \
    '/{0}withExternal_scores_lambda_{1}_r2_{2}_alpha_{3}.csv'.format(dirname.split('/')[-1],
                                                         LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF)
scores.to_csv(sfile)