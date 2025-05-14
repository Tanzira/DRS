#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 12:45:31 2025

@author: ruanlab
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import scipy
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
import time

from grenadine.Inference.regression_predictors import (TIGRESS, GENIE3,
                                                       BayesianRidgeScore)
from joblib import Parallel, delayed

from utils import format_time

RANDOM_SEED = 42
USE_SAVED_MODELS = False
DATASET_DIRECTORY = './dataset'
MODEL_DIRECTORY = './models'
SCORES_DIRECTORY = './probability_scores'

'''Reading TF file'''
#tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
tf_file = DATASET_DIRECTORY + '/DatabaseExtract_v_1.01_human_TFs.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[(human_tfs['Is TF?'] =='Yes') & (human_tfs['EntrezGene ID'] != 'None')]
human_tfs.set_index('EntrezGene ID', inplace = True)
human_tfs = human_tfs.loc[~np.isnan(human_tfs.index.astype(float)), :]
human_tfs.index = human_tfs.index.astype(int)


'''AECS data'''
aces_dirname = '{:}/ACES/'.format(DATASET_DIRECTORY)

aces_raw  = loadmat(aces_dirname + 'ACESExpr.mat')['data']
aces_p_type = loadmat(aces_dirname + 'ACESLabel.mat')['label']
aces_entrez_id = loadmat(aces_dirname + 'ACES_EntrezIds.mat')['entrez_ids']
aces_data = pd.DataFrame(aces_raw)
aces_data.columns = aces_entrez_id.reshape(-1)


'''read r2 values for each gene file'''
LAMBDA_VAL = 0.06
r2 = pd.read_csv('{:}/R2_scores/cv_r2_score_lambda_{:}.txt'.format(
    DATASET_DIRECTORY, LAMBDA_VAL), header=None)
r2.index = aces_data.columns

#-----------------------------------------
dirname = MODEL_DIRECTORY + '/ACESStratifiedKFold'
n_fold = 10
grn_builders = [TIGRESS, BayesianRidgeScore, GENIE3]
normalize = False
#-----------------------------------------

# Training DRS
print(dirname.split('/')[-1], '\n---------------')



label_names = {0: 'nmeta', 1: 'meta'}

def _do_grenadine_parallel(X_train, tf_locs, random_state = None):
    predictors_with_stochasticity = [GENIE3]
    
    def _do_grenadine(X_train, tf_locs, gene_idx, random_state = None):
        expr_gene = X_train[:, gene_idx]
        temp = X_train.copy()
        if gene_idx in tf_locs:
            temp[:, gene_idx] = 0
        expr_tf = temp[:, tf_locs]
        if Predictor in predictors_with_stochasticity:
            model = Predictor(expr_tf, expr_gene, random_state = random_state)
        else:
            model = Predictor(expr_tf, expr_gene)
        return model
    
    scores = (Parallel(n_jobs = -1,
                       require = 'sharedmem')
              (delayed(_do_grenadine)(X_train, tf_locs, i) for i in range(X_train.shape[1])))
    return np.vstack(scores).T

X = aces_data.copy()
y = aces_p_type.copy().ravel()

common_tf = np.intersect1d(X.columns, human_tfs.index)
tf_locs = [X.columns.get_loc(c) for c in common_tf]
X = X.values

skf = StratifiedKFold(n_splits = n_fold, random_state = RANDOM_SEED, shuffle = True)
indices_list = list(skf.split(X, y))

for Predictor in grn_builders:
    print(Predictor.__name__, '\n-----------------')
    for fold, (train_index, test_index) in enumerate(indices_list):
        start_time = time.time()
        if len(indices_list) <= 1:
            # only one train/test split, do not use subdirectory for folds
            dirname_fold = dirname
        else:
            dirname_fold = dirname + '/fold_{:}'.format(fold)
        os.makedirs(dirname_fold, exist_ok=True)
        
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        
        X_train_norm = X_train
        X_test_norm = X_test
        
        if normalize:
        # perform z-normalization using the mean and std of the train split
            mean, std = X_train.mean(axis = 0), X_train.std(axis = 0)
            X_train_norm = (X_train - mean) / std
            X_test_norm = (X_test - mean) / std
    
        
        # Build 2 GRNS (meta, nmeta)
        models = {}
        for label, label_name in label_names.items():
            print('\tfold={:}, train shape={:}, label={:}'.format(
                fold, X_train.shape, label_name))
            filename = '{:}/{:}_{:}'.format(dirname_fold, 
                                                Predictor.__name__,
                                                label_name)
            if USE_SAVED_MODELS and Path(filename).exists():
                scores_sparse = scipy.sparse.load_npz(filename)
            else:
                start_time = time.time()
                mask = y_train == label
                #scores = _do_grenadine_parallel(X_train[mask, :], Y_train[mask, :], Predictor)
                scores = _do_grenadine_parallel(X_train[mask, :],
                                                tf_locs,
                                                random_state = RANDOM_SEED)
                models[label_name] = scores
                print('\t', format_time(time.time() - start_time))
                scores_sparse = scipy.sparse.csr_matrix(scores)
                scipy.sparse.save_npz(filename, scores_sparse)