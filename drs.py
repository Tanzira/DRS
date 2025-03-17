#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:09:12 2025

@author: tanzira, sakhawat
"""

from joblib import Parallel, delayed
import numpy as np
from sklearn import linear_model
import pickle
import scipy
import os
import time

from utils import format_time

class DRS():
    label_names = {0: 'nmeta', 1: 'meta'}
    
    def _do_lasso_parallel(X_train, tf_locs, lambda_val, random_state = None):
        models = (Parallel(n_jobs = -1,
                           require = 'sharedmem')
                  (delayed(linear_model.Lasso(alpha = lambda_val,
                                              max_iter = 10000,
                                              random_state = random_state)
                           .fit)(X_train[:, tf_locs],
                                 X_train[:, i]) for i in range(X_train.shape[1])))
        coefficients = [model.coef_ for model in models]
        intercepts = [model.intercept_ for model in models]
        return coefficients, intercepts
    
    def train(X_train, y_train, tf_locs, lambdas, random_state = None):
        models = {}
        intercepts = {}
        start_time = time.time()
        print('Training DRS')
        for label, label_name in DRS.label_names.items():
            X_train_label = X_train[y_train == label, :]           
            models_label, intercepts_label = [], []
            #Doing lasso for 3 different lambdas
            for lambda_val in lambdas:
                coef, intercept = DRS._do_lasso_parallel(X_train_label, tf_locs, lambda_val)
                models_label.append(coef)
                intercepts_label.append(intercept)
            models_label = np.swapaxes(np.stack(models_label), 1, 2)
            intercepts_label = np.stack(intercepts_label)
            models[label_name] = models_label
            intercepts[label_name] = intercepts_label
        print('Completed [' + format_time(time.time() - start_time) + ']')
        return DRS(models, intercepts)
    
    # For compatibility with legacy code, unused
    def load_from_pickle(dirname):
        models = {}
        intercepts = {}
        for label, label_name in DRS.label_names.items():
            filename = f'{dirname}/{label_name}_#'
            with open(filename.replace('#', 'models') + '.pkl', 'rb') as file:
                models[label_name] = pickle.load(file)
            intercepts[label_name] = np.loadtxt(filename.replace('#', 'intercepts') + '.txt')
        return DRS(models, intercepts)
    
    def load_from_file(dirname):
        
        models = {}
        intercepts = {}
        for label, label_name in DRS.label_names.items():
            filename = f'{dirname}/{label_name}_#'
            intercepts[label_name] = np.loadtxt(filename.replace('#', 'intercepts') + '.txt').T
            models_label_sparse = scipy.sparse.load_npz(filename.replace('#', 'models') + '.npz')

            # Convert back to a dense numpy array, if needed
            n_lambda, n_gene_models = intercepts[label_name].shape
            models_label = models_label_sparse.toarray().reshape([n_lambda, -1, n_gene_models])
            models[label_name] = models_label
        return DRS(models, intercepts)
    
    def __init__(self, models, intercepts):
        self.models = models
        self.intercepts = intercepts
            
    def save_to_file(self, dirname):
        try:
            os.makedirs(dirname, exist_ok=True)
            for label, label_name in DRS.label_names.items():
                filename = dirname + f'/{label_name}_#'
                models_label = self.models[label_name]
                models_label_sparse = (scipy
                                      .sparse
                                      .csr_matrix(models_label
                                                  .reshape(-1,
                                                           models_label.shape[-1])))
                scipy.sparse.save_npz(filename.replace('#', 'models'), models_label_sparse)
                np.savetxt(filename.replace('#', 'intercepts') + '.txt',
                           self.intercepts[label_name].T,
                           fmt = '%.8f')
            return self
        except OSError:
            print("Error occured")
        
     
    def _get_pred_expr(self, Xtest, tf_locs, alpha_cutoff, lambda_indices):
        Mp, Mn = self.models['meta'], self.models['nmeta']
        bp, bn = self.intercepts['meta'], self.intercepts['nmeta']
        #print(f'{Mp.shape=}, {bp.shape=}, {Xtest.shape=}')
        
        if lambda_indices is not None:
            Mp, Mn = Mp[lambda_indices], Mn[lambda_indices]
            bp, bn = bp[lambda_indices], bn[lambda_indices]
        
        Mp[np.abs(Mp) < alpha_cutoff] = 0
        Mn[np.abs(Mn) < alpha_cutoff] = 0
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
    
    def _get_distance(self, pred0, pred1, actual, method):
        # pred0: expression levels predicted by non-metastatic models
        # pred1: expression levels predicted by metastatic models
        n = len(actual) #total number of patients
        if method == 'pearson':
            s0 = np.corrcoef(actual, pred0)[range(n), range(n, n*2)]
            s1 = np.corrcoef(actual, pred1)[range(n), range(n, n*2)]
            s1, s0 = 1 - np.abs(s1), 1 - np.abs(s0) #changing correlation into distance
        elif method == 'spearman':
            s0 = scipy.stats.spearmanr(actual, pred0, axis = 1)[0][range(n), range(n, n*2)]
            s1 = scipy.stats.spearmanr(actual, pred1, axis = 1)[0][range(n), range(n, n*2)]
            s1, s0 = 1 - np.abs(s1), 1 - np.abs(s0) #changing correlation into distance
            
        elif method == 'logdistance':
            s0 = np.log(np.abs((pred0 / actual))).sum(axis = 1)
            s1 = np.log(np.abs((pred1 / actual))).sum(axis = 1)
        else:
            if method == 'minkowski':
                norm_val = 1
            else:
                norm_val = 2
            s0 = np.linalg.norm((pred0 - actual), ord = norm_val, axis = 1)
            s1 = np.linalg.norm((pred1 - actual), ord = norm_val, axis = 1)
        return s0, s1
    
    def predict_proba(self, X_test, tf_locs, filtered_gene_index, 
                      lambda_indices = None, alpha_cutoff = 0.01, 
                      method = 'pearson'):
        #get predicted expressions
        #print(f'{lambda_indices=}')
        y_pred_n, y_pred_p = self._get_pred_expr(X_test, tf_locs, alpha_cutoff, lambda_indices)
        #filtering genes based on good gene locations
        X_test_filtered = X_test[:, filtered_gene_index]
        y_pred_p = y_pred_p[:, filtered_gene_index]
        y_pred_n = y_pred_n[:, filtered_gene_index]
        dn, dp = self._get_distance(y_pred_n, y_pred_p, X_test_filtered, method) #Using pearson distance
        c_pred = dn - dp
        return c_pred