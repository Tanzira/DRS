#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:53:37 2023

@author: tanzira
"""

#%%All imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.metrics import roc_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
dir_name = 'ProbabilityScores'
#%% R2 plot all the lambdas
lambdas = [1.0, 0.5, 0.25, 0.13, 0.06, 0.03, 0.01, 0.008, 0.004]
r2_scores = pd.DataFrame(columns = lambdas)
fig, axes = plt.subplots(figsize = (6, 4), constrained_layout = True)
for lambda_ in lambdas:
    file_name = 'R2CVScores/cv_r2_score_lambda_{0}.txt'.format(lambda_)
    with open(file_name, 'r') as file:
        scores = np.array(file.read().strip().split(), dtype = np.float64)
        r2_scores.loc[:, lambda_] = scores
sns.boxplot(data = r2_scores, ax = axes, showfliers = True)
plt.xlabel(r'$\lambda$ thresholds')
plt.ylabel(r'$R^2$ values')
axes.set_xticklabels([r'$2^{' + str(i) + '}$' for i in range(0, -9, -1)])
axes.invert_xaxis()
plt.savefig('Figures/r2_box_plot_bc.eps')
plt.show()

#%%correlation based Edge Stats figure with positive negative correlation cutoff
LAMBDA_VAL = 0.06
ALPHA_CUTOFF = 0.02
r2_threshold = 0.1
m_file = 'Models/NetworkModels/meta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
nm_file = 'Models/NetworkModels/nmeta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
Mp = pd.read_csv(m_file, index_col = 0)
Mn = pd.read_csv(nm_file, index_col = 0)

r2 = pd.read_csv('R2CVScores/cv_r2_score_lambda_{0}.txt'.format(LAMBDA_VAL), header=None)
min_tg_th = 10

    
goodGenes = (r2.values >= r2_threshold)
filtered_Mp = Mp.loc[goodGenes, :]
filtered_Mn = Mn.loc[goodGenes, :]
print(filtered_Mp.shape, filtered_Mn.shape)

'''Not taking coef threshold for this stats'''
filtered_Mp[np.abs(filtered_Mp) < ALPHA_CUTOFF] = 0
filtered_Mn[np.abs(filtered_Mn) < ALPHA_CUTOFF] = 0

#keeping TFs with at least min_tg_th targets
nTargets_meta = (filtered_Mp != 0).sum(axis = 0) >= min_tg_th
nTargets_nmeta = (filtered_Mn != 0).sum(axis = 0) >= min_tg_th

'''Here TFs from both networks has to have minimum target greater than the threshold
So the network shape will be the same'''
filtered_Mp = filtered_Mp.loc[:, nTargets_meta & nTargets_nmeta]
filtered_Mn = filtered_Mn.loc[:, nTargets_meta & nTargets_nmeta]

print("After min target filtration: ", filtered_Mp.shape, filtered_Mn.shape)

#calculating pearson correlation among TFs
meta_coreg = filtered_Mp.corr()
nmeta_coreg = filtered_Mn.corr()

#filling null values with 0
meta_coreg.fillna(0, inplace = True)
nmeta_coreg.fillna(0, inplace = True)

np.fill_diagonal(meta_coreg.values, 0) # Filling the diagonals with 0
np.fill_diagonal(nmeta_coreg.values, 0) # Filling the diagonals with 0
#calculating total number of non-zero correlations for meta
total_edges_m_p = (meta_coreg > 0).sum().sum()
total_edges_nm_p = (nmeta_coreg > 0).sum().sum()

total_edges_m_n = (meta_coreg < 0).sum().sum()
total_edges_nm_n = (nmeta_coreg < 0).sum().sum()

print('min_tg_th: ', min_tg_th)
print(' Meta: pos: \t', total_edges_m_p, 'neg: \t', total_edges_m_n)
print('Nmeta: pos: \t',total_edges_nm_p, 'neg: \t', total_edges_nm_n)

n_tf_meta = meta_coreg.shape[0]
n_tf_nmeta = nmeta_coreg.shape[0]
stats = []
p_cutoffs = np.arange(0.1, 0.31, 0.025)
percentage_nm, percentage_m, ci_1s, ci_2s = [], [], [], []
for t in p_cutoffs:
    a = np.sum(nmeta_coreg.values >= t)
    b = np.sum(meta_coreg.values >= t)
    c = np.sum(nmeta_coreg.values < -t)
    d = np.sum(meta_coreg.values < -t)
    print(a, b, c, d)
    stats.append([t, a, b, c, d])
    n1, n2 = (c+a), (d+b)
    p1, p2 = c / n1, d / n2
    ci_1s.append(np.sqrt(p1*(1-p1)/n1)*100)
    ci_2s.append(np.sqrt(p2*(1-p2)/n2)*100)
    percentage_nm.append(p1*100)
    percentage_m.append(p2*100)
  
columns = ['cc', 
           'Non-metastatic (+ve)', 'Metastatic (+ve)',
           'Non-metastatic (-ve)', 'Metastatic (-ve)']
stats = pd.DataFrame(stats, columns = columns).set_index('cc')
fig, axes = plt.subplots(figsize = (6, 6), sharex = True, nrows = 2,
                         gridspec_kw = {"height_ratios": [1, 1], "hspace": 0.05})

ax = axes[0]

stats.plot(ax = ax, style = ['s-', 'o-', '^--', 'x--'])

ax.set_yscale('log')
#ax.set_xticks([])
ax.set_ylabel('# of correlated\n TF pairs')
ax.legend(bbox_to_anchor = [1, 1], loc = 'upper left')


ax = axes[1]
plt.errorbar(p_cutoffs, percentage_nm, yerr = ci_1s, marker = 'o', label = 'Non-metastatic')
plt.errorbar(p_cutoffs, percentage_m, yerr = ci_2s, marker = 'x', label = 'Metastatic')
plt.legend(bbox_to_anchor = [1, 1], loc = 'upper left')
plt.ylabel('% of negatively\n correlated TF pairs')
ax.set_xlabel('Absolute correlation cut-offs')
plt.savefig('Figures/negative-edge-stats.eps', bbox_inches = 'tight', dpi = 300)
plt.show()

#%%Accuracy calculation function
''' 
    Using this method to get f1, kappa, auc and balanced accuracy for LOSO method.
    Here we used other threshold than 0.5 which is used by sklearn. 0.5 s not optimal for all classifier like RF here.
    So we took mid point of max and min probability scores for each classifier that we got from 10 fold CV.
'''
def kappa_f1_bacc(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba, drop_intermediate = False)
    '''Methods from sklearn'''
    
    prob_th = round((min(y_pred_proba) + max(y_pred_proba))/ 2, 2)
    print("proba th: ", prob_th)
    y_pred = np.array(y_pred_proba > prob_th, dtype = int)
    f1_sklearn = metrics.f1_score(y_true, y_pred)
    bacc_sklearn = balanced_accuracy_score(y_true, y_pred)
    cohen_kappa = metrics.cohen_kappa_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_pred_proba)
    return auc, f1_sklearn, cohen_kappa, bacc_sklearn

def get_fpr_tpr_kappa(y_true, pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, pred_proba, drop_intermediate = False)
    pred_proba_reshaped = np.tile(pred_proba, [len(thresholds), 1])
    thres_reshaped = np.reshape(thresholds, [-1, 1])
    c_pred_thres = np.array(pred_proba_reshaped >= thres_reshaped, dtype = int)
    c_test_thres = np.tile(y_true, [len(thresholds), 1])
    
    '''Calculating these for kappa score calculation to get kappa for different FPR'''
    TP = ((c_pred_thres == 1) & (c_test_thres == 1)).sum(axis = 1)
    FP = ((c_pred_thres == 1) & (c_test_thres == 0)).sum(axis = 1)
    TN = ((c_pred_thres == 0) & (c_test_thres == 0)).sum(axis = 1)
    FN = ((c_pred_thres == 0) & (c_test_thres == 1)).sum(axis = 1)
    
    '''Calculating kappa for different thresholds'''
    kappa = 2 * (TP*TN - FN*FP) / ((TP + FP) * (FP + TN) + (TP + FN) * (FN + TN))
    return fpr, tpr, kappa

#%% ACES 10 fold and LOSO accuracy plot
ten_fold_fpath =dir_name + '/aces_ten_fold_scores_lambda_0.06_r2_0.1_alpha_0.02.csv'
loso_fpath = dir_name + '/aces_loso_scores_lambda_0.06_r2_0.1_alpha_0.02.csv'
#reading scores for ten fold cross validation here.
tenfold_scores = pd.read_csv(ten_fold_fpath, index_col = 0)

columns = ['AUC', 'F1-score', 'kappa', 'balanced\naccuracy']
index = ['DRS', 'RF', 'MLP', 'KNN', 'LogR', 'DecTree', 'SVC']
acc_tenfold = pd.DataFrame(0, columns = columns, index = index)
for i, classifier in enumerate(index):
    # print(classifier, (min(nki_scores[classifier]) + max(nki_scores[classifier])) / 2)
    acc_tenfold.loc[classifier, :] = kappa_f1_bacc(tenfold_scores.real_class, 
                                             tenfold_scores[classifier])
fig, axes = plt.subplots(1, 2, figsize = (10, 3))
ax = axes[ 0]
acc_tenfold.T.plot.bar(legend = False, rot = 45, width = .7, ax = ax)

h, l = ax.get_legend_handles_labels()
ax.set_title('(a) ACES stratified 10-fold CV')

print("ACES 10 fold CV: ")
print(acc_tenfold.T)
#reading scores for leave one out study here.
loso_scores = pd.read_csv(loso_fpath, index_col = 0)
columns = ['AUC', 'F1-score', 'kappa', 'balanced\naccuracy']
acc_loso_cv = pd.DataFrame(0, columns = columns, index = index)
for i, classifier in enumerate(index):
    acc_loso_cv.loc[classifier, :] = kappa_f1_bacc(loso_scores.real_class, 
                                             loso_scores[classifier])
print("LOSO CV: ")
print(acc_loso_cv)
ax = axes[1]
acc_loso_cv.T.plot.bar( rot = 45, ax = ax, width = .7, legend = False)
# ax.legend(loc = 'upper left', bbox_to_anchor = [1, 1], frameon = False)
ax.set_title('(b) ACES leave-one-study-out CV')
# ax.set_xlabel('Performance metric')
plt.subplots_adjust(wspace = 0.2)
plt.legend(h, l,loc = 'upper left', bbox_to_anchor = [1, 1], frameon = False)
plt.savefig('Figures/combined-performance-r2-cutoff.png', bbox_inches = 'tight')
plt.show()

#%%Loso Study wise AUC plot
loso_fpath = dir_name + '/aces_loso_scores_lambda_0.06_r2_0.1_alpha_0.02.csv'
loso_scores = pd.read_csv(loso_fpath)
classifiers = ['DRS', 'RF', 'MLP', 'KNN', 'LogR', 'DecTree', 'SVC']

df_auc = pd.DataFrame(index = loso_scores.study.unique(), columns = classifiers)

for study in df_auc.index:
    for classifier in df_auc.columns:
        df = loso_scores[loso_scores.study == study][[classifier, 'real_class']]
        df_auc.loc[study, classifier] = roc_auc_score(df.real_class, df[classifier])
df_auc.plot.bar(figsize = (12, 4), width = .7, rot = 80, legend = False)
plt.gca().legend(loc = 'upper left', bbox_to_anchor = [1, 1])
plt.savefig('Figures/study-specific-auc-loso-cv.eps', bbox_inches = 'tight')
plt.show()
# Write table
t = loso_scores[['study', 'real_class']]
t['real_class'] = t['real_class'].replace({0.0: 'Non-metastatic', 1.0: 'Metastatic'})
t = pd.pivot_table(t.value_counts().reset_index(), values = 0, index = 'study', columns = 'real_class')
t['Total'] = t.sum(axis = 1)
t = pd.concat([t, df_auc], axis = 1)
t.to_latex('loso_table.tex',  float_format = "{:0.3f}".format)

#%%NKI performance plot
nki_val_path = dir_name + '/nki_validation_scores_lambda_0.06_r2_0.1_alpha_0.02.csv'
nki_10_fpath = dir_name + '/nki_ten_fold_scores_lambda_0.06_alpha_0.0.csv'


nki_val_scores = pd.read_csv(nki_val_path, index_col = 0)
nki_10f_scores = pd.read_csv(nki_10_fpath, index_col = 0)

columns = ['AUC', 'F1-score', 'kappa', 'balanced\naccuracy']
classifiers = ['DRS', 'RF', 'MLP', 'KNN', 'LogR', 'DecTree', 'SVC']

acc_nki_10f = pd.DataFrame(0, columns = columns, index = classifiers)
for i, classifier in enumerate(index):
    # print(classifier, (min(nki_scores[classifier]) + max(nki_scores[classifier])) / 2)
    acc_nki_10f.loc[classifier, :] = kappa_f1_bacc(nki_10f_scores.real_class, 
                                            nki_10f_scores[classifier])
print("NKI 10 fold Accuracy:")
print(acc_nki_10f)

acc_nki_val = pd.DataFrame(0, columns = columns, index = classifiers)
for i, classifier in enumerate(index):
    acc_nki_val.loc[classifier, :] = kappa_f1_bacc(nki_val_scores.real_class, 
                                                   nki_val_scores[classifier])
print("NKI Validation Accuracy: ")
print(acc_nki_val)



fig, ax = plt.subplots(1, 2, figsize = (10, 3), 
                         gridspec_kw={'width_ratios':[4, 4]})

acc_nki_10f.T.plot.bar(legend = False, rot = 45, width = .7, ax = ax[0])
ax[0].set_title('(a) NKI stratified 10-fold CV')

acc_nki_val.T.plot.bar(legend = True, rot = 45, width = .7, ax = ax[1])
ax[1].set_title('(b) NKI validation')

ax[1].legend(loc = 'upper left', bbox_to_anchor = [1, 1], frameon = False)
plt.savefig('Figures/nki-performance-r2-cutoff.eps', bbox_inches = 'tight', dpi = 300)
plt.show()