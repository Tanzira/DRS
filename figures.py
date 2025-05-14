#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:53:37 2023

@author: tanzira, sakhawat
"""

#%%All imports

import os
import pandas as pd
import numpy as np
import scipy
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from utils import scores_to_metrics, scores_to_metrics_per_fold

SCORES_DIRECTORY = './probability_scores'
FIGURES_DIRECTORY = './figures'
MODEL_DIRECTORY = './models'
EXPORT_FIGURES = True
FIGURE_FILETYPE = 'eps'

classifier_order = ['DRS', 'RF', 'MLP', 'KNN', 'LogR', 'DecTree', 'SVC']

# wrapper function for pyplot.savefig()
def savefig(filename):
    if not EXPORT_FIGURES:
        return 
    export_path = '{}/{}.{}'.format(FIGURES_DIRECTORY, filename, FIGURE_FILETYPE)
    try:
        os.makedirs(FIGURES_DIRECTORY, exist_ok=True)
        plt.savefig(export_path, bbox_inches = 'tight', dpi = 300)
    except:
        print(f'Could not save figure: "{export_path}"')
    
#%% R2 distribution plot for different lambdas
lambdas = [1.0, 0.5, 0.25, 0.13, 0.06, 0.03, 0.01, 0.008, 0.004]
r2_scores = pd.DataFrame(columns = lambdas)
fig, axes = plt.subplots(figsize = (6, 4), constrained_layout = True)
for lambda_ in lambdas:
    file_name = 'dataset/R2_scores/cv_r2_score_lambda_{0}.txt'.format(lambda_)
    with open(file_name, 'r') as file:
        scores = np.array(file.read().strip().split(), dtype = np.float64)
        r2_scores.loc[:, lambda_] = scores
sns.boxplot(data = r2_scores, ax = axes, showfliers = True)
plt.xlabel(r'$\lambda$ thresholds')
plt.ylabel(r'$R^2$ values')
axes.set_xticklabels([r'$2^{' + str(i) + '}$' for i in range(0, -9, -1)])
axes.invert_xaxis()
savefig('r2_box_plot_bc.eps')
plt.show()

#%% Correlated TF pairs

LAMBDA_VAL = 0.06
ALPHA_CUTOFF = 0.02
r2_threshold = 0.1
p_cutoffs = np.arange(0.1, 0.31, 0.025)
p_cutoffs = np.arange(0.1, 0.41, 0.025)


m_file = 'models/NetworkModels/ACES/meta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
nm_file = 'models/NetworkModels/ACES/nmeta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
Mp = pd.read_csv(m_file, index_col = 0)
Mn = pd.read_csv(nm_file, index_col = 0)

r2 = pd.read_csv('dataset/R2_scores/cv_r2_score_lambda_{0}.txt'.format(LAMBDA_VAL), header=None)
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

n_tf_meta = meta_coreg.shape[0]
n_tf_nmeta = nmeta_coreg.shape[0]
stats = []

percentage_nm, percentage_m, ci_1s, ci_2s = [], [], [], []
total_pairs_nm, total_pairs_m = [], []
for t in p_cutoffs:
    a = np.sum(nmeta_coreg.values >= t)
    b = np.sum(meta_coreg.values >= t)
    c = np.sum(nmeta_coreg.values <= -t)
    d = np.sum(meta_coreg.values <= -t)
    total_pairs_nm.append(a+c)
    total_pairs_m.append(b+d)
    stats.append([t, a, b, c, d])
    n1, n2 = (c+a), (d+b)
    p1, p2 = c / n1, d / n2
    ci_1s.append(np.sqrt(p1*(1-p1)/n1)*100)
    ci_2s.append(np.sqrt(p2*(1-p2)/n2)*100)
    percentage_nm.append(p1*100)
    percentage_m.append(p2*100)
  
columns = ['cc', 
           'Non-metastatic (# +ve)', 'Metastatic (# +ve)',
           'Non-metastatic (# -ve)', 'Metastatic (# -ve)']
stats = pd.DataFrame(stats, columns = columns).set_index('cc')
fig, axes = plt.subplots(figsize = (6, 6), sharex = True, nrows = 2,
                         gridspec_kw = {"height_ratios": [1, 1], "hspace": 0.05})
ax = axes[0]

stats.plot(ax = ax, style = ['s-', 'o-', '^--', 'x--'])

ax.set_yscale('log')
#ax.set_xticks([])
ax.set_ylabel('# of correlated\n TF pairs')
ax.legend(bbox_to_anchor = [1, 1], loc = 'upper left')

# For manual inspection
stats['Non-metastatic (% -ve)'] = percentage_nm
stats['Metastatic (% -ve)'] = percentage_m
stats['Non-metastatic (total)'] = stats['Non-metastatic (# +ve)'] + stats['Non-metastatic (# -ve)']
stats['Metastatic (total)'] = stats['Metastatic (# +ve)'] + stats['Metastatic (# -ve)']
stats = stats.T
stats['Network'] = stats.index.str.split(' ').str[0]
stats['Edge property'] = stats.index.str.split(' (', regex = False).str[1].str[:-1]
stats = stats.set_index(['Network', 'Edge property'], drop = True).sort_index()
stats.columns.name = 'cutoff'

ax = axes[1]
plt.errorbar(p_cutoffs, percentage_nm, yerr = ci_1s, marker = 'o', label = 'Non-metastatic')
plt.errorbar(p_cutoffs, percentage_m, yerr = ci_2s, marker = 'x', label = 'Metastatic')
plt.legend(bbox_to_anchor = [1, 1], loc = 'upper left')
plt.ylabel('% of negatively\n correlated TF pairs')
ax.set_xlabel('Absolute correlation cut-offs')
savefig('negative-edge-stats')
plt.show()

#%% Calculate all metrics

score_threshold_ratio = 0.5
score_files = [
    'ACESLeaveOneStudyOut_scores_lambda_0.06_r2_0.1_alpha_0.02',
    'ACESStratifiedKFold_scores_lambda_0.06_r2_0.1_alpha_0.02',
    'METABRICStratifiedKFold_scores_lambda_0.06_alpha_0.0',
    'METABRICValidation_scores_lambda_0.06_r2_0.1_alpha_0.02',
    'NKIStratifiedKFold_scores_lambda_0.06_alpha_0.0',
    'NKIValidation_scores_lambda_0.06_r2_0.1_alpha_0.02',
    'ACESBaselinesWithGeneFiltering_scores',
    'ACESStratifiedKFoldwithExternal_scores_lambda_0.06_r2_0.1_alpha_0.02'
]

scores = {}
metrics = {}
metrics_per_fold = {}
for filename in score_files:
    experiment_name = filename.split('_')[0]
    score_df = pd.read_csv(f'{SCORES_DIRECTORY}/{filename}.csv')
    scores[experiment_name] = score_df
    metrics[experiment_name] = scores_to_metrics(score_df, score_threshold_ratio)
    metrics_per_fold[experiment_name] = scores_to_metrics_per_fold(score_df, score_threshold_ratio)
    
metrics = pd.concat(metrics)
metrics.index.names = ['Experiment', 'Model']

#%% ACES 10 fold and LOSO accuracy plot

fig, axes = plt.subplots(1, 2, figsize = (10, 3))
ax = axes[0]
df = metrics.copy().T
df.index = np.hstack([df.index[:-1], [df.index[-1].replace('-', '\n')]])
experiment_name = 'ACESStratifiedKFold'
df[experiment_name][classifier_order].plot.bar(legend = False, rot = 45, width = .7, ax = ax)
ax.set_title('(a) ACES stratified 10-fold CV')
print(ax.get_title(), '\n', df[experiment_name][classifier_order].T)

ax = axes[1]
experiment_name = 'ACESLeaveOneStudyOut'
df[experiment_name][classifier_order].plot.bar( rot = 45, ax = ax, width = .7)
ax.set_title('(b) ACES leave-one-study-out CV')
print(ax.get_title(), '\n', df[experiment_name][classifier_order].T)
ax.legend(loc = 'upper left', bbox_to_anchor = [1, 1], frameon = False)
# ax.set_xlabel('Performance metric')
plt.subplots_adjust(wspace = 0.2)
savefig('combined-performance-r2-cutoff')
plt.show()

#%% NKI 10 fold and LOSO accuracy plot

fig, axes = plt.subplots(1, 2, figsize = (10, 3))
ax = axes[0]
df = metrics.copy().T
df.index = np.hstack([df.index[:-1], [df.index[-1].replace('-', '\n')]])
df['NKIStratifiedKFold'][classifier_order].plot.bar(legend = False, rot = 45, width = .7, ax = ax)
ax.set_title('(a) NKI stratified 10-fold CV')

ax = axes[1]
df['NKIValidation'][classifier_order].plot.bar( rot = 45, ax = ax, width = .7)
ax.set_title('(b) NKI validation')
ax.legend(loc = 'upper left', bbox_to_anchor = [1, 1], frameon = False)
# ax.set_xlabel('Performance metric')
plt.subplots_adjust(wspace = 0.2)
savefig('nki-performance-r2-cutoff')
plt.show()

#%% METABRIC 10 fold and LOSO accuracy plot

fig, axes = plt.subplots(1, 2, figsize = (10, 3))
ax = axes[0]
df = metrics.copy().T
df.index = np.hstack([df.index[:-1], [df.index[-1].replace('-', '\n')]])
experiment_name = 'METABRICStratifiedKFold'
df[experiment_name][classifier_order].plot.bar(legend = False, rot = 45, width = .7, ax = ax)
ax.set_title('(a) METABRIC stratified 10-fold CV')
print(ax.get_title(), '\n', df[experiment_name][classifier_order].T)

ax = axes[1]
experiment_name = 'METABRICValidation'
df[experiment_name][classifier_order].plot.bar( rot = 45, ax = ax, width = .7)
ax.set_title('(b) METABRIC validation')
print(ax.get_title(), '\n', df[experiment_name][classifier_order].T)
ax.legend(loc = 'upper left', bbox_to_anchor = [1, 1], frameon = False)
# ax.set_xlabel('Performance metric')
plt.subplots_adjust(wspace = 0.2)
savefig('metabric-performance-r2-cutoff')
plt.show()

#%% NKI+METABRIC plot

experiment_names = [
    'NKIStratifiedKFold',
    'METABRICStratifiedKFold',
    'NKIValidation',
    'METABRICValidation'
]
titles = [
    '(a) NKI stratified 10-fold CV',
    '(b) METABRIC stratified 10-fold CV',
    '(c) NKI validation',
    '(d) METABRIC validation'
]

fig, axes = plt.subplots(1, 4, figsize = (14, 3), sharey = True)
for (ax, experiment_name, plt_title) in zip(axes, experiment_names, titles):
    df = metrics.T[experiment_name][classifier_order]
    df.index = list(df.index[:-1]) + ['bal-acc.']
    df.plot.bar(rot = 40, width = 0.7, ax = ax)
    ax.set_title(plt_title)
    ax.set_xticklabels(ax.get_xticklabels(), ha = 'right')
    ax.legend(loc = 'upper left', bbox_to_anchor = [1, 1], frameon = False)
    if ax != axes[-1]:
        ax.get_legend().remove()
plt.subplots_adjust(wspace = 0.1)
savefig('nki-metabric-performance-r2-cutoff')
plt.show()
        

#%% Significance of performance difference between DRS and Baselines

# p-values calculated from pairwise t-tests on fold-by-fold performance metrics

experiment_names = [
    'ACESLeaveOneStudyOut',
    'ACESStratifiedKFold',
    'NKIStratifiedKFold',
    'METABRICStratifiedKFold'
]

pvals = {}
stat = {}
diff = {}
for experiment_name in experiment_names:
    df = metrics_per_fold[experiment_name].reorder_levels([1, 0])
    df = df.sort_index().loc[classifier_order, :]
    pvals[experiment_name] = pd.DataFrame(1.0,
                                          index = classifier_order[1:],
                                          columns = df.columns)
    stat[experiment_name] = pd.DataFrame(0.0,
                                         index = pvals[experiment_name].index,
                                         columns = pvals[experiment_name].columns)
    diff[experiment_name] = pd.DataFrame(0.0,
                                         index = pvals[experiment_name].index,
                                         columns = pvals[experiment_name].columns)
    for baseline_name in classifier_order[1:]:
        for metric_name in df.columns:
            stat_, pval_ = ttest_rel(df.loc['DRS', metric_name],
                                     df.loc[baseline_name, metric_name])
            diff_ = (df.loc['DRS', metric_name] - df.loc[baseline_name, metric_name]).mean()
            diff[experiment_name].loc[baseline_name, metric_name] = diff_
            pvals[experiment_name].loc[baseline_name, metric_name] = pval_
            stat[experiment_name].loc[baseline_name, metric_name] = stat_
pvals = pd.concat(pvals)
stat = pd.concat(stat)
diff = pd.concat(diff)
is_significant = pd.DataFrame(0, index = pvals.index, columns = pvals.columns)
is_significant[(pvals <= 0.05) & (stat > 0)] = 1
is_significant[(pvals <= 0.05) & (stat < 0)] = -1

# Export table

table = diff.map('{:.2f}'.format).map(lambda val: val.replace('-', '$-$'))
table = (pvals <= 0.05).replace({True: '\\bf ', False: ''}) + table
table.columns = list(table.columns[:-1]) + ['bal-acc.']
# table.to_latex('pvalue_comp.txt', multirow = False)

#%% DRS Parameter Tuning

filenames = ['ACESGridSearch10FCV_scores', 'ACESGridSearchLOSOCV_scores']

gs_metrics = {}
for filename in filenames:
    score_df = pd.read_csv(f'{SCORES_DIRECTORY}/{filename}.csv')
    # calculate performance metrics
    df = scores_to_metrics(score_df)
    df['r2 cutoff'] = [val.split('r2=')[1].split(')')[0] for val in df.index]
    df['alpha cutoff'] = [val.split('alpha=')[1].split(',')[0] for val in df.index]
    df['lambda'] = [val.split('lambda=')[1].split(',')[0] for val in df.index]
    df['genes'] = [val.split('n_genes=')[1].split(')')[0] for val in df.index]
    df = df.set_index(['r2 cutoff', 'alpha cutoff', 'lambda', 'genes'])
    df = df.reorder_levels([2, 0, 1, 3]).sort_index()
    gs_metrics[filename.split('_')[0]] = df
gs_metrics = pd.concat(gs_metrics)
gs_metrics.index.names = ['Experiment'] + list(gs_metrics.index.names[1:])

# Latex table
params_r2 = ['None', 0.0, 0.2]
params_alpha = [0.0, 0.02, 0.04, 0.08]
table = gs_metrics.loc[(slice(None), slice(None), 
                     [str(val) for val in params_r2], 
                     [str(val) for val in params_alpha]),
                    'AUC']
table = table.unstack(level = 0).round(3)
table.columns = [val.split('CV')[0].split('Search')[1] for val in table.columns]
table = table.reorder_levels([1, 0, 2, 3]).sort_index()
table['10F'] = table['10F'].map('{:.3f}'.format)
table['LOSO'] = table['LOSO'].map('{:.3f}'.format)
table = table.astype(str)
table = table.reset_index()
table['r2 cutoff'] = table['r2 cutoff'].replace({'0.0': '0'})
table['lambda'] = table['lambda'].replace({'0.0': '0', 'All': r'All$^*$'})
table['genes'] = table['genes'].astype(int).map('{:,}'.format)
table.columns = [r'$R^2$ cutoff', r'$\lambda$', r'$\alpha$', '\# genes'] + list(table.columns[-2:])
table = table.T
table.index.name = 'level_1'
table['level_0'] = ['Hyperparameter']*3 + ['# genes'] + ['AUC']*2
table = table.set_index(['level_0'], append = True)
table = table.reorder_levels([1, 0])
table.index.names = ['', '']
table = table.T
#%% ACES Baseline models with gene filtering

df = metrics.copy().T
df.index = np.hstack([df.index[:-1], [df.index[-1].replace('-', '\n')]])
experiment_name = 'ACESBaselinesWithGeneFiltering'
df = df[experiment_name].T
df['Base model'] = df.index.str.split(' ').str[0]
df['Genes'] = df.index.str.split('(').str.len().values-1
df['Genes'] = df['Genes'].replace({0: 'All', 1: 'Filtered'})
df = df.set_index(['Base model', 'Genes'], drop = True).sort_index().round(3)
df.columns.name = 'Metric'
print(df)

#%% External GRNS: number of regulatory TFs

N_FOLD = 10
external_method = 'BayesianRidgeScore'

fig, axes = plt.subplots(nrows = N_FOLD, ncols = 2, figsize = (5, 10),
                         sharex = True, sharey = True)
dirname = f'{MODEL_DIRECTORY}/ACESStratifiedKFold/with_{external_method}'
for fold in range(N_FOLD):
    for j, label_name in enumerate(['meta', 'nmeta']):
        ax = axes[fold, j]
        model = scipy.sparse.load_npz(f'{dirname}/fold_{fold}/{external_method}_meta.npz')
        model = np.array(model.todense())
        #ax.hist((np.abs(model) > 0).sum(axis = 0), color = f'C{j}')
        ax.hist(model.flat, color = f'C{j}')
        ax.set_yscale('log')
        if j == 0:
            ax.set_ylabel(f'fold {fold}')
        if fold == N_FOLD - 1:
            ax.set_xlabel(label_name)
plt.show()

#%% External GRNs: ACES 10 fold performance

df = metrics.loc['ACESStratifiedKFoldwithExternal'].T
df['DRS'] = metrics.loc[('ACESStratifiedKFold', 'DRS'), :]
df = df.T.sort_index()