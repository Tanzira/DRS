#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 18:10:46 2023

@author: tanzira, sakhawat
"""

#%% All imports
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.utils import resample
import os
import networkx as nx

from drs import DRS
from utils import get_network_attributes

DATASET_DIRECTORY = './dataset'
#%% Load Datasets

'''AECS data'''
aces_dirname = f'{DATASET_DIRECTORY}/ACES'
aces_raw  = loadmat(f'{aces_dirname}/ACESExpr.mat')['data']
aces_p_type = loadmat(f'{aces_dirname}/ACESLabel.mat')['label']
aces_entrez_id = loadmat(f'{aces_dirname}/ACES_EntrezIds.mat')['entrez_ids']
aces_data = pd.DataFrame(aces_raw)
aces_data.columns = aces_entrez_id.reshape(-1)
#reading protein coding genes for HGNC symbol

pc_genes = pd.read_csv(f'{DATASET_DIRECTORY}/protein-coding_gene_04_26_2023.txt',
                       index_col = 0, sep = '\t', low_memory=False)
pc_genes = pc_genes[['symbol', 'name','entrez_id', 'ensembl_gene_id']]
pc_genes.set_index('entrez_id', inplace = True)

'''Reading NKI data'''
nki_dirname = f'{DATASET_DIRECTORY}/NKI'
nki_raw = loadmat(f'{nki_dirname}/vijver.mat')['vijver']
nki_p_type = loadmat(f'{nki_dirname}/VijverLabel.mat')['label']
nki_entrez_id = loadmat(f'{nki_dirname}/vijver_gene_list.mat')['vijver_gene_list']
nki_data = pd.DataFrame(nki_raw)
nki_data.columns = nki_entrez_id.reshape(-1)


#Reading TF file and getting common TF bettween gene expression and TF file
tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[(human_tfs['Is TF?'] =='Yes') & (human_tfs['EntrezGene ID'] != 'None')]
human_tfs.set_index('EntrezGene ID', inplace = True)
human_tfs = human_tfs.loc[~np.isnan(human_tfs.index.astype(float)), :]
human_tfs.index = human_tfs.index.astype(int)

common_tf = np.intersect1d(aces_data.columns, human_tfs.index)

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

r2_file = f'{DATASET_DIRECTORY}/R2_scores/cv_r2_score_lambda'

datasets = {'ACES': (aces_data, aces_p_type),
            'NKI': (nki_data, nki_p_type),
            'METABRIC': (metabric_data, metabric_p_type)
            }
dirname = 'models/NetworkModels'
    
#%% Helper functions

def train_bootstrap_mean_models(X, y, common_tf, niter, lambda_val, dirname):
    mean_models = {}
    for i in range(niter):
        print(f'Iteration {i}')
        X_pos = X.loc[y == 1, :]
        X_neg = X.loc[y == 0, :]
        X_pos_sampled = X_pos.loc[resample(X_pos.index, n_samples = len(X_pos))]
        X_neg_sampled = X_neg.loc[resample(X_neg.index, n_samples = len(X_neg))]
        X_sampled = np.vstack([X_pos_sampled, X_neg_sampled])
        y_sampled = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        tf_locs = [X.columns.get_loc(c) for c in common_tf]
        drs_obj = DRS.train(X_sampled, y_sampled, tf_locs, [LAMBDA_VAL])
        if len(mean_models) == 0:
            mean_models = {k: v for k, v in drs_obj.models.items()}
        else:
            for k, v in drs_obj.models.items():
                mean_models[k] += v

    # Export
    try:
        os.makedirs(dirname, exist_ok=True)
        for label_name in mean_models.keys():
            A_mean = mean_models[label_name] / niter
            assert A_mean.shape[0] == 1
            A_mean = A_mean[0].T
            #scipy.sparse.save_npz(f'{dirname}/{label_name}_net_with_bootstrap_mean_{LAMBDA_VAL}.npz',
            #                      scipy.sparse.csr_matrix(A_mean))
            df_A = pd.DataFrame(A_mean, index = X.columns, columns = common_tf)
            df_A.to_csv(f'{dirname}/{label_name}_net_with_bootstrap_mean_{LAMBDA_VAL}.csv')
    except OSError:
        print("Error occured")

#%% Create bootstrap-mean models from ACES data

dirname = 'models/NetworkModels/ACES'
niter = 200 # Number of bootstrap iterations
LAMBDA_VAL = 0.06

X, y = datasets[dirname.split('/')[-1]]
X = X.copy()
y = y.ravel()
common_tf = np.intersect1d(X.columns, human_tfs.index)

train_bootstrap_mean_models(X, y, common_tf, niter, LAMBDA_VAL, dirname)

#%% TF-Target coregulatory network analysis and statistics

dirname = 'models/NetworkModels/ACES' # ACES | NKI | METABRIC
LAMBDA_VAL = 0.06
#ALPHA_CUTOFF = 0.02
ALPHA_CUTOFF = 0.00
r2_threshold = 0.1

target_dataset = datasets[dirname.split('/')[-1]][0]

# Load R2 values calculated from ACES dataset
r2 = pd.read_csv(f'{r2_file}_{LAMBDA_VAL}.txt', header=None)
# convert from df to series
r2 = r2[0]
r2.index = aces_data.columns
# any gene not in ACES will be considered to have R^2 = 1 (gene will be included in downstream analysis)
target_r2 = pd.Series(1, index = target_dataset.columns)
common_genes = np.intersect1d(aces_data.columns, target_dataset.columns)
target_r2.loc[common_genes] = r2.loc[common_genes]
goodGenes = (target_r2.values >= r2_threshold)

A = {}
attributes = {}
for label_name in ['meta', 'nmeta']:
    filename = f'{dirname}/{label_name}_net_with_bootstrap_mean_{LAMBDA_VAL}.csv'
    A_mean = pd.read_csv(filename, index_col = 0)
    A_mean = A_mean.loc[goodGenes, :]
    A_mean[np.abs(A_mean) < ALPHA_CUTOFF] = 0
    A_mean.columns = A_mean.columns.astype(int)
    A_mean.index = A_mean.index.astype(int)
    print(A_mean.shape)
    targets = A_mean.index
    tfs = A_mean.columns

    all_targets = np.union1d(A_mean.index, A_mean.columns.astype(int))
    A_mean = A_mean.reindex(all_targets, columns = all_targets, fill_value = 0).T
    A[label_name] = A_mean
    print('Summarizing network attributes for', label_name)
    attributes[label_name] = pd.Series(get_network_attributes(A_mean, tfs, targets))

attributes = pd.concat(attributes).unstack().T.convert_dtypes()
print(attributes)

#%% TF-Target network creation and node difference calculation


'''Creating TF-Target network'''

LAMBDA_VAL = 0.06
ALPHA_CUTOFF = 0.02
r2_threshold = 0.1


m_file = dirname + '/meta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
nm_file = dirname + '/nmeta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
lasso_meta = pd.read_csv(m_file, index_col = 0)
lasso_nmeta = pd.read_csv(nm_file, index_col = 0)

r2 = pd.read_csv(f'{r2_file}_{LAMBDA_VAL}.txt', header=None)

goodGenes = (r2.values >= r2_threshold)
filtered_meta = lasso_meta.loc[goodGenes, :] # filtering genes with better r2
filtered_nmeta = lasso_nmeta.loc[goodGenes, :] # filtering genes with better r2


target_tf_meta = pd.DataFrame(np.abs(filtered_meta) >= ALPHA_CUTOFF, dtype = np.int32) #setting coef >= coef_th as 1 else 0
target_tf_nmeta = pd.DataFrame(np.abs(filtered_nmeta) >= ALPHA_CUTOFF, dtype = np.int32)#setting coef >= coef_th as 1 else 0

cols = list(map(int, target_tf_meta.columns))

target_tf_meta.columns = cols
target_tf_nmeta.columns = cols


all_targets = np.union1d(target_tf_meta.index, target_tf_meta.columns)

adj_mat_meta = target_tf_meta.reindex(all_targets, columns = all_targets, fill_value = 0).T
adj_mat_nmeta = target_tf_nmeta.reindex(all_targets, columns = all_targets, fill_value = 0).T


GM = nx.convert_matrix.from_pandas_adjacency(adj_mat_meta, create_using=None)
GNM = nx.convert_matrix.from_pandas_adjacency(adj_mat_nmeta, create_using=None)

network_attribute = pd.DataFrame(columns = ['nodes', 'degree_m'], data =  GM.degree())
network_attribute.set_index('nodes', inplace = True)
# df_attr_nm = pd.DataFrame(columns = ['nodes', 'degree'], data =  GNM.degree())

network_attribute['node_type'] = ['tf' if x in target_tf_meta.columns else 'target' for x in network_attribute.index]
network_attribute['HGNC'] = [pc_genes.loc[x, 'symbol'] if x in pc_genes.index else 'N/A' for x in network_attribute.index]

'''
Calculating the Node Difference usinf Li Lu's formula'
'''
all_nodes = sorted(GM.nodes)
for p in all_nodes:
    dist_sum = 0
    for q in all_nodes:
        pij = GM.number_of_edges(p, q)
        qij = GNM.number_of_edges(p, q)
        dist_sum += np.abs(pij - qij)
    Di = np.sqrt(dist_sum)
    network_attribute.loc[p, 'node_diff'] = round(Di, 3)

for p in all_nodes:
    network_attribute.loc[p, 'degree_nm'] = GNM.degree[p]
print(network_attribute)


top_tfs = network_attribute[network_attribute['node_type'] == 'tf'].sort_values('node_diff', ascending = False)
top_targets = network_attribute[network_attribute['node_type'] == 'target'].sort_values('node_diff', ascending = False)

#Uncomment these lines if you want to save these two files

top_tfs.to_csv(dirname + '/top_tfs_based_on_diff_bootstrapped_{0}.csv'.format(LAMBDA_VAL))
top_targets.to_csv(dirname + '/top_targets_based_on_diff_bootstrapped_{0}.csv'.format(LAMBDA_VAL))

#%% TF-TF network

r2_threshold = 0.1
ALPHA_CUTOFF = 0.02
delta = 100
LAMBDA_VAL = 0.06
write_gml = False

# Read data
matfile_p = dirname + "meta_net_with_bootstrap_mean_{0}.csv".format(LAMBDA_VAL)
matfile_n = dirname + "nmeta_net_with_bootstrap_mean_{0}.csv".format(LAMBDA_VAL)

#r2file = basepath + "cv_r2_score_alpha_{0}.txt".format(lambda_)
#r2 = pd.read_csv(r2file, header = None).squeeze()
r2 = pd.read_csv(r2_file.format(LAMBDA_VAL), header=None)

M_met = pd.read_csv(matfile_p, sep = ",", index_col = 0)
M_non = pd.read_csv(matfile_n, sep = ",", index_col = 0)

assert r2.shape[0] == M_met.shape[0] and r2.shape[0] == M_non.shape[0]

tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
tfs = tfs[tfs['Is TF?'] =='Yes']
tfs.set_index('EntrezGene ID', inplace = True)
tfs = tfs.loc[~np.isnan(tfs.index.astype(float)), :]

tfs = tfs.loc[M_met.columns.astype(str), 'HGNC symbol']
tfs.index = tfs.index.astype(str)
M_met.columns = tfs.values
M_non.columns = tfs.values

print(tfs.shape)

# apply r2 cutoff
mask_r2 = (r2 >= r2_threshold).values
print('r2 >= {0}: {1}/{2} ({3:.2f}%)'.format(r2_threshold, mask_r2.sum(),
                                        mask_r2.shape[0], mask_r2.sum() * 100 / mask_r2.shape[0]))

for label_name, M_label in zip(['meta', 'nmeta'], [M_met, M_non]):
    
    M_label = M_label.loc[mask_r2, :].to_numpy()
    
    # apply alpha cutoff and change range to [-1, 1]
    M_label[(M_label < ALPHA_CUTOFF) & (M_label > -ALPHA_CUTOFF)] = 0
    M_label = pd.DataFrame(M_label, columns = tfs.values)
    
    # TF-TF network
    pos_label = np.zeros(shape = M_met.shape).astype(float)
    neg_label = np.zeros(shape = M_met.shape).astype(float)

    
    pos_label[M_met > 0] = 1
    neg_label[M_met < 0] = 1
    
    # Float matrix multiplication is faster than int
    Sc = (np.dot(pos_label.T, pos_label) + np.dot(neg_label.T, neg_label)).astype(int)
    Sa = (np.dot(pos_label.T, neg_label) + np.dot(neg_label.T, pos_label)).astype(int)

    
    np.fill_diagonal(Sc, 0)
    np.fill_diagonal(Sa, 0)
    
    # Apply delta cutoff
    Sc = pd.DataFrame((Sc >= delta).astype(int), index = tfs.values, columns = tfs.values)
    Sa = pd.DataFrame((Sa >= delta).astype(int), index = tfs.values, columns = tfs.values)

    # Aggregate
    S = Sc + 2*Sa
    
    S = pd.DataFrame(S, index = tfs, columns = tfs)
    
    SG = nx.from_pandas_adjacency(S)
    SG.remove_nodes_from(list(nx.isolates(SG)))
    
    graphname = dirname + '/TFnet-#--lambda_{0}-r2_{1}-a_{2}-D_{3}'.format(LAMBDA_VAL, r2_threshold, ALPHA_CUTOFF, delta)
    if write_gml:
        nx.write_gml(SG, graphname.replace('#', 'met') + '.gml')
