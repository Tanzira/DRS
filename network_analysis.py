#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 18:10:46 2023

@author: tanzira
"""

#%% All imports
import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy.io import loadmat
from sklearn.utils import resample
import os
# import matplotlib.pyplot as plt
import pickle
# from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
#%% Network creation for metastatic and not metastatic patients
#getting input for metastatic or non metastatic coefficient calculation.
meta_val = 0 #1 means metastatic 0 means non metastatic
data_raw  = loadmat('Dataset/ACES_Data/ACESExpr.mat')['data']
p_type = loadmat('Dataset/ACES_Data/ACESLabel.mat')['label']
entrez_id = loadmat('Dataset/ACES_Data/ACES_EntrezIds.mat')['entrez_ids']
expr_data = pd.DataFrame(data_raw)
expr_data.columns = entrez_id.reshape(-1)

#Getting only patients with metastatic/non metastatic cancer for coefficient calculation.
patients = expr_data.loc[p_type == meta_val, :]
#Reading TF file and getting common TF bettween gene expression and TF file
tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[(human_tfs['Is TF?'] =='Yes') & (human_tfs['EntrezGene ID'] != 'None')]
human_tfs.set_index('EntrezGene ID', inplace = True)
human_tfs.index = human_tfs.index.astype(int)

common_tf = np.intersect1d(expr_data.columns, human_tfs.index)

#Omitting here for bootstrapping
# tf_df = patients.loc[:, common_tf]

'''Lasso regression'''
def do_lasso(sampled_data, tf_df, lambda_val):
    coefficients = []
    for gene in sampled_data.columns:
        y = sampled_data.loc[:, gene].values.copy()
        tf_exp = tf_df.copy()
        if gene in tf_df.columns:
            tf_exp.loc[:, gene] = 0
        x = tf_exp.to_numpy()
        reg = linear_model.Lasso(alpha=lambda_val, max_iter=10000, random_state = 0)#Lasso regression model
        reg.fit(x, y)  
        coefficients.append(reg.coef_)
    return coefficients

#Doing lasso for 200 times with bbotstrapping
niter = 200
#creating model for lambda 0.03
LAMBDA_VAL = 0.06
models = []
for n in range(niter):
    sampled_data = patients.loc[resample(patients.index, n_samples=1161), :]
    tf_df = sampled_data.loc[:, common_tf]
    coef = do_lasso(sampled_data, tf_df, LAMBDA_VAL)
    print("Unique columns: ", n ,sampled_data.index.nunique(), sampled_data.shape)
    models.append(coef)
##Combining the models from different alpha values
models = np.swapaxes(np.stack(models), 1, 2)
dir_name = 'Models/NetworkModels'
try:
    os.makedirs(dir_name, exist_ok=True)
except OSError:
    print("Error occured")
if meta_val == 1:
    f2 = dir_name + '/meta_net_with_{}_bootstrap_mean.pkl'.format(niter)
if meta_val == 0:
    f2 = dir_name + '/nmeta_net_with_{}_bootstrap_mean.pkl'.format(niter)
mean_model = np.mean(models, axis = 0).T
#uncomment if you want to write
# out2 = open(f2, 'wb')
# pickle.dump(mean_model, out2)
# out2.close()

#%% TFTG coregulatory network analysis and statistics
#parameters
LAMBDA_VAL = 0.06
ALPHA_CUTOFF = 0.02
r2_threshold = 0.1
m_file = 'Models/NetworkModels/meta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
nm_file = 'Models/NetworkModels/nmeta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
lasso_meta = pd.read_csv(m_file, index_col = 0)
lasso_nmeta = pd.read_csv(nm_file, index_col = 0)

r2 = pd.read_csv('R2CVScores/cv_r2_score_lambda_{0}.txt'.format(LAMBDA_VAL), header=None)

goodGenes = (r2.values >= r2_threshold)
filtered_meta = lasso_meta.loc[goodGenes, :] # filtering genes with better r2
filtered_nmeta = lasso_nmeta.loc[goodGenes, :] # filtering genes with better r2
print(filtered_meta.shape, filtered_nmeta.shape)

#Keeping only the connections that are greater than coefficient/alpha threshold
filtered_meta[np.abs(filtered_meta) < ALPHA_CUTOFF] = 0
filtered_nmeta[np.abs(filtered_nmeta) < ALPHA_CUTOFF] = 0

target_tf_meta = pd.DataFrame(filtered_meta)
target_tf_nmeta = pd.DataFrame(filtered_nmeta)
cols = list(map(int, target_tf_meta.columns))

target_tf_meta.columns = cols
target_tf_nmeta.columns = cols

all_targets = np.union1d(target_tf_meta.index, target_tf_meta.columns)
adj_mat_meta = target_tf_meta.reindex(all_targets, columns = all_targets, fill_value = 0).T
adj_mat_nmeta = target_tf_nmeta.reindex(all_targets, columns = all_targets, fill_value = 0).T

pos_edge_m, neg_edge_m = (adj_mat_meta > 0).sum().sum(), (adj_mat_meta < 0).sum().sum()
pos_edge_nm, neg_edge_nm = (adj_mat_nmeta > 0).sum().sum(), (adj_mat_nmeta < 0).sum().sum()

GM = nx.convert_matrix.from_pandas_adjacency(adj_mat_meta, create_using=nx.DiGraph())
GNM = nx.convert_matrix.from_pandas_adjacency(adj_mat_nmeta, create_using=nx.DiGraph())

GM.edges(data = True)
GNM.edges(data = True)

print(GM)
print(GNM)

n_edges_m, n_edges_nm = nx.number_of_edges(GM), nx.number_of_edges(GNM)
n_singletons_m, n_singletons_nm = len(list(nx.isolates(GM))), len(list(nx.isolates(GNM)))

def get_in_out_degree_dist(G1, G2, targets, tfs):
    in_degrees_m = [G1.in_degree(n) for n in targets]
    in_degrees_nm = [G2.in_degree(n) for n in targets]
    out_degrees_m = [G1.out_degree(n) for n in tfs]
    out_degrees_nm = [G2.out_degree(n) for n in tfs]
    
    ##Removing zeors from the list for the TFs. For targets we have at least 1 TF that's regulating it.
    out_degrees_m = [i for i in out_degrees_m if i != 0]
    out_degrees_nm = [i for i in out_degrees_nm if i != 0]
    return in_degrees_m,in_degrees_nm, out_degrees_m,out_degrees_nm
   

#Getting Tfs and targets
targets = target_tf_meta.index
tfs = target_tf_meta.columns

in_meta, in_nmeta, out_meta, out_nmeta = get_in_out_degree_dist(GM, GNM, targets, tfs)

#getting number of TFs and targets with at least 1 target or TF in it.
n_tf_meta, n_tf_nmeta = len(out_meta), len(out_nmeta)
n_tg_meta, n_tg_nmeta = len(in_meta), len(in_nmeta)

#Getting minimum or maximum in and out degree
max_out_m, max_out_nm = max(out_meta), max(out_nmeta)
max_in_m, max_in_nm = max(in_meta), max(in_nmeta)

#Getting average and median in and out degrees.
avg_in_meta, avg_in_nmeta = np.mean(in_meta), np.mean(in_nmeta)
avg_out_meta, avg_out_nmeta = np.mean(out_meta), np.mean(out_nmeta)
med_in_meta, med_in_nmeta = np.median(in_meta), np.median(in_nmeta)
med_out_meta, med_out_nmeta = np.median(out_meta), np.median(out_nmeta)

# #removing the singletons
GM.remove_nodes_from(list(nx.isolates(GM)))
GNM.remove_nodes_from(list(nx.isolates(GNM)))

##calculating nodes after removing the sinfletons
nodes_m, nodes_nm = GM.number_of_nodes(), GNM.number_of_nodes()
#average degree
avg_deg_m, avg_deg_nm = n_edges_m /nodes_m ,  n_edges_nm / nodes_nm

#largest degree
l_deg_m = sorted(GM.degree, key=lambda x: x[1], reverse=True)[0][1]
l_deg_nm = sorted(GNM.degree, key=lambda x: x[1], reverse=True)[0][1]
#Clustering coefficient
avg_cc_m,  avg_cc_nm= nx.average_clustering(GM), nx.average_clustering(GNM)

#getting average shortest path
# avg_spl_m, avg_spl_nm = nx.average_shortest_path_length(GM), nx.average_shortest_path_length(GNM)

#Converting the graph into undriected graph because for directed graph some functions are unavailable.
HGM, HGNM = GM.to_undirected(), GNM.to_undirected()

#getting largest connected components
gcc_m = sorted(nx.connected_components(HGM), key=len, reverse=True)
lcc_m = len(HGM.subgraph(gcc_m[0]))
gcc_nm = sorted(nx.connected_components(HGNM), key=len, reverse=True)
lcc_nm = len(HGNM.subgraph(gcc_nm[0]))


#getting average shortest path for largest connected component
avg_spl_m_ud, avg_spl_nm_ud = nx.average_shortest_path_length(HGM.subgraph(gcc_m[0])),\
                                nx.average_shortest_path_length(HGNM.subgraph(gcc_nm[0]))

#getting diameter for largest connected component
diameter_m, diameter_nm = nx.diameter(HGM.subgraph(gcc_m[0])),\
                                nx.diameter(HGNM.subgraph(gcc_nm[0]))

#Clustering coefficient
avg_cc_m_ud,  avg_cc_nm_ud= nx.average_clustering(HGM), nx.average_clustering(HGNM)

#average degree
nodes_m_ud, nodes_nm_ud = HGM.number_of_nodes(), HGNM.number_of_nodes()
avg_deg_m_ud, avg_deg_nm_ud = 2*HGM.number_of_edges() /nodes_m_ud ,\
                                2*HGNM.number_of_edges() / nodes_nm_ud

print('Network stats \t\t\t meta \t non-meta')

print('# of nodes \t\t\t\t{} \t{}'.format(nodes_m, nodes_nm))
print('# of edges \t\t\t\t{} \t{}'.format(n_edges_m, n_edges_nm))


print('# of TFs \t\t\t\t{} \t\t{}'.format(n_tf_meta, n_tf_nmeta))
print('# of Targets \t\t\t{} \t{}'.format(n_tg_meta, n_tg_nmeta))



print('+ve edge \t\t\t\t{} \t {}'.format(pos_edge_m, pos_edge_nm))
print('-ve edge \t\t\t\t{} \t {}'.format(neg_edge_m, neg_edge_nm))


print('max out degree(TF) \t\t{} \t\t {}'.format(max_out_m, max_out_nm))
print('max in degree(TG) \t\t{} \t\t {}'.format(max_in_m, max_in_nm))
print('mean out degree(TF) \t\t{:.2f} \t {:.2f}'.format(avg_out_meta, avg_out_nmeta))
print('mean in degree(TG)\t\t{:.2f} \t {:.2f}'.format(avg_in_meta, avg_in_nmeta ))
print('median out degree(TF) \t{:.2f} \t {:.2f}'.format(med_out_meta, med_out_nmeta))
print('median in degree(TG)\t\t{:.2f} \t {:.2f}'.format(med_in_meta, med_in_nmeta))

print('singletons \t\t\t\t{} \t\t {}'.format(n_singletons_m, n_singletons_nm))
print('mean degree \t\t\t\t{:.2f} \t {:.2f}'.format(avg_deg_m, avg_deg_nm))

print('largest degree \t\t\t{} \t\t {}'.format(l_deg_m, l_deg_nm))

print('Average CC \t\t\t\t{:.2f} \t {:.2f}'.format(avg_cc_m, avg_cc_nm))
# print('Average Shortest path \t{:.2f} \t {:.2f}'.format(avg_spl_m, avg_spl_nm))

print('mean degree UD \t\t\t{:.2f} \t {:.2f}'.format(avg_deg_m_ud, avg_deg_nm_ud))
print('Average CC UD \t\t\t{:.2f} \t {:.2f}'.format(avg_cc_m_ud,  avg_cc_nm_ud))

print('Largest Conne Comp UD \t\t{} \t {}'.format(lcc_m, lcc_nm))

print('Largest connected comp avg shortest path UD \t\t{:.2f} \t {:.2f}'.format(avg_spl_m_ud, avg_spl_nm_ud))
print('Largest connected comp diameter UD \t\t\t\t{} \t\t {}'.format(diameter_m, diameter_nm))

#%% Creating the coreg-antireg network using the mean coefficient from 200 bootstrap
LAMBDA_VAL = 0.06
ALPHA_CUTOFF = 0.02
r2_threshold = 0.1
m_file = 'Models/NetworkModels/meta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
nm_file = 'Models/NetworkModels/nmeta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
lasso_meta = pd.read_csv(m_file, index_col = 0)
lasso_nmeta = pd.read_csv(nm_file, index_col = 0)

r2 = pd.read_csv('R2CVScores/cv_r2_score_lambda_{0}.txt'.format(LAMBDA_VAL), header=None)

goodGenes = (r2.values >= r2_threshold)
filtered_meta = lasso_meta.loc[goodGenes, :] # filtering genes with better r2
filtered_nmeta = lasso_nmeta.loc[goodGenes, :] # filtering genes with better r2

print(filtered_meta.shape, filtered_meta.shape)

filtered_meta[filtered_meta >= ALPHA_CUTOFF] = 1
filtered_meta[filtered_meta < -ALPHA_CUTOFF] = -1
filtered_meta[np.abs(filtered_meta) < 1] = 0
#creating a mask for antiregulatory and coregulatory network
mask_p_meta = filtered_meta == 1
mask_n_meta = filtered_meta == -1

meta_plus = filtered_meta.copy()
meta_minus =  filtered_meta.copy()

#making all the positive relationships to 1 and the others as 0
meta_plus[mask_p_meta] = 1
meta_plus[~mask_p_meta] = 0

#making all the negative relationships to 1 and the others as 0
meta_minus[mask_n_meta] = 1
meta_minus[~mask_n_meta] = 0

filtered_nmeta[filtered_nmeta >= ALPHA_CUTOFF] = 1
filtered_nmeta[filtered_nmeta < -ALPHA_CUTOFF] = -1
filtered_nmeta[np.abs(filtered_nmeta) < 1] = 0
#creating a mask for antiregulatory and coregulatory network
mask_p_nmeta = filtered_nmeta == 1
mask_n_nmeta = filtered_nmeta == -1

nmeta_plus = filtered_nmeta.copy()
nmeta_minus =  filtered_nmeta.copy()

#making all the positive relationships to 1 and the others as 0
nmeta_plus[mask_p_nmeta] = 1
nmeta_plus[~mask_p_nmeta] = 0
#making all the negative relationships to 1 and the others as 0
nmeta_minus[mask_n_nmeta] = 1
nmeta_minus[~mask_n_nmeta] = 0

#creating coregulatory and antiregulatory network for metastatic
coreg_meta = (meta_plus.T @ meta_plus) + (meta_minus.T @ meta_minus)
antireg_meta = (meta_plus.T @ meta_minus) + (meta_minus.T @ meta_plus)
np.fill_diagonal(coreg_meta.values, 0) # Filling the diagonals with 0
np.fill_diagonal(antireg_meta.values, 0) # Filling the diagonals with 0

#creating coregulatory and antiregulatory network for not metastatic
coreg_nmeta = (nmeta_plus.T @ nmeta_plus) + (nmeta_minus.T @ nmeta_minus)
antireg_nmeta = (nmeta_plus.T @ nmeta_minus)  + (nmeta_minus.T @ nmeta_plus)
np.fill_diagonal(coreg_nmeta.values, 0) # Filling the diagonals with 0
np.fill_diagonal(antireg_nmeta.values, 0) # Filling the diagonals with 0
'''Saving the TF TF coregulatory network'''
def saving_tf_tf_coreg_antireg_network(tf_tf_coreg, tf_tf_antireg, fname):
    #Different edge threshold for picking a sparser network
    edge_ths =[100]
    for edge_th in edge_ths:
        #pval_th = -np.log10(pval)
        tf_coreg = tf_tf_coreg.copy()
        tf_coreg = pd.DataFrame(tf_coreg >= edge_th, dtype = np.int32)
        if isinstance(tf_tf_antireg, pd.DataFrame):
            tf_antireg = tf_tf_antireg.copy()
            tf_antireg = pd.DataFrame(tf_antireg>= edge_th, dtype = np.int32)*2
            tf_net = tf_coreg + tf_antireg
        else:
            tf_net = tf_coreg
        G = nx.from_numpy_matrix(tf_net.values, create_using=None)
        label_mapping = {idx: val for idx, val in enumerate(human_tfs.loc[tf_net.columns.astype(int), 'HGNC symbol'])}
        G = nx.relabel_nodes(G, label_mapping)
        G1 = G.copy()
        G.remove_nodes_from(list(nx.isolates(G)))
        sf = 'CoregNetRecomb_{0}/tf_tf_{1}_edge_th_{2}_coef_{3}_r2_{4}_1.gml'.\
                format(LAMBDA_VAL, fname, edge_th, ALPHA_CUTOFF, r2_threshold)
        #returning the whole network, network without singletons and the filename.
        return G, G1, sf
        
meta_net, meta_net2, sf1 = saving_tf_tf_coreg_antireg_network(coreg_meta, antireg_meta, 'meta')

nmeta_net, nmeta_net2, sf2 = saving_tf_tf_coreg_antireg_network(coreg_nmeta, antireg_nmeta, 'nmeta')

common_nodes = np.intersect1d(meta_net.nodes, nmeta_net.nodes)
total_nodes =  np.union1d(meta_net.nodes, nmeta_net.nodes)
# print(len(total_nodes), len(common_nodes))

'''Getting networks with same nodes(TFs) from union of two networks'''
meta_net = meta_net2.subgraph(total_nodes)
nmeta_net = nmeta_net2.subgraph(total_nodes)

print(meta_net, nmeta_net)

# nx.write_gml(meta_net, sf1)
# nx.write_gml(nmeta_net, sf2)

#%% TF-Target network creation and node difference calculation

#reading protein coding genes for HGNC symbol

pc_genes = pd.read_csv('Dataset/protein-coding_gene_04_26_2023.txt',
                       index_col = 0, sep = '\t', low_memory=False)
pc_genes = pc_genes[['symbol', 'name','entrez_id', 'ensembl_gene_id']]
pc_genes.set_index('entrez_id', inplace = True)

'''Creating TF-Target network'''

LAMBDA_VAL = 0.06
ALPHA_CUTOFF = 0.02
r2_threshold = 0.1
m_file = 'Models/NetworkModels/meta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
nm_file = 'Models/NetworkModels/nmeta_net_with_bootstrap_mean_{0}.csv'.format(LAMBDA_VAL)
lasso_meta = pd.read_csv(m_file, index_col = 0)
lasso_nmeta = pd.read_csv(nm_file, index_col = 0)

r2 = pd.read_csv('R2CVScores/cv_r2_score_lambda_{0}.txt'.format(LAMBDA_VAL), header=None)

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

top_tfs.to_csv('CoregNetRecomb_{0}/top_tfs_based_on_diff_bootstrapped_{0}.csv'.format(LAMBDA_VAL))
top_targets.to_csv('CoregNetRecomb_{0}/top_targets_based_on_diff_bootstrapped_{0}.csv'.format(LAMBDA_VAL))