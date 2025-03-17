#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:39:50 2025

@author: tanzira, sakhawat
"""

import pandas as pd
import numpy as np
import networkx as nx

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def qtnorm_with_ref_columnwise(Xref, X):
    '''
    Column-by-column normalization of X using Xref as reference
    '''
    # both datasets should have same number of features
    assert Xref.shape[1] == X.shape[1]
    X_new = np.quantile(Xref, np.linspace(0, 1, X.shape[0]), axis = 0)
    rank = pd.DataFrame(X).rank(axis = 0, method = 'min').astype(int).values - 1
    X_new = np.take_along_axis(X_new, rank, axis = 0)
    return X_new


def get_filtered_expr(expr1, expr2, human_tfs):
    common_genes = np.intersect1d(expr1.columns, expr2.columns)
    common_tfs = np.intersect1d(common_genes, human_tfs.index)
    #filtering ACES and NKI data based on common number of genes both has.
    expr1_filtered = expr1.loc[:, common_genes]
    expr2_filtered = expr2.loc[:, common_genes]
    return expr1_filtered, expr2_filtered, common_tfs

def get_network_attributes(adj, tfs, targets):
    attributes = {}
    
    attributes['+ve edges'] = (adj > 0).sum().sum()
    attributes['-ve edges'] = (adj < 0).sum().sum()
    # G should be a networkx graph
    G = nx.convert_matrix.from_pandas_adjacency(adj, create_using=nx.DiGraph())
    
    
    m = nx.number_of_edges(G)

    # In and out-degrees
    deg_in = np.array([G.in_degree(n) for n in targets])
    deg_out = np.array([G.out_degree(n) for n in tfs])
    #deg_out = [i for i in deg_out if i != 0]
    deg_out = deg_out[deg_out > 0]
    
    
    attributes['# of TFs'] = len(deg_out)
    attributes['# of Targets'] = len(deg_in)
    
    # highest in and out-degrees
    attributes['max out degree (TF)'] = max(deg_out)
    attributes['max in degree (TG)'] = max(deg_in)
    
    #Getting average and median in and out degrees.
    attributes['avg in degree (TG)']  = np.mean(deg_in)
    attributes['avg out degree (TF)'] = np.mean(deg_out)
    attributes['median in degree (TG)'] = np.median(deg_in)
    attributes['median out degree (TF)'] = np.median(deg_out)
    
    # Remove singletons
    # TODO: make this non-destructive
    attributes['singletons'] = len(list(nx.isolates(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    
    n = G.number_of_nodes()
    attributes['# of nodes'] = n
    attributes['# of edges'] = m   
    
    attributes['avg degree'] = m / n
    
    # Largest degree   
    attributes['largest degree'] = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][1]
    
    # Clustering coefficient
    cc_avg = nx.average_clustering(G)
    attributes['average clustering coefficient'] = cc_avg
    
    # Convert to undirected for the following attributes
    # TODO: make this non-destructive
    Gu = G.to_undirected()

    gcc = Gu.subgraph(sorted(nx.connected_components(Gu), key = len, reverse = True)[0])

    #average degree
    nodes_ud = Gu.number_of_nodes()
    
    attributes['avg degree (u)'] = 2*Gu.number_of_edges() /nodes_ud
    attributes['avg clustering coef (u)'] = nx.average_clustering(Gu)
    attributes['largest conn. comp (u)'] = len(gcc)
    attributes['LCC avg shortest path (u)'] = nx.average_shortest_path_length(gcc)
    attributes['LCC diameter (u)'] = nx.diameter(gcc)
    
    return attributes