#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import scvi
import scanpy as sc
import time

from anndata import AnnData
from scipy import linalg
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
from scipy import stats, sparse

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, r2_score
from sklearn.preprocessing import label_binarize
from math import sqrt
from scipy.stats import gaussian_kde

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import argparse
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import QED
from tqdm import tqdm
sys.path.append('../')
from perturbnet.perturb.util import * 
from perturbnet.perturb.cinn.module.flow import * 
from perturbnet.perturb.genotypevae.genotypeVAE import *
from perturbnet.perturb.data_vae.modules.vae import *
from perturbnet.perturb.cinn.module.flow_generate import SCVIZ_CheckNet2Net
from matplotlib.colors import ListedColormap




def Seq_to_Embed_ESM(ordered_trt, batch_size, model, alphabet, save_path = None):
    data = []
    count = 1
    batch_converter = alphabet.get_batch_converter()
    for i in ordered_trt:
        data.append((count,i))
        count += 1

    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    for j in tqdm(range(len(batches))):
        batch = batches[j]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len + 1].mean(0).numpy().reshape(1, -1))
    if save_path:
        np.save(save_path,sequence_representations)
    
    return(sequence_representations)


def create_train_test_splits_by_key(adata, train_ratio, add_key, split_key, control, random_seed=None):
    """
    Splits the observations in an AnnData object into training and testing sets based on unique values of a specified key,
    with certain control values always included in the training set.
    
    Parameters:
    adata (AnnData): The AnnData object containing the dataset.
    train_ratio (float): The proportion of unique values to include in the train split (0 < train_ratio < 1), excluding control values.
    add_key (str): The key to be added to adata.obs, where the train/test labels will be stored.
    split_key (str): The key in adata.obs used to determine the unique values for making splits.
    control (list): A list of values from the split_key that should always be included in the training set.
    random_seed (int, optional): The seed for the random number generator for reproducibility. If None, the seed is set based on the current time.

    Returns:
    None: The function adds a new column to adata.obs indicating train/test split.
    """
    if random_seed is None:
        random_seed = int(time.time())
    np.random.seed(random_seed)
    unique_values = adata.obs[split_key].unique()
    non_control_values = [value for value in unique_values if value not in control]
    np.random.shuffle(non_control_values)
    # Calculate the number of unique non-control values to include in the train set
    num_train = int(np.floor(train_ratio * len(non_control_values)))
    # Select training values from non-control values
    train_values = set(non_control_values[:num_train])
    # Combine train values with control values
    train_values.update(control)
    # Assign 'train' or 'test' to the observations based on the split of unique values
    adata.obs[add_key] = adata.obs[split_key].apply(lambda x: 'train' if x in train_values else 'test')

    
def build_cinn(adata, cell_repre_model, perturbation_key,  perturbation_type = ["chem", "genetic", "protein"], 
               trt_key = "ordered_all_trt", embed_key = "ordered_all_embedding", 
               random_seed = 42):
    if perturbation_type == "protein":
        scvi_model = cell_repre_model
        perturb_with_onehot = np.array(adata.obs[perturbation_key])
        trt_list = np.unique(perturb_with_onehot)
        embed_idx = []
        for i in range(len(trt_list)):
            trt = trt_list[i]
            idx = np.where(adata.uns[trt_key] == trt)[0][0]
            embed_idx.append(idx)
        embeddings = adata.uns[embed_key][embed_idx]
        
        perturbToEmbed = {}
        for i in range(trt_list.shape[0]):
            perturbToEmbed[trt_list[i]] = i
        torch.manual_seed(42)
        
        flow_model = ConditionalFlatCouplingFlow(conditioning_dim=1280,
                                    embedding_dim=10,
                                    conditioning_depth=2,
                                     n_flows=20,
                                     in_channels=10,
                                     hidden_dim=1024,
                                     hidden_depth=2,
                                     activation="none",
                                     conditioner_use_bn=True)
        model_c = Net2NetFlow_scVIFixFlow(configured_flow = flow_model,
                               cond_stage_data = perturb_with_onehot,
                               perturbToEmbedLib = perturbToEmbed,
                               embedData = embeddings,
                               scvi_model = cell_repre_model)
        return model_c, embeddings, perturbToEmbed
    
    
def predict_perturbation_protein(perturbnet_model, perturbation_embeddings, library_latent, n_cell = 100, random_seed = 42):
    np.random.seed(random_seed)
    Lsample_idx = np.random.choice(range(library_latent.shape[0]), n_cell, replace=True)
    onehot_indice_trt = np.tile(perturbation_embeddings, (n_cell, 1))
    trt_onehot = onehot_indice_trt + np.random.normal(scale = 0.001, size = onehot_indice_trt.shape)
    library_trt_latent = library_latent[Lsample_idx]
    fake_latent, fake_data = perturbnet_model.sample_data(trt_onehot, library_trt_latent)
    
    return fake_latent, fake_data, trt_onehot

def evaluation_metrics_PerturbNet_not_use(adata, perturbnet_model,library_latent, perturbation_key, 
                       trt_key = "ordered_all_trt", embed_key = "ordered_all_embedding",
                       Rsquare = True, Pearson = True, Hellinger = True, gene_set = ["all", "deg", "leg"], 
                       random_seed = 42, n_cell = None):
    if sparse.issparse(adata.X):
        usedata = adata.X.A
    else:
        usedata = adata.X
    if sparse.issparse(adata.layers["counts"]):
        usedata_count = adata.layers["counts"].A
    else:
        usedata_count = adata.layers["counts"]
    normModel = NormalizedRevisionRSquare(largeCountData = usedata_count)
    normModelVar = NormalizedRevisionRSquareVar(norm_model=normModel)
    adata.var["gene_idx"] = np.arange(0,adata.var.shape[0],1)
    perturb = []
    ncell = []
    n_large = []
    r2 = []
    r2_deg = []
    r2_large = []
    pear = []
    pear_deg = []
    pear_large = []
    hd = []
    hd_deg = []
    hd_large = []
    np.random.seed(random_seed)
    trt_unseen_list = adata.obs[perturbation_key].unique()
    perturb_with_onehot_removed = adata.obs[perturbation_key]
    for indice_trt in tqdm(range(len(trt_unseen_list))):
        trt_type = trt_unseen_list[indice_trt]
        pert = trt_type
        pert_idx = np.where(adata.uns[trt_key] == pert)[0][0]
        pert_embed = adata.uns[embed_key][pert_idx]

        idx_trt_type = np.where(perturb_with_onehot_removed == trt_type)[0]
        if idx_trt_type.shape[0] > 1000:
            idx_trt_type  =  np.random.choice(idx_trt_type, 1000, replace = False)
        if n_cell:
            _, fake_data,_ = predict_perturbation_protein(perturbnet_model, pert_embed, library_latent, n_cell = 100, random_seed = random_seed)
        else:
            n_cell = idx_trt_type.shape[0]
            _, fake_data,_ = predict_perturbation_protein(perturbnet_model, pert_embed, library_latent, n_cell = 100, random_seed = random_seed)
        DEG_gene = adata.uns["rank_genes_groups"]["names"][trt_type]
        DEG_idx = np.array(adata.var.loc[DEG_gene]["gene_idx"])
        real_data = usedata_count[idx_trt_type]
        r2_value, real_norm, rfake_norm = normModel.calculate_r_square(real_data, fake_data)
        pear_value = normModel.calculate_pearson(real_data, fake_data)
        hd_value = normModel.calculate_Hellinger_by_gene(real_data,fake_data)


        real_data_deg = real_data[:,DEG_idx]
        fake_data_deg = fake_data[:,DEG_idx]


        r2_deg_value,_,_ = normModel.calculate_r_square(real_data_deg, fake_data_deg)
        pear_deg_value = normModel.calculate_pearson(real_data_deg, fake_data_deg)
        hd_deg_value =  normModel.calculate_Hellinger_by_gene(real_data_deg,fake_data_deg)
   
        large_effect_idx =   DEG_idx[abs(adata.uns["rank_genes_groups"]["logfoldchanges"][pert])>=1]
        num_large =  sum(abs(adata.uns["rank_genes_groups"]["logfoldchanges"][pert])>=1)
    
        if num_large <=1:
            hd_large_value = 1.5
            r2_large_value = 1.5
            pear_large_value = 1.5
        else:
            real_data_large = real_data[:,large_effect_idx ]
            fake_data_large = fake_data[:,large_effect_idx ]
            hd_large_value = fidscore_cal.calculate_Hellinger_by_gene(real_data_large, fake_data_large)
            r2_large_value,_ ,_  = normModel.calculate_r_square(real_data_large, fake_data_large)
            pear_large_value = normModel.calculate_pearson(real_data_large, fake_data_large)
        

        perturb.append(pert)
        ncell.append(len(idx_trt_type))
        n_large.append(num_large)
        r2.append(r2_value)
        r2_deg.append(r2_deg_value)
        r2_large.append(r2_large_value)
        pear.append(pear_value)
        pear_deg.append(pear_deg_value)
        pear_large.append(pear_large_value)
        hd.append(hd_value)
        hd_deg.append(hd_deg_value)
        hd_large.append(hd_large_value)
    
    results = pd.DataFrame({"perturbation":perturb, "number_real_cells_used":ncell,"n_large":n_large,
                        "r2":r2,"pear":pear, "hd":hd,
                        "r2_deg":r2_deg, "pear_deg":pear_deg, "hd_deg":hd_deg,
                        "r2_large":r2_large, "pear_large": pear_large,"hd_large":hd_large
                       })
    return(results)
                                                
def evaluation_metrics_PerturbNet(adata, perturbnet_model, library_latent, perturbation_key,
                                  trt_key="ordered_all_trt", embed_key="ordered_all_embedding",
                                  calc_rsquare=True, calc_pearson=True, calc_hellinger=True,
                                  calc_deg=True, calc_large_effect=True,
                                  random_seed=42, n_cell=None):
    if sparse.issparse(adata.X):
        usedata = adata.X.A
    else:
        usedata = adata.X
    if sparse.issparse(adata.layers["counts"]):
        usedata_count = adata.layers["counts"].A
    else:
        usedata_count = adata.layers["counts"]
    
    normModel = NormalizedRevisionRSquare(largeCountData=usedata_count)
    adata.var["gene_idx"] = np.arange(adata.var.shape[0])

    results = []
    np.random.seed(random_seed)
    trt_unseen_list = adata.obs[perturbation_key].unique()

    for trt_type in trt_unseen_list:
        pert_idx = np.where(adata.uns[trt_key] == trt_type)[0][0]
        pert_embed = adata.uns[embed_key][pert_idx]
        idx_trt_type = np.where(adata.obs[perturbation_key] == trt_type)[0]

        if idx_trt_type.shape[0] > 1000:
            idx_trt_type = np.random.choice(idx_trt_type, 1000, replace=False)
        n_cell_used = n_cell if n_cell else idx_trt_type.shape[0]
        
        _, fake_data, _ = predict_perturbation_protein(perturbnet_model, pert_embed, library_latent, n_cell=n_cell_used, random_seed=random_seed)

        real_data = usedata_count[idx_trt_type]
        DEG_gene = adata.uns["rank_genes_groups"]["names"][trt_type]
        DEG_idx = np.array(adata.var.loc[DEG_gene]["gene_idx"])

        row = {"perturbation": trt_type, "number_real_cells_used": len(idx_trt_type)}

        if calc_rsquare:
            r2_value, _, _ = normModel.calculate_r_square(real_data, fake_data)
            row["r2"] = r2_value
            if calc_deg:
                real_data_deg = real_data[:, DEG_idx]
                fake_data_deg = fake_data[:, DEG_idx]
                r2_deg_value, _, _ = normModel.calculate_r_square(real_data_deg, fake_data_deg)
                row["r2_deg"] = r2_deg_value

        if calc_pearson:
            pear_value = normModel.calculate_pearson(real_data, fake_data)
            row["pear"] = pear_value
            if calc_deg:
                pear_deg_value = normModel.calculate_pearson(real_data_deg, fake_data_deg)
                row["pear_deg"] = pear_deg_value

        if calc_hellinger:
            hd_value = normModel.calculate_Hellinger_by_gene(real_data, fake_data)
            row["hd"] = hd_value
            if calc_deg:
                hd_deg_value = normModel.calculate_Hellinger_by_gene(real_data_deg, fake_data_deg)
                row["hd_deg"] = hd_deg_value

        if calc_large_effect:
            large_effect_idx = DEG_idx[abs(adata.uns["rank_genes_groups"]["logfoldchanges"][trt_type]) >= 1]
            num_large = sum(abs(adata.uns["rank_genes_groups"]["logfoldchanges"][trt_type]) >= 1)
            if num_large <= 1:
                pseudo_value = 1.5
                row.update({"r2_large": pseudo_value, "pear_large": pseudo_value, "hd_large": pseudo_value})
            else:
                real_data_large = real_data[:, large_effect_idx]
                fake_data_large = fake_data[:, large_effect_idx]
                if calc_rsquare:
                    r2_large_value, _, _ = normModel.calculate_r_square(real_data_large, fake_data_large)
                    row["r2_large"] = r2_large_value
                if calc_pearson:
                    pear_large_value = normModel.calculate_pearson(real_data_large, fake_data_large)
                    row["pear_large"] = pear_large_value
                if calc_hellinger:
                    hd_large_value = normModel.calculate_Hellinger_by_gene(real_data_large, fake_data_large)
                    row["hd_large"] = hd_large_value

        results.append(row)

    results_df = pd.DataFrame(results)
    return results_df
    
def evaluation_metrics_Random(adata_train,adata_test, perturbation_key,
                                  trt_key="ordered_all_trt", embed_key="ordered_all_embedding",
                                  calc_rsquare=True, calc_pearson=True, calc_hellinger=True,
                                  calc_deg=True, calc_large_effect=True,
                                  random_seed=42, n_cell=None):

    def calculate_r_square(real_data, fake_data,norm_vec_real,norm_vec_fake):
        real_data_norm = real_data.copy()
        real_data_norm_sum = norm_vec_real 
        real_data_norm  = real_data_norm[real_data_norm_sum != 0,:]

        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = norm_vec_fake
        fake_data_norm  = fake_data_norm[fake_data_norm_sum != 0,:]

        real_data_norm = real_data_norm /  real_data_norm_sum[:, None] * 1e4
        fake_data_norm = fake_data_norm /  fake_data_norm_sum[:, None] * 1e4
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(fake_data_norm)
        # important to make sure x is y_true and y is y_pred since it will affect which y_bar to be used
        x = np.average(real_data_norm, axis = 0)
        y = np.average(fake_data_norm, axis = 0)
        r2_value = r2_score(x, y)
        return(r2_value)

    def calculate_pearson(real_data, fake_data,norm_vec_real,norm_vec_fake):
        real_data_norm = real_data.copy()
        real_data_norm_sum = norm_vec_real 
        real_data_norm  = real_data_norm[real_data_norm_sum != 0,:]

        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = norm_vec_fake
        fake_data_norm  = fake_data_norm[fake_data_norm_sum != 0,:]

        real_data_norm = real_data_norm /  real_data_norm_sum[:, None] * 1e4
        fake_data_norm = fake_data_norm /  fake_data_norm_sum[:, None] * 1e4
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(fake_data_norm)
        # important to make sure x is y_true and y is y_pred since it will affect which y_bar to be used
        x = np.average(real_data_norm, axis = 0)
        y = np.average(fake_data_norm, axis = 0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        return(r_value)

    def calculate_Hellinger_by_gene(real_data, fake_data,norm_vec_real,norm_vec_fake):
        real_data_norm = real_data.copy()
        real_data_norm_sum = norm_vec_real 
        real_data_norm  = real_data_norm[real_data_norm_sum != 0,:]

        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = norm_vec_fake
        fake_data_norm  = fake_data_norm[fake_data_norm_sum != 0,:]

        real_data_norm = real_data_norm / real_data_norm_sum[:, None] * 1e4
        fake_data_norm = fake_data_norm /  fake_data_norm_sum[:, None] * 1e4
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(fake_data_norm)
    
        fake_data_var = np.var(real_data_norm,axis = 0)
        real_data_var = np.var(fake_data_norm,axis = 0)
        non_zero_var_col = np.where((real_data_var > 1e-6) & (fake_data_var > 1e-6))[0]     
        real_data_norm = real_data_norm[:,non_zero_var_col]
        fake_data_norm = fake_data_norm[:,non_zero_var_col]      
        sum_dis = 0
        for i in range(real_data_norm.shape[1]):
            real_sub = real_data_norm[:,i]
            fake_sub = fake_data_norm[:,i]
            bh_coef = bhatta_coef(real_sub ,fake_sub)
            if bh_coef>=1:
                continue
            sum_dis += sqrt(1 - bh_coef)
        return(sum_dis/real_data.shape[1])

    adata_test.var["gene_idx"] = np.arange(adata_test.var.shape[0])

    results = []
    np.random.seed(random_seed)
    trt_unseen_list = adata_test.obs[perturbation_key].unique()

    for trt_type in trt_unseen_list:

        
        
        seen_data = adata_train.layers["counts"].A
        seen_data_idx = list(range(seen_data.shape[0]))
        
        real_data_idx = adata_test.obs[adata_test.obs["variant_seq"] == trt_type].index
        if real_data_idx.shape[0] > 1000:
            real_data_idx = np.random.choice(real_data_idx, 1000, replace = False)
    
        real_data = adata_test[real_data_idx, :].copy().layers["counts"].A
        norm_vec_real = adata_test[real_data_idx, :].obs["n_counts_total"].to_numpy()
        
        n_cell_used = n_cell if n_cell else real_data.shape[0]
        
        idx_rsample = np.random.choice(seen_data_idx, n_cell_used, replace=True)
        fake_data = seen_data[idx_rsample]
        norm_vec_fake = adata_train.obs["n_counts_total"].to_numpy()[idx_rsample]

        DEG_gene = adata_test.uns["rank_genes_groups"]["names"][trt_type]
        DEG_idx = np.array(adata_test.var.loc[DEG_gene]["gene_idx"])

        row = {"perturbation": trt_type, "number_real_cells_used": len(real_data_idx)}

        if calc_rsquare:
            r2_value = calculate_r_square(real_data, fake_data, norm_vec_real,norm_vec_fake)
            row["r2"] = r2_value
            if calc_deg:
                real_data_deg = real_data[:, DEG_idx]
                fake_data_deg = fake_data[:, DEG_idx]
                r2_deg_value = calculate_r_square(real_data_deg, fake_data_deg, norm_vec_real,norm_vec_fake)
                row["r2_deg"] = r2_deg_value

        if calc_pearson:
            pear_value = calculate_pearson(real_data, fake_data, norm_vec_real,norm_vec_fake)
            row["pear"] = pear_value
            if calc_deg:
                pear_deg_value = calculate_pearson(real_data_deg, fake_data_deg, norm_vec_real,norm_vec_fake)
                row["pear_deg"] = pear_deg_value

        if calc_hellinger:
            hd_value = calculate_Hellinger_by_gene(real_data, fake_data, norm_vec_real, norm_vec_fake)
            row["hd"] = hd_value
            if calc_deg:
                hd_deg_value = calculate_Hellinger_by_gene(real_data_deg, fake_data_deg, norm_vec_real,norm_vec_fake)
                row["hd_deg"] = hd_deg_value

        if calc_large_effect:
            large_effect_idx = DEG_idx[abs(adata_test.uns["rank_genes_groups"]["logfoldchanges"][trt_type]) >= 1]
            num_large = sum(abs(adata_test.uns["rank_genes_groups"]["logfoldchanges"][trt_type]) >= 1)
            if num_large <= 1:
                pseudo_value = 1.5
                row.update({"r2_large": pseudo_value, "pear_large": pseudo_value, "hd_large": pseudo_value})
            else:
                real_data_large = real_data[:, large_effect_idx]
                fake_data_large = fake_data[:, large_effect_idx]
                if calc_rsquare:
                    r2_large_value = calculate_r_square(real_data_large, fake_data_large, norm_vec_real,norm_vec_fake)
                    row["r2_large"] = r2_large_value
                if calc_pearson:
                    pear_large_value = calculate_pearson(real_data_large, fake_data_large, norm_vec_real,norm_vec_fake)
                    row["pear_large"] = pear_large_value
                if calc_hellinger:
                    hd_large_value = calculate_Hellinger_by_gene(real_data_large, fake_data_large, norm_vec_real,norm_vec_fake)
                    row["hd_large"] = hd_large_value

        results.append(row)

    results_df = pd.DataFrame(results)
    return results_df

#####################################
# PLOT FUNCTION
######################################
def umapPlot_latent_check(real_latent, fake_latent, path_file_save = None):
    all_latent = np.concatenate([fake_latent, real_latent], axis = 0)
    cat_t = ["Real"] * real_latent.shape[0]
    cat_g = ["Fake"] * fake_latent.shape[0]
    cat_rf_gt = np.append(cat_g, cat_t)
    trans = umap.UMAP(random_state=42, min_dist = 0.5, n_neighbors=30).fit(all_latent)
    X_embedded_pr = trans.transform(all_latent)
    df = X_embedded_pr.copy()
    df = pd.DataFrame(df)
    df['x-umap'] = X_embedded_pr[:,0]
    df['y-umap'] = X_embedded_pr[:,1]
    df['category'] = cat_rf_gt
    
    chart_pr = ggplot(df, aes(x= 'x-umap', y= 'y-umap', colour = 'category') ) \
    + geom_point(size=0.5, alpha = 0.5) \
    + ggtitle("UMAP dimensions")

    if path_file_save is not None:
        chart_pr.save(path_file_save, width=12, height=8, dpi=144)
    return chart_pr


def boxplot_metrics(model_dict, metric_key, path_file_save = None):
    
    shared_cols = ["perturbation", metric_key]
    df_list = []
    for model, results in model_dict.items():
        results = results[shared_cols]
        results["model"] = np.repeat(model,results.shape[0])
        df_list.append(results)
    df = pd.concat(df_list, ignore_index=True)
    
    chart_pr = ggplot(df, aes(x= "model", y= metric_key, fill = "model") ) \
    + geom_boxplot() \

    if path_file_save is not None:
        chart_pr.save(path_file_save, width=12, height=8, dpi=144)
    return chart_pr

def contourplot_prepare_embeddings(adata, Lsample_obs, perturbnet_model, highlight, 
                                   trt_key = "ordered_all_trt", embed_key = "ordered_all_embedding", 
                                   n_cell = 50, random_seed = 42):
    background_pert = []
    background_cell = []
    
    for mut in adata.obs.mutation_name.unique():
        if mut == highlight:
            continue
        
        pert = adata.obs[adata.obs.mutation_name == mut]["variant_seq"].unique()[0]
        pert_idx = np.where(adata.uns[trt_key] == pert)[0][0]
        pert_embed = adata.uns[embed_key][pert_idx]
        fake_cell_latent_background, _, fake_pert_latent_background = predict_perturbation_protein(
            perturbnet_model,
            perturbation_embeddings=pert_embed,
            library_latent=Lsample_obs,
            n_cell=n_cell,
            random_seed=random_seed
        )
        
        background_pert.append(fake_pert_latent_background)
        background_cell.append(fake_cell_latent_background)
    
    background_pert = np.concatenate(background_pert)
    background_cell = np.concatenate(background_cell)
    
    return background_pert, background_cell

def contourplot_space_mapping_pca(embeddings_cell, embeddings_pert, background_pert, background_cell, highlight_label,
                                  random_state=42, n_pcs=50, bandwidth=0.2):
    # Apply PCA to reduce the dimensions of perturbations to 50 principal components
    pca = PCA(n_components=n_pcs)
    pert_pca = pca.fit_transform(np.concatenate([background_pert, embeddings_pert]))

    # Concatenate embeddings after PCA
    embeddings_cell_all = np.concatenate([background_cell, embeddings_cell])
    embeddings_pert_all = pert_pca

    # Labels for plotting
    cat_pert = ["Other"] * background_pert.shape[0] + [highlight_label] * embeddings_pert.shape[0]
    cat_cell = ["Other"] * background_cell.shape[0] + [highlight_label] * embeddings_cell.shape[0]

    # Create UMAP transformers and transform data
    trans_pert = umap.UMAP(random_state=random_state, min_dist=0.5, n_neighbors=30).fit(embeddings_pert_all)
    trans_cell = umap.UMAP(random_state=random_state, min_dist=0.5, n_neighbors=30).fit(embeddings_cell_all)
    Y_embedded = trans_pert.transform(embeddings_pert_all)
    Z_embedded = trans_cell.transform(embeddings_cell_all)

    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    # Define a plotting function for each subplot
    def plot_with_contours(ax, data, categories, title, highlight, add_contour = True):
        highlight_data = data[categories == highlight]
        other_data = data[categories != highlight]

        # Plot background data
        ax.scatter(other_data[:, 0], other_data[:, 1], color='gray', s=1, label='Other')

        # Plot highlight data
        ax.scatter(highlight_data[:, 0], highlight_data[:, 1], color='red', s=1, label=highlight_label)

        if add_contour and highlight_data.size > 0:
            x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
            y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
            x_grid = np.linspace(x_min, x_max, 100)
            y_grid = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            kde = gaussian_kde(highlight_data.T, bw_method=bandwidth)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(X, Y, Z, levels=5, colors='red')
        
        ax.set_xlim(np.min(data[:, 0]), np.max(data[:, 0]))
        ax.set_ylim(np.min(data[:, 1]), np.max(data[:, 1]))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plotting for each representation
    plot_with_contours(ax1, Y_embedded, np.array(cat_pert), 'Perturbation Representation', highlight_label, add_contour = False)
    plot_with_contours(ax2, Z_embedded, np.array(cat_cell), 'Cellular Representation', highlight_label)

    # Draw lines for highlighted points
    transFigure = fig.transFigure.inverted()
    highlight_indices = np.where(np.array(cat_pert) == highlight_label)[0]
    if len(highlight_indices) > 30:
        highlight_indices = np.random.choice(highlight_indices, 30, replace=False)  # Randomly pick 30 indices

    for index in highlight_indices:
        xy1 = transFigure.transform(ax1.transData.transform(Y_embedded[index]))
        xy2 = transFigure.transform(ax2.transData.transform(Z_embedded[index]))
        line = matplotlib.lines.Line2D((xy1[0], xy2[0]), (xy1[1], xy2[1]), transform=fig.transFigure, color='red', linewidth=0.5)
        fig.lines.append(line)

    # Place a legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    plt.show()