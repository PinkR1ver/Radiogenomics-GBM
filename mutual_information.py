from email.mime import base
from lib2to3.pgen2.pgen import generate_grammar
import os
import pandas as pd
import numpy as np
from sklearn import feature_selection
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

main_gene_symbol = ["OLIG2", "PDPN" ,"POSTN" ,"MECOM" ,"EZH2" ,"HIF1A" ,"BIRC5" ,"ID1" ,"ID2" ,"IGFBP2" ,"ITGA6" ,"DANCR" ,"MET" ,"MYC" ,"NOS2" ,"PDGFRA" ,"PI3" ,"TGFBR2" ,"TNFAIP3", "PROM1", "CD44"]
interesting_region = ["expression_energy_it", "expression_energy_ct", "expression_energy_le"]

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi

def find_common_gene_expression(gene_file):
    gene_expression = pd.read_csv(gene_file)
    gene_expression = gene_expression.fillna(0)
    patients = gene_expression.tumor_name.unique()
    it_gene_df = pd.DataFrame(index=patients, columns=main_gene_symbol)
    ct_gene_df = pd.DataFrame(index=patients, columns=main_gene_symbol)
    le_gene_df = pd.DataFrame(index=patients, columns=main_gene_symbol)
    for gene in main_gene_symbol:
        for patient in patients:
            df = gene_expression[gene_expression['tumor_name'] == patient]
            df = df[df['gene_symbol'] == gene]
            for region in interesting_region:
                gene_mean = df[region].mean()
                if region == "expression_energy_it":
                    it_gene_df.at[patient, gene] = gene_mean
                if region == "expression_energy_ct":
                    ct_gene_df.at[patient, gene] = gene_mean
                if region == "expression_energy_le":
                    le_gene_df.at[patient, gene] = gene_mean
    
    for index in it_gene_df.index:
        if "-2" in index:
            it_gene_df = it_gene_df.drop([index])
        else:
            it_gene_df = it_gene_df.rename(index={index:index.split('-')[0]})
    
    it_gene_df = it_gene_df.sort_index()

    for index in ct_gene_df.index:
        if "-2" in index:
            ct_gene_df = ct_gene_df.drop([index])
        else:
            ct_gene_df = ct_gene_df.rename(index={index:index.split('-')[0]})
    
    ct_gene_df = ct_gene_df.sort_index()

    for index in le_gene_df.index:
        if "-2" in index:
            le_gene_df = le_gene_df.drop([index])
        else:
            le_gene_df = le_gene_df.rename(index={index:index.split('-')[0]})
    
    le_gene_df = le_gene_df.sort_index()

    return it_gene_df, ct_gene_df, le_gene_df
                

def feature_sum_up(feature_file):
    features = pd.read_csv(feature_file)
    patients = features.Patient.unique()
    feature_col = features.columns[28:-1]
    
    feature_df = pd.DataFrame(index=patients, columns=feature_col)

    for patient in patients:
         df = features[features['Patient'] == patient]
         for feature in feature_col:
             feature_mean = df[feature].mean()
             feature_df.at[patient, feature] = feature_mean

    return feature_df.sort_index()
    

if __name__ == '__main__':
    base_path = os.path.dirname(__file__)
    gene_file = os.path.join(base_path, 'data', 'gene_expression_details.csv')
    it_gene_df, ct_gene_df, le_gene_df = find_common_gene_expression(gene_file)

    feature_file = os.path.join(base_path, "data", "feature_extraction.csv")
    feature_df_origin = feature_sum_up(feature_file)

    feature_df = feature_df_origin.copy(deep=True)

    mutual_info_df = pd.DataFrame(index=feature_df.columns, columns=main_gene_symbol)

    # print(it_gene_df)
    # print(feature_df)

    # for index in it_gene_df.index:
    #     if index not in feature_df.index:
    #         it_gene_df = it_gene_df.drop([index])

    # for index in feature_df.index:
    #     if index not in it_gene_df.index:
    #         feature_df = feature_df.drop([index])

    for index in ct_gene_df.index:
        if index not in feature_df.index:
            ct_gene_df = ct_gene_df.drop([index])

    for index in feature_df.index:
        if index not in ct_gene_df.index:
            feature_df = feature_df.drop([index])

    # print(it_gene_df)
    # print(feature_df)

    # feature_np = feature_df.to_numpy()

    MI_df = pd.DataFrame(index=feature_df.columns, columns=ct_gene_df.columns)

    for col1 in feature_df.columns:
        for col2 in ct_gene_df.columns:
            feature_vec = feature_df[col1].to_numpy()
            gene_vec = ct_gene_df[col2].to_numpy()

            # print(feature_np.shape)
            # print(gene_vec.shape)
            # MU = feature_selection.mutual_info_classif(X=feature_np, y=gene_vec)
            # print(MU)

            MI = calc_MI(feature_vec, gene_vec, 5)
            MI_df.at[col1, col2] = MI

    # print(MI_df)
    fig = plt.figure(figsize=(30, 30))
    sns.heatmap(MI_df.fillna(0), annot=True, fmt="g")
    heatmap_path = os.path.join(base_path, 'data', 'Mutual_Information', 'Radiomics_with_ctAreaGeneExpression.svg')
    plt.savefig(heatmap_path, format='svg', dpi=1800)
    plt.close(fig)

    
    feature_df = feature_df_origin.copy(deep=True)

    for index in it_gene_df.index:
        if index not in feature_df.index:
            it_gene_df = it_gene_df.drop([index])

    for index in feature_df.index:
        if index not in it_gene_df.index:
            feature_df = feature_df.drop([index])

    MI_df = pd.DataFrame(index=feature_df.columns, columns=it_gene_df.columns)

    for col1 in feature_df.columns:
        for col2 in it_gene_df.columns:
            feature_vec = feature_df[col1].to_numpy()
            gene_vec = it_gene_df[col2].to_numpy()

            MI = calc_MI(feature_vec, gene_vec, 5)
            MI_df.at[col1, col2] = MI

    # print(MI_df)
    fig = plt.figure(figsize=(30, 30))
    sns.heatmap(MI_df.fillna(0), annot=True, fmt="g")
    heatmap_path = os.path.join(base_path, 'data', 'Mutual_Information', 'Radiomics_with_itAreaGeneExpression.svg')
    plt.savefig(heatmap_path, format='svg', dpi=1800)
    plt.close(fig)

    feature_df = feature_df_origin.copy(deep=True)

    for index in le_gene_df.index:
        if index not in feature_df.index:
            le_gene_df = le_gene_df.drop([index])

    for index in feature_df.index:
        if index not in le_gene_df.index:
            feature_df = feature_df.drop([index])

    MI_df = pd.DataFrame(index=feature_df.columns, columns=le_gene_df.columns)

    for col1 in feature_df.columns:
        for col2 in le_gene_df.columns:
            feature_vec = feature_df[col1].to_numpy()
            gene_vec = le_gene_df[col2].to_numpy()

            MI = calc_MI(feature_vec, gene_vec, 5)
            MI_df.at[col1, col2] = MI

    # print(MI_df)
    fig = plt.figure(figsize=(30, 30))
    sns.heatmap(MI_df.fillna(0), annot=True, fmt="g")
    heatmap_path = os.path.join(base_path, 'data', 'Mutual_Information', 'Radiomics_with_leAreaGeneExpression.svg')
    plt.savefig(heatmap_path, format='svg', dpi=1800)
    plt.close(fig)


    
        
