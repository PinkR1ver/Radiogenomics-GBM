import os
from re import L
from turtle import title
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rich import print
import gc

from torch import FloatStorage

basePath = r''
dataPath = os.path.join(basePath, 'data')
genePath = os.path.join(dataPath, 'gene')
savePath = os.path.join(genePath, 'gene_expression_histogram')
geneFile = 'gene_expression_details.csv'

if __name__ == '__main__':
    gene_expression_details = pd.read_csv(os.path.join(dataPath, geneFile))
    tumor_regions = gene_expression_details.columns[18:-1]

    
    # Get tumor_name
    tumor_name = gene_expression_details['tumor_name']
    tumor_name = tumor_name.drop_duplicates()
    tumor_name = tumor_name.reset_index(drop=True)

    for tumor_region in tumor_regions:

        for i in tumor_name:

            if os.path.isfile(os.path.join(savePath, tumor_region, i, 'log.txt')):
                f = open(os.path.join(savePath, tumor_region, i, 'log.txt'), "w")
                f.close()
            if os.path.isfile(os.path.join(savePath, tumor_region, i, 'loss.txt')):
                f = open(os.path.join(savePath, tumor_region, i, 'loss.txt'), "w")
                f.close()

            # Get sub group
            sub_gene_expression_details = gene_expression_details.loc[gene_expression_details['tumor_name'] == i]
            sub_gene_expression_details = sub_gene_expression_details.reset_index(drop=True)
            ## print(sub_gene_expression_details)

            # Get gene symbol from subgroup
            gene_symbol = sub_gene_expression_details['gene_symbol']
            gene_symbol = gene_symbol.drop_duplicates()
            gene_symbol = gene_symbol.reset_index(drop=True)
            ## print(gene_symbol)

            # Extract one tumor, one gene

            fig_all = plt.figure(num="Together", figsize=(30, 30))

            color_may = len(gene_symbol)

            labels = []

            for flag, j in enumerate(gene_symbol):


                sub_sub_gene_expression_details = sub_gene_expression_details.loc[sub_gene_expression_details['gene_symbol'] == j]
                
                specific_tumor_region_expression_energy = sub_sub_gene_expression_details[tumor_region]
                specific_tumor_region_expression_energy = specific_tumor_region_expression_energy.reset_index(drop=True)
                remove_null_specific_tumor_region_expression_energy = specific_tumor_region_expression_energy[~(specific_tumor_region_expression_energy.isnull())]
                remove_null_specific_tumor_region_expression_energy = remove_null_specific_tumor_region_expression_energy.reset_index(drop=True)
                sort_specific_tumor_region_expression_energy = np.sort(remove_null_specific_tumor_region_expression_energy.to_numpy())


                if len(remove_null_specific_tumor_region_expression_energy) > 8:
                    fig = plt.figure(num="Current", figsize=(6, 6))

                    if not os.path.isdir(os.path.join(savePath, tumor_region)):
                        os.mkdir(os.path.join(savePath, tumor_region))
                    if not os.path.isdir(os.path.join(savePath, tumor_region, i)):
                        os.mkdir(os.path.join(savePath, tumor_region, i))
                    if not os.path.isfile(os.path.join(savePath, tumor_region, i, 'log.txt')):
                        f = open(os.path.join(savePath, tumor_region, i, 'log.txt'), "x")
                        f.close()

                    print(j)
                    print(remove_null_specific_tumor_region_expression_energy)

                    sns.histplot(remove_null_specific_tumor_region_expression_energy, stat='density', kde=True)
                    plt.title(j)
                    plt.savefig(os.path.join(savePath, tumor_region, i, j+'.png'))
                    plt.close(fig)

                    plt.figure("Together")
                    sns.scatterplot(x=flag ,y=remove_null_specific_tumor_region_expression_energy.values, size=1000 ,color=(sns.color_palette("hls", color_may))[flag])
                    labels.append(j)
                    
                    f = open(os.path.join(savePath, tumor_region, i, 'log.txt'), "a")
                    f.write(f'{j}:\n')
                    for k in sort_specific_tumor_region_expression_energy:
                        f.write(f'{str(k)}\n')
                    f.write('---------------------------------------------------\n\n\n')
                    f.close()
                    gc.collect()
                
                else:

                    print(j)

                    if len(remove_null_specific_tumor_region_expression_energy) >= 1:
                        plt.figure("Together")
                        sns.scatterplot(x=flag ,y=remove_null_specific_tumor_region_expression_energy.values, size=1000 ,color=(sns.color_palette("hls", color_may))[flag])
                        labels.append(j)

                    if not os.path.isdir(os.path.join(savePath, tumor_region)):
                        os.mkdir(os.path.join(savePath, tumor_region))
                    if not os.path.isdir(os.path.join(savePath, tumor_region, i)):
                        os.mkdir(os.path.join(savePath, tumor_region, i))
                    if not os.path.isfile(os.path.join(savePath, tumor_region, i, 'loss.txt')):
                        f = open(os.path.join(savePath, tumor_region, i, 'loss.txt'), "x")
                        f.close()
                    
                    f = open(os.path.join(savePath, tumor_region, i, 'loss.txt'), "a")
                    f.write(f'{j}:\n')
                    for k in sort_specific_tumor_region_expression_energy:
                        f.write(f'{str(k)}\n')
                    f.write('---------------------------------------------------\n\n\n')
                    f.close()

                    plt.close(fig)
                    gc.collect()

            plt.figure("Together")
            plt.legend(title='Gene', loc='upper right', labels=labels, ncol=6)
            plt.title("Compare different gene expression energy in " + tumor_region)
            plt.xlabel("Different Genes")
            plt.ylabel("Expression Energy")
            plt.savefig(os.path.join(savePath, tumor_region, i, 'ALL.png'))
            plt.close(fig_all)
            gc.collect()
