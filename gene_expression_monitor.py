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

            if os.path.isfile(os.path.join(savePath, tumor_region, i, 'main_trend.txt')):
                f = open(os.path.join(savePath, tumor_region,
                         i, 'main_trend.txt'), "w")
                f.close()
            if os.path.isfile(os.path.join(savePath, tumor_region, i, 'loss.txt')):
                f = open(os.path.join(
                    savePath, tumor_region, i, 'loss.txt'), "w")
                f.close()
            if os.path.isfile(os.path.join(savePath, tumor_region, i, 'unique.txt')):
                f = open(os.path.join(
                    savePath, tumor_region, i, 'unique.txt'), "w")
                f.close()

            # Get sub group
            sub_gene_expression_details = gene_expression_details.loc[
                gene_expression_details['tumor_name'] == i]
            sub_gene_expression_details = sub_gene_expression_details.reset_index(
                drop=True)
            # print(sub_gene_expression_details)

            # Get gene symbol from subgroup
            gene_symbol = sub_gene_expression_details['gene_symbol']
            gene_symbol = gene_symbol.drop_duplicates()
            gene_symbol = gene_symbol.reset_index(drop=True)
            # print(gene_symbol)

            # Extract one tumor, one gene

            fig_all = plt.figure(num="Together", figsize=(16, 16))
            fig_unique = plt.figure(num="Unique", figsize=(30, 30))

            color_may = len(gene_symbol)
            color_select_random = np.arange(color_may)
            np.random.shuffle(color_select_random)

            labels_main_trend = []
            labels_unique = []

            flag_main_trend = 0
            flag_unique = 0

            for j in gene_symbol:

                sub_sub_gene_expression_details = sub_gene_expression_details.loc[
                    sub_gene_expression_details['gene_symbol'] == j]

                specific_tumor_region_expression_energy = sub_sub_gene_expression_details[
                    tumor_region]
                specific_tumor_region_expression_energy = specific_tumor_region_expression_energy.reset_index(
                    drop=True)
                remove_null_specific_tumor_region_expression_energy = specific_tumor_region_expression_energy[~(
                    specific_tumor_region_expression_energy.isnull())]
                remove_null_specific_tumor_region_expression_energy = remove_null_specific_tumor_region_expression_energy.reset_index(
                    drop=True)
                sort_specific_tumor_region_expression_energy = np.sort(
                    remove_null_specific_tumor_region_expression_energy.to_numpy())

                if len(remove_null_specific_tumor_region_expression_energy) > 8:
                    fig = plt.figure(num="Current", figsize=(6, 6))

                    if not os.path.isdir(os.path.join(savePath, tumor_region)):
                        os.mkdir(os.path.join(savePath, tumor_region))
                    if not os.path.isdir(os.path.join(savePath, tumor_region, i)):
                        os.mkdir(os.path.join(savePath, tumor_region, i))
                    if not os.path.isfile(os.path.join(savePath, tumor_region, i, 'main_trend.txt')):
                        f = open(os.path.join(savePath, tumor_region,
                                 i, 'main_trend.txt'), "x")
                        f.close()

                    print(j)
                    print(remove_null_specific_tumor_region_expression_energy)

                    sns.histplot(
                        remove_null_specific_tumor_region_expression_energy, stat='density', kde=True)
                    plt.title(j)
                    plt.savefig(os.path.join(
                        savePath, tumor_region, i, j+'.png'))
                    plt.close(fig)

                    plt.figure("Together")
                    sns.scatterplot(x=flag_main_trend, y=remove_null_specific_tumor_region_expression_energy.values, size=1000, color=(
                        sns.color_palette("hls", color_may))[color_select_random[flag_main_trend]])
                    plt.annotate(j + ':' + str(np.amax(remove_null_specific_tumor_region_expression_energy.values)),
                                 (flag_main_trend, np.amax(remove_null_specific_tumor_region_expression_energy.values) + 1))
                    labels_main_trend.append(j)
                    flag_main_trend += 1

                    f = open(os.path.join(savePath, tumor_region,
                             i, 'main_trend.txt'), "a")
                    f.write(f'{j}:\n')
                    for k in sort_specific_tumor_region_expression_energy:
                        f.write(f'{str(k)}\n')
                    f.write(
                        '---------------------------------------------------\n\n\n')
                    f.close()
                    gc.collect()

                elif len(remove_null_specific_tumor_region_expression_energy) == 0:

                    print(j)

                    if not os.path.isdir(os.path.join(savePath, tumor_region)):
                        os.mkdir(os.path.join(savePath, tumor_region))
                    if not os.path.isdir(os.path.join(savePath, tumor_region, i)):
                        os.mkdir(os.path.join(savePath, tumor_region, i))
                    if not os.path.isfile(os.path.join(savePath, tumor_region, i, 'loss.txt')):
                        f = open(os.path.join(
                            savePath, tumor_region, i, 'loss.txt'), "x")
                        f.close()

                    f = open(os.path.join(
                        savePath, tumor_region, i, 'loss.txt'), "a")
                    f.write(f'{j}:\n')
                    f.close()

                    plt.close(fig)
                    gc.collect()

                else:
                    print(j)

                    if not os.path.isdir(os.path.join(savePath, tumor_region)):
                        os.mkdir(os.path.join(savePath, tumor_region))
                    if not os.path.isdir(os.path.join(savePath, tumor_region, i)):
                        os.mkdir(os.path.join(savePath, tumor_region, i))
                    if not os.path.isfile(os.path.join(savePath, tumor_region, i, 'unique.txt')):
                        f = open(os.path.join(
                            savePath, tumor_region, i, 'unique.txt'), "x")
                        f.close()

                    plt.figure("Unique")
                    sns.scatterplot(x=flag_unique, y=remove_null_specific_tumor_region_expression_energy.values, size=1000, color=(
                        sns.color_palette("hls", color_may))[color_select_random[flag_unique]])
                    if sort_specific_tumor_region_expression_energy[-1] > 10:
                        plt.annotate(j + ':' + str(np.amax(remove_null_specific_tumor_region_expression_energy.values)),
                                    (flag_unique, np.amax(remove_null_specific_tumor_region_expression_energy.values) + 1))
                    labels_unique.append(j)
                    flag_unique += 1

                    f = open(os.path.join(
                        savePath, tumor_region, i, 'unique.txt'), "a")
                    f.write(f'{j}:\n')
                    for k in sort_specific_tumor_region_expression_energy:
                        f.write(f'{str(k)}\n')
                    f.write(
                        '---------------------------------------------------\n\n\n')
                    f.close()

                    plt.close(fig)
                    gc.collect()

            plt.figure("Together")
            plt.legend(title='Gene', loc='upper right',
                       labels=labels_main_trend, ncol=2)
            plt.title(
                "Compare different gene expression energy in " + tumor_region)
            plt.xlabel("Different Genes")
            plt.ylabel("Expression Energy")
            plt.savefig(os.path.join(
                savePath, tumor_region, i, 'main_trend.png'))
            plt.close(fig_all)

            plt.figure("Unique")
            plt.legend(title='Gene', loc='upper right',
                       labels=labels_unique, ncol=6)
            plt.title(
                "Compare different gene expression energy in " + tumor_region)
            plt.xlabel("Different Genes")
            plt.ylabel("Expression Energy")
            plt.savefig(os.path.join(savePath, tumor_region, i, 'Unique.png'))
            plt.close(fig_unique)

            gc.collect()
