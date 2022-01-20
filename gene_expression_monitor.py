import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rich import print

basePath = r''
dataPath = os.path.join(basePath, 'data')
genePath = os.path.join(dataPath, 'gene')
savePath = os.path.join(genePath, 'gene_expression_histogram')
geneFile = 'gene_expression_details.csv'

if __name__ == '__main__':
    gene_expression_details = pd.read_csv(os.path.join(dataPath, geneFile))
    
    # Get tumor_name
    tumor_name = gene_expression_details['tumor_name']
    tumor_name = tumor_name.drop_duplicates()
    tumor_name = tumor_name.reset_index(drop=True)

    for i in tumor_name:

        if not os.path.isdir(os.path.join(savePath, i)):
            os.mkdir(os.path.join(savePath, i))


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
        fig = plt.figure(figsize=(6, 6))
        for j in gene_symbol:
            sub_sub_gene_expression_details = sub_gene_expression_details.loc[sub_gene_expression_details['gene_symbol'] == j]
            expression_energy_le = sub_sub_gene_expression_details['expression_energy_le']
            expression_energy_le = expression_energy_le.reset_index(drop=True)
            if len(expression_energy_le) > 15:
                print(j)
                print(expression_energy_le)
                sns.kdeplot(sub_sub_gene_expression_details['expression_energy_le'])
                plt.savefig(os.path.join(savePath, i, j+'.png'))
                plt.close()
            


