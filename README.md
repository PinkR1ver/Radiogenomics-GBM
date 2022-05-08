# Radiogenomics analysis on Dataset from Ivy Glioblastoma Atlas Project (Ivy GAP)
## What is Radiogenomics
![](Pictures\RadiogenomicsResearchTypicalPathway.png)

## Which part this repositories focus on?
1. GBM MRI image segmentation
2. Feature Extracion from ROI
3. Data anaylsis, including:
   * GBM subtypes prediction,which is based on Wang research on 2017, prediction (Accuracy:92%, using Multilayer Perceptron)
   * Mutual Information Analysis to Radiomics feature and gene expression energy.

## File Structure
```
├─data # Store data, both data for training and generated results
│  ├─classification results # Classification results
│  │  ├─model
│  │  ├─Simplified
│  │  │  ├─Test
│  │  │  │  ├─Normalize
│  │  │  │  │  └─PCA
│  │  │  │  └─PCA
│  │  │  │      └─Normalize
│  │  │  └─Train
│  │  │      ├─Normalize
│  │  │      │  └─PCA
│  │  │      └─PCA
│  │  │          └─Normalize
│  │  ├─Test
│  │  │  ├─Normalize
│  │  │  │  └─PCA
│  │  │  └─PCA
│  │  │      └─Normalize
│  │  └─Train
│  │      ├─Normalize
│  │      │  └─PCA
│  │      └─PCA
│  │          └─Normalize
│  ├─gene # Gene analysis (Not important, pls ignore)
│  │  └─...
│  ├─Images # Training Images, store based on patient and MRI series
│  │  ├─W10_FLAIR_AX
│  │  └─...
│  ├─Masks # Training Masks, store based on patient and MRI series
│  │  ├─W10_AX
│  │  └─...
│  ├─Mutual_Information # Mutual Information between feature
│  ├─result # Store 
│      └─FLAIR
│          ├─monitor
│          │  ├─test
│          │  ├─train
│          │  └─validation
│          ├─ROC_curve
│          ├─test
│          └─train
│  ├─feature_extraction.csv # radiomics feature, extracted by radiomics, plz check their doc
│  ├─GBM_MRI_Dataset.csv # contianing every slice location, using for training
│  ├─gene_expression_details.csv # gene expression energy
│  ├─tumor_details.csv # Basic Database info, containing every patients' subtype
│  └─Params.yaml # using for feature extraction, configure file for pyradiomics, plz check their doc
├─exception_in_trainning # Store error message, will send to your email.
├─model # Store model
│  ├─FLAIR
│  ├─Stack
│  ├─T1
│  └─T2
├─Pictures # Pictures for README
├─requirement # Store Python environment
├─train.py  # Training Main
├─data.py # Training dataset
├─utils.py # Training utils
├─unet.py # Training net
├─trainHelper.py # Help training module
├─classification.py # Classification main
├─mutual_information.py # mutual information analysis
├─feature_extraction.py # Get radiomics features from mask
├─get_graph_csv.py # Old version file, ignore it, used to get data in csv to replot fancy graph
├─generate_gif_results.py # Get gif, old version file, only use as reference, can not run directly
└─gene_expression_monitor.py # Get some statistic information about gene expression, old version code, you can ignore it
```

## How to use it

### Segmentation
Install python environment by requirements (pip)
```
pip install -r requirements.txt
```

GBM MRI image segmentation using `train.py, data.py, utils.py, unet.py, trainHelper.py`. The main python script is `train.py`, using command to run it:
```
python train.py [argv1] [argv2]
```
`argv1` is the MRI series you can choose, containing `T1, T2, FLAIR, Stack`. `argv2` is the epoches you want.

Also, see the `trainHelper.py`, you can use your own mail to remind you the train's situation.

### Classification

Just run `classification.py`

### Mutual Information analysis

Just run `mutual_information.py`

## Some results show

### Segmentation
![](Pictures\Segmentation_Table.png)
![](Pictures\Segmentation_Vis.png)

### Classification
![](Pictures\MLP.png)

### Mutual Information
![](Pictures\Radiomics_with_ctAreaGeneExpression.png)

## TODO List
- [] Change old version generate_gif_results.py to new version


