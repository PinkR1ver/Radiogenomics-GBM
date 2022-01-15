import os
from statistics import mode
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from io import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
import warnings
from sklearn.exceptions import ConvergenceWarning

basePath = r''
dataPath = os.path.join(basePath, 'data')
subtypeFileName = 'tumor_details.csv'
featureFileName = 'feature_extraction.csv'
saveResultPath = os.path.join(dataPath, 'classification results')

if __name__ == '__main__':
    '''
    subtype = pd.read_csv(os.path.join(dataPath, subtypeFileName))
    subtype = subtype.replace(np.nan, '')
    subtype = subtype[subtype['molecular_subtype'] != '']
    subtype = subtype[['tumor_name', 'molecular_subtype']]
    subtype = subtype.reset_index(drop=True)
    print(subtype)
    for i in range(len(subtype)):
        subtype.at[i,'tumor_name'] = subtype.at[i,'tumor_name'].split('-')[0]
    subtype = subtype.rename(columns={'tumor_name':'Patient'})
    print(subtype.columns)

    featureTable = pd.read_csv(os.path.join(dataPath, featureFileName))
    print(featureTable.columns)
    subtype_list = subtype['Patient'].to_list()
    molecular_subtype = pd.DataFrame({}, columns=['molecular_subtype'])
    #featureTable = featureTable[featureTable['Patient'] in subtype['Patient'].to_list()]
    for i in range(len(featureTable)):
        if featureTable.at[i, 'Patient'] not in subtype_list:
            featureTable = featureTable.drop(index=i)
        else:
            for index, m_s in (subtype[subtype['Patient'] == featureTable.at[i, 'Patient']]).molecular_subtype.items():
                df_tmp = pd.DataFrame([[m_s]], columns=['molecular_subtype'])
            molecular_subtype = molecular_subtype.append(df_tmp, ignore_index=True)
    featureTable = featureTable.reset_index(drop=True)
    molecular_subtype = molecular_subtype.reset_index(drop=True)
    print(molecular_subtype)
    classificationDataset = pd.concat([featureTable, molecular_subtype], axis=1)
    #featureTable = featureTable.reset_index(drop=True)
    print(classificationDataset)
    classificationDataset.to_csv(os.path.join(dataPath, 'valid_data_to_classify.csv'), index=False)
    '''
    f = open(os.path.join(saveResultPath, 'log.txt'), "w")
    f.close()

    # loading Data
    classificationDataset = pd.read_csv(os.path.join(dataPath, 'valid_data_to_classify.csv'))
    classificationDataset = classificationDataset[classificationDataset['Slice'] < 150]
    classificationDataset = classificationDataset[classificationDataset['Slice'] > 50]
    classificationDataset = classificationDataset.reset_index(drop=True)
    feature_cols = classificationDataset.columns[6:-1]
    X = classificationDataset[feature_cols]
    y = classificationDataset.molecular_subtype



    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% for training and 30% for testing

    # Test Normalize X-train and PCA normalize X_train  and normalize PCA X_train
    normal_X_train = StandardScaler().fit_transform(X_train)

    # print(np.mean(normal_X_train), np.std(normal_X_train)) #this sentence can check normalize results, the mean should be 0, the std should be 1

    ## Normalize First
    normal_X_train = pd.DataFrame(normal_X_train, columns=feature_cols)
    
    normal_X_test = StandardScaler().fit_transform(X_test)
    normal_X_test = pd.DataFrame(normal_X_test, columns=feature_cols)

    ## Example to print PCA graph

    #pca = PCA(n_components=2)
    #pca_normal_X_train = pca.fit_transform(normal_X_train)
    #pca_normal_X_train = pd.DataFrame(pca_normal_X_train, columns = ['principal component 1', 'principal component 2'])
    #pca_normal_full_dataset = pca_normal_X_train
    #pca_normal_full_dataset['molecular_subtype'] = y_train

    
    #plt.figure(figsize=(10,10))
    #sns.scatterplot(
    #   x="principal component 1", y="principal component 2",
    #    hue="molecular_subtype",
    #    palette=sns.color_palette("hls", 8),
    #    data=pca_normal_full_dataset,
    #    legend="full",
    #    alpha=0.3
    #)
    #plt.savefig(os.path.join(saveResultPath, 'PCA Normalize X_train to 2 dims.png'))

    ## PCA after normalize
    pca = PCA(0.99)
    pca = pca.fit(normal_X_train)
    pca_normal_X_train = pca.transform(normal_X_train)
    print(f'PCA Dataset shape:{pca_normal_X_train.shape}') # detect pca features
    print(f'PCA variance ratio:{pca.explained_variance_ratio_}') # detect information loss

    ### Detect pca number to write pca columns
    pca_cols = []
    for i in range(pca_normal_X_train.shape[1]):
        pca_cols.append(f'principal component {i+1}')

    pca_normal_X_train = pd.DataFrame(pca_normal_X_train, columns=pca_cols)

    pca_normal_X_test = pca.transform(normal_X_test)
    pca_normal_X_test = pd.DataFrame(pca_normal_X_test, columns=pca_cols)


    # PCA First
    pca = pca.fit(X_train)
    pca_X_train = pca.transform(X_train)
    
    pca_cols = []
    for i in range(pca_X_train.shape[1]):
        pca_cols.append(f'principal component {i+1}')
    
    pca_X_train = pd.DataFrame(pca_X_train, columns=pca_cols)

    pca_X_test = pca.transform(X_test)
    pca_X_test = pd.DataFrame(pca_X_test, columns=pca_cols)

    ## Normalize after PCA
    normal_pca_X_train = StandardScaler().fit_transform(pca_X_train)
    normal_pca_X_train = pd.DataFrame(normal_pca_X_train, columns=pca_cols)

    normal_pca_X_test = StandardScaler().fit_transform(pca_X_test)
    normal_pca_X_test = pd.DataFrame(normal_pca_X_test, columns=pca_cols)




    #Build Decision Tree classifer to original data
    DecisionTree_clf = DecisionTreeClassifier(criterion='entropy')
    DecisionTree_clf = DecisionTree_clf.fit(X_train, y_train)

    y_pred = DecisionTree_clf.predict(X_test)
    y_pred_in_train = DecisionTree_clf.predict(X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Decision Trees to original Data in TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')
   
    f.write('Decision Trees to original Data in TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')

    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=DecisionTree_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=DecisionTree_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    #Build Decision Tree classifer to normalize data
    DecisionTree_clf = DecisionTreeClassifier(criterion='entropy')
    DecisionTree_clf = DecisionTree_clf.fit(normal_X_train, y_train)

    y_pred = DecisionTree_clf.predict(normal_X_test)
    y_pred_in_train = DecisionTree_clf.predict(normal_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Decision Trees to normalize Data in TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')
   
    f.write('Decision Trees to normalize Data in TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')

    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=DecisionTree_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=DecisionTree_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'Normalize','DecisionTrees ConfusionMatrix.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train', 'Normalize', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()


    #Build Decision Tree classifer to PCA data
    DecisionTree_clf = DecisionTreeClassifier(criterion='entropy')
    DecisionTree_clf = DecisionTree_clf.fit(pca_X_train, y_train)

    y_pred = DecisionTree_clf.predict(pca_X_test)
    y_pred_in_train = DecisionTree_clf.predict(pca_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Decision Trees to PCA Data in TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')
   
    f.write('Decision Trees to PCA Data in TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')

    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=DecisionTree_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=DecisionTree_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'PCA','DecisionTrees ConfusionMatrix.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train', 'PCA', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    #Build Decision Tree classifer to PCA Normalized data
    DecisionTree_clf = DecisionTreeClassifier(criterion='entropy')
    DecisionTree_clf = DecisionTree_clf.fit(pca_normal_X_train, y_train)

    y_pred = DecisionTree_clf.predict(pca_normal_X_test)
    y_pred_in_train = DecisionTree_clf.predict(pca_normal_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Decision Trees to PCA Normalized Data in TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')
   
    f.write('Decision Trees to PCA Normalized  Data in TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')

    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=DecisionTree_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=DecisionTree_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'Normalize', 'PCA', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train', 'Normalize', 'PCA', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    #Build Decision Tree classifer to PCA Normalized data
    DecisionTree_clf = DecisionTreeClassifier(criterion='entropy')
    DecisionTree_clf = DecisionTree_clf.fit(normal_pca_X_train, y_train)

    y_pred = DecisionTree_clf.predict(normal_pca_X_test)
    y_pred_in_train = DecisionTree_clf.predict(normal_pca_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Decision Trees to Normalized PCA Data in TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')
   
    f.write('Decision Trees to Normalized PCA Data in TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')

    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=DecisionTree_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=DecisionTree_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'PCA', 'Normalize', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train', 'PCA', 'Normalize', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    #Build RandomForest classifer to Original data
    RandomForest_clf = RandomForestClassifier()
    RandomForest_clf = RandomForest_clf.fit(X_train, y_train)

    y_pred = RandomForest_clf.predict(X_test)
    y_pred_in_train = RandomForest_clf.predict(X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Random Forest to orginal Data in TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
   
    f.write('Random Forest to original Data in TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=RandomForest_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=RandomForest_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'Random Forest.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train', 'Random Forest.png'))
    plt.close()

    #Build RandomForest classifer to Original data, adding tree numbers
    RandomForest_clf = RandomForestClassifier(500)
    RandomForest_clf = RandomForest_clf.fit(X_train, y_train)

    y_pred = RandomForest_clf.predict(X_test)
    y_pred_in_train = RandomForest_clf.predict(X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Random Forest to orginal Data in TestDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
   
    f.write('Random Forest to original Data in TrainDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=RandomForest_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=RandomForest_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'Random Forest, TreeNumbers=500.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train', 'Random Forest, TreeNumbers=500.png'))
    plt.close()

    #Build RandomForest classifer to normalized data, adding tree numbers
    RandomForest_clf = RandomForestClassifier(500)
    RandomForest_clf = RandomForest_clf.fit(normal_X_train, y_train)

    y_pred = RandomForest_clf.predict(normal_X_test)
    y_pred_in_train = RandomForest_clf.predict(normal_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Random Forest to normalized Data in TestDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
   
    f.write('Random Forest to normalized Data in TrainDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=RandomForest_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=RandomForest_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'Normalize','Random Forest, TreeNumbers=500.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train', 'Normalize', 'Random Forest, TreeNumbers=500.png'))
    plt.close()
    
    #Build RandomForest classifer to PCA normalized data, adding tree numbers
    RandomForest_clf = RandomForestClassifier(500)
    RandomForest_clf = RandomForest_clf.fit(pca_normal_X_train, y_train)

    y_pred = RandomForest_clf.predict(pca_normal_X_test)
    y_pred_in_train = RandomForest_clf.predict(pca_normal_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Random Forest to PCA normalized Data in TestDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
   
    f.write('Random Forest to PCA normalized Data in TrainDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=RandomForest_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=RandomForest_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'Normalize', 'PCA', 'Random Forest, TreeNumbers=500.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train', 'Normalize', 'PCA', 'Random Forest, TreeNumbers=500.png'))
    plt.close()


    #Build SVM classifer to PCA normalized data
    SVM_clf = svm.SVC(kernel='linear', C=1.0)
    SVM_clf = SVM_clf.fit(pca_normal_X_train, y_train)

    y_pred = SVM_clf.predict(pca_normal_X_test)
    y_pred_in_train = SVM_clf.predict(pca_normal_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=SVM_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('SVM to PCA normalized Data in TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=SVM_clf.classes_))
    f.write('\n\n')
   
    f.write('SVM to PCA normalized Data in TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=SVM_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=SVM_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=SVM_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=SVM_clf.classes_, yticklabels=SVM_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'Normalize', 'PCA', 'SVM.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=SVM_clf.classes_, yticklabels=SVM_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train', 'Normalize', 'PCA', 'SVM.png'))
    plt.close()

    #Bulid NN to classify, data must be normalize
    MLP_clf = MLPClassifier(hidden_layer_sizes=(1024,1024,512,256,64,32), max_iter=500)
    MLP_clf = MLP_clf.fit(normal_X_train, y_train)

    y_pred = MLP_clf.predict(normal_X_test)
    y_pred_in_train = MLP_clf.predict(normal_X_train)

    print(metrics.classification_report(y_test, y_pred, target_names=MLP_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('MLP to Normalized Data in TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=MLP_clf.classes_))
    f.write('\n\n')
   
    f.write('MLP to Normalized Data in TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=MLP_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=MLP_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=MLP_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=MLP_clf.classes_, yticklabels=MLP_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'Normalize', 'MLP.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=MLP_clf.classes_, yticklabels=MLP_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train',  'Normalize', 'MLP.png'))
    plt.close()


    # Bulid NN to classify, data must be normalize
    MLP_clf = MLPClassifier(hidden_layer_sizes=(256,256,128,64))
    MLP_clf = MLP_clf.fit(pca_normal_X_train, y_train)

    y_pred = MLP_clf.predict(pca_normal_X_test)
    y_pred_in_train = MLP_clf.predict(pca_normal_X_train)

    print(metrics.classification_report(y_test, y_pred, target_names=MLP_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('MLP to PCA Normalized Data in TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=MLP_clf.classes_))
    f.write('\n\n')
   
    f.write('MLP to PCA Normalized Data in TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=MLP_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=MLP_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=MLP_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=MLP_clf.classes_, yticklabels=MLP_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Test', 'Normalize', 'PCA', 'MLP.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=MLP_clf.classes_, yticklabels=MLP_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Train',  'Normalize', 'PCA', 'MLP.png'))
    plt.close()

    # -------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------



    ## Simplify Dataset to single type
    simlified_subtype_list = ['Classical', 'Proneural', 'Mesenchymal', 'Neural']
    simlified_classificationDataset = pd.DataFrame({}, columns=classificationDataset.columns)
    for i in range(len(classificationDataset)):
        if classificationDataset.at[i, 'molecular_subtype'] not in simlified_subtype_list:
            classificationDataset = classificationDataset.drop(index=i)
    simlified_classificationDataset = classificationDataset.reset_index(drop=True)
    X = simlified_classificationDataset[feature_cols]
    y = simlified_classificationDataset.molecular_subtype

    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% for training and 30% for testing

    # Test Normalize X-train and PCA normalize X_train  and normalize PCA X_train
    normal_X_train = StandardScaler().fit_transform(X_train)

    # print(np.mean(normal_X_train), np.std(normal_X_train)) #this sentence can check normalize results, the mean should be 0, the std should be 1

    ## Normalize First
    normal_X_train = pd.DataFrame(normal_X_train, columns=feature_cols)
    
    normal_X_test = StandardScaler().fit_transform(X_test)
    normal_X_test = pd.DataFrame(normal_X_test, columns=feature_cols)

    ## PCA after normalize
    pca = PCA(0.99)
    pca = pca.fit(normal_X_train)
    pca_normal_X_train = pca.transform(normal_X_train)
    print(f'PCA Dataset shape:{pca_normal_X_train.shape}') # detect pca features
    print(f'PCA variance ratio:{pca.explained_variance_ratio_}') # detect information loss

    ### Detect pca number to write pca columns
    pca_cols = []
    for i in range(pca_normal_X_train.shape[1]):
        pca_cols.append(f'principal component {i+1}')

    pca_normal_X_train = pd.DataFrame(pca_normal_X_train, columns=pca_cols)

    pca_normal_X_test = pca.transform(normal_X_test)
    pca_normal_X_test = pd.DataFrame(pca_normal_X_test, columns=pca_cols)


    # PCA First
    pca = pca.fit(X_train)
    pca_X_train = pca.transform(X_train)
    
    pca_cols = []
    for i in range(pca_X_train.shape[1]):
        pca_cols.append(f'principal component {i+1}')
    
    pca_X_train = pd.DataFrame(pca_X_train, columns=pca_cols)

    pca_X_test = pca.transform(X_test)
    pca_X_test = pd.DataFrame(pca_X_test, columns=pca_cols)

    ## Normalize after PCA
    normal_pca_X_train = StandardScaler().fit_transform(pca_X_train)
    normal_pca_X_train = pd.DataFrame(normal_pca_X_train, columns=pca_cols)

    normal_pca_X_test = StandardScaler().fit_transform(pca_X_test)
    normal_pca_X_test = pd.DataFrame(normal_pca_X_test, columns=pca_cols)

    


    #Build Decision Tree classifer to original data
    DecisionTree_clf = DecisionTreeClassifier(criterion='entropy')
    DecisionTree_clf = DecisionTree_clf.fit(X_train, y_train)

    y_pred = DecisionTree_clf.predict(X_test)
    y_pred_in_train = DecisionTree_clf.predict(X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Decision Trees to original Data in Simplified TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')
   
    f.write('Decision Trees to original Data in Simplified TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')

    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=DecisionTree_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=DecisionTree_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    #Build Decision Tree classifer to normalize data
    DecisionTree_clf = DecisionTreeClassifier(criterion='entropy')
    DecisionTree_clf = DecisionTree_clf.fit(normal_X_train, y_train)

    y_pred = DecisionTree_clf.predict(normal_X_test)
    y_pred_in_train = DecisionTree_clf.predict(normal_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Decision Trees to normalize Data in Simplified TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')
   
    f.write('Decision Trees to normalize Data in Simplified TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')

    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=DecisionTree_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=DecisionTree_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'Normalize','DecisionTrees ConfusionMatrix.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train', 'Normalize', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()


    #Build Decision Tree classifer to PCA data
    DecisionTree_clf = DecisionTreeClassifier(criterion='entropy')
    DecisionTree_clf = DecisionTree_clf.fit(pca_X_train, y_train)

    y_pred = DecisionTree_clf.predict(pca_X_test)
    y_pred_in_train = DecisionTree_clf.predict(pca_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Decision Trees to PCA Data in Simplified TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')
   
    f.write('Decision Trees to PCA Data in Simplified TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')

    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=DecisionTree_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=DecisionTree_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'PCA','DecisionTrees ConfusionMatrix.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train', 'PCA', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    #Build Decision Tree classifer to PCA Normalized data
    DecisionTree_clf = DecisionTreeClassifier(criterion='entropy')
    DecisionTree_clf = DecisionTree_clf.fit(pca_normal_X_train, y_train)

    y_pred = DecisionTree_clf.predict(pca_normal_X_test)
    y_pred_in_train = DecisionTree_clf.predict(pca_normal_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Decision Trees to PCA Normalized Data in Simplified TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')
   
    f.write('Decision Trees to PCA Normalized  Data in Simplified TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')

    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=DecisionTree_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=DecisionTree_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'Normalize', 'PCA', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train', 'Normalize', 'PCA', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    #Build Decision Tree classifer to PCA Normalized data
    DecisionTree_clf = DecisionTreeClassifier(criterion='entropy')
    DecisionTree_clf = DecisionTree_clf.fit(normal_pca_X_train, y_train)

    y_pred = DecisionTree_clf.predict(normal_pca_X_test)
    y_pred_in_train = DecisionTree_clf.predict(normal_pca_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Decision Trees to Normalized PCA Data in Simplified TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')
   
    f.write('Decision Trees to Normalized PCA Data in Simplified TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=DecisionTree_clf.classes_))
    f.write('\n\n')

    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=DecisionTree_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=DecisionTree_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'PCA', 'Normalize', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=DecisionTree_clf.classes_, yticklabels=DecisionTree_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train', 'PCA', 'Normalize', 'DecisionTrees ConfusionMatrix.png'))
    plt.close()

    #Build RandomForest classifer to Original data
    RandomForest_clf = RandomForestClassifier()
    RandomForest_clf = RandomForest_clf.fit(X_train, y_train)

    y_pred = RandomForest_clf.predict(X_test)
    y_pred_in_train = RandomForest_clf.predict(X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Random Forest to orginal Data in Simplified TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
   
    f.write('Random Forest to original Data in Simplified TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=RandomForest_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=RandomForest_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'Random Forest.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train', 'Random Forest.png'))
    plt.close()

    #Build RandomForest classifer to Original data, adding tree numbers
    RandomForest_clf = RandomForestClassifier(500)
    RandomForest_clf = RandomForest_clf.fit(X_train, y_train)

    y_pred = RandomForest_clf.predict(X_test)
    y_pred_in_train = RandomForest_clf.predict(X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Random Forest to orginal Data in Simplified TestDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
   
    f.write('Random Forest to original Data in Simplified TrainDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=RandomForest_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=RandomForest_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'Random Forest, TreeNumbers=500.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train', 'Random Forest, TreeNumbers=500.png'))
    plt.close()

    #Build RandomForest classifer to normalized data, adding tree numbers
    RandomForest_clf = RandomForestClassifier(500)
    RandomForest_clf = RandomForest_clf.fit(normal_X_train, y_train)

    y_pred = RandomForest_clf.predict(normal_X_test)
    y_pred_in_train = RandomForest_clf.predict(normal_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Random Forest to normalized Data in Simplified TestDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
   
    f.write('Random Forest to normalized Data in Simplified TrainDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=RandomForest_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=RandomForest_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'Normalize','Random Forest, TreeNumbers=500.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train', 'Normalize', 'Random Forest, TreeNumbers=500.png'))
    plt.close()
    
    #Build RandomForest classifer to PCA normalized data, adding tree numbers
    RandomForest_clf = RandomForestClassifier(500)
    RandomForest_clf = RandomForest_clf.fit(pca_normal_X_train, y_train)

    y_pred = RandomForest_clf.predict(pca_normal_X_test)
    y_pred_in_train = RandomForest_clf.predict(pca_normal_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('Random Forest to PCA normalized Data in Simplified TestDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
   
    f.write('Random Forest to PCA normalized Data in Simplified TrainDataset, adding tree numbers to 500: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=RandomForest_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=RandomForest_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=RandomForest_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'Normalize', 'PCA', 'Random Forest, TreeNumbers=500.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=RandomForest_clf.classes_, yticklabels=RandomForest_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train', 'Normalize', 'PCA', 'Random Forest, TreeNumbers=500.png'))
    plt.close()


    #Build SVM classifer to PCA normalized data
    SVM_clf = svm.SVC(kernel='linear', C=1.0)
    SVM_clf = SVM_clf.fit(pca_normal_X_train, y_train)

    y_pred = SVM_clf.predict(pca_normal_X_test)
    y_pred_in_train = SVM_clf.predict(pca_normal_X_train)
    #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=SVM_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('SVM to PCA normalized Data in Simplified TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=SVM_clf.classes_))
    f.write('\n\n')
   
    f.write('SVM to PCA normalized Data in Simplified TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=SVM_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=SVM_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=SVM_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=SVM_clf.classes_, yticklabels=SVM_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'Normalize', 'PCA', 'SVM.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=SVM_clf.classes_, yticklabels=SVM_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train', 'Normalize', 'PCA', 'SVM.png'))
    plt.close()

    #Bulid NN to classify, data must be normalize
    MLP_clf = MLPClassifier(hidden_layer_sizes=(1024,1024,512,256,64,32), max_iter=500)
    MLP_clf = MLP_clf.fit(normal_X_train, y_train)

    y_pred = MLP_clf.predict(normal_X_test)
    y_pred_in_train = MLP_clf.predict(normal_X_train)

    print(metrics.classification_report(y_test, y_pred, target_names=MLP_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('MLP to Normalized Data in Simplified TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=MLP_clf.classes_))
    f.write('\n\n')
   
    f.write('MLP to Normalized Data in Simplified TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=MLP_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=MLP_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=MLP_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=MLP_clf.classes_, yticklabels=MLP_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'Normalize', 'MLP.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=MLP_clf.classes_, yticklabels=MLP_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train',  'Normalize', 'MLP.png'))
    plt.close()

    # Bulid NN to classify, data must be normalize
    MLP_clf = MLPClassifier(hidden_layer_sizes=(256,256,128,64))
    MLP_clf = MLP_clf.fit(pca_normal_X_train, y_train)

    y_pred = MLP_clf.predict(pca_normal_X_test)
    y_pred_in_train = MLP_clf.predict(pca_normal_X_train)

    print(metrics.classification_report(y_test, y_pred, target_names=MLP_clf.classes_))

    f = open(os.path.join(saveResultPath, 'log.txt'), "a")

    f.write('MLP to PCA Normalized Data in Simplified TestDataset: \n')
    f.write(metrics.classification_report(y_test, y_pred, target_names=MLP_clf.classes_))
    f.write('\n\n')
   
    f.write('MLP to PCA Normalized Data in Simplified TrainDataset: \n')
    f.write(metrics.classification_report(y_train, y_pred_in_train, target_names=MLP_clf.classes_))
    f.write('\n\n')
    f.close()


    cm_test = confusion_matrix(y_test, y_pred, labels=MLP_clf.classes_)
    cm_train = confusion_matrix(y_train, y_pred_in_train, labels=MLP_clf.classes_)


    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_test / np.sum(cm_test), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=MLP_clf.classes_, yticklabels=MLP_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Test', 'Normalize', 'PCA', 'MLP.png'))
    plt.close()

    fig = plt.figure(figsize=(16, 16))
    sns.heatmap(cm_train / np.sum(cm_train), cmap="YlGnBu", annot=True, fmt='.2%', square=1, linewidth=2., xticklabels=MLP_clf.classes_, yticklabels=MLP_clf.classes_)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(saveResultPath, 'Simplified', 'Train',  'Normalize', 'PCA', 'MLP.png'))
    plt.close()





