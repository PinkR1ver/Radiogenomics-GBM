import os
from PIL import Image
import numpy as np

dataPath = r'C:\Users\83549\Github Projects\Radiogenemics--on-Ivy-Gap\data\classification results'
testPath = os.path.join(dataPath, 'Simplified' ,'Test')
trainPath = os.path.join(dataPath, 'Train')

if __name__ == '__main__':
    a = Image.open(os.path.join(testPath, r'DecisionTrees ConfusionMatrix.png'))
    a = np.array(a)
    b = Image.open(os.path.join(testPath, r'Random Forest.png'))
    b = np.array(b)
    c = Image.open(os.path.join(testPath, 'Normalize', 'PCA', r'MLP.png'))
    c= np.array(c)
    e = Image.open(os.path.join(testPath, 'Normalize', 'PCA', r'SVM.png'))
    e= np.array(e)
    d = np.hstack((a, b, c, e))
    im = Image.fromarray(d)
    im.save(r'Results.png')