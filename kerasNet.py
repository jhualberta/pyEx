# windows: conda install graphviz after installed graphviz by pip install graphviz
# ubuntu: sudo apt-get install graphviz
# pip3 install keras
# pip3 install ann_visualizer
# pip3 install graphviz
# pip3 install tensorflow
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve
from random import sample 
import numpy as np  
from sklearn.neural_network import MLPClassifier
import seaborn as sn # beauty plot
from keras.models import Sequential
from keras.layers import Dense
import ann_visualizer
from ann_visualizer.visualize import ann_viz

colNames = ["GOS", "brain", "Synek", "N20", "III", "V", "III-V/I-III", "FOUR", "GCS", "lesion"]
trainSet = pd.read_csv(r'~/exercise/pyEx/trainSet.csv', sep = ",", header=0, index_col=False, names= colNames)
testSet = pd.read_csv(r'~/exercise/pyEx/testSet.csv', sep = ",", header=0, index_col=False, names= colNames)
nPar = 9 # nPar-1, number of parameters for fitting
nParTot = 10 # total 10-1 pars, 
nrows = trainSet.shape[0]

features = trainSet.iloc[:,1:nPar]
target = trainSet.iloc[:,0]

x_test = trainSet.iloc[:,1:nPar]
y_test = trainSet.iloc[:,0]

nlayer = [8,12,1] # default [12, 8, 1]
# create model based on trainSet
model = Sequential()
model.add(Dense(nlayer[0], input_dim=8, activation='relu'))
model.add(Dense(nlayer[1], activation='relu'))
model.add(Dense(nlayer[2], activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(np.array(features), np.array(target), epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(features, target)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
ann_viz(model, title="Medical index neural network")

y_pred = model.predict(x_test)
#print(x_test)

Y_pred = [1 if k > 0.5 else 0 for k in y_pred] # Ternary Operator in Python
#[k for k in y_pred]
# !!! Note: in neuro network, no binary output but continuous values 
print(Y_pred)
#
cnf_matrix = metrics.confusion_matrix(y_test, Y_pred) 
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred))
print("Precision:",metrics.precision_score(y_test, Y_pred))
print("Recall:",metrics.recall_score(y_test, Y_pred))
fig, ax = plt.subplots(figsize=(8, 8))
#ax.imshow(cnf_matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
#for i in range(2):
#    for j in range(2):
#        ax.text(j, i, cnf_matrix[i, j], ha='center', va='center', color='red')
sn.heatmap(cnf_matrix, annot=True)

plt.show()

#ROC
fpr, tpr, roc = roc_curve(y_test,  y_pred, drop_intermediate=False)
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label="ROC curve of test data 1")#, roc="+str(roc))
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.legend(loc=4)

# AUC
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
print("AUC", auc)
plt.plot(fpr,tpr,label="AUC curve of test data 1, auc="+str(auc))
plt.legend(loc=4)

plt.show()
