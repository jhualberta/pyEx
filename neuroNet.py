import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve
from random import sample 
import numpy as np  
from sklearn.neural_network import MLPClassifier
import seaborn as sn # beauty plot
colNames = ["GOS", "brain", "Synek", "N20", "III", "V", "III-V/I-III", "FOUR", "GCS", "lesion"]
trainSet = pd.read_csv(r'~/exercise/pyEx/trainSet.csv', sep = ",", header=0, index_col=False, names= colNames)
testSet = pd.read_csv(r'~/exercise/pyEx/testSet.csv', sep = ",", header=0, index_col=False, names= colNames)
nPar = 9 # nPar-1, number of parameters for fitting
nParTot = 10 # total 10-1 pars, 
nrows = trainSet.shape[0]


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

features = trainSet.iloc[:,1:nPar]
target = trainSet.iloc[:,0]
clf.fit(features, target)

x_test = testSet.iloc[:,1:nPar]
y_test = testSet.iloc[:,0]

y_pred = clf.predict(x_test)
#print(y_pred)

print([coef.shape for coef in clf.coefs_])

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cnf_matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
sn.heatmap(cnf_matrix, annot=True)

y_pred_proba = clf.predict_proba(x_test)[::,1]

#ROC
fpr, tpr, roc = roc_curve(y_test,  y_pred_proba, drop_intermediate=False)
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
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC", auc)
plt.plot(fpr,tpr,label="AUC curve of test data 1, auc="+str(auc))
plt.legend(loc=4)

plt.show()
