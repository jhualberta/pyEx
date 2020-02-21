import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve
import click

colNames = ["GOS", "brain", "Synek", "N20", "III", "V", "III-V/I-III", "FOUR", "GCS", "lesion"]
#neurodata = pd.read_csv(r'~/exercise/pyEx/trainData1.csv', sep = " ", header=0, index_col=False, names= colNames) 
#testdata =  pd.read_csv(r'~/exercise/pyEx/testData.csv', sep = " ", header=0, index_col=False, names= colNames)
neurodata = pd.read_csv(r'~/exercise/pyEx/rawData.csv', sep = ",", header=0, index_col=False, names= colNames)

nPar = 9 # nPar-1, number of parameters for fitting
nParTot = 10 # total 10-1 pars, 
nrows = neurodata.shape[0]

#print("raw data")
print("number of raw dataset: ", nrows)
#print(neurodata)
#print("use parameters: ", neurodata.columns)

#inputData = neurodata.iloc[:,1:9]
#outputData = neurodata['GOS']

def fillNan(data):
    # for the missing data (NaN), replace them by mode/mean, interpolate values
    valName = ""
    for title in colNames:
        val = data[title].mode()[0]
        valName = "mode"
        #val = data[title].mean()
        #valName = "mean"
        data[title] = data[title].fillna(val)
        #print("NaNs in", title.ljust(20,'-'),"are replaced by", valName, val) 
    data_filtered = data
    return data_filtered

def reject_outliers(data, item):
    # remove all the records containing NaN values
    data.fillna(1000, inplace=True)
    data_filtered = data[data[item]!=1000] 
    return data_filtered

def bootstrap( data ):
   ######## use bootstrap method to select train/test data (after filtering) #######
   kSample = data.shape[0]
   ## Note: the below calculation can be dangerous if kSample is very large!
   prob_nonDuplication = (1-1./kSample)**kSample
   print("Sample non-duplication probability = ",prob_nonDuplication)
   print( "Expected test set number: ", int(kSample*prob_nonDuplication) )
   sampleRow = []
   for k in range(kSample):
     random_subset = data.sample(n=1,replace=True)# sample and put back
     #print(random_subset)
     sampleRow.append(random_subset)
   
   trainSet = pd.concat(sampleRow)
   #trainSet.drop_duplicates(keep='first', inplace=True)
   
   #print("-----------random sampled index---------")
   #print(trainSet.index)
   testSet = data.drop(trainSet.index)
   trainSet = data.drop(testSet.index)
   testSet.sort_index()
   print("number of test data: ", testSet.shape[0])
   print("----test set (filtered set - train set)")
   #print(testSet)
   #print(testSet.to_string())
   trainSet.sort_index()
   print("number of train data: ", trainSet.shape[0])
   print("----train set --------------------")
   #print(trainSet)
   #print(trainSet.to_string())
   return( trainSet, testSet)

### for the raw data, we can filter all the records with NaN values
### or we can replace NaN with means/modes/interpolate

## filtering N/A values
#data_filtered = reject_outliers(neurodata, colNames[0]) # check row by row

## replace the N/A values by mode
data_filtered = fillNan(neurodata)

for item in colNames[1:nParTot]:
    data_filtered = reject_outliers(data_filtered, item)
print("number of dataset after filtering: ", data_filtered.shape[0], ", ", nrows-data_filtered.shape[0], "are removed.")
#print(data_filtered)
#print(data_filtered.to_string())

trainSet = neurodata
testSet = neurodata

## use saved train, test set or bootstrap sampling again ########################################
saveKey = False
if click.confirm("Use the saved tables? (Make sure they exist)", default=True):
   saveKey = True
   trainSet = pd.read_csv(r'trainSet.csv', sep = ",", header=0, index_col=False, names= colNames)
   testSet = pd.read_csv(r'testSet.csv', sep = ",", header=0, index_col=False, names= colNames)
else:
   ### Separate the data into trainSet and testSet by using bootstrap method
   trainSet, testSet = bootstrap(data_filtered)

features = trainSet.iloc[:,1:nPar]
target = trainSet.iloc[:,0]
#print(target)

logreg = LogisticRegression()
result = logreg.fit(features, target)
print("trained coeffieciency = ", logreg.coef_)

testdata = testSet
testdata_filtered = reject_outliers(testdata, colNames[0])

for item in colNames[1:nParTot]:
    testdata_filtered = reject_outliers(testdata_filtered, item)
print("number of test data: ", testdata_filtered.shape[0], ",", testdata.shape[0] - testdata_filtered.shape[0], "are removed.")

x_test = testdata_filtered.iloc[:,1:nPar]
y_test = testdata_filtered.iloc[:,0]

y_pred = logreg.predict(x_test)
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

for i in range(2):
    for j in range(2):
        ax.text(j, i, cnf_matrix[i, j], ha='center', va='center', color='red')
#plt.show()

y_pred_proba = logreg.predict_proba(x_test)[::,1]

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

################### classifiers ########################
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
# check parameters 2 by 2

# assign which two parameters you want to check, pick up in range: [1, 8]
# colNames = ["GOS", "brain", "Synek", "N20", "III", "V", "III-V/I-III", "FOUR", "GCS", "lesion"]
#              0      1        2        3      4      5    6              7       8      9
par1 = 1 
par2 = 2 

featuresCheck = data_filtered.iloc[:, [par1, par2]]
target = data_filtered.iloc[:,0]
clf = LogisticRegression().fit(featuresCheck, target)
h = .02  # step size in the mesh
x_min, x_max = float(x_test[colNames[par1]].min()) - 1, float(x_test[colNames[par1]].max()) + 1
y_min, y_max = float(x_test[colNames[par2]].min()) - 1, float(x_test[colNames[par2]].max()) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.title("Decision surface of LogisticRegression %s, %s " %(colNames[par1],colNames[par2]) )
plt.axis('tight')

plt.scatter(featuresCheck[colNames[par1]], featuresCheck[colNames[par2]], c=target, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel(colNames[par1])
plt.ylabel(colNames[par2])

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

#plt.show()

if saveKey == False:
   if click.confirm("Saved the tables?", default=True):
      testSet.to_csv('trainSet.csv', index=False) #, compression=compression_opts)
      trainSet.to_csv('testSet.csv', index=False) #, compression=compression_opts)
else:
   print("You used the saved data, no need to save them again. Good bye.") 
