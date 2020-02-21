import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve
import click
from sklearn.neural_network import MLPClassifier
import seaborn as sn # beauty plot

colNames = ["GOS", "brain", "Synek", "N20", "III", "V", "III-V/I-III", "FOUR", "GCS", "lesion"]
#neurodata = pd.read_csv(r'~/exercise/pyEx/trainData1.csv', sep = " ", header=0, index_col=False, names= colNames) 
#testdata =  pd.read_csv(r'~/exercise/pyEx/testData.csv', sep = " ", header=0, index_col=False, names= colNames)
neurodata = pd.read_csv(r'~/exercise/pyEx/rawData.csv', sep = ",", header=0, index_col=False, names= colNames)

nPar = 9 # nPar-1, number of parameters for fitting
nParTot = 10 # total 10-1 pars, 
nrows = neurodata.shape[0]

#print("raw data")
#print("number of raw dataset: ", nrows)
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
   #print("Sample non-duplication probability = ",prob_nonDuplication)
   #print( "Expected test set number: ", int(kSample*prob_nonDuplication) )
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
   #print("number of test data: ", testSet.shape[0])
   #print("----test set (filtered set - train set)")
   #print(testSet)
   #print(testSet.to_string())
   trainSet.sort_index()
   #print("number of train data: ", trainSet.shape[0])
   #print("----train set --------------------")
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
#print("number of dataset after filtering: ", data_filtered.shape[0], ", ", nrows-data_filtered.shape[0], "are removed.")
#print(data_filtered)
#print(data_filtered.to_string())

trainSet = neurodata
testSet = neurodata

nBootstrap = 100
default_nBootstrap = 100 
nBoot = input("How many bootstraps do you want? (default=100) Press Enter\n")
if not nBoot:
   print("Alright, set", default_nBootstrap ,"tests by default")
   nBootstrap = default_nBootstrap
else:
   nBootstrap = int(nBoot)

auc_nBootstrap = []
coeff_nBootstrap = []
accuracy_nBootstrap = []
precision_nBootstrap = []
recall_nBootstrap = []

print("Waiting for bootstraping "+str(nBootstrap)+" samples ...")
for iBoot in range(nBootstrap):
   #   ### Separate the data into trainSet and testSet by using bootstrap method
   trainSet, testSet = bootstrap(data_filtered)
   
   features = trainSet.iloc[:,1:nPar]
   target = trainSet.iloc[:,0]
   #print(target)
  
   clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
   
   features = trainSet.iloc[:,1:nPar]
   target = trainSet.iloc[:,0]
   clf.fit(features, target)
   
   x_test = testSet.iloc[:,1:nPar]
   y_test = testSet.iloc[:,0]
   
   y_pred = clf.predict(x_test)

   #coeff = clf.coef_ # fitted parameters
   #print("trained coeffieciency = ", coeff)
   #coeff_nBootstrap.append(coeff) 
   testdata = testSet
   testdata_filtered = reject_outliers(testdata, colNames[0])
   
   for item in colNames[1:nParTot]:
       testdata_filtered = reject_outliers(testdata_filtered, item)
   #print("number of test data: ", testdata_filtered.shape[0], ",", testdata.shape[0] - testdata_filtered.shape[0], "are removed.")
   
   x_test = testdata_filtered.iloc[:,1:nPar]
   y_test = testdata_filtered.iloc[:,0]
   
   y_pred = clf.predict(x_test)
   cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

   accu = metrics.accuracy_score(y_test, y_pred)
   preci = metrics.precision_score(y_test, y_pred)
   recall = metrics.recall_score(y_test, y_pred)
   #print("Accuracy:", accu)
   #print("Precision:", preci)
   #print("Recall:", recall) 

   accuracy_nBootstrap.append(accu)
   precision_nBootstrap.append(preci)
   recall_nBootstrap.append(recall)

   #fig, ax = plt.subplots(figsize=(8, 8))
   #ax.imshow(cnf_matrix)
   #ax.grid(False)
   #ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
   #ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
   #ax.set_ylim(1.5, -0.5)
   
   #for i in range(2):
   #    for j in range(2):
   #        ax.text(j, i, cnf_matrix[i, j], ha='center', va='center', color='red')
   #plt.show()
   
   y_pred_proba = clf.predict_proba(x_test)[::,1]
   
   #ROC
   fpr, tpr, roc = roc_curve(y_test,  y_pred_proba, drop_intermediate=False)
   #plt.figure()
   ##Adding the ROC
   #plt.plot(fpr, tpr, color='red', lw=2, label="ROC curve of test data 1")#, roc="+str(roc))
   ##Random FPR and TPR
   #plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
   ##Title and label
   #plt.xlabel('FPR')
   #plt.ylabel('TPR')
   #plt.title('ROC curve')
   #plt.legend(loc=4)
   #plt.show()
   
   # AUC
   fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
   auc = metrics.roc_auc_score(y_test, y_pred_proba)
   #print("AUC", auc)
   auc_nBootstrap.append(auc)

   #plt.plot(fpr,tpr,label="AUC curve of test data 1, auc="+str(auc))
   #plt.legend(loc=4)
   #plt.show()
   
   #testSet.to_csv('sub_trainSet_%i.csv'%iBoot, index=False)
   #trainSet.to_csv('sub_testSet_%i.csv'%iBoot, index=False)

########### Now plot bootstrap results ##########################


precisionArray = np.array(precision_nBootstrap)
accuracyArray = np.array(accuracy_nBootstrap)
recallArray = np.array(recall_nBootstrap)
aucArray = np.array(auc_nBootstrap)
#coeffArray = np.array(coeff_nBootstrap)
print("------------------------------")
print("precision:", precision_nBootstrap)
print("best precision achieved:", max(precisionArray ), "dataSet id:", np.argmax(precisionArray))
print("------------------------------")
print("accuracy:", accuracy_nBootstrap)
print("best accuracy achieved:", max(accuracyArray ), "dataSet id:", np.argmax(accuracyArray))
print("------------------------------")
print("recall:", recall_nBootstrap)
print("best recall achieved:", max(recallArray ), "dataSet id:", np.argmax(recallArray))
print("------------------------------")
print("auc:", auc_nBootstrap)
print("best auc achieved:", max(aucArray), "dataSet id:", np.argmax(aucArray))


####### confidence levels #################
alpha = 0.683 #0.9954 # 0.683, 0.9954, 0.997, 0.999999426697 [1,2,3,5 sigma]
p_lo = ((1.0-alpha)/2.0)*100 
p_hi = (alpha+(1.0-alpha)/2.0)*100

lower_confident = max(0.0, np.percentile(accuracyArray, p_lo))
upper_confident = max(0.0, np.percentile(accuracyArray, p_hi))
print("The accuracy has a %.5f%% confidence interval of [%0.3f, %0.3f]" %(alpha*100, lower_confident, upper_confident))

lower_confident = max(0.0, np.percentile(aucArray, p_lo))
upper_confident = max(0.0, np.percentile(aucArray, p_hi))
print("The AUC has a %.5f%% confidence interval of [%0.3f, %0.3f]" %(alpha*100, lower_confident, upper_confident))

alpha = 0.683 #0.9954 # 0.683, 0.9954, 0.997, 0.999999426697 [1,2,3,5 sigma]
lower_confident = max(0.0, np.percentile(precisionArray, p_lo))
upper_confident = max(0.0, np.percentile(precisionArray, p_hi))
print("The precision has a %.5f%% confidence interval [%0.3f, %0.3f]" %(alpha*100, lower_confident, upper_confident))

plt.subplot(221)
plt.xlim([min(aucArray)-0.1, max(aucArray)+0.1])
plt.hist(aucArray, bins=50, alpha=1)
plt.title('AUC distributions')
#plt.xlabel('AUC values')
plt.ylabel('count')

plt.subplot(222)
plt.xlim([min(precisionArray)-0.1, max(precisionArray)+0.1])
plt.hist(precisionArray, bins=50, alpha=1)
plt.title('Precision distributions')
#plt.xlabel('Precision values')
plt.ylabel('count')

plt.subplot(223)
plt.xlim([min(accuracyArray)-0.1, max(accuracyArray)+0.1])
plt.hist(accuracyArray, bins=50, alpha=1)
plt.title('Accuracy distributions')
#plt.xlabel('Accuracy values')
plt.ylabel('count')


plt.subplot(224)
plt.xlim([min(aucArray)-0.1, max(aucArray)+0.1])
plt.hist(aucArray, bins=50, alpha=1)
plt.title('Recall distributions')
#plt.xlabel('Recall values')
plt.ylabel('count')
plt.suptitle('Distributions of bootstrap parameters from'+str(nBootstrap)+" bootstrap samples")
plt.show()

######## print fitted 8 parameters ###############
#par = []
#for i in range(8):
#  par.append(coeffArray[i][0])
#print(par[0])
#
#plt.subplot(331)
#plt.xlim([min(par[0])-1, max(par[0])+1])
#plt.hist(par[0], bins=100, alpha=1)
#plt.title('par[0]')
##plt.xlabel('Recall values')
#plt.ylabel('count')
#
#plt.subplot(332)
#plt.xlim([min(par[1])-1, max(par[1])+1])
#plt.hist(par[1], bins=100, alpha=1)
#plt.title('par[1]')
##plt.xlabel('Recall values')
#plt.ylabel('count')
#
#plt.subplot(333)
#plt.xlim([min(par[2])-1, max(par[2])+1])
#plt.hist(par[2], bins=100, alpha=1)
#plt.title('par[2]')
##plt.xlabel('Recall values')
#plt.ylabel('count')
#
#plt.subplot(334)
#plt.xlim([min(par[3])-1, max(par[3])+1])
#plt.hist(par[3], bins=100, alpha=1)
#plt.title('par[3]')
##plt.xlabel('Recall values')
#plt.ylabel('count')
#
#plt.subplot(335)
#plt.xlim([min(par[4])-1, max(par[4])+1])
#plt.hist(par[4], bins=100, alpha=1)
#plt.title('par[4]')
##plt.xlabel('Recall values')
#plt.ylabel('count')
#
#plt.subplot(336)
#plt.xlim([min(par[5])-1, max(par[5])+1])
#plt.hist(par[5], bins=100, alpha=1)
#plt.title('par[5]')
##plt.xlabel('Recall values')
#plt.ylabel('count')
#
#plt.subplot(337)
#plt.xlim([min(par[6])-1, max(par[6])+1])
#plt.hist(par[6], bins=100, alpha=1)
#plt.title('par[6]')
##plt.xlabel('Recall values')
#plt.ylabel('count')
#
#plt.subplot(338)
#plt.xlim([min(par[7])-1, max(par[7])+1])
#plt.hist(par[7], bins=100, alpha=1)
#plt.title('par[7]')
##plt.xlabel('Recall values')
#plt.ylabel('count')
#plt.suptitle('Distributions of bootstrap parameters from'+str(nBootstrap)+" bootstrap samples")

plt.show()
