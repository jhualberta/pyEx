import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve
from random import sample 
import numpy as np  
# SPSS test value, not used
pred_p = [0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.25,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.25,0.91304,0.25,0.25,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304,0.91304]

# record is true, observe is predict

record = [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
observe = [1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1]

cnf_matrix = metrics.confusion_matrix(record, observe)
print("Accuracy:",metrics.accuracy_score(record, observe))
print("Precision:",metrics.precision_score(record, observe))
print("Recall:",metrics.recall_score(record,observe))
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

logreg = LogisticRegression()




#probabilities = logreg.predict(np.array(record))
#predictions = probabilities[:, 1]
#fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
#roc_auc = metrics.auc(fpr, tpr)
#
#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
#plt.legend(loc='lower right')
#
#plt.plot([0, 1], [0, 1], 'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()



#fpr, tpr, roc = roc_curve(observe, a)#, drop_intermediate=False)
#plt.figure()
###Adding the ROC
#plt.plot(fpr, tpr, color='red', lw=2, label="ROC curve of test data 1")
#
#fpr, tpr, _ = metrics.roc_curve(observe,  a)
#auc = metrics.roc_auc_score(observe, a)
#print("AUC", auc)
#plt.plot([0,1],[0,1],'r--')
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.plot(fpr,tpr,label="AUC curve of test data 1, auc="+str(auc))
#plt.legend(loc=4)
plt.show()
