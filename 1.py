import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X=data.data
y=data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1] #Only positive values is considered

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ",cm)

TN, FP, FN,TP = cm.ravel()

print("\nTP =", TP)
print("TN =", TN)
print("FP =", FP)
print("FN =", FN)

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = (2 * precision * recall)/(precision + recall)
specificity = TN/(TN+FP)
npv = TN/(TN+FN)
mcc = ((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

print("\n--- Custom Metrics ---")

print("Accuracy   :", accuracy)
print("Precision  :", precision)
print("Recall     :", recall)
print("F1 Score   :", F1)
print("Specificity:", specificity)
print("NPV        :", npv)
print("MCC        :", mcc)

print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Precision  :", precision_score(y_test,y_pred))
print("Recall     :", recall_score(y_test,y_pred))
print("F1 Score   :", f1_score(y_test,y_pred))
print("MCC        :", matthews_corrcoef(y_test,y_pred))

fpr,tpr,_ =roc_curve(y_test,y_prob)
model_auc = roc_auc_score(y_test,y_prob)

random_probs = np.random.rand(len(y_test))
rfpr,rtpr,_ = roc_curve(y_test,random_probs)
random_auc = roc_auc_score(y_test,random_probs)

plt.plot(fpr,tpr)
plt.plot(rfpr,rtpr,'--')
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve")
plt.savefig("roc.png")


print("Model AUC: ",model_auc)
print("Random AUC: ",random_auc)
