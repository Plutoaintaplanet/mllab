import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error

X = np.array([
    [2,3,1,5],
    [3,4,2,6],
    [4,5,3,7],
    [5,6,4,8],
    [6,7,5,9],
    [7,8,6,10]
])

y_class = np.array([0,0,0,1,1,1])
y_reg = np.array([10,12,14,16,18,20])

X_train, X_test, yc_train, yc_test = train_test_split(X,y_class,test_size=0.3,random_state=1)
_,_,yr_train,yr_test = train_test_split(X,y_reg,test_size=0.3,random_state=1)

log_model = LogisticRegression()
log_model.fit(X_train,yc_train)
y_pred_class = log_model.predict(X_test)

lin_model = LinearRegression()
lin_model.fit(X_train, yr_train)
y_pred_reg = lin_model.predict(X_test)

print("Confusion matrix: ",confusion_matrix(yc_test,y_pred_class))
print("Accuracy: ",accuracy_score(yc_test,y_pred_class))
print("Precision: ",precision_score(yc_test,y_pred_class))
print("Recall: ",recall_score(yc_test,y_pred_class))

mse = mean_squared_error(yr_test,y_pred_reg)
rmse = np.sqrt(mse)

print("MSE: ",mse)
print("RMSE: ",rmse)

new_data=np.array([[8,9,7,11]])

print("Logistic Regression: ",log_model.predict(new_data))
print("Linear Regression: ",lin_model.predict(new_data))
