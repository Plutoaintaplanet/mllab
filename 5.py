import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Dataset URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Column names for dataset
columns = ['Pregnancies', 'Glucose', 'BloodPressure',
           'SkinThickness', 'Insulin', 'BMI',
           'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(url,names=columns)
print(df)

X = df.drop('Outcome',axis=1)
y = df['Outcome'].values

X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def KNN(X_train,y_train, X_test, k):
  predications = []

  for test in X_test:
    distances=np.sqrt(np.sum((X_train - test)**2,axis=1))
    k_labels = y_train[np.argsort(distances)[:k]]
    pred = np.bincount(k_labels).argmax()

    predications.append(pred)

  return predications

for k in [3,5,7,9]:
  y_pred = KNN(X_train,y_train,X_test,k)

  accuracy = accuracy_score(y_test,y_pred)
  print("K=",k, " Accuracy=",accuracy*100)


sample = [X_test[0]]
pred = KNN(X_train,y_train,X_test,7)

if(pred[0]==1):
  print("Person has diabetes")

else:
  print("Person doesnt have diabetes")
