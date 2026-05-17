import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak'],
    'play': ['No', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)
print(df)

le = LabelEncoder()

for col in df.columns:
  df[col]= le.fit_transform(df[col])

print("Encoded: ")
print(df)

X = df.drop('play',axis=1)
y = df['play']

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
model = GaussianNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))

new = [[2,1,0,1]]
pred= model.predict(new)
if(pred[0]==1):
  print("play tennis")
else:
  print("No tennis")
