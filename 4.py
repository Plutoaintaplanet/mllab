import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

data = {
    'Outlook':['Sunny','Sunny','Overcast','Rainy','Rainy'],
    'Humidity': ['High', 'High', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Strong'],
    'Play':['No','No','Yes','Yes','No']
}

df= pd.DataFrame(data)
print("Dataset\n")
print(df)

le = LabelEncoder()
for col in df.columns:
  df[col]=le.fit_transform(df[col])

print("Encoded dataset")
print(df)

X=df.drop('Play',axis=1)
y=df['Play']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X,y)

new_data=[[2,0,1]]
pred = model.predict(new_data)

print("Predication:")
if(pred[0]==1):
  print("Can play tennis")
else:
  print("Not allowed to play tennis")

print("Decision Tree\n")
rules = tree.export_text(
    model,
    feature_names=X.columns.tolist()
)
print(rules)
