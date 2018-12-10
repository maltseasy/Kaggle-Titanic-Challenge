import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np


data = pd.read_csv('train.csv')
test_d = pd.read_csv('test.csv')
data["Age"] = data["Age"].fillna(data["Age"].median())
y = data['Survived']
train,test = data.drop(["PassengerId","Name","Cabin","Ticket","Survived","Embarked"],axis=1), test_d.drop(["PassengerId","Name","Cabin","Ticket","Embarked"],axis=1)
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

def sexdiscrim(x):
    if (x=="male"):
        return 1
    else:
        return 0
def farediscrim(x):
    if (x < 30):
        return 0
    elif (x < 50):
        return 1
    elif (x<250):
        return 2
    elif (x>=250):
        return 3
def agediscrim(x):
    if (x < 10):
        return 0
    elif (x < 20):
        return 1
    elif (x<50):
        return 2
    elif (x>=50):
        return 3

for a in [train, test]:
    a['Age'] = a['Age'].apply(agediscrim)

for a in [train, test]:
    a['Sex'] = a['Sex'].apply(sexdiscrim)

for a in [train, test]:
    a['Fare'] = a['Fare'].apply(farediscrim)

for a in [train, test]:
    a['FS'] = a['Parch'] + a['SibSp']

for a in [train, test]:
    a['FamilySize'] = a['Parch'] + a['SibSp']

a["Pclass"] = a["Pclass"].fillna(0)
a["Sex"] = a["Sex"].fillna(0)
a["Age"] = a["Age"].fillna(2)
a["SibSp"] = a["SibSp"].fillna(0)
a["Parch"] = a["Parch"].fillna(0)
a["Fare"] = a["Fare"].fillna(0)
a["FamilySize"] = a["FamilySize"].fillna(0)

train_y = data["Survived"].ravel()
train_x = train
for a in [train_x, test]:
    for col in a.columns:
        a[col] = a[col].astype('int')

xt, xv, yt, yv = train_test_split(train_x, train_y, test_size=0.3)
x_test = test.copy()

xt.index = np.arange(len(xt))
xv.index = np.arange(len(xv))

log_regression = LogisticRegression()
log_regression.fit(xt,yt)

preds = log_regression.predict(test)
rd = {
    "Passenger": test_d.PassengerId,
    "Survived": preds
}

df = pd.DataFrame(rd, columns = ['PassengerId', 'Survived'])
#df.to_csv("out.csv")