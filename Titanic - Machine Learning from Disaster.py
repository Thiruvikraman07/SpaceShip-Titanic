#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('train.csv')
train.head()


# In[3]:


train.describe()


# In[4]:


train.info()


# In[5]:


train = train.drop(['Cabin','PassengerId','Name','Ticket'],axis=1)


# In[6]:


for i in range(0,891):
    if(train['Sex'][i]=='male'):
        train['Sex'][i] = 1
    else:
        train['Sex'][i] = 0
for i in range(0,891):
    if(train['Embarked'][i]=='C'):
        train['Embarked'][i] = 0
    elif(train['Embarked'][i]=='Q'):
        train['Embarked'][i] = 1
    elif(train['Embarked'][i]=='S'):
        train['Embarked'][i] = 3
train['Embarked']


# In[7]:


train.info()


# In[8]:


train = train.dropna()


# In[12]:


from sklearn.model_selection import train_test_split
X = train.drop('Survived',axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X,y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print("%f (%f) with: %r" % (mean, stdev, param))


# In[13]:


from sklearn.model_selection import train_test_split
X = train.drop('Survived',axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=101)
logmodel = LogisticRegression(solver='lbfgs',penalty='l2',C=0.1)
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[14]:


test = pd.read_csv('test.csv')
a = test
test = test.drop(['Cabin','PassengerId','Name','Ticket'],axis=1)
for i in range(0,418):
    if(test['Sex'][i]=='male'):
        test['Sex'][i] = 1
    else:
        test['Sex'][i] = 0
for i in range(0,418):
    if(test['Embarked'][i]=='C'):
        test['Embarked'][i] = 0
    elif(test['Embarked'][i]=='Q'):
        test['Embarked'][i] = 1
    elif(test['Embarked'][i]=='S'):
        test['Embarked'][i] = 3
import math as m
for i in range(0,418):
    if(m.isnan(test['Age'][i])== True):
        if(test['Pclass'][i]==1):
            test['Age'][i]= 37.0
        elif(test['Pclass'][i]==2):
            test['Age'][i]=30.0
        elif(test['Pclass'][i]==3):
            test['Age'][i]= 25.0
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
predictions = logmodel.predict(test)
import csv

with open('titanic.csv', 'w', newline='') as file:
    fieldnames = ['PassengerId', 'Survived']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(0,418):
        writer.writerow({'PassengerId': a['PassengerId'][i], 'Survived': predictions[i]})


# In[15]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[16]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)


# In[17]:


grid.best_params_


# In[18]:


grid.best_estimator_


# In[19]:


grid_predictions = grid.predict(X_test)


# In[20]:


print(classification_report(y_test,grid_predictions))


# In[28]:


predictions = grid.predict(test)


# In[30]:


with open('titanic.csv', 'w', newline='') as file:
    fieldnames = ['PassengerId', 'Survived']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(0,418):
        writer.writerow({'PassengerId': a['PassengerId'][i], 'Survived': predictions[i]})


# In[ ]:




