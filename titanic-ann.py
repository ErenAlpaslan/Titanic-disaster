# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 21:58:02 2019

@author: eren
"""
#importing libraries
import pandas as pd
import numpy as np


df = pd.read_csv("./titanic/train.csv")
#%%
df.drop(['Name','PassengerId','Cabin','Ticket'],axis = 1, inplace = True)
df.dropna(subset=['Embarked'], how='all', inplace = True)
dataset = df.iloc[:,:].values
dataset[pd.isnull(dataset)] = 'NaN'


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset[:,2] = le.fit_transform(dataset[:,2])
dataset[:,-1] = le.fit_transform(dataset[:,-1])


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = 'median')
dataset[:,3:4] = imputer.fit_transform(dataset[:,3:4])


dt = pd.DataFrame(dataset)
dt.info()

X = dataset[:,1:]
y = dataset[:,:1].astype(int)
y = pd.DataFrame(y)
# %% Splitting train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# %% ANN model
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import BatchNormalization

classifier = Sequential()

classifier.add(Dense(output_dim = 50, init= 'uniform', activation= 'relu', input_dim = 7))
classifier.add(Dropout(rate = 0.25))
classifier.add(Dense(output_dim = 50, init= 'uniform', activation= 'relu'))
classifier.add(Dropout(rate = 0.5))
classifier.add(Dense(output_dim = 1, init= 'uniform', activation= 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=128, epochs= 500, validation_data = (X_test,y_test))
# %% predict test values
y_head = classifier.predict(X_test)
y_head = [1 if each >0.5 else 0 for each in y_head]

# %% grid search
 from keras.wrappers.scikit_learn import KerasClassifier
 from sklearn.model_selection import GridSearchCV
 
 def build(optimizer):
     classifier = Sequential()
     classifier.add(Dense(output_dim = 50, init= 'uniform', activation= 'relu', input_dim = 7))
     classifier.add(Dropout(rate = 0.25))
     classifier.add(Dense(output_dim = 25, init= 'uniform', activation= 'relu'))
     classifier.add(Dropout(rate = 0.5))
     classifier.add(Dense(output_dim = 1, init= 'uniform', activation= 'sigmoid'))
     classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
     return classifier
 
 classifier = KerasClassifier(build_fn = build)
 
 #dictionary for hyperparameters
 parameters = {'batch_size' : [64, 128],
               'epochs' : [500, 1000],
               'optimizer' : ['adam']
               }
 
 grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
 grid_search = grid_search.fit(X_train,y_train)
 
 best_parameters = grid_search.best_params_
 best_accuracy = grid_search.best_score_
 
# %% confusion matrix
from sklearn.metrics import precision_score
precision_score(y_test, y_head, average='weighted')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_head)
print((cm[0,0]+cm[1,1])/178)


# %% Prepare test set
test_dataframe = pd.read_csv("./titanic/test.csv")
test_dataframe.drop(['Name','PassengerId','Cabin','Ticket'],axis = 1, inplace = True)
test_dataframe.dropna(subset=['Embarked'], how='all', inplace = True)
test_dataframe = test_dataframe.iloc[:,:].values
test_dataframe[pd.isnull(test_dataframe)] = 'NaN'
test_dataframe = pd.DataFrame(test_dataframe)
test_dataframe.dropna(axis = 0,how='all', inplace = True)


#%%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
test_dataframe[1] = le.fit_transform(test_dataframe[1])
test_dataframe[6] = le.fit_transform(test_dataframe[6])

# %%
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = 'median')
test_dataframe.iloc[:,2:3] = imputer.fit_transform(test_dataframe.iloc[:,2:3])
# Scaling
sc = MinMaxScaler()
test_dataframe = scaler.fit_transform(test_dataframe)
#%%
# test predict
y_pred = classifier.predict(test_dataframe)
y_pred = [1 if each >0.5 else 0 for each in y_pred]



#%% comparing

submission = pd.read_csv("./titanic\gender_submission.csv")
submission = submission.iloc[:,1].values



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(submission, y_pred)
print((cm[0,0]+cm[1,1])/418)


#%%
p_id = pd.read_csv("./titanic\gender_submission.csv")
p_id = p_id.iloc[:,0].values
p_id = pd.DataFrame(p_id)
y_pred = pd.DataFrame(y_pred)



predictions = pd.concat([p_id,y_pred], axis = 1)
predictions = pd.DataFrame(predictions.iloc[:,:].values,columns = ['PassengerId','Survived'])
predictions.to_csv(r'./titanic\prediction2.csv', index = False)


#%%
p = pd.read_csv('./titanic/prediction.csv')





