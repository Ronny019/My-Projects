import pandas as pd
import numpy as np

dataset=pd.read_excel("./CCPP dataset/Folds5x2_pp.xlsx",sheet_name = None)

def getdata(dataframe):
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    return X,y

X_train = pd.DataFrame()
y_train = pd.DataFrame()

for i in range(1,5):
    sheet_no = "Sheet"+str(i)
    X_temp,y_temp = getdata(dataset[sheet_no])
    X_train = pd.concat([X_train,X_temp],axis = 0)
    y_train = pd.concat([y_train,y_temp])


X_train = X_train.values
y_train = y_train.values
y_train = y_train.reshape(len(y_train),1)

X_test = dataset["Sheet5"].iloc[:, :-1].values
y_test = dataset["Sheet5"].iloc[:, -1].values
y_test = y_test.reshape(len(y_test),1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

#Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')


from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 2,
                           n_jobs = -1,
                           verbose = 4)
grid_search = grid_search.fit(X_train, y_train)
best_acc = grid_search.best_score_
best_par = grid_search.best_params_

print(best_acc)
print(best_par)