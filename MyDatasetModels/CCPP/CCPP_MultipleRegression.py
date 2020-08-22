import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_excel("./CCPP dataset/Folds5x2_pp.xlsx",sheet_name = None)

#print(dataset)

def getdata(dataframe):
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    return X,y

#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, -1].values
#y = y.reshape(len(y),1)

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

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score,mean_squared_error
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

#r2 score: 0.9286960898122537
#mean_squared_error: 20.76739753253501

from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 1, kernel = 'rbf')
X_test = kpca.fit_transform(X_test)


#Visualising the Multiple Linear regression results
plt.scatter(X_test,y_test, color = 'red',s=1)
plt.scatter(X_test,y_pred, color = 'blue',s=1)
plt.title('Truth or Bluff (SVR)')
plt.xlabel('1D of input')
plt.ylabel('Power Output')
plt.show()

#for i in range(4):
#    plt.scatter(X_test[:,i],y_test, color = 'red',s=1)
#    plt.scatter(X_test[:,i],y_pred, color = 'blue',s=1)
#    plt.title('Truth or Bluff (SVR)')
#    plt.xlabel('1D of input')
#    plt.ylabel('Power Output')
#    plt.show()
