
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
regressor = SVR(kernel = 'rbf',C=10,gamma=100.0,verbose=True)
regressor.fit(X_train, y_train)



#Predicting the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score, mean_squared_error
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

#C=1 gamma=3.0
#0.9628018990435875
#10.834016647658496

#C=100 gamma=2.0
#r2 score: 0.9734564825827955
#mean_squared_error: 7.730849215189109

#C=1,gamma=100.0
#r2 score: 0.9908616603911649
#mean_squared_error: 2.66155854488615

#C=1,gamma=1000.0
#0.9903475770802015
#2.811286273078155

#C=1, gamma = 500
#0.9903924721920089
#2.7982104875887317

#C=1000, gamma = 1000
#0.9903478044421267
#2.811220053491134

#C=10,gamma=100.0
#0.9909044308083992
#2.649101580674778

# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 1, kernel = 'rbf')
X_test = kpca.fit_transform(X_test)

#Visualising the Multiple Linear regression results
plt.scatter(X_test,y_test, color = 'red',s=1)
plt.scatter(X_test,y_pred, color = 'blue',s=1)
plt.title('Power Output plot(SVR)')
plt.xlabel('1D of input')
plt.ylabel('Power Output')
plt.show()