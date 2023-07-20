# Energy Prediction of a Comibined Cycle Power Plant

## Data preporcessing
### 1. Import the libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
```

### 2. Read the excel file


```python
dataset=pd.read_excel("./CCPP dataset/Folds5x2_pp.xlsx",sheet_name = None)
```

### 3. Preparing the Training and Test Data

#### In this project we take the first four excel sheet data as the training data and the last sheet as the test data


```python
def getdata(dataframe):
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    return X,y

X_train = pd.DataFrame()
y_train = pd.DataFrame()
```

#### Train Data:


```python
for i in range(1,5):
    sheet_no = "Sheet"+str(i)
    X_temp,y_temp = getdata(dataset[sheet_no])
    X_train = pd.concat([X_train,X_temp],axis = 0)
    y_train = pd.concat([y_train,y_temp])


X_train = X_train.values
y_train = y_train.values
y_train = y_train.reshape(len(y_train),1)
```

#### Test Data:


```python
X_test = dataset["Sheet5"].iloc[:, :-1].values
y_test = dataset["Sheet5"].iloc[:, -1].values
y_test = y_test.reshape(len(y_test),1)
```

## Data Normalization
### In this Project we employ the use of Standard Scaler library to bring the data to  mean and unity variance.


```python
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)
```

## Model building
#### In this project we use the RBF SVM Kernel to model our predictor


```python
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf',C=10,gamma=100.0,verbose=True)
regressor.fit(X_train, y_train)
```

    c:\users\hp\pycharmprojects\pythonproject\venv\lib\site-packages\sklearn\utils\validation.py:1143: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

    [LibSVM]




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVR(C=10, gamma=100.0, verbose=True)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">SVR</label><div class="sk-toggleable__content"><pre>SVR(C=10, gamma=100.0, verbose=True)</pre></div></div></div></div></div>



### Prediction of Test Set Results


```python
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(len(y_test),1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

    [[478.77 480.48]
     [447.46 445.75]
     [440.47 438.76]
     ...
     [464.25 465.96]
     [452.64 450.93]
     [453.35 451.67]]
    

### Evaluating Model Performance
#### We evaluate the model performance based on the accuracy and the R2 score of the model


```python
from sklearn.metrics import r2_score, mean_squared_error
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
```

    0.9909044308083992
    2.649101580674778
    

### Visualizing the results
#### Applying PCA to reduce the features to one dimension


```python
# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 1, kernel = 'rbf')
X_test = kpca.fit_transform(sc_X.transform(X_test))

```

#### Plotting y_test and y_pred in Matplotlib


```python
#Visualising the Multiple Linear regression results
plt.scatter(X_test,y_test, color = 'red',s=1)
plt.scatter(X_test,y_pred, color = 'blue',s=1)
plt.title('Power Output plot(SVR)')
plt.xlabel('1D of scaled input')
plt.ylabel('Power Output')
plt.show()
```


    
![png](output_22_0.png)
    

