# Classification of Wheat Kernels

#### Importing Libraries


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Data Preprocessing

#### Importing Dataset


```python
# Importing the dataset
dataset = pd.read_csv('./cleaned_seeds_dataset.csv')
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:, -1].values
```

#### Splitting Data into Train and Test data


```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

#### Standardizing Data


```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Building the classfier Model

#### After experimenting with the data, the KNN classifier chosen for this model, using Euclidean distance as the distance metric


```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
```




## Testing the model


```python
y_pred = classifier.predict(X_test)
```

#### Printing the confusion matrix and accuracy


```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

    [[15  0  2]
     [ 1 20  0]
     [ 0  0 15]]
    


```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)
```

    0.9433962264150944
    
