
## Classification

### Types of machine learning
Unsupervised learning: Uncovering hidden patterns from unlabeled data. \
For example, we have clustering, where we group customers into distinct categories.

Supervised learning: The predicted values are known.\
Our aim: Predict the target values of unseen data, given the features.\
Types: Classification and Regression.


### scikit-learn syntax

For each model, we'll use the following syntax:
```python
from scikit.module import Model
model = Model()
model.fit(X.y)
predictions = model.predict(X_new)
print(predictions)
```

### k-Nearest Neighbors

Let's focus on predicting the label of a data point by:
..* Looking at the k closest labeled data points
..* Taking a majority vote

#### knn syntax

Suppose we want to predict whether an individual has churned or not by their total day and evening charge.

```python
from sklearn.neighbors import KNeighborsClassifier
X = df[ ['total_day_charge', 'total_eve_charge'] ].values
y = df['churn'].values
#.values are used to covert dataframe into numpy arrays
print(X.shape, y.shape)

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X,y)

X_new = np.array( [1,1],[2,2],[3,3] )
# for example
predictions = knn.predict(X_new)
print('Predictions: {}'.format(predictions))
```


### Measuring the model's performance

In classification, accuracy is a commonly used metric.
Accuracy = #correct predictions/#total number of observations

To correctly measure performance, we split data into a training set and a test set.

#### train/test split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                        random_state=21, stratify=y)
#stratify ensures data is splitted for y values having the same proportions as the split

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test,y_test))
# this returns the accuracy of our model
```

#### Model complexity and over/underfitting

```python
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1,26)
for neighbor in neighbors:
  knn = KNeighborsClassifier(n_neighbors = neighbor)
  knn.fit(X_train, y_train)
  train_acurracies[neighbor] = knn.score(X_train,y_train)
  test_accuracies[neighbor] = knn.score(X_test, y_test)

# then we plot our results

plt.figure(figsize=(8,6))
plt.title('KNN: Varying Number of Neighbors')
plt.plot(neighbors, train_accuracies.values() , label='Training Accuracy')
plt.plot(neighbors, test_accuracies.values(), label='Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
```


## Regression

Let's predict blood glucose levels and store it in a pandas dataframe

```python
import pandas as pd
diabetes_df = pd.read_csv('diabetes.csv')
print(diabetes_df.head())
```



Now let's create feature and target arrays,

```python
X = diabetes_df.drop('glucose',axis=1).values
y = diabetes_df['glucose'].values
print(type(X),type(y))
X_bmi = X[:,3]
```

Let's fit a regression model,

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

plt.scatter(X_bmi.y)
plt.plot(X_bmi,y)
```


#### Linear regression using all features
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_Test_split(X,y,test_size=0.3,
                                                    random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
```

The default metric for LR is R-squared.\
$R^2$ quantifies the variance in target values explained by the features and its values range from 0 to 1.\

R-squared in scikit-learn:
```python
reg_all.score(X_test,y_test)
```