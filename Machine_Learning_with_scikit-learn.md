
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

Another way to assess the accuracy of our model, is using the MSE.
```python
# RMSE
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred,squared=False)
```


### Cross validation

Motivation:
..* Model performance is dependent on the way we split up the data
..* Not representative of the model's ability to generalize to unseen data

We split the data into $n$ groups (or folds). Train our data leaving one fold aside and compute out metric, then do this for each fold, leaving us $n$ different values for our metric which we can compute the mean, median, or 95% confidence interval of.

```python
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=6,shuffle=True, random_state=42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X,y, cv=kf)
```

We can study these results accordingly
```python
#results
print(cv_results)

#mean and std
print(np.mean(cv_results),np.std(cv_results))

#95% C.I.
print( np.quantile(cv_results, [0.025,0.975]))
```


### Regularized Regression

A Linear regression minizes a loss function. If we allow these coefficient to be very large, we might be overfitting. Regularization penalizes big coefficients. 

#### Ridge regression

..* Loss function = OLS loss function + 
$$\alpha * \sum_{i=1}^n a_i^2$$

This penalizes large positive or negative coefficients.

```python
from sklearn.linear_model import Ridge
scores = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
  ridge = Ridge(alpha=alpha)
  ridge.fit(X_train, y_train)
  y_pred = ridge.predict(X_test)
  scores.append( ridge.score(X_test,y_test))
```

#### Lasso regression

..* Loss function = OLS loss function + 
$$\alpha * \sum_{i=1}^n \lVert a_i \rVert$$

```python
from sklearn.linear_model import Lasso
scores = []
for alpha in [0.01, 1.0, 10.0, 20.0, 50.0]:
  lasso = Lasso(alpha=alpha)
  lasso.fit(X_train, y_train)
  lasso_pred = lasso.predict(X_test)
  scores.append(lasso.score(X_test, y_test))
```

Lasso can select important features of a dataset, shrinking the coefficients of less important features to zero.

```python
from sklearn.linear_model import Lasso
X = diabetes_df.drop('glucose', axis=1).values
y = diabetes_df['glucose'].values
names = diabetes_df.drop('glucose',axis=1).columns
lasso = Lassa(alpha=0.1)
lasso_coef = lasso.fit(X,y).coef_
plt.bar(names,lasso_coef)
plt.xticks(rotation=45)
plt.show()
```

## Fine tuning our model

Accuracy is not always a useful metric. If, in a classification problem, 99% of values are 'a' and 1% of values are 'b', then the model setting all values to 'a' would have an accuracy of 0.99, predicting no 'b' values. This is known are class imbalance.

Let's set:\
tp = true positives\
tn = true negatives\
fp = false positives\
fn = false negatives

Accuracy: correct predictions
$$\frac{ tp + tn}{tp+tn+fp+fn}$$

Precision: From all positive predictions, how many are correct
$$\frac{tp}{tp+fp}$$

Recall or sensitivity: From all positive values, how many are correctly predicted. (positive meaning that they are the '1' value)
$$\frac{tp}{tp+fn}$$

F1 score: harmonic mean between precision and recall
$$2* \frac{\text{precision}*\text{recall}}{\text{precision}+\text{recall}}$$


#### Confusion matrix in scikit-learn

```python
from sklearn.metrics import classification_report, confusion_matrix
knn = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
```


### Logistic regression and the ROC curve

Logistic regression is used for classification problems, and outputs probabilities.
If p>0.5 (threshhold), we classify as 1, if not, as 0.
```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

#two dimensional array, for probabilities for both clases. We slice for the positive class
y_pred_probs = logreg.predict_proba(X_test)[:,1]
```

By default, threshold is 0.5. If we vary this threshold, we use the ROC curve.

```python
from sklearn.metrics import roc_curve
#false positive rates, true positive rates, thresholds
fpr, tpr, thresholds = roc_curve(y_test,y_pred_probs)

plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
```

Let's calculate ROC AUC in scikit-learn

```python
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test,y_pred_probs))
```



### Hyperparamenter tuning

Our problem is to choose the best hyperparameters for our models. One tool is the Grid search cross-validation


```python
from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits = 5, shuffle=True, random_state=42)
param_grid = {'alpha': np.arange(0.0001,1,10),
              'solver': ['sag','lsqr']}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)

ridge_cv.fit(X_train,y_train)

print(ridge_cv.best_params_, ridge_cv.best_score_)
```

This doesn't really scale well when we have a large group of hyperparameters. We  can perform a random search.

#### RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
kf = KFold(n_splits = 5, shuffle=True, random_state=42)
param_grid = {'alpha': np.arange(0.0001,1,10),
              'solver': ['sag','lsqr']}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf,n_iter=2)
ridge_cv.fit(X_train,y_train)

print(ridge_cv.best_params_, ridge_cv.best_score_)

test_score = ridge_cv.score(X_test,y_test)
```


## Processing and Pipelines

Scikit-learn requires numeric data and no missing values. 
Dealing with categorical features, we need to convert it to numerical features.

```python
import pandas as pd
music_df = pd.read_csv('music.csv')
music_dummies = pd.get_dummies( music_df['genre'], drop_first=True)
print(music_dummies.head())

music_dummies = pd.concat([music_df, music_dummies], axis=1)
music_dummies = music_dummies.drop('genre', axis=1)
```

#### Encoding dummy variables

```python
music_dummies = pd.get_dummies(music_df, drop_first=True)
print(music_dummies.columns)
```

#### Linear regression with dummy variables
```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
X = music_dummies.drop('popularity',axis=1).values
y = music_dummies['popularity'].values
X_train, X_test, y_train, y_test = train_test_split( X, y , test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
linreg = LinearRegression()
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
print(np.sqrt(-linreg_cv))
```


### Handling missing data

```python
Lets print the amount of missing data on our dataframe
print(music_df.isna().sum().sort_values())
```

#### Dropping missing data
```python
music_df = music_df.dropna(subset=['column1','column2'])
```

#### Imputing values
We can impute the mean to replace missing data, or median, or frequent values.

```python
from sklearn.impute import SimpleImputer
X_cat = music_df['genre'].values.reshape(-1,1)
X_num = music_df.drop(['genre','popularity'],axis=1).values
y = music_df['popularity'].values
X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y,test_size=0.2,random_state=12)

X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y,test_size=0.2,random_state=12)

imp_cat = SimpleImputer(strategy='most_frequent')
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

imp_num = SimpleImputer()
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat,axis=1)
```

#### Imputng within a pipeline

```python
from sklearn.pipeline import Pipeline
music_df = music_df.dropna(subset = ['genre','popularity'])
music_df['genre'] = np.where( music_df['genre']=='Rock' , 1,0)

X = music_df.drop('genre',axis=1).values
y = music_df['genre'].values

steps = [('imputation',SimpleImputer()),
          ('logistic_regression',LogisticRegression())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
pipeline.fit(X_train,y_train)
pipeline.score(X_test,y_test)
```


### Centering and scaling

```python
print( music_df[['duration_ms','loudness','speechiness']].describe())
```

How to scale our data?

- Subtract the mean and divide by variance -> standardization
- Substract the minimum and divide by range -> from 0 to 1

```python
from sklearn.preprocessing import StandardScaler
X = music_df.drop('genre',axis=1).values
y = music_df['genre'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print( np.mean(X), np.std(X))
print( np.mean(X_train_scaled), np.std(X_train_scaled))
```

#### Scaling in a pipeline

```python
steps = [('scaler',StandardScaler()), ('knn',KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)
print(knn_scaled.score(X_test,y_test))
```


#### CV and scaling in a pipeline

```python
from sklearn.model_selection import GridSearchCV
steps = [('scaler',StandardScaler()), ('knn',KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbords': np.arange(1,50)}

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train,y_train)
y_pred = cv.predict(X_test)

print(cv.best_score_)
print(cv.best_params_)
```



### Evaluating multiple models

```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

X = music.drop('genre',axis=1).values
y = music['genre'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {'Logistic Regression': LogisticRegression(),
          'KNN': KNeighborsClassifier(),
          'Decision Tree': DecisionTreeClassifier()}

results = []
for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled, y_train , cv=kf)
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()
```

#### Test set performance
```python
for model, model in models.items():
  model.fit(X_train_scaled, y_train)
  test_score = model.score(X_test_scaled, y_test)
  print('{} Test Set Accuracy: {}'.format(name, test_score))
```

