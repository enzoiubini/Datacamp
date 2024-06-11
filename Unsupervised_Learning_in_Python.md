## Unsupervised Learning

Unsupervided learning is a class of machine learning techniques for discovering patterns in data. 
eg, clustering customers by their purchases, compressing the data using purchase patterns (dimension reduction)


Supervised learning finds patterns for a prediction task. In this case, the pattern discovery is guided or 'supervised', so that the patterns are as useful as possible for predicting the label.

Unsupervised learning find patters without labels, finding pure patterns.

We use the iris dataset, measuring many iris plants.
-Three species of iris: setosa, versicolor and virginica
-measurements: petal length, petal width, sepal length, sepal width

Let's see examples of k-means clustering

```python
print(samples)

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples)

labels = model.predict(samples)
```

K-means predicts to which cluster a new sample belongs.

#### Create scatter plots
```python
import matplotlib.pyplot as plt
xs = samples[:,0]
ys = samples[:,2]
plt.scatter(xs,ys,c=labels)
plt.show()
```


### Evaluating a clustering

We can check corresponder with the labels we know, but what if there are none to check? 

Let's first compare the clusters with the species.

```python
import pandas as pd
df = pd.DataFrame({'labels':labels, 'species':species})
print(df)

ct = pd.crosstab(df['labels'],df['species'])
print(ct)
```

How do we measure without preknown labels now? We measure the spread of the clusters. Inertia measures how spread the clusters are.

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(samples)
print(model.intertia_)
```

### Transforming features for better clusterings

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
labels = model.fit_predict(samples)

df = pd.DataFrame({'labels':labels, 'varieties':varieties})
ct = pd.crosstab( df['labels'],df['varieties']))

# We must use standardscaler to make the distributions equal, otherwise the model does not pick up on the different variances

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled = scaler.transform(samples)
```

To make all these steps, we work with pipelines

```python
from sklearn.preprocessing import StandardScaler()
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(caler,kmeans)
pipeline.fit(samples)

labels = pipeline.predict(samples)
```

### Visualizing hierarchies

Clusters are contained in another. 
Example: countries gave scores to songs performed at the Eurovision 2016. Data is arranged in a rectangular array. The samples are the countries. The result can be visualized as a tree-like diagram. (dendrogram)

In the beginning, each country is in a separate cluster. At each step, the two closest clusters are merged, all until all countries are in a single cluster.

Given samples (array of scores) and country_names
```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method='complete')
dendrogram( mergings, labels=country_names,leaf_rotation=90, leaf_font_size=6)
plt.show()
```

### Cluster labels in hierarchical clustering

Height of dendrogram = distance between merging clusters.

Distance between clusters is defined by a 'linkage method'.

#### Extracting cluster labels

We use the fcluster() function
```python
from scipy.cluster.hierarchy import linkage
mergings = linkage(samples, method='complete')
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 15,criterion='distance')
print(labels)

import pandas as pd
pairs = pd.DataFrame({'labels':labels, 'countries':country_names})
print( pairs.sort_values('labels')
```
