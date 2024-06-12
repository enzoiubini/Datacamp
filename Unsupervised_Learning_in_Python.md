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


### t-SNE for 2-dimensional maps

t-SNE: unsupervised learning method

t-SNE = 't-distributed stochastic neighbor embedding'

map approximately preserves nearness of samples

```python
print(samples)
# two dimensional data array

print(species)
# labels of each sample 0,1,2

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(samples)

xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs,ys, c=species)
plt.show()
```

t-SNE only has fit_transform() method, and simultaneously fits the model and transform the data. Since it has no separate fit() and transform() methods, it can't extend the map to include new data samples.

The learning rate must be chosen usually between 50-200. wrong choice: points bunch together.

Also, the axes have no interpretable meaning. And each time the model is applied, a different plot is shown, but they always keep their relative distance to one another.

Exercise to annotate each company
```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)
# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs,ys,alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
```

### Visualizing the PCA transformation

More efficient storage and computation, and removal of less-informative 'noise' features.

- first step: decorrelation
PCA aligns data samples to be aligned with axes, and shifts data samples so they have mean 0.

PCA is a scikit-learn components, fit() learns the transformation and transform() applies it.

```python
from sklearn.decomposition import PCA
model = PCA()
model.fit(samples)

transformed = model.transform(samples)
```

New columns are the 'PCA features'.
Due to the rotation it performs, it de-correlates the data.

#### Principal components
Available as components_ attribute of PCA object
```python
print(model.components_)
```


### Intrinsic dimension 

intrinsic dimension: number of features needed to approximate the dataset

Let's consider 3 features of versicolor dataset: sepal length, sepal width and petal width
3-d scatter looks like a 2-d plane.

PCA features are ordered by variance descending. Intrinsic dimension is number of PCA features with significant variance.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(samples)

features = range( pca.n_components_)

plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()
```


Example drawing arrow and extracting first PCA

```python
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]
# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()
```


### Dimension reduction with PCA

Specify how many features to keep -> PCA(n_components=2)

```python
#samples is an array of 4 features
# species list of iris species numbers

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(samples)

tranformed = pca.transform(samples)
print( transformed.shape)
```

Let's consider a word frequency array.

This matrix is sparse, but scikit-learn PCA doesn't support csr_matrix, so we'll use TruncatedSVD instead.

```python
from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3)
model.fit(documents)  #documents in csr_matrix
transformed = model.transform(documents)
```



### Non-negative matrix factorization

NNF is a dimension reduction technique.
These models are interpretable unlike PCA, easy to explain!
However, all sample features must be positve!

Using scikit-learn NMF

```python
from sklearn.decomposition import NMF
model = NMF(n_components=2)
model.fit(samples)
nmf_features = model.transform(samples)

print(model.components_)

print(nmf_features)
```

Reconstruction of a sample

```python
print(samples[i,:])
# [0.12, 0.18, 0.32, 0.14]

print(nmf_features[i,:])
# [0.15, 0.12]

# we can combine them to approximately reconstruct the sample

# can also be expressed as a product of matrices

```

NMF fits to non-negative data only


### NMF learns interpretable parts

```python
from sklearn.decomposition import NMF
nmf = NMF(n_components=10)
nmf.fit(articles)
```

Each row (or component) live in a 800-dimensional space (in this example).
Choosing a components, and looking at which words have higher value, we get the topic of the document.

If NMF is applied to images, then NMF components represent patterns.

##### Grayscale images

grayscale images: no colors, only shades of gray.
Measure pixel brightness, represented with a value between 0 and 1.


These 2-dimensional array can be flatened, represented by a flat array of non negative numbers. To encode images, each row corresponds to an image, and each column to a pixel.
-> we can apply NMF

```python
print(sample)
# two dimensional array

bitmap = sample.reshape((2,3))
print(bitmap)
# image

# to restore the image
from matplotlib import pyplot as plt
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.show()
```



### Building recommender systems using NMF

Task: recommend articles of similar topics.

```python
from sklearn.decomposition import NMF
nmf = NMF(n_components=6)
nmf_features = nmf.fit_transform(articles)

# strategy -> compare articles using nmf features
```

Different versions of the same document have the same topic proportions. while exact feature values may be different (due to one version using many meaningless words), all versions lie on the same lie through the origin. To compare, we compare the angle between lines

### Cosine similarity

Higher values ( cos 0 = 1) means more similar.

```python
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)

#if has index 23
current_article = norm_features[23,:]
similarities = norm_features.dot(current_article)
print(similarities)
```

Dataframes and labels.
Label similarities with the article titles, using a DataFrame. titles are given as a list: titles

```python
import pandas as pd
norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features, index=titles)
current_article = df.loc['Dog bites man']
similarities = df.dot(current_article)

print( similarities.nlargest())
```



