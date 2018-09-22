# K-Means Clustering
Implementation of the k-Mean clustering algorithm to cluster words belonging to four categories: animals, countries,
fruits and veggies, arranged into four different files. 
The first entry in each line is a word followed by 300 features (word embedding) describing the meaning of that word.

Implementation of the k-means is tested with k values between 1-10 and the precision, recall, and F-score is computed for each set of clusters, which is then plotted in a graph.

Several tests are computed with varying distence measures:
* Euclidean distance
* Euclidean distance with a unit L2 normilization
* Manhattan distance
* Manhattan distance with a unit L2 normilization
* Cosine Similarity distance

#### Dependencies
* Numpy
* Matplotlib

To conduct the test:

```
python kMeansClustering.py
```




