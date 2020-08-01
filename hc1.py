#Importing th libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing the dataset
dataset= pd.read_csv("Mall_Customers.csv")
x= dataset.iloc[:,[3,4]].values

#To find the optimal number of clusters
import scipy.cluster.hierarchy as sc
dendogram= sc.dendrogram(sc.linkage(x, method="ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

#To fit the algorithm to the dataset
from sklearn.cluster import AgglomerativeClustering as ac
hc= ac(n_clusters= 5, linkage="ward")
y=hc.fit_predict(x)

#Visualizing the graph
plt.scatter(x[y == 0, 0], x[y == 0,1], s=100, c='red', label=' Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1,1], s=100, c='magenta', label=' Cluster 2')
plt.scatter(x[y == 2, 0], x[y == 2,1], s=100, c='green', label=' Cluster 3')
plt.scatter(x[y == 3, 0], x[y == 3,1], s=100, c='yellow', label=' Cluster 4')
plt.scatter(x[y == 4, 0], x[y == 4,1], s=100, c='cyan', label=' Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()