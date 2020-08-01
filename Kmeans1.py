#Importing the libraries
import pandas as pd
import matplotlib.py_kmeansplot as plt
import numpy_kmeans as np

#Importing the dataset
dataset= pd.read_csv('Mall_Customers.csv')
x= dataset.iloc[:,[3,4]].values

#To find the optimal number of clusters
from sklearn.cluster import KMeans
wcss= []
for __ in range(1,11):
    kmeans= KMeans(n_clusters= __, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("WCSS Plot")
plt.xlabel("No of Clusters")
plt.y_kmeanslabel("Value of WCSS")
plt.show()

#To Apply_kmeans Kmeans to Dataset
kmeans= KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans_kmeans= kmeans.fit_predict(x)

#To visualize the plot
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s= 100, c='red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='cyan', label='Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, c='magenta', label='Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=100, c='yellow', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s=300, c='green', label='Centroid')
plt.title('Clusters of customers')
plt.xlabel('Salary')
plt.ylabel('Spending Score')
plt.legend()
plt.show()