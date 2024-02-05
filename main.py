import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('USArrests.csv')
newDf = df[["Murder", "Assault", "UrbanPop", "Rape"]]
ar = np.array(newDf)
Z = hierarchy.linkage(ar, 'single')
plt.figure(figsize=(10, 6))
dn = hierarchy.dendrogram(Z, labels=df['State'].tolist(), orientation='top', distance_sort='descending')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

k = 3
clusters = hierarchy.fcluster(hierarchy.linkage(ar, method='single'), k, criterion='maxclust')
df['Cluster'] = clusters
print(pd.DataFrame({'State': df['State'], 'Cluster': df['Cluster']}))

# Standardize the variables to have standard deviation one
scaler = StandardScaler()
scaled_data = scaler.fit_transform(newDf)

# Perform hierarchical clustering with complete linkage and Euclidean distance
Z = hierarchy.linkage(scaled_data, method='complete', metric='euclidean')

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dn = hierarchy.dendrogram(Z, labels=df['State'].tolist(), orientation='top', distance_sort='descending')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('States')
plt.ylabel('Euclidean Distance')
plt.show()
