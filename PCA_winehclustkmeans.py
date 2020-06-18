# -*- coding: utf-8 -*-
"""
Created on Mon May 11 22:04:37 2020

@author: Rohith
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\PCA\\winehclust.csv")

df.describe()
df.dtypes

df_new=scale(df.iloc[:,3:16])

#Applying PCA

pca=PCA(n_components=9)
pca_values=pca.fit_transform(df_new)

#The amount of varience that each PCA explains
var=pca.explained_variance_ratio_
var

#Cumulative varience
var1=np.cumsum(np.round(var,decimals=4)*100)
var1

#Variance plot for PCA Components
plt.plot(var1,color='green')


########To apply Hierarchical clustering ###################

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch


z=linkage(df_new,method='complete',metric='euclidean')

plt.figure(figsize=(15,5));
plt.title("Hierarchical Culster Dendogram");
plt.xlabel("Index");
plt.ylabel('Distance');
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=8);

help(AgglomerativeClustering)

h_clust=AgglomerativeClustering(n_clusters=10,linkage='complete', affinity='euclidean').fit(df_new)
h_clust.labels_
df["clust"]=h_clust.labels_
df=df.iloc[:,[16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]

##################To apply Kmeans Clustering ################

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#To plot elbow curve and screw plot

k=list(range(2,15))
k
TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i).fit(df_new)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(df_new[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_new.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS,'ro-')


#KMeans model building:

kmeans=KMeans(n_clusters=11)
kmeans.fit(df_new)
kmeans.labels_

df['clust']=kmeans.labels_

df=df.iloc[:,[16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]


















