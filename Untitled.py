#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[23]:


iris=pd.read_csv('Iris.csv')
iris.head(5)


# In[ ]:





# x=iris.iloc[:,[0,1,2,3]].values
# 
# from sklearn.cluster import KMeans
# wcss=[]
# 
# for i in range(1,11):
#     kmeans=KMeans(n_clusters=i, init ='k-means++',
#                   max_iter=300, n_init =10 ,random_state =0)
#     kmeans.fit(x)
#     wcss.append(kmeans.inertia_)
#     
# 
# plt.plot(range(1,11),wcss)
# plt.title('The elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('wcss')
# plt.show()

# In[25]:


kmeans =KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state =0)
y_kmeans=kmeans.fit_predict(x)


# In[28]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],
            s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],
           s=100,c='blue',label='Iris-versicolour')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],
           s=100,c='green',label='Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
           s=100,c='yellow',label='centroids')
plt.legend()


# In[ ]:




