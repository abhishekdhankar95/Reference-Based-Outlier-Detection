# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:05:49 2019

@author: Animesh
"""

import numpy as np
from sklearn.cluster import KMeans
import math  

np.random.seed(5)

data = np.loadtxt('t7.10k.dat', delimiter=" ")

number_of_reference_points = int(math.sqrt(data.shape[0])/2) 
kmeans = KMeans(n_clusters=number_of_reference_points, random_state=5).fit(data)

temp_reference_points = kmeans.cluster_centers_

reference_points=[]
for i in range(number_of_reference_points):
    reference_points.append(tuple (temp_reference_points[i]))