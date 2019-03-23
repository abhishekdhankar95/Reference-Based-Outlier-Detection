# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 01:22:23 2019

@author: abhis
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:05:49 2019
@author: Animesh
"""

import numpy as np
from sklearn.cluster import KMeans
import math  

def reference_points_kMeans(file_name):
    
    np.random.seed(5)
    
    data = np.loadtxt(file_name, delimiter=" ")
    
    number_of_reference_points = int(math.sqrt(data.shape[0])/2) 
    kmeans = KMeans(n_clusters=number_of_reference_points, random_state=5).fit(data)
    
    #return (kmeans.cluster_centers_, data)
    center = np.array([[5,1],[0,0]])
    return (center,data)