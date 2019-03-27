import numpy as np
from sklearn.cluster import KMeans
import math  

def reference_points_kMeans(file_name):
    
    np.random.seed(5)
    
    data = np.loadtxt(file_name, delimiter=" ")
    
    number_of_reference_points = int(math.sqrt(data.shape[0])/2) *6
    print(number_of_reference_points)
    kmeans = KMeans(n_clusters=number_of_reference_points, random_state=5).fit(data)
    
    return (kmeans.cluster_centers_, data)
    #center = np.array([[350,250],[-100,-100],[1400,1400], [0, 1400], [1400, 0]])
    #center = np.array([[350,250]])
    #return (center,data)