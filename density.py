# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:59:53 2019

@author: abhis
"""

import numpy as np
import find_reference_points as ref_point_finder
import sys

class ref_distance_point_pair:
    def __init__(self, ref_distance, ref_point):
        self.ref_distance = ref_distance
        self.ref_point = ref_point
        


def absolute_distance(coordinate_tuple_1, coordinate_tuple_2, distance_type = "euclidean"):
    if distance_type == "euclidean":
        return np.linalg.norm(coordinate_tuple_1-coordinate_tuple_2)
    elif distance_type == "cosine":
        return np.dot(coordinate_tuple_1, coordinate_tuple_2)/(np.linalg.norm(coordinate_tuple_1)*np.linalg.norm(coordinate_tuple_2))

def ref_point_absolute_distance(ref_points, X):
    
    ref_points_distances = np.array(X.shape[0]*[0])
    
    for ref_point in ref_points:
        ref_point_distances = np.array([])
        for data_point in X:
            ref_point_distances = np.hstack((ref_point_distances, absolute_distance(ref_point, data_point)))
            
        #print(ref_points_distances.shape)
        ref_points_distances = np.vstack((ref_points_distances, ref_point_distances))
    
    
    ref_points_distances = ref_points_distances[1:]
    return ref_points_distances

def compute_kNN_sum(X_i_args_index, X_i_args, ref_point_distances_i, k):
    
    
    ref_distance_sum = 0
    lower_bound = max(X_i_args_index - 1, 0)
    upper_bound = min(X_i_args_index + 1, X_i_args.size - 1)
    
    curr = ref_point_distances_i[X_i_args[X_i_args_index]]
    
    while lower_bound >= 0 and upper_bound < ref_point_distances_i.size and k > 0:
        
        lower = ref_point_distances_i[X_i_args[lower_bound]]
        upper = ref_point_distances_i[X_i_args[upper_bound]]
        if(abs(curr - lower) < abs(curr - upper)):
            ref_distance_sum += abs(curr - lower)
            k -= 1
            lower -= 1
        else:
            ref_distance_sum += abs(curr - upper)
            k -= 1
            upper += 1
    
    while lower_bound > 0 and k > 0:
        
        lower = ref_point_distances_i[X_i_args[lower_bound]]
        ref_distance_sum += abs(curr - lower)
        k -= 1
        lower -= 1
    
    while upper_bound < ref_point_distances_i.size and k > 0:
        
        upper = ref_point_distances_i[X_i_args[upper_bound]]
        ref_distance_sum += abs(curr - upper)
        k -= 1
        upper += 1
        
    #print("ref "+str(ref_distance_sum))
    
    return ref_distance_sum
    
    
def minimum_density_computation(ref_points, X, ref_points_distances, k):
    
    
    ref_point_1 = ref_points[0];
    X_1_args = np.argsort(ref_points_distances[0])
    
    X_1 = X[X_1_args]
    
    min_density = np.array(X.shape[0] * [0.0])
    
    for i in range(X_1_args.shape[0]):
        data_point_index = X_1_args[i]
        #print(i, compute_kNN_sum(data_point_index, X_1_args, ref_points_distances[0], k))
        temp = (compute_kNN_sum(data_point_index, X_1_args, ref_points_distances[0], k)+1)/k
        
        ##print(1/temp)
        
        min_density[data_point_index] = 1/temp
        
    
        
    for j in range(1, ref_points.shape[0]):
        #print(j)
        ref_point_j = ref_points[j]
        X_j_args = np.argsort(ref_points_distances[j])
        X_j = X[X_j_args]
        
        for i in range(X_j_args.size):
            data_point_index = X_j_args[i]
            temp = (compute_kNN_sum(data_point_index, X_1_args, ref_points_distances[0], k)+1)/k            
            temp_min_density = 1/temp
            min_density[data_point_index] = min(min_density[data_point_index], temp_min_density)
    return min_density
        

if __name__ == "__main__":
    
    
    print(absolute_distance(np.array([5.0, 6.0, 7.0, 8.0]), np.array([1.0, 2.0, 3.0, 4.0]), "cosine"))
    
    k = int(sys.argv[1])
    file_name = sys.argv[2]
    
    ref_points, X = ref_point_finder.reference_points_kMeans(file_name)
    
    print(ref_points.shape, X.shape)
    
    ref_points_distances = ref_point_absolute_distance(ref_points, X)
    
    min_density = minimum_density_computation(ref_points, X, ref_points_distances, k)
    
    max_min_density = max(min_density)
    ros_of_X = 1 - min_density/max_min_density
    
    #for density in min_density:
    #       print(density)
    
    
    
    
    
    
    
    
    
    
    
    
    
    