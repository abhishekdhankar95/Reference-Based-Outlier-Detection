import numpy as np
import find_reference_points as ref_point_finder
import sys
import matplotlib.pyplot as plt
import scipy as scipy

class ref_distance_point_pair:
    def __init__(self, ref_distance, ref_point):
        self.ref_distance = ref_distance
        self.ref_point = ref_point
        


def absolute_distance(coordinate_tuple_1, coordinate_tuple_2, distance_type = "euclidean"):
    if distance_type == "euclidean":
        return np.linalg.norm(coordinate_tuple_1-coordinate_tuple_2)
    elif distance_type == "cosine":
        return np.dot(coordinate_tuple_1, coordinate_tuple_2)/(np.linalg.norm(coordinate_tuple_1)*np.linalg.norm(coordinate_tuple_2))
    elif distance_type == "mahalanobis":
        diff = coordinate_tuple_1 - coordinate_tuple_2
        X = np.vstack([coordinate_tuple_1,coordinate_tuple_2])
        V = np.cov(X.T)
        VI = np.linalg.inv(V)
        return np.sqrt(np.sum(np.dot(diff,VI) * diff, axis = 1))


def ref_point_absolute_distance(ref_points, X):
    
    ref_points_distances = np.array(X.shape[0]*[0])
    
    for ref_point in ref_points:
        ref_point_distances = np.array([])
        for data_point in X:
            ref_point_distances = np.hstack((ref_point_distances, absolute_distance(ref_point, data_point)))
    
        ref_points_distances = np.vstack((ref_points_distances, ref_point_distances))
    
    ref_points_distances = ref_points_distances[1:]
    return ref_points_distances

def compute_kNN_sum(X_i_args_index, X_i_args, ref_point_distances_i, k):
    
    
    ref_distance_sum = 0
    lower_bound = X_i_args_index - 1
    upper_bound = X_i_args_index + 1
    
    
    
    curr = ref_point_distances_i[X_i_args[X_i_args_index]]
    
    while lower_bound >= 0 and upper_bound < ref_point_distances_i.shape[0] and k > 0:
        
        lower = ref_point_distances_i[X_i_args[lower_bound]]
        upper = ref_point_distances_i[X_i_args[upper_bound]]
        
        if(abs(curr - lower) < abs(curr - upper)):
            
#            print(abs(curr - lower))
            ref_distance_sum += abs(curr - lower)
            k -= 1
            lower_bound -= 1
        else:
            
#            print(abs(curr - upper))
            ref_distance_sum += abs(curr - upper)
            k -= 1
            upper_bound += 1
            
    
    
    while lower_bound >= 0 and k > 0:
        
        
        lower = ref_point_distances_i[X_i_args[lower_bound]]
#        print(abs(curr - lower))
        ref_distance_sum += abs(curr - lower)
        k -= 1
        lower_bound -= 1
    
    while upper_bound < ref_point_distances_i.shape[0] and k > 0:
        
        
        upper = ref_point_distances_i[X_i_args[upper_bound]]        
#        print(abs(curr - upper))
        ref_distance_sum += abs(curr - upper)
        k -= 1
        upper_bound += 1
        
#    print("ref_distance_sum:" + str(ref_distance_sum))
    return ref_distance_sum
    
    
def minimum_density_computation(ref_points, X, ref_points_distances, k):
    
    
    ref_point_1 = ref_points[0];
    X_1_args = np.argsort(ref_points_distances[0])
    X_1 = X[X_1_args]
    
    
    min_density = np.array(X.shape[0] * [0.0])
    
    for i in range(X_1_args.shape[0]):
        data_point_index = X_1_args[i]
        temp = (compute_kNN_sum(i, X_1_args, ref_points_distances[0], k))/k        
        min_density[data_point_index] = 1/temp
        
# =============================================================================
#     print()
#     print()
# =============================================================================
        
    for j in range(1, ref_points.shape[0]):
        #print(j)
        ref_point_j = ref_points[j]
        X_j_args = np.argsort(ref_points_distances[j])
        X_j = X[X_j_args]
        
        for i in range(X_j_args.shape[0]):
            data_point_index = X_j_args[i]
            temp = (compute_kNN_sum(i, X_j_args, ref_points_distances[j], k))/k            
            temp_min_density = 1/temp
            min_density[data_point_index] = min(min_density[data_point_index], temp_min_density)
    return min_density
        

def takeSecond(elem):
    return elem[1]


if __name__ == "__main__":
    
    
    #print(absolute_distance(np.array([5.0, 6.0, 7.0, 8.0]), np.array([1.0, 2.0, 3.0, 4.0]), "cosine"))
    
    k = int(sys.argv[1])
    file_name = sys.argv[2]
    
    ref_points, X = ref_point_finder.reference_points_kMeans(file_name)
    
    ref_points_distances = ref_point_absolute_distance(ref_points, X)
    
    min_density = minimum_density_computation(ref_points, X, ref_points_distances, k)
    
    max_min_density = max(min_density)
    ros_of_X = 1 - (min_density/max_min_density)
    ros_arg = np.argsort(ros_of_X)
    
    X_ros_sorted = X[ros_arg]
    X_ros_sorted_dec= np.flip(X_ros_sorted,axis=0)
    
    number_of_top_outliers = 700
    
    class_label=[]
    
    for i in range(X_ros_sorted.shape[0]):
        class_label.append(1)
        
    '''
    for i in range(number_of_top_outliers):
        class_label[int (sorted_ros_of_X[i][0])]=2
    '''
    
    for i in range(number_of_top_outliers):
        class_label[X_ros_sorted.shape[0] - 1 - i] = 2  
        
    plt.figure(figsize=(20,10))
    plt.scatter(X_ros_sorted[:,0], X_ros_sorted[:,1], c=class_label)
    