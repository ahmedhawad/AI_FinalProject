
import pandas as pd
from numpy import zeros_like
from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import math

file_path = "User_Data.csv"

dataset = np.array(pd.read_csv(file_path))

dict_ds = {}
count = 1

for i in dataset: #makes dictionary { CustomerID : [numpy array of variables to eb used for dbscan]}
    dict_ds[count] =  i
    count+=1


"""
loops through everyone and if there is someone matchign the minimum distance combines them in
"""
def euclidean_distance(point1, point2):
    """
    calcualates euclidean distance of 2 points. 
    expectation is in form point1 = (x,y)
    """
    sums = 0
    for x1,x2 in zip(point1, point2):
        sums+= ((x1-x2)**2)
    return math.sqrt(sums)

def cluster_check(clusters, point):
    """
    True if point is in the list
    False if the point is not in the list
    """
    for thelist in clusters:
        if point in thelist:
            return True
    return False

def noise(dataset,clusters):
    """
    not working yet
    """
    noise = []
    unpacked_clusters = []
    # for the_list in clusters:
    #     for i in the_list:
            
                
    for i in clusters:
        for j in i:
            unpacked_clusters.append(j)
            
            
    for i in dataset:
        for j in dataset[i]:
            if j not in unpacked_clusters:
                noise.append(j)

def dbscan(dataset,epsilon, min_points):
    """
    
    dataset is dict {key: [np array of variables used for dataset]}
    not working yet
    """
    clusters  = [] #list of lists of all the clusters
    for i in dataset: #main point
        i_list = []
        for j in dataset: #secondary point
            # print(j)
            if i != j:
                distance = euclidean_distance(dataset[i],dataset[j])#distance  between the two
                #problem is that the our scale is diff than sklearn. I think they normalize their data, we don'y
                within_bounds = distance <= epsilon #boolean
                # print(distance,within_bounds)
                if not within_bounds: #if it meets epsilon criteria  
                    if not cluster_check(clusters, j): #it is already not part of a group
                         
                        # for thelist in clusters:
                        #     if point in thelist:
                        #         return True
                        # return False
                        #no need to check if it's in i_list because it can only ever be added there once
                        
                        i_list.append(j)
        """
        Add a method to get noise points
        """
        if len(i_list) >= min_points:  #if it meets min points criteria
                clusters.append(i_list)
    return len(clusters), noise(dataset,clusters), clusters
def dbscan_new(dataset, epsilon, min_points, cluster_list):
    """
    
    dataset is dict {key: [np array of variables used for dataset]}
    not working yet
    """
    
    if (len(dataset) >= min_points): #checks if min points of sufficient for the entire dataset
        clusters = []
        for i in dataset: #main point
            i_list = []
            i_tolist = dataset[i].tolist()
            for j in dataset: #secondary point
                if i != j:
                    j_tolist = dataset[j].tolist()
                    distance = euclidean_distance(i_tolist,j_tolist)#distance  between the two
                    # print(distance)
                    #problem is that the our scale is diff than sklearn. I think they normalize their data, we don'y
                    within_bounds = distance <= epsilon #boolean
                    # print(distance,within_bounds)
                    if within_bounds: #if it meets epsilon criteria  
                        if not cluster_check(cluster_list, j_tolist): #it is already not part of a group
                            
                            #no need to check if it's in i_list because it can only ever be added there once
                            
                            i_list.append(j_tolist)
                            # print(i_list)
                            # [array: [1, 1, 1, 1], array: [2, 2, 2, 2]]
            if len(i_list) >= min_points:  #if it meets min points criteria
                # clusters.append(i_list)
                for k in i_list:
                  clusters.append(k)
        # print(len(clusters))
        cluster_duplicate_free = [] 
        [cluster_duplicate_free.append(x) for x in clusters if x not in cluster_duplicate_free] 
        # print(len(cluster_duplicate_free))
        cluster_list.append(cluster_duplicate_free)
        dataset_update = dataset.copy() # dictionary has a weird copy method
        
        for x in dataset.keys(): # dataset cannot change key while iterating, so i use an identical dict as the loop-runner
          card = True
          # print(dataset_update[x], "dataset")
          for y in cluster_duplicate_free:
            # print(y, "cluster")
            if dataset_update[x].tolist() == y:
              # print("hello_")
              card = False
          if not card:
            del dataset_update[x]
        # print(len(dataset_update))
        # print(len(dataset_update))
        # print("loop", cluster_list, len(cluster_list))
        dbscan_new(dataset_update, epsilon, min_points, cluster_list)
    else:
        return cluster_list, len(dataset)
            # if not i_list: #if the list is not empty
            
            #     for list in i_list:
                
            # if euclidean_distance(dataset[i],dataset[j]) <= epsilon and :
epsilon = 1
min_clusters = 3
cluster_1 = []
    
print("Number of Clusters:",dbscan(dict_df,epsilon,min_clusters))
d = {
    "a": [1,2,4],
    "b": [1,2,4],
    "c": [1,2,4]
    }
cluster_2 = []
# [[first cluster]]
epsilon = 10
min_clusters = 3
# print(len(dict_df))
# print("Number of Clusters:",dbscan_new(dict_df,epsilon,min_clusters, cluster_2))
dbscan_new(dict_df,epsilon,min_clusters, cluster_2)
# print(type(dict_df[1]))
