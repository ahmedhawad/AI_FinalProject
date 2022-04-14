


import pandas as pd
from numpy import zeros_like
from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import math


dataset = "Use_Data.csv"

old_df = pd.read_csv(dataset)



variables = ["CustomerID","Gender","Age","AnnualIncome","SpendingScore" ]



df = old_df[variables].replace({"Male": 1, "Female": 0,}).astype(int)


array_df = np.array(df)

dict_df = {}

for i in array_df: #makes dictionary { CustomerID : [numpy array of variables to eb used for dbscan]}
    dict_df[i[0]] = i[1:]

# for i in dict_df:
#     print (i, dict_df[i])


for i in dict_df:
    print(i, dict_df[i])



"""
loops through everyone and if there is someone matchign the minimum distance combines them in
"""



def euclidean_distance( point1, point2):
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

            if i != j:


                distance = euclidean_distance(dataset[i],dataset[j])#distance  between the two
                #problem is that the our scale is diff than sklearn. I think they normalize their data, we don'y
                within_bounds = distance <= epsilon #boolean


                # print(distance,within_bounds)
                if not within_bounds: #if it meets epsilon criteria  

                    if not cluster_check(clusters, j): #it is already not part of a group
                         
                        #no need to check if it's in i_list because it can only ever be added there once
                        
                        i_list.append(j)

        """
        Add a method to get noise points
        """

        if len(i_list) >= min_points:  #if it meets min points criteria
                clusters.append(i_list)

    return len(clusters), noise(dataset,clusters)


            # if not i_list: #if the list is not empty
            
            #     for list in i_list:
                

            # if euclidean_distance(dataset[i],dataset[j]) <= epsilon and :
                
    


epsilon = 1
min_clusters = 3

    
print("Number of Clusters:",dbscan(dict_df,epsilon,min_clusters))






d = {
    "a": [1,2,4],
    "b": [1,2,4],
    "c": [1,2,4]

    }













