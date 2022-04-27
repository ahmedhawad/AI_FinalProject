


import pandas as pd
from numpy import count_nonzero, zeros_like
from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import make_moons
import math
import matplotlib.pyplot as plt
from kneed import KneeLocator

"""

# remove male female from dataset


Looking to fix dbscan 

get new datasets



name = main




"""



file_path = "User_Data.csv"



dataset = np.array(pd.read_csv(file_path))
dataset, not_used =  make_moons(
    n_samples=750, noise=0.03
)


# dataset = "Use_Data.csv"







# array_df = np.array(df)

dict_ds = {}
count = 1

for i in dataset: #makes dictionary { CustomerID : [numpy array of variables to eb used for dbscan]}
    dict_ds[count] =  i
    count+=1


# for i in dict_ds:
#     print(i, dict_ds[i])



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
        


def findEpsilon(dataset, minPts):
    kNearestDist = []
    #List of Lists, whose index corresponds to the point in the dataset. ie: index 2 is the list of distances from all points to point 2
    distances = []

    # print(dataset)
    for p1 in dataset:
        distP1 = []
        for p2 in dataset:
            distP1.append(euclidean_distance(dataset[p1], dataset[p2]))
        distances.append(distP1)

    for i in range(len(distances)):
        sortedDist = sorted(distances[i])
        kNearestDist.append(list(sortedDist)[minPts])
    kNearestDist.sort()
    plt.plot(kNearestDist)
    plt.show()

    maxCurve = []
    for i in range(1, len(kNearestDist)-1):
        secDerivative = kNearestDist[i+1] + kNearestDist[i-1] - 2 * kNearestDist[i]
        maxCurve.append(secDerivative)
    maxValue = max(maxCurve)
    maxIndex = maxCurve.index(maxValue) + 1
    epsilon = kNearestDist[maxIndex]

    kn = KneeLocator(range(0, 200), kNearestDist, curve='convex', direction='increasing', interp_method='interp1d')
    epsilon2 = kNearestDist[kn.knee]

    print(epsilon)
    print(epsilon2)

    return epsilon2



def dbscan(dataset,epsilon, min_points):
    """
    
    dataset is dict {key: [np array of variables used for dataset]}

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

    return len(clusters), noise(dataset,clusters), clusters


            # if not i_list: #if the list is not empty
            
            #     for list in i_list:
                

            # if euclidean_distance(dataset[i],dataset[j]) <= epsilon and :
                
    


min_clusters = 2
# epsilon = findEpsilon(dict_ds,min_clusters)
epsilon = .1

output = dbscan(dict_ds,epsilon,min_clusters)
    
print("Number of Clusters:")
















