


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




# file_path = "User_Data.csv"



# dataset = np.array(pd.read_csv(file_path))
dataset, not_used =  make_moons(
    n_samples= 700, noise=0.08
)


# print(dataset)
# print(not_used)

# dataset = "Use_Data.csv"

# for i in not_used:
#     print(i)




# print(len(not_used))


new = []
for i in range(10):

    new.append(1)

# print(len(new))
x = dataset

count = 0

# for i in x:
#     if i[0]>1:
#         x[i] == [1]
#     else:
#         x[i] == [0]
        
        

labeling = list([1]*len(dataset))

# array_df = np.array(df)

dict_ds = {}
count = 1

for i in dataset: #makes dictionary { CustomerID : [numpy array of variables to eb used for dbscan]}
    dict_ds[count] =  i
    count+=1


# print(len(dataset))
# print(dataset)

# for i in dict_ds:
#     print(i, dict_ds[i])

cluster_label = np.shape(len(dict_ds[1]))

# print(cluster_label)



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



def already_clustered(clusters, point):

    """
    True if point is in the list
    False if the point is not in the list


    [ [1,2,4] ]
    """

    for thelist in clusters:
        if point in thelist:
            return True
    return False


def noise(dataset,clusters):
    """
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
            # if i != j:
                distance = euclidean_distance(dataset[i],dataset[j])#distance  between the two
                within_bounds = distance >= epsilon #boolean

                if within_bounds: #if it meets epsilon criteria  

                    if not already_clustered(clusters, j):
                        i_list.append(j) #it is already not part of a group   
                        print(i_list)               
        if len(i_list) >= min_points:  #if it meets min points criteria
                clusters.append(i_list)

    label = [-1]*(len(dataset))
    count = 0
    for i in clusters:
        for j in i: 
            label[j-1] = count
        count +=1





    return len(clusters), noise(dataset,clusters), clusters,label

# print(3)


            # if not i_list: #if the list is not empty
            
            #     for list in i_list:
                

            # if euclidean_distance(dataset[i],dataset[j]) <= epsilon and :
                
    


min_clusters = 10 # epsilon = findEpsilon(dict_ds,min_clusters)
epsilon = .5

output = dbscan(dict_ds,epsilon,min_clusters)

new = output[3]
    
print("Number of Clusters:", output[0], )
# print(output[3])

# print(dataset[:,0],output[2])

# for i in output[2]:
#     print(i)

plt.scatter(x= dataset[:,0], y= dataset[:,1], c = new)

plt.show()



















