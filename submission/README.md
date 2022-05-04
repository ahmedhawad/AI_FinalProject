# b351-21-project
Final Project Group 21


Density Based Spatial Clustering Algorithm with Noise (DBSCAN)

B351 Introduction to AI 
This is the final project for Group 21
By: Ahmed Awad, Ting Wei Chou, Matthew Hughes, Nawaf Aloufi	


Project Goal:
    - Build our own custom dbscan
    - Compare it's accuracy to sklean's dbscan
    - Showcase how our dbscan compares to other clustering algorithms
    - Allow users to input their own file to be clustered to be used for general purposes




There are 2 files, Clustering.py,  CreditCard_User_Data.csv



1. Clusterings.py :

    __This is where all of our code is.__ We created a custom DBSCAN and compared it to sklearn's DBSCAN, Kmeans, Affinity Propogation, and MeanShift Clustering algorithms. The datasets we used are make_moons and make_circles from sklearn.dbscan and User_Data.csv from our own directory. 

    __When you run the file,__ it will output 15 graphs to showcase the clustering algorithms at once. It will also print out our accuracy via homogoneity score, rand score, and completness score for our custom built dbscan and sklearn's dbscan. 

   __To cluster your own dataset,__ run dbscan_my_dataset(), as an input in the temrinal, enter your file path, epsilon, and minimum number of points per cluster. By default this function shows 2d graphs and is looking for a csv. To update either of these, just change what the file type is in pandas.readtype and plt.scatter() axis. Please ensure all your dataset only contains numbers.



2. User_Data.csv:

    This holds our sample dataset of users to be used to cluster. 

