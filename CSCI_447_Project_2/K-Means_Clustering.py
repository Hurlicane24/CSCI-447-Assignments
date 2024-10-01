#DATA ATTRIBUTES
#----------------------------------------------------------------------------------------------------------
#df_train is a pandas dataframe holding the information from the training dataset

#df_test is a pandas dataframe holding the information from the test dataframe (could be a hyperparamter tuning set)

#distance_matrix is a 2-D numpy array that holds the distances between each pair of training examples 

#k is an integer representing the number of clusters. This is a hyperparameter that will be tuned

#sigma is a float representing the bandwidth of the Guassian kernel. This is a hyperparameter that will be tuned

#centroids is a list holding the current centroids of the dataset. This is updated by cluster_data()

#clusters is a list that will hold dictionaries mapping cluster IDs to the examples contained in the cluster

#features is a list that will hold dictionaries mapping feature vector IDs to a list of its feature values
#----------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import random

class KMeansClustering:
    def __init__(self, df_train, df_test, distance_matrix, k):
        self.df_train = df_train
        self.df_test = df_test
        self.distance_matrix = distance_matrix
        self.k = k
        self.sigma = 0.1
        self.centroids = []
        self.clusters = []
        self.features = []

#----------------------------------------------------------------------------------------------------------

    #Creates a list of feature vectors in the training set which are represented as dictionaries
    def get_features(self):
        for i in range(len(self.df_train)):
            feature_vector = []
            for column in self.df_train.columns:
                if(column != "class" and column != "id"):
                    feature_vector.append(self.df_train.loc[i, column])
            self.features.append(feature_vector)

#----------------------------------------------------------------------------------------------------------

    def cluster_data(self):
        converged = False
        first_time = True

        #Selects k random training examples to be the initial centroids
        for i in range(self.k):
            centroid = random.randint(0, len(self.features) - 1)
            while(self.features[centroid] in self.centroids):
                print(centroid)
                centroid = random.randint(0, len(self.features) - 1)
            self.centroids.append(self.features[centroid])
        print(self.centroids)

        old_centroids = []
        while(converged == False):

            #Fill keys of the dictionaries in self.clusters with the current centroids
            for i in range(len(self.centroids)):
                cluster_dict = {}
                cluster_dict[tuple(self.centroids[i])] = []
                self.clusters.append(cluster_dict)

            #If it's the first time through the loop, access distance_matrix for distances and assign vectors to
            #appropriate clusters
            if(first_time):            
                for i in range(len(self.features)):
                    centroid_distances = []
                    for centroid in self.centroids:
                        centroid_distances.append(self.distance_matrix[i][self.features.index(centroid)])
                    index = centroid_distances.index(min(centroid_distances))
                    self.clusters[index][tuple(self.centroids[index])].append(i)
                print(self.clusters)
                first_time = False
                
            #If it's not the first time, use calculate_distance() to get distances and assign vectors to
            #appropriate clusters
            else:
                pass

            #If convergence property is not met, calculate new centroids and run again

            #If convergence property is met, stop

#----------------------------------------------------------------------------------------------------------

    def classify_one():
        pass

#----------------------------------------------------------------------------------------------------------

    def classify_all():
        pass

#----------------------------------------------------------------------------------------------------------
    
    def regress_one():
        pass

#----------------------------------------------------------------------------------------------------------

    def regress_all():
        pass

#----------------------------------------------------------------------------------------------------------

    def tune_hypers():
        pass

#----------------------------------------------------------------------------------------------------------

    def reset_hypers():
        pass

#----------------------------------------------------------------------------------------------------------

#This is purely for bug testing
data = {
    'Temperature': [78.5, 97.1, 99.002, 88.5, 66.907],
    'Precipitation': [10.08, 7.99, 78.64, 33.33, 23.89],
    'wind speed': [78, 0.9, 65, 25, 7.9]
}

distance_matrix = np.zeros((5,5))
skip = []
for i in range(5):
    for j in range(5):
        if(i == j):
            distance_matrix[i][j] = 0

        elif((j,i) in skip):
            pass

        else:
            distance = random.uniform(0, 15)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
            skip.append((j,i))

df_train = pd.DataFrame(data)
df_test = pd.DataFrame()
k = 2
clustering = KMeansClustering(df_train, df_test, distance_matrix, k)
clustering.get_features()
print(distance_matrix)
clustering.cluster_data()