#DATA ATTRIBUTES
#----------------------------------------------------------------------------------------------------------
#df_train is a pandas dataframe holding the information from the training dataset

#df_test is a pandas dataframe holding the information from the test dataframe (could be a hyperparamter tuning set)

#distance_matrix is a 2-D numpy array that holds the distances between each pair of training examples 

#k is an integer representing the number of clusters. This is a hyperparameter that will be tuned

#sigma is a float representing the bandwidth of the Guassian kernel. This is a hyperparameter that will be tuned

#centroids is a list holding the current centroids of the dataset. This is updated by cluster_data()

#clusters is a list that will hold dictionaries mapping cluster IDs to the examples contained in the cluster

#features is a list that will hold the feature vectors in the training set
#----------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import random
import math

class KMeansClustering:
    def __init__(self, df_train, df_test, distance_matrix, categorical_columns, numerical_columns, k):
        self.df_train = df_train
        self.df_test = df_test
        self.distance_matrix = distance_matrix
        self.categorical = categorical_columns
        self.numerical = numerical_columns
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
                if(column != "class" and column != "id" and column != "value"):
                    feature_vector.append(self.df_train.loc[i, column])
            self.features.append(feature_vector)
        print(self.features)
        return(self.features)

#----------------------------------------------------------------------------------------------------------

    def cluster_data(self):
        converged = False
        first_time = True
        old_centroids = []

        #Selects k random training examples to be the initial centroids
        for i in range(self.k):
            centroid = random.randint(0, len(self.features) - 1)
            while(self.features[centroid] in self.centroids):
                print(centroid)
                centroid = random.randint(0, len(self.features) - 1)
            self.centroids.append(self.features[centroid])
        print(self.centroids)

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
                converged = True
                
            #If it's not the first time, use calculate_distance() to get distances and assign vectors to
            #appropriate clusters
            else:
                for i in range(len(self.features)):
                    centroid_distances = []
                    for centroid in self.centroids:
                        centroid_distances.append(self.calculate_distance(self.features[i], centroid))
                    index = centroid_distances.index(min(centroid_distances))
                    self.clusters[index[tuple(self.centroids[index])]].append(i)
                print(self.clusters)   

            #Save old centroids before updating 
            old_centroids = self.centroids

            #Update centroids 
            self.centroids.clear()
            for cluster in self.clusters:
                pass



#----------------------------------------------------------------------------------------------------------

    def classify_one(self):
        pass

#----------------------------------------------------------------------------------------------------------

    def classify_all(self):
        pass

#----------------------------------------------------------------------------------------------------------
    
    def regress_one(self):
        pass

#----------------------------------------------------------------------------------------------------------

    def regress_all(self):
        pass

#----------------------------------------------------------------------------------------------------------

    def tune_hypers(self):
        pass

#----------------------------------------------------------------------------------------------------------

    def reset_hypers(self):
        pass
        
#----------------------------------------------------------------------------------------------------------

    #Calculates the Euclidean distance between to feature vectors. This function is only used for numerical columns
    def euclidean_distance(self, vector1, vector2):
        numerical_indices = []
        for i in range(len(self.df_train.columns)):
            if(self.df_train.columns[i] in self.numerical):
                numerical_indices.append(i)

        sum_of_squared_differences = 0
        index = 0
        for column in self.numerical:
            val1 = vector1[numerical_indices[index]]
            val2 = vector2[numerical_indices[index]]
            squared_difference = pow((val2 - val1), 2)
            sum_of_squared_differences += squared_difference
            index += 1

        euclidean_distance = math.sqrt(sum_of_squared_differences)
        return(euclidean_distance)

#----------------------------------------------------------------------------------------------------------

    #Calculates the value difference metric between two feature vectors. This function is only used for categorical
    #columns
    def value_difference_metric(self, vector1, vector2):
    
        #If the task is regression, use the value column for VDM (NOT SURE IF THIS IS CORRECT)
        if("value" in self.df_train.columns):
            value_difference_metric = 0
            for column in self.categorical:
                val1 = self.df_train.loc[vector1, column]
                val2 = self.df_train.loc[vector2, column]

                sum_of_targets1 = 0
                sum_of_targets2 = 0
                mean1 = 0
                mean2 = 0
                num_of_instances1 = 0
                num_of_instances2 = 0

                #Need to change
                for i in range(len(self.df_train)):
                    if(self.df_train.loc[i, column] == val1):
                        sum_of_targets1 += self.df_train.loc[i, "value"]
                        num_of_instances1 += 1
                mean1 = sum_of_targets1/num_of_instances1

                for i in range(len(self.df_train)):
                    if(self.df_train.loc[i, column] == val2):
                        sum_of_targets2 += self.df_train.loc[i, "value"]
                        num_of_instances2 += 1
                mean2 = sum_of_targets2/num_of_instances1

                value_difference_metric += abs(mean1 - mean2)
            
            return(value_difference_metric)
            

        #If the task is classification, use the class column for VDM
        else:
            
            categorical_indices = []
            for i in range(len(self.df_train.columns)):
                if(self.df_train.columns[i] in self.categorical):
                    categorical_indices.append(i)

            #Obtain list of classes
            classes = []
            for Class in self.df_train["class"]:
                if(Class not in classes):
                    classes.append(Class)

            #Intitialize variables to store steps of calculation
            value_difference_sum = 0
            value_difference_metric = 0
            index = 0

            #For each categorical column, find the value difference between vector1 and vector2
            for column in self.categorical:
                
                #Initialize variables to store steps of calculation
                val1 = vector1[categorical_indices[index]]
                val2 = vector2[categorical_indices[index]]
                print("Val1: {}, Val2: {}".format(val1, val2))
                C_i1 = 0
                C_i2 = 0
                C_i_a1 = 0
                C_i_a2 = 0
                sum_over_classes = 0

                #For each class, find (abs((C_i,a/C_i) - (C_j,a/C_j)))^2
                for Class in classes:  
                    calculation = 0    
                    for i in range(len(self.df_train)):
                        if(self.df_train.loc[i, column] == val1): 
                            C_i1 += 1
                        if(self.df_train.loc[i, column] == val1 and self.df_train.loc[i, "class"] == Class):
                            C_i_a1 += 1
                    for i in range(len(self.df_train)):
                        if(self.df_train.loc[i, column] == val2):
                            C_i2 += 1
                        if(self.df_train.loc[i, column] == val2 and self.df_train.loc[i, "class"] == Class):
                            C_i_a2 += 1

                    calculation = pow((abs((C_i_a1/C_i1) - (C_i_a2/C_i2))), 2)

                    #Accumulate results to obtain delta(v_i, v_j)
                    sum_over_classes += calculation
                    C_i1 = 0
                    C_i2 = 0
                    C_i_a1 = 0
                    C_i_a2 = 0

                #Accumulate the delta(v_i, v_j)s over each categorical feature
                value_difference_sum += sum_over_classes

                index += 1

            #Take sqrt(value_difference_sum) to obtain the value distance metric
            value_difference_metric = math.sqrt(value_difference_sum)
            return(value_difference_metric)
            
#----------------------------------------------------------------------------------------------------------

    #Calculates the total distance between vector1 and vector2 using euclidean_distance() and value_difference_metric()
    def calculate_distance(self, vector1, vector2):
        total_distance = ((len(self.numerical)/len(self.features))*self.euclidean_distance(vector1, vector2)) + ((len(self.categorical)/len(self.features))*self.value_difference_metric(vector1, vector2))
        return(total_distance)
    
#----------------------------------------------------------------------------------------------------------

#This is purely for bug testing
data = {
    #numerical columns
    'Temperature': [78.5, 97.1, 99.002, 88.5, 66.907],
    'Precipitation': [10.08, 7.99, 78.64, 33.33, 23.89],
    'wind speed': [78, 0.9, 65, 25, 7.9],
    #categorical columns
    'color': ["red", "blue", "blue", "red", "red"],
    'weather': ["sunny", "overcast", "sunny", "sunny", "overcast"],
    'class': [1, 2, 2, 1, 2]
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
numerical = ["Temperature", "Precipitation", "wind speed"]
categorical = ["color", "weather"]
k = 2
clustering = KMeansClustering(df_train, df_test, distance_matrix, categorical, numerical, k)
features = clustering.get_features()
print(distance_matrix)
clustering.cluster_data()
print("Euclidean Distance:", clustering.euclidean_distance(features[0], features[1]))
print("Value Difference Metric:", clustering.value_difference_metric(features[0], features[1]))