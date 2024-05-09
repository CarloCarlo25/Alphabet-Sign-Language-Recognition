# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:33:19 2024

@ Algorithm Creator: Dr. Mark Lexter D. De Lara
@ Code Author: John Carlo Alonzo
@ Affiliation: University of the Philippines Los Baños
@ Reference: De Lara, Mark Lexter D. "Persistent homology classification algorithm.
             PeerJ Computer Science 9 (2023): e1195.
"""

##############################################################
###### PERSISTENT HOMOLOGY CLASSIFICATION ALGORITHM ##########
##############################################################
# The following assumptions should be satisfied before using this module
# 1. The feature dataset must be cleaned. 
# 2. The datapoints are expected to be at least 2-dimensional vector.
# 3. The dimension of all data points should be consistent, i.e. dim(a) = dim(b) for all a,b in the dataset.
# 4. The classes should be represented by positive integers.
# 5. The data type of the dataset must be pandas dataframe
# 6. On predict function, the data point should have the same dimension as the training data
# 7. The predictee should be a numpy array


import pandas as pd
import numpy as np 
from ripser import ripser

class PersistentHomologyClassifier: ## Creates a blueprint of the object PersistentHomologyClassifier(PHC)
    
    def __init__(self, X_train, Y_train, maxsc): #Defines the attributes of the object
        self.X_train = X_train 
        self.Y_train = Y_train 
        self.maxsc = maxsc     


    def fit(self): # This is the fit method of the object(PHC)
        diagram_list = [] # Stores the PD obtained for each classes after the pred was added into each
        class_list = [] # Stores the classes obtained from the dataset

        self.X_train.index = self.Y_train # Calls the attribute X_train and concatenatesa label array on the first column of the
        strata = self.X_train.groupby(self.X_train.index) # Stratifies the dataset according to classes and assigned to the variable strata
        for category, group_data in strata: # this for loop will create the PD for each class
            category_data = group_data.iloc[:, 0:].to_numpy() # This will select all of the data point in a specific class 
        
            # Compute the persistence diagram using Ripser
            diagrams = ripser(category_data, thresh = self.maxsc)['dgms'][0] # after selecting the point cloud on a specific class, a VR filtration was applied with a specified maximum epsilon
            diagrams[np.isinf(diagrams)] =self.maxsc # this is to remove the infinity value in diagram and replace it with a specified maximum epsilon
            
            class_list.append(category) # This will add the current class in class_list
            
            # Append the diagrams to the list
            diagram_list.append(diagrams) # This will add the current diagram into diagrams_list
    
        return class_list, diagram_list # when the PersistenceHomologyClassifier.fit() was called, those two output was returned to the workspace
    
    
    
    def predict(self, pred, diagram_list, class_list): # This is the predict method of the object PHC
        
        diagram_list_test = [] # Stores the PD obtained for each classes after the pred was added into each
        self.X_train.index = self.Y_train 
        strata = self.X_train.groupby(self.X_train.index)
        
        for category, group_data in strata: # Creates PD for each class, this time the pred was added into each
            category_data = np.append(group_data.iloc[:, 0:].to_numpy(), pred, axis=0) # Adds the point to be predicted into the current class
    
            # Compute the persistence diagram using Ripser
            diagrams = ripser(category_data, thresh = self.maxsc)['dgms'][0] # Computes the PD for the current class using VR Filtration with maximum epsilon
            diagrams[np.isinf(diagrams)] = self.maxsc # removes infinity value in PD and replaces with maximum epsilon
    
            # Append the diagrams to the list
            diagram_list_test.append(diagrams) # Stores the current diagram to the diagrams_list
            
            
            ##### Score 
        np.set_printoptions(precision=8) # to ensure that the machine is not truncating the value
        k = [] # This is where we store the score of classwise difference of PD
        for j in range(0, len(diagram_list)): # This loop computes the classwise difference of PD
            sum_X_i = np.sum(diagram_list[j][:,1]) # Training PD indexing
            sum_Y_i = np.sum(diagram_list_test[j][:,1]) # Testing PD indexing
            score_X_i = np.abs(sum_X_i - sum_Y_i) # Computing the difference
            k = np.append(k, score_X_i) # Stores the current difference to k
        l=min(k) # returns the minimum of k
        
        for h in range(0, len(k)): # this for loop selects the index at which the minimum occurs, if minimum was found 
            if k[h] == l:
                pred = class_list[h]
                
                
        return pred