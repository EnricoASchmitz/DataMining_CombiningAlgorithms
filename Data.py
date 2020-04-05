# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:25:11 2020

@author: enric
"""
import sys, getopt
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import Methods
import VisualizePlot as plotScript
import gc
import matplotlib.pyplot as plt
import itertools
from sklearn import preprocessing
# In[Standard values]: 
def main(argv):

    #standard values for the script, these are implemented to reduce time needed to run script.
    inputfile = './Data/heart.csv'#'./Data/Iris.csv'#
    trueclass =  'target'#"Species"#
    skip = []#["Id"]#
    output_dir = "./images/"

# In[Getopt]
    """
    Use getopt to make it possible to run script in the terminal/commandprompt
    If a input is missing or the filetype is not right the script will stop
    """
     
    options = 'Data.py -i <inputfile csv> -y <trueclass> -o <output directory> -s <columns to skip>'
    try:
        opts, args = getopt.getopt(argv,"hi:y:k:o:s",["ifile=", "trueclass=", "output=", "skip="])
    except getopt.GetoptError:
        print(options)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(options)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-y", "--trueclass"):
            trueclass = arg
        elif opt in ("-o", "--output"):
            output_dir = arg
        elif opt in ("-s", "--skip"):
            skip = arg
# In[Show and check input paramets]
    #print given parameters
    print('Input file is ', os.path.abspath(inputfile))
    print('True class is ', trueclass)
    print('Output directory is ', os.path.abspath(output_dir))    
    if skip:
        print('Column(s) to skip is/are ', skip)
        
    """
    Check if file is the right type
    And if output directory doesn't exist we should create it
    """
    
    if not inputfile.lower().endswith(".csv"):
        print("\nError: File needs to be .csv")
        sys.exit()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    

    #Get the file name without path and extension
    filename = os.path.splitext(os.path.basename(inputfile))[0]
    
    return pd.read_csv(inputfile),trueclass, output_dir, skip, filename 


if __name__ == "__main__":
# In[Setup data and parameters]:
    #Get input from getopt  
    Data, trueLabel, output_dir, skip, filename = main(sys.argv[1:])
    #Skip columns which are specified before
    for col in skip:
        del Data[str(col)]

    # Get the X and y values from the data
    y = Data[trueLabel]
    X = Data.drop([trueLabel], axis=1)
    
    #If plot is set to True it will plot all different combinations of collumns and shows how the classes are distributed
    plot=False
    if plot:
        for subset in itertools.combinations(X.columns, 2):
            plt.scatter(X[subset[0]],X[subset[1]], c=y)
            plt.show()
            
    # remove Data since we have splitted into X and y
    del Data
    gc.collect()
    
    #Get the amount of unique items in y
    Nc = len(np.unique(y))
    """
    Do RepeatedStratifiedKfold with n_splits and n_repeats, 
    we set splits to 5 since a high value gave an warning that one class was getting low frequencies. 
    And repeats is 3 to get a good idea of the accuracy which can still run without needing days to finish
    """
    n_splits=5
    n_repeats=3
    rskf = RepeatedStratifiedKFold(n_splits, n_repeats, random_state=1)
    
    """
    These values where chosen to be able to compare to the paper of 
    Van Der Merwe, D. W., & Engelbrecht, A. P. (2003).
    Data clustering using particle swarm optimization. 
    2003 Congress on Evolutionary Computation, CEC 2003 - Proceedings, 1, 215–220. https://doi.org/10.1109/CEC.2003.1299577
    Which used the same values
    """
    itt = 1000
    NParticles = 10
    sValue = 0.0001
    
    # Create a dictonairy to save the accuracy and the crossentropy of each kfold
    HC_ = {'acc':[],'je':[]}
    Kmeans_ = {'acc':[],'je':[]}
    PSO_ = {'acc':[],'je':[]}
    HC_Kmeans_ = {'acc':[],'je':[]}
    Kmeans_PSO_ = {'acc':[],'je':[]}
    HC_PSO_ = {'acc':[],'je':[]}
    HC_Kmeans_PSO_ = {'acc':[],'je':[]}
    Kmeans_SK_={'acc':[],'je':[]}
    Kmeans_SKInit_={'acc':[],'je':[]}
    
    # We use the label encoder to change Y to [0,1,...] because these values are easier to handle
    le = preprocessing.LabelEncoder()
    
# In[Kfold]
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = le.fit_transform(y.iloc[train_index]),le.transform(y.iloc[test_index])
        HC_clusters, acc, je = Methods.heirachical_clustering(X_train,y_train, X_test, y_test, Nc)
        HC_['acc'].append(acc)
        HC_['je'].append(je)
        gc.collect()
        Kmeans_clusters, acc, je = Methods.start_Kmeans(X_train,y_train, X_test, y_test,Nc, trueLabel,itt,sValue)
        Kmeans_['acc'].append(acc)
        Kmeans_['je'].append(je)
        gc.collect()
        acc, je = Methods.start_PSO(X_train, X_test, y_train, y_test,Nc,NParticles, itt)
        PSO_['acc'].append(acc)
        PSO_['je'].append(je)
        gc.collect()
        HC_Kmeans_clusters, acc, je = Methods.HC_Kmeans(X_train,X_test,y_train,y_test,Nc,HC_clusters,itt,sValue)
        HC_Kmeans_['acc'].append(acc)
        HC_Kmeans_['je'].append(je)
        gc.collect()
        acc, je = Methods.Hybrid_PSO(X_train,X_test,y_train,y_test,Nc,Kmeans_clusters,NParticles, itt)
        Kmeans_PSO_['acc'].append(acc)
        Kmeans_PSO_['je'].append(je)
        gc.collect()
        acc, je = Methods.Hybrid_PSO(X_train,X_test,y_train,y_test,Nc,HC_clusters,NParticles, itt)
        HC_PSO_['acc'].append(acc)
        HC_PSO_['je'].append(je)
        gc.collect()
        acc, je = Methods.Hybrid_PSO(X_train,X_test,y_train,y_test,Nc,HC_Kmeans_clusters,NParticles, itt)
        HC_Kmeans_PSO_['acc'].append(acc)
        HC_Kmeans_PSO_['je'].append(je)
        gc.collect()
        acc, je = Methods.Kmeans_SK(X_train,X_test,y_train,y_test,Nc,itt)
        Kmeans_SK_['acc'].append(acc)
        Kmeans_SK_['je'].append(je)
        gc.collect()
        acc, je = Methods.Kmeans_SKInit(X_train,X_test,y_train,y_test,Nc,itt)
        Kmeans_SKInit_['acc'].append(acc)
        Kmeans_SKInit_['je'].append(je)
        gc.collect()

# In[Save output to directory and delete all lists ]    
    accuracy_dict={"HC":HC_['acc'],"Kmeans":Kmeans_['acc'],"PSO":PSO_['acc'],"HC_Kmeans":HC_Kmeans_['acc'],"Kmeans_PSO":Kmeans_PSO_['acc'],"HC_PSO":HC_PSO_['acc'],"HC_Kmeans_PSO":HC_Kmeans_PSO_['acc'], "Kmeans_SK": Kmeans_SK_['acc'],"Kmeans_SKInit":Kmeans_SKInit_['acc']}
    Je_dict= {"HC":HC_['je'],"Kmeans":Kmeans_['je'],"PSO":PSO_['je'],"HC_Kmeans":HC_Kmeans_['je'],"Kmeans_PSO":Kmeans_PSO_['je'],"HC_PSO":HC_PSO_['je'],"HC_Kmeans_PSO":HC_Kmeans_PSO_['je'], "Kmeans_SK": Kmeans_SK_['je'],"Kmeans_SKInit":Kmeans_SKInit_['je']}
    del HC_, Kmeans_, PSO_, HC_Kmeans_, Kmeans_PSO_, HC_PSO_, Kmeans_SK_, Kmeans_SKInit_
    gc.collect()
    
# In[Plotting]
    # Random colors we tought where easy to see.
    color = ["red","black","navy","orange","green","purple","yellow","cyan","pink"]
    
    # Plot all different accuracy plots
    plotScript.plotM(accuracy_dict,output_dir,filename,n_splits,n_repeats, color)
    Basic= ["HC","Kmeans","PSO"]
    plotScript.plot_function(Basic, accuracy_dict,color, "Basic", output_dir, filename, n_splits, n_repeats)
    Kmeans = ["Kmeans","HC_Kmeans","Kmeans_PSO","HC_Kmeans_PSO","Kmeans_SK","Kmeans_SKInit"]
    plotScript.plot_function(Kmeans, accuracy_dict,color, "Kmeans", output_dir, filename, n_splits, n_repeats)
    HC = ["HC","HC_Kmeans","HC_PSO","HC_Kmeans_PSO"]
    plotScript.plot_function(HC, accuracy_dict,color, "HC", output_dir, filename, n_splits, n_repeats)
    PSO = ["PSO","HC_PSO","Kmeans_PSO","HC_Kmeans_PSO"]
    plotScript.plot_function(PSO, accuracy_dict,color, "PSO", output_dir, filename, n_splits, n_repeats)
    
    # plot the cross entropy
    plotScript.plot_function(Je_dict.keys(), Je_dict,color, "Methods", output_dir, filename, n_splits, n_repeats, ylab = "Quantization error", index = 0, outer ="min")
    

    
    


      
    