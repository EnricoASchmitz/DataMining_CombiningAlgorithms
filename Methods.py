# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:26:20 2020

@author: enric
"""
# In[Imports]:


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from sklearn.cluster import AgglomerativeClustering, KMeans
from tqdm import tqdm

# In[Functions]:
# Function to check the difference between 2 listst
def diff(list1, list2):
    c = set(list1).union(set(list2))  # or c = set(list1) | set(list2)
    d = set(list1).intersection(set(list2))  # or d = set(list1) & set(list2)
    return list(c - d)

# remap the prediction to y so we can calculate accuracy
def ReMap(y, y_pred):
    pred = ["Missing"] * len(y_pred)
    # get for each value the most occurring counter part this value can be the same
    df= pd.crosstab(index = y, columns = np.asarray(y_pred))
    df = pd.DataFrame(df.idxmax(axis=1))
    df.columns = ["pred"] 
    df.index.name = 'True'
    df.reset_index(inplace=True)
    # replace the values in the prediction to map to y
    for number, row in df.iterrows():
        indices = [i for i, x in enumerate(y_pred) if x == row[1]]
        for index in indices:
            pred[index] = row[0]
    # fill in the missing values
    if "Missing" in pred:
        indices = [i for i, x in enumerate(pred) if x == "Missing"]
        difference = diff(list(y),list(pred))
        difference.remove("Missing")
        for index in indices:
            pred[index] = difference[0]
    return pred

# can calculate the accuracy of a prediciton
def accuracy(listA,listB):
    listA = list(listA)
    listB = list(listB)
    correct = 0
    for element in range(len(listA)):
        if (listA[element] == listB[element]):
            correct += 1
    acc = (correct/len(listA)) * 100
    return round(acc,2)

# function used to reduce the memory needed
def round_centriods(Centriods):
    Centriods = [round(num, 4) for num in Centriods]
    return Centriods

#calculate quatinazation error
def quatinazation_error(dist,Nc):
    # calculate quatinazation error
    denominator = []
    for cluster, points in dist.items():
        Cij = len(points)
        distSum = []
        for dist in points:
            distSum.append(sum([x / Cij for x in dist]))
        denominator.append(sum(distSum))
    Je = sum(denominator)/Nc
    return [Je]
# In[Heirachical]:

def heirachical_clustering(X_train,y_train,X_test,y_test, Nc):
    # use SKlearn Hierachical clustering since doing this manualy was to heavy for our PC.
    aggl = AgglomerativeClustering(n_clusters = Nc).fit(X_train)
    # Make the prediction have the same labeling as y
    pred_y_train = ReMap(y_train,aggl.labels_)
    # Pick the middle of each cluster as centriod to use in other clustering algorithms
    cluster = []
    for item in np.unique(pred_y_train):
        indices = [i for i, x in enumerate(pred_y_train) if x == item]
        centriod= X_train.iloc[indices].mean(axis = 0).tolist()
        # round the centriods at cost of some precision to reduce memory
        centriod= round_centriods(centriod)
        cluster.append(centriod)
    # calculate training and testing accuracy        
    train_acc = accuracy(y_train,pred_y_train)
    pred_y_test = ReMap(y_test,aggl.fit_predict(X_test))
    test_acc = accuracy(y_test,pred_y_test)
    # get the labels and how often they occure 
    labels = np.unique(y_test,return_counts=True)
    dummie, dist = PSO_predictor(X_test,labels,cluster)
    Je = quatinazation_error(dist,Nc)
    return cluster, [train_acc,test_acc], Je


# In[Kmeans]:    
# kmeans
def kmeans_Fit(X,y, Nc, clusters):
    # get the labels and how often they occure 
    labels = np.unique(y,return_counts=True)
    y_pi = {}
    centriods = []
    # itterate over all rows/samples
    for index, row in X.iterrows():
        # get one sample
        Zp = list(row)
        distances = []
        
        # loop over all samples and get the closest centriod
        for Mij in clusters:
            d= []
            for k in range(len(Mij)):
                d.append((Zp[k] - Mij[k]) ** 2)
            distances.append(math.sqrt(sum(d)))
        key = labels[0][distances.index(min(distances))]
        # write the sample to the closest centriod
        if key in y_pi:
            y_pi[key].append(Zp)
        else:
            y_pi[key] = [Zp]
    # adjust the centriods
    for cluster, points in y_pi.items():
            nj = len(points)
            Mj = [element * (1/nj) for element in [sum(i) for i in zip(*points)]]
            centriods.append(Mj)
    # predict the accuracy of the training set
    acc, dummie = kmeans_predict(clusters, X, y, Nc, plot = False)
    return centriods, acc

# very similair to the fitting but doesn't adjust centriods and this time calculates cross entropy for test
def kmeans_predict(clusters, X, y, Nc, plot = False):
    y_kmeans = []
    y_pi= {}
    #loop over all samples
    for index, row in X.iterrows():
        Zp = list(row)
        distances = []
        labels = np.unique(y,return_counts=True)
        #get one centroid
        for Mij in clusters:
            d= []
            # for all points compare to all centroids
            for k in range(len(Mij)):
                #euclidean distance
                d.append((Zp[k] - Mij[k]) ** 2)
            distances.append(math.sqrt(sum(d)))
        # get the label of the closest centroid
        key = labels[0][distances.index(min(distances))]
        # save prediction
        y_kmeans.append(key)
        if key in y_pi:
            y_pi[key].append(distances)
        else:
            y_pi[key] = [distances]
    Je = quatinazation_error(y_pi,Nc)
    # remap prediction to y
    y_k = ReMap(y,y_kmeans)
    acc = accuracy(y,y_k)
    # if given plot == True then will plot the first and second feature
    if plot:
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
        centers = clusters
        plt.scatter([item[0] for item in centers],[item[1] for item in centers], c='black', s=200, alpha=0.5);
        plt.show()
    return acc, Je

def Kmeans(X_train,X_test,y_train,y_test,Nc,clusters,itt,sValue):
    Stop = False
    i = 0 
    # set the best_entropy as high as possible since you want it low
    best_Je= [math.inf]
    for itteration in range(itt):
        if Stop==False:
            # fit the training set and get accuracy and updated clusters
            clust, train_acc = kmeans_Fit(X_train,y_train, Nc, clusters)
            # predict for the training set
            test_acc, Je = kmeans_predict(clust, X_test, y_test, Nc)
            # Early stop such that if little to no change is observed for 5 itterations it will end.
            if abs(sum(np.array(clust).flatten())-sum(np.array(clusters).flatten())) < sValue:
                i += 1       
                if i == 5:
                    Stop = True
            else:
                i = 0
            # only update the centriods if the cross entropy is lower
            if  best_Je[0] > Je[0]:
                clusters = clust
                best_Je = Je
    return clusters, train_acc, test_acc, best_Je

def start_Kmeans(X_train,y_train, X_test, y_test,Nc, trueLabel,itt,sValue, kmeans = True):
    y_set = np.unique(y_train)
    clusters= []
    # create random centriods
    for ylabel in y_set:
        centriod = list(X_train.iloc[random.choice(np.where(y_train== ylabel)[0].tolist())])
        centriod= round_centriods(centriod)
        clusters.append(centriod)
    # perform kmeans with the centriods
    clusters, train_acc, test_acc, Je = Kmeans(X_train,X_test,y_train,y_test,Nc,clusters,itt,sValue)
    
    return clusters, [train_acc,test_acc] , Je


# In[PSO]:


#PSO
def initialize_PSO(NParticles,X,y ):
    y_set = np.unique(y)
    particles = []
    # for each particle create a random point (similair to a centriod)
    for particle in range(NParticles):
        clusters= []
        for ylabel in y_set:
            clus = round_centriods(X.iloc[random.choice(np.where(y== ylabel)[0].tolist())])
            clusters.append(clus)
        particles.append(clusters)
    return(particles)

def start_PSO(X_train, X_test, y_train, y_test,Nc,NParticles, itt):
    # create the particles
    init_particles = initialize_PSO(NParticles,X_train,y_train)
    # Do the prediction for test and training
    pred,Je = PSO(X_train, X_test, y_train, y_test, NParticles, itt, init_particles, Nc)
    # calculate accuracy and cross entropy
    train_acc = accuracy(y_train,pred[0])
    test_acc = accuracy(y_test,pred[1])
    return [train_acc, test_acc], Je

def PSO_predictor(X, labels, Pi):
    y_pi = {}
    pred = []
    # itterate over X and find the closest centriod
    for index, row in X.iterrows():
        Zp = list(row)
        distances = []
        for Mij in Pi:
            d= []
            for k in range(len(Mij)):
                #euclidean distance
                d.append((Zp[k] - Mij[k]) ** 2)
            distances.append(math.sqrt(sum(d)))
        # pick the closest centriod as label
        key = labels[0][distances.index(min(distances))]
        if key in y_pi:
            y_pi[key].append(distances)
        else:
            y_pi[key] = [distances]
        pred.append(key)
    return pred, y_pi 



def PSO(X_train, X_test, y_train, y_test, NParticles, itt, init_particles, Nc):
    # get all occurences for all labels in y test and y train
    labels_test = np.unique(y_test,return_counts=True)
    labels_train = np.unique(y_train,return_counts=True)
    Inf = math.inf
    # set local best en global best best quatinazation error
    local_best = [[Inf,[]]] * NParticles
    global_best = [-1,Inf]
    # parameters for PSO taken from the original paper 
    v = [0] * NParticles
    w = 0.72
    c1 = 1.49
    c2 = 1.49
    # get a progressbar to see what is happening since it can take a long time
    with tqdm(total=itt, position=0, leave=True) as pbar:
        for itteration in tqdm(range(itt), position=0, leave=True):
            for i, Pi in enumerate(init_particles):
                # create 2 random variables
                r1 = random.random()
                r2 = random.random()
                # predict test
                pred_test, y_pi = PSO_predictor(X_test,labels_test,Pi)
                # predict train
                pred_train, dummie = PSO_predictor(X_train,labels_train,Pi)
                Je = quatinazation_error(y_pi,Nc)[0]
                # set the global best to the best score over all particles
                if global_best[1] > Je:
                    global_best = [i,Je, Pi, pred_train , pred_test]
                # set the local best which is given to all particles
                if local_best[i][0] > Je:
                    local_best[i] = [Je,Pi]
                # calculate the change and direction of the particle
                Vik = w*v[i] + c1 * r1 * np.subtract(local_best[i][1], Pi) + c2*r2*np.subtract(global_best[2],Pi)
                v[i] = Vik
                xit1 = Pi + Vik
                init_particles[i] = xit1
            pbar.update()
        return [global_best[3],global_best[4]], [global_best[1]]
            
        
    


# In[HC_Kmeans]:
# perform kmeans with the clusters from hierachical clustering
def HC_Kmeans(X_train,X_test,y_train,y_test,Nc,HC_clusters,itt,sValue):
    clusters, train_acc, test_acc, Je = Kmeans(X_train,X_test,y_train,y_test,Nc,HC_clusters,itt,sValue)
    return clusters,[train_acc,test_acc], Je

# In[Kmeans_PSO & HC_PSO]:

# perform PSO with given centriods, this can for example come from heirachical clustering or kmeans
def Hybrid_PSO(X_train,X_test,y_train,y_test,Nc,Cluster,NParticles, itt):
    init_particles =[]
    #The first particle is from the other method
    init_particles.append(Cluster)
    # all other particles are again random
    init_particles[1:] = initialize_PSO(NParticles-1,X_train,y_train)
    # get the training and test accuracy and cross entropy
    pred, Je = PSO(X_train, X_test, y_train, y_test, NParticles, itt, init_particles, Nc)
    train_acc = accuracy(y_train,pred[0])
    test_acc = accuracy(y_test,pred[1])
    return [train_acc, test_acc], Je


# In[Kmeans]
    
def Kmeans_SK(X_train,X_test,y_train,y_test,Nc,itt):
    # Use kmeans with random init from sklearn which means you get a normal kmeans
    KM_SK = KMeans(n_clusters=Nc,init="random", random_state=1,max_iter = itt).fit(X_train)
    # compute test and training accuracy  and use test for the cross-entropy
    pred_y_train = ReMap(y_train,KM_SK.labels_)
    train_acc = accuracy(y_train,pred_y_train)
    pred_y_test = ReMap(y_test,KM_SK.predict(X_test))
    test_acc = accuracy(y_test,pred_y_test)
    labels = np.unique(y_test,return_counts=True)
    cluster = (KM_SK.cluster_centers_).tolist()
    dummie, dist = PSO_predictor(X_test,labels,cluster)
    Je = quatinazation_error(dist,Nc)
    return [train_acc, test_acc], Je

# In[Kmeans++]
    
def Kmeans_SKInit(X_train,X_test,y_train,y_test,Nc,itt):
    # use kmeans++ from sklearn
    KMI_SK = KMeans(n_clusters=Nc,max_iter = itt).fit(X_train)
    # compute test and training accuracy  and use test for the cross-entropy
    pred_y_train = ReMap(y_train,KMI_SK.labels_)
    train_acc = accuracy(y_train,pred_y_train)
    pred_y_test = ReMap(y_test,KMI_SK.predict(X_test))
    test_acc = accuracy(y_test,pred_y_test)
    labels = np.unique(y_test,return_counts=True)
    cluster = (KMI_SK.cluster_centers_).tolist()
    dummie, dist = PSO_predictor(X_test,labels,cluster)
    Je = quatinazation_error(dist,Nc)
    return [train_acc, test_acc], Je