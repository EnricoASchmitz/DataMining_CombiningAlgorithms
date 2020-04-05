# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:43:50 2020

@author: enric
"""
import matplotlib.pyplot as plt
from statistics import mean
import math

# Used to see if the number of subplots to be made is a prime number
def isPrime(n) : 
  
    # Corner cases 
    if (n <= 1) : 
        return False
    if (n <= 3) : 
        return True
  
    # This is checked so that we can skip  
    # middle five numbers in below loop 
    if (n % 2 == 0 or n % 3 == 0) : 
        return False
  
    i = 5
    while(i * i <= n) : 
        if (n % i == 0 or n % (i + 2) == 0) : 
            return False
        i = i + 6
  
    return True

# This function plots the test and training accuracy of all methods
def plotM(accuracy_dict,output_dir,filename,n_splits,n_repeats, color):
    imgname = "%s(%s)_methods.png" %(output_dir, filename)
    # we have 9 clustering methods thus is 3x3 where they share x and y enough.
    fig, axs = plt.subplots(3, 3, sharex='all', sharey='all',
                            gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    
    fig.suptitle('Accuracy of the methods')
    # change the xticks to the fold and repeat this
    plt.setp(axs, xticks=list(range(0,(n_splits*n_repeats)+n_repeats)), xticklabels=list(range(0,n_splits))*n_repeats)
    axes = axs.flat
    plots = []
    labels = []
    for i, key in enumerate(accuracy_dict.keys()):
        #plot test accuracy in one subplot
        pl0 = axes[i].plot([item[1] for item in accuracy_dict[key]], color[i], alpha=0.6)
        #save handles
        plots.append(pl0[0])
        #save label
        name = key.replace("_", " ")
        labels.append(name+ " test")
        #plot training accuracy
        pl1 = axes[i].plot([item[0] for item in accuracy_dict[key]], color[i], linestyle='dashed', label=key+" train", alpha=0.6)
        #save handle
        plots.append(pl1[0])
        #save label
        labels.append(name+ " train")
    fig.legend(plots, labels,     # The line objects and labels
           loc="lower center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           ncol = 3,            # amount of columns in the legend
           title="Clustering methods"  # Title for the legend
           )
    fig.text(0.5, 0.40, 'kfold', ha='center') # add xlabel
    fig.text(0.04, 0.65, 'Accuracy (%)', va='center', rotation='vertical')# add y label
    for ax in axs.flat:
        ax.label_outer() #remove all ticks execept the needed ones
    plt.subplots_adjust(bottom =0.5) # make space for the legend
    plt.savefig(imgname, bbox_inches='tight') # save the image
    plt.show()
    
def plot_function(listofkeys, dict_,color, name, output_dir, filename, n_splits, n_repeats, ylab = "Accuracy (%)", xlab = "Kfold", index= 1, outer = "max"):
    imgname = "%s(%s)%s_%s.png" %(output_dir, filename,name,ylab)
    # Save the length of the list
    length=len(listofkeys)
    # Calculate how to distribute the subplots, thus calculating number of rows and columns needed
    N = math.sqrt(length)
    # if length <= 7 and it is a prime we will create 1 column, more than 7 will follow the normal path because then the column becomes to big
    if length<=7 and isPrime(length):
        col = 1
        row = length
    # check if sides can be the same size
    elif length % N == 0:
        col = int(N)
    # Get the ceiling of the sqrt and use that as col
    else:
        col = math.ceil(N)
    # calculate row based on col
    row = math.ceil(length/col)
    # create the subplot grid
    fig, axs = plt.subplots(nrows = row, ncols=col, sharex='all', sharey='all',
                            gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    fig.suptitle("%s Test %s"%(name,ylab))
    # change columns and rows if col == 1 to make the legend easier to read
    if col == 1:
        col = row
        row = 1       
    plt.setp(axs, xticks=list(range(0,(n_splits*n_repeats)+n_repeats)), xticklabels=list(range(0,n_splits))*n_repeats)
    axes = axs.flat
    plots = []
    labels = []
    for i, key in enumerate(listofkeys):
        # get the highest score of a method
        if outer == "max":
            high_score = max([item[index] for item in dict_[key]])
        if outer == "min":
            high_score = abs(min([item[index] for item in dict_[key]]))
        # get the avg score of a method
        avg_score = mean([item[index] for item in dict_[key]])
        # plot and save handels
        pl = axes[i].plot([item[index] for item in dict_[key]],marker='.', alpha=0.4, c=color[i])
        plots.append(pl[0])
        # save label
        name = key.replace("_", " ")
        labels.append("%s: \n(avg=%s, %s=%s)"%(name,round(avg_score,2),outer,round(high_score,2)))
    fig.legend(plots, labels,     # The line objects and labels
           loc="lower center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           ncol = col,          # use same amount of columns
           title="Clustering methods"  # Title for the legend
           )
    for ax in axs.flat:
        ax.label_outer()
    # depending on the amount of rows create space for the legend
    ad = 0.15 + 0.1 * row
    plt.subplots_adjust(bottom =ad)
    # set x and y label
    fig.text(0.5, ad-0.1, xlab, ha='center')
    fig.text(0.04, 0.65, ylab, va='center', rotation='vertical')
    
    plt.savefig(imgname, bbox_inches='tight')
    plt.show()
