# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:31:03 2019

@author: rober
"""

import matplotlib.pyplot as plt



def heatmap(data, row_labels, col_labels, cmap, ax=None, title=None):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
        title      : Title for the plot
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
           All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, cmap, interpolation='nearest', aspect='auto')
    print(type(im))

    
    ax.set_title(title, y=1.09)
    
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False) 

    return im





