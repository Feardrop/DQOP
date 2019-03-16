# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:18:43 2019

@author: Feardrop
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def pareto_plot(plot_data, clean=True, filename=""):
    
    plot_data = np.array(plot_data)
    
    x = plot_data[:,0]
    y = plot_data[:,1]
    ub_x = max(x)*1.02
    lb_x = min(x)*0.98
    ub_y = min(1, max(y)*1.1)
    lb_y = max(0, min(y)*0.9)
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                   figsize=(10, 8))
    
    ax1.plot(x, y,zorder=1)
    ax1.scatter(x=x, y=y, marker='o', c='r', edgecolor='b',zorder=2)
    
    ax1.set_title('Scatter: full')
    ax1.set_yscale("linear")
    ax1.set_xscale("linear")
    ax1.set_xlabel('$Kosten$')
    ax1.set_ylabel('$Qualität$')
    ax1.set_xlim([lb_x, ub_x])
    ax1.set_ylim([lb_y, ub_y])
    ax1.invert_yaxis()

    plt.show()
    if not clean:
        os.makedirs("plots", exist_ok = True)
        fig.savefig(os.path.join("plots","plot_"+filename+".png"))
        fig.savefig(os.path.join("plots","plot_"+filename+".pdf"), format="pdf") # save the figure to file
    plt.close(fig)