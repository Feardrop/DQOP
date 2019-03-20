# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:42:47 2019

@author: Norman Philipps
"""

from solve_script import SolverInstance
from time import time
import numpy as np


start1 = time()
sol1=SolverInstance("cplex_local")
sol1.setProperties(tee=False, load_solutions=False, gap_tol=0.0) 
sol1.weightedSumMethod(steps=200, steps_per_power_of_ten=15, log=False, plot_this=False, live_output=False)
time1 = time() - start1


start2 = time()
sol2 =SolverInstance()
sol2.setProperties(tee=False, load_solutions=False, gap_tol=0.0)
sol2.weightedSumMethod(steps=200, steps_per_power_of_ten=15, log=False, plot_this=False, live_output=False)
time2 = time() - start2


start3 = time()
sol3 =SolverInstance("glpk_local")
# sol3.setProperties(tee=True)
sol3.weightedSumMethod(steps=100, steps_per_power_of_ten=10, log=True, plot_this=False, live_output=True)
time3 = time() - start3


# Plot results
SolverInstance.pareto_plot(sol1.WSM_Points, sol2.WSM_Points)#, sol3.WSM_Points)

print("cplex=",time1)
print("cbc=", time2) 

# -- Get all solutions from this run --
# for key, val in sol1.results.items():
#     if sol1.results[key].solution.gap == 0:
#         print(sol1.results[key].solution().Variable["Costs_ges"]["Value"],
#               sol1.results[key].solution().Variable["DQ_ges"]["Value"],
#               sol1.results[key].solution.gap)

set(np.array(sol1.WSM_Points)[:,0]), set(np.array(sol2.WSM_Points)[:,0])