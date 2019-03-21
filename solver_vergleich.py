# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:42:47 2019

@author: Norman Philipps
"""
import os
import csv
from solve_script import SolverInstance
from time import perf_counter, time
from progressbar import ProgressBar



# call class


def initCSV(csv_file):
    fields=["piecewise_function", "piecewise_pieces",  "model_size", 
            "solver_name", "timelimit", "u_value", 
            "building", "loadingdata", "generating", "solving", 
            "DQ", "Costs"]
    with open(os.path.join("tests",csv_file+".csv"), 'w') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(fields)
        
def appendToCSV(csv_file, fields):
    with open(os.path.join("tests",csv_file+".csv"), 'a') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(fields)
    
def analyze(settings, csv_filename):
    pbar_max = 1
    for key, value in settings.items():
        pbar_max = pbar_max*len(value)
    print("++  Analyzing", pbar_max,"options!\n\n")
    
    # initCSV(csv_filename)
    
    sol = SolverInstance(clean=True, live_output=False, load_solutions=True, 
                         analyze_mode=True, log_this=False)
    
    pbar = ProgressBar(max_value=pbar_max).start()
    
    pbar_cnt = 0
    for piecewise_function in settings["piecewise_function"]:
        for piecewise_pieces in settings["piecewise_pieces"]:
            start_building = time()
            sol.buildModel(repn_DQ=piecewise_function, 
                           PieceCnt_DQ=piecewise_pieces)
            building = time() - start_building
    
            for model_size in settings["model_size"]:
                # Specify filename
                datafile = "database6_"+str(model_size)+".xlsm"
                
                # Load Data
                start_loadingdata = time()
                sol.loadData(datafile, saveall=True)
                loadingdata = time() - start_loadingdata

                # Build Instance String
                name = "_".join([str(piecewise_function), 
                                 str(piecewise_pieces), str(model_size)])

                # Generate Instance
                start_generating = time()
                sol.generateInstance(name)
                generating = time() - start_generating

                for solver_name in settings["solver_name"]:
                    # Set Solver Name
                    if piecewise_function == 2 and solver_name.startswith("cbc"):
                        
                        fields = [str(piecewise_function), str(piecewise_pieces), 
                        str(model_size), str(solver_name), "SOS2 not supported", 
                        "SOS2 not supported", 
                        str(building), str(loadingdata), str(generating), 
                        "SOS2 not supported", "No Solution", "No Solution"]
                        
                        appendToCSV(csv_filename, fields)
                        
                        pbar_cnt += 1
                        pbar.update(value=pbar_cnt)
                        continue
                    sol.setSolver(solver_name)

                    for timelimit in settings["timelimit"]:
                        # Set timelimit
                        sol.setSolverOptions(timelimit=timelimit)

                        for u_value in settings["u"]:
                            # Set different weights.
                            
                            try:
                                start_solving = time()
                                sol.solveInstance(u=u_value)
                                solving = time() - start_solving
                            except KeyboardInterrupt:
                                raise KeyboardInterrupt
                            except:
                                fields = [str(piecewise_function), 
                                       str(piecewise_pieces), 
                                       str(model_size), solver_name, str(timelimit), 
                                       str(u_value), str(building), str(loadingdata), 
                                       str(generating), " SolverError after", str(time() - start_solving), "seconds.", "No Solution", "No Solution"]
                                appendToCSV(csv_filename, fields)
                                pbar_cnt += 1
                                pbar.update(value=pbar_cnt)
                                continue
                            
                            
                            
                            DQ = sol.solved_instance.DQ_ges()
                            Costs = sol.solved_instance.Costs_ges()
                            
                            if DQ is None:
                                DQ = "No Solution"
                            if Costs is None:
                                Costs = "No Solution"
                            
                            fields = [str(piecewise_function), 
                                       str(piecewise_pieces), 
                                       str(model_size), solver_name, str(timelimit), 
                                       str(u_value), str(building), str(loadingdata), 
                                       str(generating), str(solving), str(DQ), str(Costs)]
                            
                            appendToCSV(csv_filename, fields)
                            
                            pbar_cnt += 1
                            pbar.update(value=pbar_cnt)
    
    pbar.finish()



if __name__ == "__main__":
    settings = {"piecewise_function": [4],#2,3,4,5,6,7,8
                "piecewise_pieces": [3],#,50,100,5,10,20
                "model_size": [5],#3,4,6,8,10,15,20,50,100,150,200
                "u": [1, 0],
                "solver_name": ["cplex_local"],#, "cbc_local"
                "timelimit": [10000], # seconds
                }
    
    test_name = "FUNCTION_TEST3_pwf(4)_pwp(3)_mdls(5)_u(1.0)_slv(cplex)_tlim(10000)"
    
    analyze(settings, test_name)





"""
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

set(np.array(sol1.WSM_Points)[:,0]), set(np.array(sol2.WSM_Points)[:,0])"""