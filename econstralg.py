# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:45:20 2019

@author: Feardrop

"""
from progressbar import ProgressBar
from solve_script import SolverInstance
from pareto_plot import pareto_plot

class EpsilonConMeth:
    
    def __init__(self, min_step=-0.01):
        self.setSettings()
        self.min_step = min_step
        self.Points = []
        
    def setSettings(self):    
        self.solver_name = "glpk_local"
        self.time_constr = None

    def createInstance(self):
        self.solver_instance = SolverInstance(solver_name=self.solver_name, 
                                              live_output=False, clean=True)

    def solve(self, **kwargs):
        self.solver_instance.solveInstance(**kwargs)
        
    # def updateResults(self, r_dict=self.solver_instanve.results):
    #     try:
    #         self.results.update(r_dict)
    #     except AttributeError:
    #         self.results = r_dict
    
    def run(self):
        self.createInstance()
        
        self.solve(u=0)
        self.epsilon_max = self.solver_instance.solved_instance.Costs_ges()
        self.current_result = self.solver_instance.solved_instance.DQ_ges()
        
        self.solve(u=1)
        self.epsilon_min = self.solver_instance.solved_instance.Costs_ges()
        
        self.epsilon = self.epsilon_max
        
        print("solving from", self.epsilon_max, "to", self.epsilon_min)
        print("minimal step =", self.min_step)
        
        pbar = ProgressBar(min_value=self.epsilon_min, 
                           max_value=self.epsilon_max,
                           initial_value=self.epsilon_max).start()
        
        while self.epsilon > self.epsilon_min:
            self.solve(u=0, Costs_ges_max=self.epsilon)
            self.new_epsilon = self.solver_instance.solved_instance.Costs_ges()
            self.new_result = self.solver_instance.solved_instance.DQ_ges()
            if self.current_result >= self.new_result:

                self.current_result = self.new_result
                self.Points.append([self.new_epsilon, self.current_result])
            
                pbar.update(value=self.new_epsilon)
                
            self.epsilon = self.new_epsilon + self.min_step

            
        pbar.finish()
        

if __name__ == "__main__":
    e = EpsilonConMeth()
    e.run()
    pareto_plot(e.Points, filename="Test1")
