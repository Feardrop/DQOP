# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:53:28 2019

@author: Feardrop
"""
import logging
import os
import model_MINLP
from loadexcel import Dataset
from pyomo.opt import SolverFactory, SolverManagerFactory
# from pyomo.opt.parallel import SolverManagerFactory as SMF
# from pyomo.opt.parallel.manager import ActionManagerError
# from pprint import pprint
# from pyomo.environ import TransformationFactory
from defandsave import savedata
from excelread import deep_merge
import time
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import os
from progressbar import ProgressBar


logger = logging.getLogger('pyomo_script')

# Einstellungen

class SolverInstance:
    """Creates a Solver Instance.

    Parameters:

        solver_name :: string
            Defines the used Solver.
            For possible solvers use the ``helpSolvers()`` method.
        model_module :: model_MINLP
            Imported model-file/module
        model_properties :: {}
            See ``buildModel()`` for possible keywords.
        datafile :: string
            Excel file. ``databaseX.xlsm``
        live_output :: True/(False)
            Enables output. False for clean run without any feedback.

    List of supported Solvers:
        - cbc_local
        - cbc_neos
        - couenne_local
        - couenne_neos
        - bonmin_local
        - bonmin_neos
        - cplex_local
        - cplex_nl_local
        - cplex_neos
        - minos_neos
        - glpk_local
        - glpk_neos
    """
    def __init__(self,
                 solver_name="cbc_local",
                 live_output = True,
                 model_module=model_MINLP,
                 model_properties={},
                 datafile="Database5.xlsm", **kwargs):
        """Set solver (``default="cbc_local"``) and properties.
        """

        # Arguments
        self.solver_name = solver_name
        self.live_output = live_output
        self.model_module = model_module
        self.buildModel(**model_properties)  # Sets self.model
        self.kwargs = kwargs
        self.datafile = datafile
        self.solver_options = {}

        # Parameters
        self.solver_dict = {
            "cbc_local": {
                    "solver_name": "cbc",
                    "EXE_path": "./solvers/cbc-win64/cbc",
                    "solver_io": None,
                    "solver_manager_name": None},
            "cbc_neos": {
                    "solver_name": "cbc",
                    "EXE_path": None,
                    "solver_io": None,
                    "solver_manager_name": "neos"},
            "couenne_local": {
                    "solver_name": "couenne",
                    "EXE_path": "./solvers/couenne-win64/couenne",
                    "solver_io": None,
                    "solver_manager_name": None},
            "couenne_neos": {
                    "solver_name": "couenne",
                    "EXE_path": None,
                    "solver_io": None,
                    "solver_manager_name": "neos"},
            "bonmin_local": {
                    "solver_name": "bonmin",
                    "EXE_path": "./solvers/bonmin-win64/bonmin",
                    "solver_io": None,
                    "solver_manager_name": None},
            "bonmin_neos": {
                    "solver_name": "bonmin",
                    "EXE_path": None,
                    "solver_io": None,
                    "solver_manager_name": "neos"},
            "cplex_local": {
                    "solver_name": "cplex",
                    "EXE_path": None,
                    "solver_io": None,
                    "solver_manager_name": None,
                    # "optimalitytarget": 3,
                    },
            "cplex_nl_local": {
                    "solver_name": "cplex",
                    "EXE_path": None,
                    "solver_io": "nl",
                    "solver_manager_name": None,
                    "optimalitytarget": 3,
                    },
            "cplex_neos": {
                    "solver_name": "cplex",
                    "EXE_path": None,
                    "solver_io": None,
                    "solver_manager_name": "neos"},
            "minos_neos": {
                    "solver_name": "minos",
                    "EXE_path": None,
                    "solver_io": None,
                    "solver_manager_name": "neos"},
            "glpk_local": {
                    "solver_name": "glpk",
                    "EXE_path": None,
                    "solver_io": None,
                    "solver_manager_name": None},
            "glpk_neos": {
                    "solver_name": "glpk",
                    "EXE_path": None,
                    "solver_io": None,
                    "solver_manager_name": "neos"},
            }

        # Set Solver for this SolverInstance
        self.setSolver(self.solver_name)

        # Set default Properties for this SolverInstance
        self.setProperties(**self.kwargs)
        if self.live_output:
            self.getProperties()

        self.loadData(self.datafile)

    def setSolver(self, solver_name):
        """Sets the Solver specific values.

        Arguments:
            solver_name
                Defines the solver to be used.
                For possible solvers use **helpSolvers()** method.
        """
        self.solver_name = solver_name
        self.solver = self.solver_dict[solver_name]["solver_name"]
        self.solver_io = self.solver_dict[solver_name]["solver_io"]
        self.solver_manager_name = self.solver_dict[
                solver_name]["solver_manager_name"]
        self.EXE_path = self.solver_dict[solver_name]["EXE_path"]

        for key, val in self.solver_dict[solver_name].items():
            if key not in ("solver_name", "EXE_path",
                           "solver_io", "solver_manager_name"):
                self.solver_options.update({key: val})

        if self.live_output:
            self.getSolverInformation()

    def setSolverOptions(self, **solver_options):
        """Sets the Solver specific values.

        Arguments:
            solver_name
                Defines the solver to be used.
                For possible solvers use **helpSolvers()** method.
        """
        self.solver_options.update(solver_options)

        if self.live_output:
            self.getSolverInformation()
#            for key, value in self.solver_options.items():
#                print("# Option{:8}:".format("["+key+"]"), value)

    def getSolverInformation(self):
        print("\n################ Solver Info ##################")
        print("# Solver:         ", self.solver)
        print("# Solver_io:      ", self.solver_io)
        print("# SolverManager:  ", self.solver_manager_name)
        print("# Exe-Path:       ", self.EXE_path)
        print("#")
        print("# Set Solver-Options with setSolverOptions()...")
        try:
            for key, value in self.solver_options.items():
                print("# Option{:8}:".format("["+key+"]"), value)
        except AttributeError:
            pass
        print("################################################\n")

    def helpSolvers(self):
        """Prints a list of all avaliable solvers."""
        solver_list = ['{0}'.format(k) for k, v in self.solver_dict.items()]
        print(', '.join(solver_list))

    def setProperties(self,  # !!!
                      symbolic_solver_labels=True,
                      tee=False,  # True prints solver output to screen
                      keep_files=False,  # True prints intermediate file names
                      create_LP_files=False,
                      create_NL_files=False,
                      clean=False):
        """Sets Properties for the solving process."""
        self.symbolic_solver_labels = symbolic_solver_labels
        self.tee = tee
        self.keep_files = keep_files
        self.create_LP_files = create_LP_files
        self.create_NL_files = create_NL_files
        self.clean = clean

    def getProperties(self):
        """Prints all current properties."""
        print("\n############# Current properties: ##############")
        print("# symbolic_solver_labels:",   self.symbolic_solver_labels)
        print("# tee (live log):        ",   self.tee)
        print("# keep_files:            ",   self.keep_files)
        print("# create_LP_files:       ",   self.create_LP_files)
        print("# create_NL_files:       ",   self.create_NL_files)
        print("# clean (no files):      ",   self.clean)
        print("#\n# To manipulate properties use setProperties().")
        print("################################################\n")
###############################################################################

    def solve(self):  # TODO
        """Solves the current instance.

        Performs a loop over ``x`` ``steps+1``.
        """
        start = time.time()
        for x in range(self.steps+1):
            y = 10**(-2*x/11)
            eps = x/100
            instance_str = "instance_{0:1.3f}".format(y)
            start_instance = time.time()
            self.solveInstance(instance_str, u=y, epsilon=eps)
            print("Time for {}:".format(instance_str),
                  time.time()-start_instance)
        print("Time:", time.time()-start)

    def single(self, DQ_ges_max=None, Costs_ges_max=None):  # TODO
        """Set either DQ or Costs to get a solution for the other."""

        if DQ_ges_max is not None and Costs_ges_max is None:
            self.solveInstance(DQ_ges_max=DQ_ges_max)
        elif Costs_ges_max is not None and DQ_ges_max is None:
            self.solveInstance(Costs_ges_max=Costs_ges_max)
        else:
            raise KeyError("Specify DQ_ges_max OR Cost_ges_max!")

    def weightedSumMethod(self):  # TODO
        """Set steps or a list of specific values for ``u``."""
        start = time.time()
        for x in range(self.steps+1):
            y = 10**(-2*x/11)
            instance_str = "instance_{0:1.3f}".format(y)
            start_instance = time.time()
            self.solveInstance(u=y)
            print("Time for {}:".format(instance_str),
                  time.time()-start_instance)
        print("Time:", time.time()-start)

    def descendingStepMethod(self, min_step, time_constr=None, clean=True,
                                 plot_this=True, mintomax=False,
                                 epsilon_min=None, epsilon_max=None,
                                 solveInstanceOptions={}):  # TODO
        """Set steps for epsilon.

        To access generated Points use ``self.DSM_Points``.
        """
        live_output_state = self.live_output
        self.live_output = False
        
        clean_state = self.clean
        self.clean = True
        
        if min_step <= 0:
            raise ValueError('min_step as to be greater than zero!')



        if time_constr is not None:
            if self.solver_name.startswith("cplex"):
                self.setSolverOptions(timelimit=time_constr)
            if self.solver_name.startswith("cbc"):
                self.setSolverOptions(timelimit=time_constr)

        self.DSM_Points = []

        plotname_list = ["descending_step", "step({})".format(min_step)]
        if mintomax:
            plotname_list.append("min-max")
        else:
            plotname_list.append("max-min")
        if epsilon_min is not None:
            plotname_list.append("emin({})".format(epsilon_min))
        if epsilon_max is not None:
            plotname_list.append("emax({})".format(epsilon_max))
        if time_constr is not None:
            plotname_list.append("time({}sec)".format(time_constr))

        plot_name = ("_").join(plotname_list)


        if epsilon_min is None:
            self.solveInstance(u=1, **solveInstanceOptions)
            self.epsilon_min = self.solved_instance.Costs_ges()
        else:
            # this needs to be going into a different direction
            # a Cost_ges_min needs to be specified?
            self.epsilon_min = epsilon_min  # TODO
            self.solveInstance(u=0, Costs_ges_max=self.epsilon_min, **solveInstanceOptions)
        result_min = self.solved_instance.DQ_ges()

        if epsilon_max is None:
            self.solveInstance(u=0, **solveInstanceOptions)
            self.epsilon_max = self.solved_instance.Costs_ges()
        else:
            self.epsilon_max = epsilon_max
            self.solveInstance(u=0, Costs_ges_max=self.epsilon_max, **solveInstanceOptions)
        result_max = self.solved_instance.DQ_ges()


        if mintomax:  # !!! This part generates shitty solutions
            epsilon = self.epsilon_min
            current_result = result_min
            print("\n# Solving from", self.epsilon_min, "to", self.epsilon_max)
            print("# Minimal step =", min_step)

            pbar = ProgressBar(min_value=self.epsilon_min,
                               max_value=self.epsilon_max,
                               initial_value=self.epsilon_min).start()

            while epsilon + min_step < self.epsilon_max:
                epsilon += min_step
                self.solveInstance(u=0, Costs_ges_max=epsilon, **solveInstanceOptions)
                new_epsilon = self.solved_instance.Costs_ges()
                new_result = self.solved_instance.DQ_ges()
                if new_result >= current_result:

                    current_result = new_result
                    self.DSM_Points.append([new_epsilon, current_result])

                pbar.update(value=epsilon)
            pbar.finish()

        if not mintomax:
            epsilon = self.epsilon_max
            current_result = result_max
            print("\n# Solving from", self.epsilon_max, "to", self.epsilon_min)
            print("# Minimal step =", min_step)

            pbar = ProgressBar(min_value=self.epsilon_min,
                               max_value=self.epsilon_max,
                               initial_value=self.epsilon_max).start()

            while epsilon > self.epsilon_min:
                self.solveInstance(u=0, Costs_ges_max=epsilon, **solveInstanceOptions)
                new_epsilon = self.solved_instance.Costs_ges()
                new_result = self.solved_instance.DQ_ges()
                if current_result >= new_result:

                    current_result = new_result
                    self.DSM_Points.append([new_epsilon, current_result])

                pbar.update(value=epsilon)

                epsilon = new_epsilon - min_step  # minus!

            pbar.finish()
        
        # Postprocess
        self.live_output = live_output_state
        self.clean = clean_state
        
        if plot_this:
            self.pareto_plot(self.DSM_Points, filename=plot_name)

    def adBEConstMethod(self, n_p=100, x_tol = 0.01, y_tol=1e-10, n1_max=100, 
                        b_max=10, solveInstanceOptions={},  plot_this=True, 
                        log=False, x_max=float('inf'), y_min=0):

        """Adaptive bisection :math:`\\varepsilon`-constraint method.
        
        To access generated Points use ``self.ABE_Points``.
        
        ***********
        Parameters
        ***********
        
        :param n_p: (:const:`int=100`)
            Number of generated Points
        :param x_tol: (:const:`float=0.01`)
            x-tolerance
        :param y_tol: (:const:`float=1e-10`)
            y-tolerance
        :param n1_max: (:const:`int=100`)
            Number of maximum iterations before a segment is ignored.
        :param b_max: (:const:`int=10`)
            Number of maximum ignored segments.
        :param x_max: (:const:`float=float('inf')`)
            Maximum x value.
        :param y_min: (:const:`float=0`)
            Maximum y value.
        :param solveInstanceOptions: (:const:`dict={}`) 
            Additional Options for ``solveInstance()``
        :param plot_this: (:const:`boolean=True`)
            Plot generated data after completition.
        :param log: (:const:`boolean=False`)
            Log every algorithm-step.
        """
        
        from operator import itemgetter
        def log(*args):
            if log:
                print(*args)

        #Start
        live_output_state = self.live_output
        self.live_output = False
        clean_state = self.clean
        self.clean = True

        results = []
        points_used = {}

        plotname_list = ["adBisecEpsil", "points({})".format(n_p)]
        plotname_list.append("x_tol({})".format(x_tol))
        plotname_list.append("y_tol({})".format(y_tol))
        plotname_list.append("n1({})".format(n1_max))
        plotname_list.append("b({})".format(b_max))

        plot_name = ("_").join(plotname_list)

        # Find anchor points
        self.solveInstance(u=1, DQ_ges_min=y_min, **solveInstanceOptions)
        self.mu1opt = self.solved_instance.DQ_ges()
        y1 = self.solved_instance.DQ_ges()
        x1 = self.solved_instance.Costs_ges()
        results.append((x1, y1))

        self.solveInstance(u=0, Costs_ges_max=x_max, **solveInstanceOptions)
        self.mu2opt = self.solved_instance.Costs_ges()
        y2 = self.solved_instance.DQ_ges()
        x2 = self.solved_instance.Costs_ges()
        results.append((x2, y2))

        # Add entry points_used array with x and y
        # coordinates of anchor points and n1 = 1
        points_used = [(x1, y1, x2, y2)]
        n1 = 1
        
        n = 2  # Two solutions already exist.
        
        # Compute espilons
        def get_e_ub(n1,x1,x2):  # compute upper bound. 1/2, 3/4, 7/8
            return x2 - (x2-x1)/(2**n1)
        def get_e_lb(y1):
            return y1
        b = 1
        while True:
            if b > b_max+1:
                print("\nMaximum number of ignored segments reached.")
                break
            
            if n1 > n1_max:
                log("n1 =",n1)
                log("maximum number of iterations exeeded")
                
                b += 1
                log("now searching for "+str(b)+". highest distance")
                continue
            log("\n")
            log("using points:", points_used[-1])
            e_ub = get_e_ub(n1, points_used[-1][0], points_used[-1][2])
            e_lb = get_e_lb(points_used[-1][1])

            # Is the smallest tolerance gap reached?
            log("gap:", points_used[-1][2]-e_ub)
            
            if points_used[-1][2]-e_ub <= x_tol: # Yes -> Stop
                log("Tolerance limit exceeded.")
                b += 1
                log("now searching for "+str(b)+". highest distance")
                continue

            # Solve Optimization Problem
            self.solveInstance(u=0, DQ_ges_min=e_lb, Costs_ges_max=e_ub, **solveInstanceOptions)

            # Is a feassible solution found?
            if self.solved_instance.DQ_ges() is None:  # No
                # Widen the search range. (1/2 -> 3/4 -> 7/8 -> ...)
                n1 += 1
                log("n1 =",n1)
                continue
            x = self.solved_instance.Costs_ges()
            y = self.solved_instance.DQ_ges()

            # Add new entry to results array.
            results.append((x, y))
            log("new result:", (x, y))
            
            # Sort the results array in descending order of the x column
            results.sort()  # sorts by 1st column (x)
            
            # Have n_p points been reached?
            if n == n_p:  # Yes -> Stop
                print("\nMaximum number of points reached!")
                break
            
            # Calculate the euclidian distance between
            # successive rows in the results array.
            distances = np.diff(results, axis=0)  # Compute delta(x,y)
            segdists = np.sqrt((distances ** 2).sum(axis=1))  # Compute euclidian distance
            def getIndices(a, n):
                # M = a.shape[0]
                # perc = (np.arange(M-n,M)+1.0)/M*100
                # return np.percentile(a,perc)[::-1]
                return np.argpartition(a, -n)[-n:][::-1]
            
            index_max = getIndices(segdists, b)[-1]  # Identify maximum distance index
            # log("distance:",segdists[index_max])
            
            # Identify corresponding points
            new_points = results[index_max] + results[index_max+1]
            
            # Are the points existent in a single row of the points_used array?
            if new_points[3]-new_points[1] < y_tol:
                b += 1
                log("now searching for "+str(b)+". highest distance")
                continue
            
            if new_points in points_used:
                log("new points:", new_points, "already used..")
                n1 += 1
                log("n1 =",n1)
                continue

            # Add entry to points_used array and set n1 = 1
            points_used.append(new_points)
            log("new points appended:", new_points)
            n1 = 1
            
        # Postprocess
        self.ABE_Points = results
        self.live_output = live_output_state
        self.clean = clean_state

        if plot_this:
            self.pareto_plot(self.ABE_Points, filename=plot_name)


    # Compute nondominated Points
    @staticmethod
    def is_pareto(all_points):
        """Masks pareto efficient Points.
        
        :param all_points: An (n_points, n_all_points) array
        
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(all_points.shape[0], dtype = bool)
        for i, c in enumerate(all_points):
            if is_efficient[i]:
                # custom fit for this use case
                is_efficient[is_efficient] = np.any( 
                        all_points[is_efficient][0]<=c[0], axis=0) and np.any(
                        all_points[is_efficient][1]>=c[1], axis=0) # Remove dominated points
        return is_efficient

    @classmethod
    def pareto_plot(self, *plot_data_arrays, filename="", save_all=False,
                    plot_scatter=True, plot_pareto_front=True,
                    lb_x = None, ub_x = None,
                    lb_y = None, ub_y = None):
        
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                       figsize=(6, 4))
        
        # Initialize lists for boundaries
        ub_x_max, lb_x_min, ub_y_max, lb_y_min = ([] for _ in range(4))
        
        # Plot multiple graphs
        for plot_data in plot_data_arrays:
            scatter_data = np.sort(plot_data, axis=0)
            x = scatter_data[:,0]
            y = scatter_data[:,1]
            
            # Fill boundary_lists
            ub_x_max.append(max(x)*1.02)
            lb_x_min.append(min(x)*0.98)
            ub_y_max.append(min(1, max(y)*1.1))
            lb_y_min.append(max(0, min(y)*0.9))
    

            if len(scatter_data) > 1 and plot_pareto_front:
                pareto_front = scatter_data[self.is_pareto(scatter_data)]
                x2 = pareto_front[:,0]
                y2 = pareto_front[:,1]
                ax1.plot(x2, y2,zorder=1)    
            
            if plot_scatter:
                ax1.scatter(x=x, y=y, marker='x')#, c='r', edgecolor='b',zorder=2)
            
        ub_x = ub_x or max(ub_x_max)
        lb_x = lb_x or min(lb_x_min)
        ub_y = ub_y or max(ub_y_max)
        lb_y = lb_y or min(lb_y_min)
            
        ax1.set_title('Pareto Plot: '+filename)
        ax1.set_yscale("linear")
        ax1.set_xscale("linear")
        ax1.set_xlabel('$Kosten$')
        ax1.set_ylabel('$Qualität$')
        ax1.set_xlim([lb_x, ub_x])
        ax1.set_ylim([lb_y, ub_y])
        ax1.invert_yaxis()

        plt.show()

        if not self.clean or save_all:
            os.makedirs("plots", exist_ok = True)
            fig.savefig(os.path.join("plots","plot_"+filename+".png"))
            fig.savefig(os.path.join("plots","plot_"+filename+".pdf"), format="pdf") # save the figure to file
        plt.close(fig)

    def loadData(self, filename):
        """Loads the data from an excel-file a dict.

        Saves the dict as pickle and txt in ./logs/input_data.

        Keyword Arguments:
            filename *string*
                Filename incl. extention. Can be a path.
        """
        self.data = Dataset(filename).pyomo_dict
        if not self.clean:
            print("\nData successfully loaded.\n")
            savedata(self.data, "input_data", "logs/input_data", human=True)

    def addData(self, **kwargs):
        """Adds data to the loaded ``self.data`` variable.

        Usage::

            addData('func_factor_DQ'={5: 9})

        """
        for key, value in kwargs.items():
            if type(value) is dict:
                deep_merge(self.data[None],{key: value})
            else:
                self.data[None].update({key: {None: value}})


    def printModel(self):
        self.model.pprint()

    def updateResults(self, r_dict):
        try:
            self.results.update(r_dict)
        except AttributeError:
            self.results = r_dict

    def getResults(self):
        for inst_name_, result_ in self.results.items():
            print("Results for "+inst_name_)
            result_.write()

    def buildModel(self, **kwargs):
        """Builds the model with given ``**kwargs``

        Parameters
        ----------
        repn_DQ: :const:`int=3`
            Representation of the piecewise function for DataQualities::

                0: BIGM_SOS1,   5: DLOG,
                1: BIGM_BIN,    6: LOG,
                2: SOS2,        7: MC,
                3: CC,          8: INC,
                4: DCC,      None: None

        constr_DQ: :const:`int=2`
            Constraint method::

                0: UB,          2: EQ,
                1: LB,       None: None

        PieceCnt_DQ: :const:`int=10`
            Piecewise section count.

        func_factor_DQ: :const:`int=2`
            Sets the power ``**(1/func_factor_DQ)`` to the calculation function :py:func:`Compute_DQ_rule`.

        kwargs:
            Add additional Values. (Currently unused.)
        """

        self.model = self.model_module.createModel(**kwargs)

    def solveGeneratedInstance(self, instance, inst_name):
        # Nutze Solver Manager None/string
        if self.solver_manager_name is not None:
            with SolverManagerFactory(self.solver_manager_name) as solver_manager:
                if self.EXE_path is None:
                    opt = SolverFactory(self.solver, solver_io=self.solver_io)
                else:
                    opt = SolverFactory(self.solver, solver_io=self.solver_io,
                                        executable=self.EXE_path)
                # Solve the instance :: Clean gets no Logs
                if not self.clean:
                    self.result = solver_manager.solve(instance, opt=opt,
                                                  tee=self.tee,
                                                  logfile=inst_name+".log")
                else:
                    self.result = solver_manager.solve(instance, opt=opt,
                                                  tee=self.tee,)
        else:
            path_log = ["logs", "logfiles", inst_name+".log"]
            if self.EXE_path is None:
                with SolverFactory(self.solver,
                                   solver_io=self.solver_io) as opt:

                    # Set Solver Options
                    try:
                        for key, value in self.solver_options.items():
                            opt.options[key] = value
                    except AttributeError:
                        pass

                    # Solve the instance :: Clean gets no Logs
                    if not self.clean:
                        self.result = opt.solve(instance, tee=self.tee,
                                           symbolic_solver_labels=
                                           self.symbolic_solver_labels,
                                           logfile=os.path.join(*path_log))
                    else:
                        self.result = opt.solve(instance, tee=self.tee,
                                           symbolic_solver_labels=
                                           self.symbolic_solver_labels)
            else:
                with SolverFactory(self.solver, solver_io=self.solver_io,
                                   executable=self.EXE_path) as opt:

                    # Set Solver Options
                    try:
                        for key, value in self.solver_options.items():
                            opt.options[key] = value
                    except AttributeError:
                        pass

                    # Solve the instance :: Clean gets no Logs
                    if not self.clean:
                        self.result = opt.solve(instance, tee=self.tee,
                                           symbolic_solver_labels=
                                           self.symbolic_solver_labels,
                                           logfile=os.path.join(*path_log))
                    else:
                        self.result = opt.solve(instance, tee=self.tee,
                                           symbolic_solver_labels=
                                           self.symbolic_solver_labels)
        self.updateResults({inst_name: self.result})

        return instance

    def solveInstance(self, **kwargs):
        """Generates an Instance of the given model and invokes the Solver.
        All attributes of the instance can be modified via given
        keyword-arguments.

        Possible Keyword-Arguments:
            u
                Defines the cost/dataquality-ratio. 1 for Costs only.
            Costs_ges_max
                Sets maximum allowed Costs.
            DQ_ges_max
                Sets minimum required Data-Quality.
        """
        allowed_keys = ["u", "DQ_ges_max", "DQ_ges_min", "Costs_ges_max", "Costs_ges_min", 
                        "n", "DQ_func_type", "R_J_func_type"]
        allowed_keys.sort()

        # Build Instance Name
        inst_name_parts = []  # Dump for Instance Name
        inst_name_parts.append(self.solver_name)  # Solver Name on first place.
        for key, value in sorted(kwargs.items()):  # add kwargs alphabetically
            # if key in allowed_keys:
                if value is not None:
                    inst_name_parts.append(key+"("+str(value)+")")
        inst_name = "-".join(inst_name_parts)  # set delimeter

        # Create Instance
        self.instance = None  # Clean-up
        self.instance = self.model.create_instance(data=self.data,
                                                   name=inst_name)

        # Assign Parameters
        for key, value in kwargs.items():
            if key in allowed_keys:
                if value is not None:
                    self.instance.__setattr__(key, value)
            else:
                warning_str = ('''
                               KEYWORD WARNING: "{0}" = {1} is not allowed.
                               Allowed Keys are: "{2}" and "{3}".
                               '''.format(key, value,
                                          '", "'.join(allowed_keys[:-1]),
                                          allowed_keys[-1]))
                logger.warning(textwrap.dedent(warning_str))
                if value is not None:
                    self.instance.__setattr__(key, value)



        # Write generated Model-Instance
        if not self.clean:
            path_generated = ["logs", "generated_model_instances",
                              "model_"+inst_name+".txt"]
            self.instance.pprint(filename=os.path.join(*path_generated))

        # Write generated NL-File
        if self.create_NL_files and not self.clean:
            path_nl = ["logs", "nl-files", "nlfile_"+inst_name+".nl"]
            self.instance.write(os.path.join(*path_nl), format="nl")

        if self.create_LP_files and not self.clean:
            path_lp = ["logs", "lp-files", "lpfile_"+inst_name+".lp"]
            io_opt_lp = {"symbolic_solver_labels": self.symbolic_solver_labels}
            self.instance.write(os.path.join(*path_lp), format="lp",
                                io_options=io_opt_lp)

        self.solved_instance = None  # Clean-up
        self.solved_instance = self.solveGeneratedInstance(self.instance,
                                                           inst_name)

        # Save Results to Files
        if not self.clean:
            path_resutls = ["logs", "results", "results_"+inst_name+".txt"]
            with open(os.path.join(*path_resutls), "w") as f:
                f.write("######### RESULTS: "+inst_name+" #########\n\n")
                self.solved_instance.pprint(ostream=f)

        # Output Objective Values
        if self.live_output:
            print("CombObjective for {0}: {1:12.2f}"
                  .format(inst_name, self.solved_instance.CombinedValue()))
            print("           DQ_ges: {1:12.7f}"
                  .format(inst_name, self.solved_instance.DQ_ges()))
            print("        Costs_ges: {1:12.2f}"
                  .format(inst_name, self.solved_instance.Costs_ges()))