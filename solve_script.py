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

    def epsilonConstrainedMethod(self):  # TODO
        """Set steps for epsilon."""
        pass

###############################################################################

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

    def updateResults(self, result_dict):
        try:
            self.results.update(result_dict)
        except AttributeError:
            self.results = result_dict

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
                    result = solver_manager.solve(instance, opt=opt,
                                                  tee=self.tee,
                                                  logfile=inst_name+".log")
                else:
                    result = solver_manager.solve(instance, opt=opt,
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
                        result = opt.solve(instance, tee=self.tee,
                                           symbolic_solver_labels=
                                           self.symbolic_solver_labels,
                                           logfile=os.path.join(*path_log))
                    else:
                        result = opt.solve(instance, tee=self.tee,
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
                        result = opt.solve(instance, tee=self.tee,
                                           symbolic_solver_labels=
                                           self.symbolic_solver_labels,
                                           logfile=os.path.join(*path_log))
                    else:
                        result = opt.solve(instance, tee=self.tee,
                                           symbolic_solver_labels=
                                           self.symbolic_solver_labels)
        self.updateResults({inst_name: result})

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
        allowed_keys = ["u", "DQ_ges_max", "Costs_ges_max", "n",
                        "DQ_func_type", "R_J_func_type"]
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
                  .format(inst_name, self.instance.CombinedValue()))
            print("           DQ_ges: {1:12.7f}"
                  .format(inst_name, self.instance.DQ_ges()))
            print("        Costs_ges: {1:12.2f}"
                  .format(inst_name, self.instance.Costs_ges()))


if __name__ == "__main__":
    sol = SolverInstance(solver_name="cbc_local",
                         tee=False,
                         keep_files=False,
                         create_LP_files=False,
                         clean=False)
    sol.addData(func_factor_DQ={3: 10})
    # sol.solveInstance(u=0.99 , Costs_ges_max=None, n=1)
    # sol.solveInstance(u=0.99, Costs_ges_max=None, n=1, break_points_type=1)
    # sol.solveInstance(u=0.99, Costs_ges_max=None, n=1, break_points_type=2)
    # sol.solveInstance(u=0.99, Costs_ges_max=None, n=1, func_factor_DQ=2)
    # sol.buildModel(repn_DQ=5, func_factor_DQ=2)
    # sol.solveInstance(u=0.5, Costs_ges_max=None, n=1)#,  **{func_factor_DQ[3]:5}, **{func_factor_DQ[4]:5})