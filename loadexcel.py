# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:23:25 2019

@author: Norman Philipps


"""

import openpyxl as op
from excelread import sheetToDict
from munch import munchify
from pprint import pprint
import copy

class Dataset:

    def __init__(self, filestring, filtered = True, keytype="tupled"): # load_as_dict = False
        """Create data-dictionary-object.
        ::
           obj = Dataset("example.xlsx", filtered = True/False)
           obj.data_dict   ... Complete dictionary.
           obj.m(unched)   ... dot-accessible dictionary.
           obj.*sheetname* ... 1st layer object
           
           obj.data_dict["*sheetname*"] 
           == obj.m.*sheetname* 
           == obj.*sheetname*
    
        Parameters:
            filestring  
                String of file incl. extention.
            filtered    
                Get only visible data.
            
        Objects:
            data_dict
                Raw dictionary.
            trans_dict
                Translated dictionary.
            pyomo_dict
                For pyomo. Different keys.
        """
        self.workbook = op.load_workbook(filestring, data_only=True)

        self.data_dict = dict()
        self.keytype = keytype
        
        def loadSheets(self, sheetname, ID_cols):
            """Einlesen als Dictionary."""
            self.data_dict[sheetname] = dict()
            self.data_dict[sheetname] = sheetToDict(sheetname, ID_cols, 
                                          wb=self.workbook, 
                                          filtered=filtered,
                                          keytype=self.keytype) #  1 ID
            
            # Einlesen der Blätter als Variablen
            self.__dict__.update({sheetname: 
                sheetToDict(sheetname, ID_cols, wb=self.workbook, 
                            filtered=filtered, keytype=self.keytype)})


        for sheetname in self.workbook.sheetnames: # Anpassen der ID-Spalten
            if (sheetname.startswith("table_") or
                sheetname.startswith("Translator")):
                loadSheets(self, sheetname, 1)

            elif sheetname.startswith("dict_"):   
                loadSheets(self, sheetname, 2)
            else: 
                pass
            
        
        self.m = munchify(self.data_dict) # Create m version
        
        
        def translateDict(self, input_dict, *trans): # returns a dictionary
            """Translates the 1st layer keys of **input_dict**, 
            using a provided **\*trans** dictionary or, 
            if no argument is passed, uses an internal *Translator* dictionary.
            
            AttributeError gets raised if no suitable dictionary is found.
            """
            
            def getIdxKeys(self, translate_dict, prefix="table_"):
                """Returns all Index keys with the Prefix **table_**"""
                
                idx_keys = list()
                
                for keys in translate_dict.keys():
                    try:
                        if keys.startswith(prefix):
                            idx_keys.append(translate_dict[keys])
                    except:
                        pass
                
                return idx_keys
            
            
            if len(trans) > 1:
                raise TypeError("translateDict() expected at most 2 arguments, got %d" 
                                % (len(trans) + 1))
            if trans:
                tr = trans[0]#.copy()
            else:
                tr = input_dict["Translator"]
                   
            output_dict = dict() # new dict to keep orig intact
            for (key, value) in input_dict.items():
                # output_dict.update({tr[key]: value})
                try: # if key is found -> translate
                    output_dict.update({tr[key]: value})
                except: # else 
                    output_dict.update({key: value})
                    
            idx_list = getIdxKeys(self, tr)
            output_dict["idx_list"] = idx_list
            
            return output_dict
        
        self.trans_dict = translateDict(self, self.data_dict)

        
        def dictToNoneType(self, input_dict, filter_kw="kval_active"): # returns a dictionary
            """Changes the Dictionary to a Format usable by pyomo.
            
            Bsp::
                
                data = {None: {
                            "I": {None: [1,2,]},
                            "J": {None: [1,2,3,]},
                            "G": {None: [1,2,]},
                            "H": {None: [IDS, ...]},
                            "S": {None: [IDs, ...]},
                            "V": {None: [IDS, ...]},   
                            "z_J_ub": {(1,1): 30, (1,2): 40, (2,3): 70,},
                            "z_lb": {1: 5,},
                            "K_hat_J": {1: 2000, 2: 3000, 3:200},
                            "builds": {None: 2000},
                            "k_hat_J": {1: 30, 2: 20, 3:40},
                            ...
                        }}
                
            Input:      
                translated Dict with correct Keys
                
                - filter Keyword to filter incoming data
                
            Output:     
                Dict interpretabele by pyomo with "None:"-keys and
                
                - individual variables
            """
            
            
            def filterActive(unfiltered, filter_kw=filter_kw):
                """
                Input
                    Unfiltered dictionary.
                
                Output
                    Filtered dictionary filtered by "filter_kw".
                
                
                Searches for keys three layers down.
                    - First layer
                        - Table
                    - Second layer
                        - Indexes
                    - Third layer
                        - Dictionary with "filter_kw" key.
                    
                    Value is Boolean.
                """
                filtered = copy.deepcopy(unfiltered) # don't alter orig. dict 
                
                for items in filtered:
                    try:
                        for items2 in list(filtered[items]):
                            try:
                                # if row not active -> delete whole entry
                                if filtered[items][items2][filter_kw] is False:
                                    filtered[items].pop(items2)
                            except KeyError: # catch if filter_kw isn't present
                                pass        
                    except TypeError: # catch if dict is not nested.
                        pass

                return filtered             
            
            # filter the input dictionary 
            input_dict = filterActive(input_dict)
            
            # create output dictionary
            output_dict = {None: {}}
            
            # go one layer down
            write_dict = output_dict[None]
            
            
            def createIndexListDictItem(self, idx_dict, key):
                """Builds a dictionary in the form::
                    
                    {key: {None: [i1, i2, ...]}}

                from::
                   
                    key: {i1: {...}, i2: {...}}
                
                To create index-lists.
                """
                none_style_idx = {key: {None: []}}
                
                for idx in idx_dict[key].keys():
                    none_style_idx[key][None].append(idx)
                
                return none_style_idx
            

            # add Index-Lists
            for idx in input_dict["idx_list"]:
                write_dict.update(createIndexListDictItem(
                        self, input_dict, idx))
            
            ####################
            # SHOULD BE DYNAMIC 
            ####################
            # keep certain keys
            binary_keys = ["alpha_ub",
                           "beta_ub",
                           "delta_ub",
                           "gamma_ub",  
                           "tau_ub",
                           "lambda_tilde",
                           "upsilon_tilde",
                           "kappa_ub",
                           ]
            
            def createBinaryParameters(a, w, key_list, filter_kw=filter_kw):
                for key in key_list:
                    try:
                        w[key] = {}
                        for i in list(a[key]):
                            w[key][i] = a[key][i][filter_kw]
                            w[key][i] = int(w[key][i]) # turns bool -> 0/1
                    except KeyError:
                        print("key '{0}' has not been found.".format(key))
                        write_dict.pop(key)
                        
            createBinaryParameters(input_dict, write_dict, binary_keys)

            
            def writeDict(table, column, write_key, w=write_dict):
                w[write_key] = {}
                for k in input_dict[table].keys():
                    w[write_key][k] = input_dict[table][k][column]
                
            translate_all = [
                ["J",       "investcost",               "K_hat_J"       ],
                ["J",       "processcost_per_build",    "k_hat_J"       ],
                ["kappa_ub","processcost_per_unit",     "k_var_J"       ],
                ["H",       "investcost",               "K_hat_H"       ],
                ["H",       "cost_per_bandwidth",       "k_hat_H"       ],
                ["S",       "investcost",               "K_hat_S"       ],
                ["S",       "cost_per_bandwidth",       "k_hat_S"       ],
                ["P",       "cost_per_hour",            "k_var_P"       ],
                ["G",       "weight_factor",            "w"             ],
                ["G",       "z_ub",                     "z_ub"          ],
                ["G",       "z_lb",                     "z_lb"          ],
                ["kappa_ub","z_J_lb",                   "z_J_lb"        ],
                ["kappa_ub","z_J_ub",                   "z_J_ub"        ],
                ["tau_ub",  "time_in_minutes",          "t_tilde"       ],
                ["P",       "max_time_in_minutes",      "t_ub"          ],
                ["V",       "temp_ub",                  "theta_V_ub"    ],
                ["V",       "temp_lb",                  "theta_V_lb"    ],
                ["J",       "betrieb_temp_max",         "theta_J_ub"    ],
                ["J",       "betrieb_temp_min",         "theta_J_lb"    ],
                ["kappa_ub","bandwidth_fix",            "R_hat_GJ"      ],
                ["kappa_ub","bandwidth_per_z",          "R_var_GJ"      ],
                ["H",       "bandwidth_ub",             "R_H_ub"        ],
                ["S",       "bandwidth_ub",             "R_S_ub"        ],
                ["Q",       "unit_factor",              "unit_factor"   ],
                ["J",       "kbit_per_sec_base",        "R_J_base"      ],
                ["J",       "bits_base",                "D_J_base"      ],
                ["Q",       "unit_factor",              "r"             ],
                ["Q",       "direction_normal",         "ny"            ],
                ["Q",       "direction_invers",         "ny_invers"     ],
                ["",        "",                         ""              ],
                ]
            for translator in translate_all:
                try:
                    writeDict(translator[0],translator[1],translator[2])
                except KeyError:
                    pass
            
            return output_dict
        
        if self.keytype is "tupled":  
            self.pyomo_dict = dictToNoneType(self, self.trans_dict)
        else:
            print("\nNo 'pyomo_dict' created. Please set keytype='tupled'!\n")
        
        
        self.workbook.close()


    def preprocess(self):
        pass
        # weights preprocessing
        # z_werte?