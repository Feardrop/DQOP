# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:59:50 2019

@author: philipps
"""
import collections
from copy import deepcopy
# =============================================================================
# import openpyxl as op
# 
# # File als Dict
# def loadExcelToDict(filename, filtered = True):
#     workbook = op.load_workbook(filename, data_only = True)
#     
#     workbook_dict = dict()
#     for sheetname in workbook.sheetnames:
#         if sheetname.startswith("table_"):
#             workbook_dict[sheetname] = dict()
#             workbook_dict[sheetname] = sheetToDict(
#                     sheetname,  ID_cols = 1, wb = workbook, filtered = filtered) #  1 ID
#         elif sheetname.startswith("dict_"):   
#             workbook_dict[sheetname] = dict()
#             workbook_dict[sheetname] = sheetToDict(
#                     sheetname, ID_cols = 2, wb = workbook, filtered = filtered) #  2 IDs
#         else: 
#             pass
#     
#     workbook.close()
#     return workbook_dict
# =============================================================================
    
    # Tabelle als Dict
def sheetToDict(sheetname, ID_cols, wb, keytype="", filtered = True):
    """
    Parameters:
        sheetname   Name des Arbeitsblattes
        ID_cols     Anzahl der Index-Spalten
        wb          openpyxl-workbook
        filtered    Lesen aller Daten (False)/ nur sichtbare (True)
        keytype     "nested" = {1: {1: value}} oder 
                    "tupled" = {(1,1): value}
    """
    
    sheet = wb[sheetname]
    sheet_dict = dict()

    for row in sheet:
        try:
            if filtered:
                if not sheet.row_dimensions[row[0].row].hidden: 
                    # gefiltert Daten einlesen
                    deep_merge(sheet_dict, row_to_dict(row, sheet, ID_cols, keytype))
            else:
                deep_merge(sheet_dict, row_to_dict(row, sheet, ID_cols, keytype))
        except AttributeError:
            pass
        
    return sheet_dict
    
def row_to_dict(row, sheet, ID_cols, keytype):
    
    row_dict = dict()
    row_dict_keys = list()
    
    if sheet.row_dimensions[row[0].row].index != 1:
        # Nested Dict for layers
        for cell in row[0:ID_cols]:
            row_dict_keys.append(cell.value)
        
        if keytype is "nested":
            row_dict = createDeepDict(createRecord(
                    row, readHeader(sheet[1], ID_cols), ID_cols), row_dict_keys)
            
        elif keytype is "tupled":
            try:
                if len(row_dict_keys) == 1:
                    index = row_dict_keys
                else:
                    index = tuple(row_dict_keys)
        
                row_dict = {index: createRecord(
                    row, readHeader(sheet[1], ID_cols), ID_cols)}
            except:
                row_dict = createDeepDict(createRecord(
                    row, readHeader(sheet[1], ID_cols), ID_cols), row_dict_keys)
        else:
            raise KeyError('row_to_dict(keytype)-argument expected values "nested" or "tupled", got %d' % keytype)
    
    # Only return active data
    return row_dict

# Header als Liste auslesen
def readHeader(head_row, ID_cols, IncludeID=False):
    headers = list()
    for cell in head_row:
        headers.append(cell.value)  
        
    if not IncludeID:
        headers = headers[ID_cols:]
        
    return headers

# Erstellt einen Eintrag
def createRecord(row, sorted_keys_list, ID_cols):
            
    from distutils.util import strtobool
            
    record_dict = dict()
    
    for key_idx in range(len(sorted_keys_list)):
        cell_value = row[ID_cols+key_idx].value
        if (sorted_keys_list[key_idx].startswith("val_")):
        # (sorted_keys_list[key_idx] == "bool" or
        #     sorted_keys_list[key_idx] == "variable_name"):
            try:
                cell_value = strtobool(cell_value)
            except:
                cell_value = cell_value
            return cell_value # only 1 value
        else:
            if (sorted_keys_list[key_idx].startswith("kval_")):
            # (sorted_keys_list[key_idx] == "used" or 
            #  sorted_keys_list[key_idx] == "recorded"):
                try: 
                    cell_value = strtobool(cell_value)
                except:
                    cell_value = cell_value
            if cell_value is None or cell_value is "":
                cell_value = None
            record_dict[sorted_keys_list[key_idx]] = cell_value
    # print(record_dict)
    return record_dict
    

def createDeepDict(value, layers):
    """Create a dictionary off a list
    """
    orig_data = dict()
    data = orig_data
    last_layer = layers[-1]

    for layer in layers[:-1]:
        data[layer] = dict()
        data = data[layer]

    data[last_layer] = value

    return orig_data

def deepDictValue(data, layers):
    """recurse createDeepDict
    """
    for layer in layers:
        data = data[layer]

    return data

def deep_merge(d, u):
   """Do a deep merge of one dict into another.

   This will update d with values in u, but will not delete keys in d
   not found in u at some arbitrary depth of d. That is, u is deeply
   merged into d.

   Args -
     d, u: dicts

   Note: this is destructive to d, but not u.

   Returns: None
   
   link: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
   """
   stack = [(d,u)]
   while stack:
      d,u = stack.pop(0)
      for k,v in u.items():
         if not isinstance(v, collections.Mapping):
            # u[k] is not a dict, nothing to merge, so just set it,
            # regardless if d[k] *was* a dict
            d[k] = v
         else:
            # note: u[k] is a dict

            # get d[k], defaulting to a dict, if it doesn't previously
            # exist
            dv = d.setdefault(k, {})

            if not isinstance(dv, collections.Mapping):
               # d[k] is not a dict, so just set it to u[k],
               # overriding whatever it was
               d[k] = v
            else:
               # both d[k] and u[k] are dicts, push them on the stack
               # to merge
               stack.append((dv, v))



def custom_pprint(*dicts):
    """ Print a dictionary pretty with indent = 4
    """
    import json
    for dictionary in dicts:
        print('\nStructure:')
        print(json.dumps(dictionary, sort_keys = True, 
                          indent = 4, ensure_ascii = False))

