# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 10:13:05 2018

@author: Administrator

TEILGENERIERTE WERTE
"""


import numpy as np
import pickle
import os
import datetime
# import time
import pandas as pd
# import rapidjson as rjson
# import ast
import pprint

def defvar():
    # Open Dictionary
    data = {}
    # Menge der Messgrößen
    I = {
         0: ("Schichtdicke", "µm", "Hz"),
         1: ("Laserleistung", "W", "Hz"),
         2: ("Bauraumtemperatur", "°C", "Hz"),
         3: ("Rakelgeschwindigkeit", "m/s", "Hz"),
         4: ("Größe4", "Einheit1", "Hz")
         }
    data['I'] = I
    # Menge der Messmethoden, Fixkosten, Variable Kosten pro Baujob
    J = {
         0: ("Schichtdickenmesser1",    600,    5),
         1: ("Schichtdickenmesser2",    400,    5),
         2: ("Schichtdickenmesser3",    800,    1),
         3: ("Laserleistungssensor1",   1300,   30),
         4: ("Laserleistungssensor2",   1900,   40),
         5: ("Laserleistungssensor3",   2300,   25),
         6: ("Temperatursensor1",       100,     0.5),
         7: ("Temperatursensor2",       10,    0.3),
         8: ("Temperatursensor3",       1000,   0.7),
         9: ("KombiSchichtLeistung1",   2000,   8.3),
        10: ("KombiLeistungTemperatur1",2500,   43.2),
        11: ("RakelgeschwSensor1",      30,     0.3),
        12: ("RakelgeschwSensor2",      10,     0.9),
        13: ("Größe4Sensor1",          15,     1  ),
        14: ("Größe4Sensor2",          60,     5   ),
        15: ("Größe4Sensor3",          80,     2  ),
        16: ("Größe4Sensor4",          200,    0.1   ),
        17: ("Größe4Sensor5",          45,     8  )
        }
    data['J'] = J
    # Menge der Gütekriterien
    G = {
         0: "Qualität",
         1: "Frequenz"
         }
    data['G'] = G
    # Menge der möglichen Eigenschaften
    C = {
         0: "?"
         }
    data['C'] = C
    # Erlaubte Kombination Messgröße → Messmethode !! ji
    alpha_zul_lst = [[1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                     [0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]]
    data['alpha_zul_lst'] = alpha_zul_lst

    # Gütewerte der Sensoren
    z_J_max = {}
    for i in I:
        factor_i = np.random.randint(1,10)
        for g in G:
            factor_g = np.random.randint(1,10)
            for j in J:
                z_J_max[i,g,j] = [np.random.randint(1, 10 + 1) * factor_i * factor_g]
    data['z_J_max'] = z_J_max

    # Gegebene Mindestgüten
    z_I_min = {}
    for i in I:
        for g in G:
            z_I_min[i,g] = 20
    data['z_I_min'] =  z_I_min

    # Gegebene Gewichte
    w = {}
    for i in I:
        for g in G:
            w[i,g] = np.round(np.random.randint(1, 20)/20, 2)
    data['w'] = w

    # Umwandlung List-Dict
    alpha_zul = {}
    for i in I:
        for j in J:
            alpha_zul[i,j] = alpha_zul_lst[i][j]
    data['alpha_zul'] = alpha_zul

    # Berechung Maximum
    z_I_max = {}
    for i in I:
        for g in G:
            liste_j = []
            for j in J:
                liste_j.append(alpha_zul[i,j] * z_J_max[i,g,j][0])
            z_I_max[i,g] = np.amax(liste_j)
    data['z_I_max'] = z_I_max

    return data

def defrndvar(l_I = 4, l_J = 20, l_G = 3, r_alpha = 0.4, zJmax_rnd_i_max = 10, zJmax_rnd_g_max = 10, zJmax_rnd_max = 10, zImin_rnd_i_max = 10, zImin_rnd_g_max = 10, feinheit_w = 20, ik_max = 10000, bk_fix_max = 20, bk_var_max = 0.3, *args, **kwargs):

    # Open Dictionary
    data = {}
    # Menge der Messgrößen
    I = {}
    for i in range(0,l_I):
        I[i] = ("Messgröße_{}".format(i),)
        for g in range(0,l_G):
            I[i] += ("Einh.G{}_I{}".format(g,i),)
        print(I[i])
    data['I'] = I

    # Erlaubte Kombination Messgröße → Messmethode !! i,j
    alpha_zul_lst = np.random.choice([0, 1], size=(l_I,l_J), p=[(1-r_alpha),r_alpha])
#    print(alpha_zul_lst)
    data['alpha_zul_lst'] = alpha_zul_lst

    # Umwandlung List-Dict
    alpha_zul = {}
    for i in range(0,l_I):
        for j in range(0,l_J):
            alpha_zul[i,j] = alpha_zul_lst[i][j]
    data['alpha_zul'] = alpha_zul

    # Menge der Messmethoden, Fixkosten, Variable Kosten pro Baujob
    J = {}
    # hier alpha hinzufügen und individuelle kosten
#    if alpha_zul_lst[i][j] == 1:
#        factor_fix[j]

    for j in range(0,l_J):
        J[j] = ("Messmethode_{}".format(j), np.random.randint(1,ik_max), np.random.randint(1,bk_fix_max))
        print(J[j])
    data['J'] = J

    # Menge der Gütekriterien
    G = {}
    for g in range(0,l_G):
        G[g] = ("Kriterium_{}".format(g))
        print(G[g])
    data['G'] = G

    # Menge der möglichen Eigenschaften
    C = {
         0: "?"
         }
    data['C'] = C

#    zJmax_rnd_i_max = 10,
#    zJmax_rnd_g_max = 10,
#    zJmax_rnd_max = 10,
#    zImin_rnd_i_max = 10,
#    zImin_rnd_g_max = 10,

    # Gütewerte der Sensoren
    z_J_max = {}
    for i in range(0,l_I):
        factor_i = np.random.randint(1, zJmax_rnd_i_max + 1)
        for g in range(0,l_G):
            factor_g = np.random.randint(1, zJmax_rnd_g_max + 1)
            for j in range(0,l_J):
                z_J_max[i,g,j] = np.random.randint(1, zJmax_rnd_max + 1) * factor_i * factor_g
    data['z_J_max'] = z_J_max

    # Berechung Maximum
    z_I_max = {}
    for i in range(0,l_I):
        for g in range(0,l_G):
            liste_j = []
            for j in range(0,l_J):
                liste_j.append(alpha_zul[i,j] * z_J_max[i,g,j])
            z_I_max[i,g] = np.amax(liste_j)
    data['z_I_max'] = z_I_max

    # Gegebene Mindestgüten
    z_I_min = {}
    for i in range(0,l_I):
        factor_i = np.random.randint(1, zImin_rnd_i_max + 1)
        for g in range(0,l_G):
            factor_g = np.random.randint(1, zImin_rnd_g_max + 1)
            z_I_min[i,g] = 1 * factor_i * factor_g
            while z_I_max[i,g] < z_I_min[i,g]:
                print("z_I_min[{},{}]:".format(i,g),z_I_min[i,g])
                z_I_min[i,g] = z_I_min[i,g]-10
            z_I_min[i,g] = z_I_min[i,g]-10
            if z_I_min[i,g] < 0:
                z_I_min[i,g] = 0
    data['z_I_min'] =  z_I_min

    # Gegebene Gewichte
    w = {}
    for i in I:
        for g in G:
            w[i,g] = np.round(np.random.randint(1, feinheit_w)/feinheit_w, 2)
    data['w'] = w

    # Variable Prozesskosten
    bk_var = {}
    for i in I:
        for j in J:
            for g in G:
                bk_var[i,g,j] = np.random.randint(1,bk_var_max*1000)/1000
    data['bk_var'] = bk_var

    return data


# Speicherung in Datei




def savedata(data_dict,filename,directory,human=False):
    """Saves Dictionaries as Files.
    
    Can be specified to save an additional TXT-file.
    
    Args:
        data_dict: Input Dictionary.
        filename: Filename you want for pickle/TXT.
        directory: Where you want to save the File. Can be given as "abc/def".
        human: Optional Boolean. True for saving TXT file.
    
    Returns:
        Nothing.
    """
    # Save as pickle binary-file
    pickling_on = open(os.path.join(directory,filename+".pickle"),"wb")
    pickle.dump(data_dict, pickling_on)
    pickling_on.close()
    # Save as human readable txt file
    if human == True:
        with open(os.path.join(directory,filename+".txt"), "w") as f:
            pprint.pprint(data_dict, stream=f, width=80)
#            f.write(str(data_dict))
            # print("Save TXT file as", filename+".txt")
    # with open(filename+".json", "w") as f:
    #     rjson.dump(data_dict.items(), f, indent=4)
    #     print("Save JSON file as", filename+".json")

def loadfile(filename,directory):
    pickle_off = open(os.path.join(directory,filename+".pickle"),"rb")
    data_load = pickle.load(pickle_off)
    return data_load




# # save: convert each tuple key to a string before saving as json object
# with open('/tmp/test', 'w') as f: dump({str(k):v for k, v in x.items()}, f)

# # load in two stages:#
# # (i) load json object
# with open('/tmp/test', 'r') as f: obj = load(f)

# # (ii) convert loaded keys from string back to tuple
# d={literal_eval(k):v for k, v in obj.items()}





def createnewdatafile(directory='.\\Data\\' ,rnd=False,*args,**kwargs):
    now = datetime.datetime.now()

    if rnd == False:
        newfile = "Daten_{}".format(now.strftime('%Y-%m-%d_%H%M%S'))
        directory = os.path.join('Data','results_'+newfile)
        os.makedirs(os.path.join(directory,"fig"))
        savedata(defvar(),newfile,directory,human=True)
    else:
        newfile = "Daten_{}_random".format(now.strftime('%Y-%m-%d_%H%M%S'))
        directory = os.path.join('Data','results_'+newfile)
        os.makedirs(os.path.join(directory,"fig"))
        savedata(defrndvar(*args,**kwargs),newfile,directory,human=True)

    print("\n +++ Neue Variablen definiert. +++ \n {} \n String in Zwischenablage".format(newfile))
    pd.DataFrame(["'"+newfile+"'"]).to_clipboard(index=False,header=False)

def confirm(prompt=None, resp=False):
    """prompts for yes or no response from the user. Returns True for yes and
    False for no.

    'resp' should be set to the default value assumed by the caller when
    user simply types ENTER.

    >>> confirm(prompt='Create Directory?', resp=True)
    Create Directory? [y]|n:
    True
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y:
    False
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: y
    True

    """

    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')

    while True:
        ans = input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print('Please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False

