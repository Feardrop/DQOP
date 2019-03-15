# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:48:02 2019

@author: Feardrop

"""


import pyomo.environ as pe

def createModel(repn_DQ=3, constr_DQ=2, PieceCnt_DQ=10):

    model = pe.AbstractModel(name="MI(N)LP")

    model.instance_name = ""  # Define Instance Name

    ###########################################################################
    # Parameters for the Bitrate Function

    # 1-simple / 2-advanced
    model.R_J_func_type = pe.Param(default=1, mutable=True)


    ###########################################################################
    # Parameters for the Piecewise Linear Function for DataQualities

    # 1-linear / 2-quadratic / 3-cubic / etc
    model.func_factor_DQ  = pe.Param(default=2, mutable=True)

    model.translate_repn = {0:'BIGM_SOS1',
                            1:'BIGM_BIN',
                            2:'SOS2',
                            3:'CC',
                            4:'DCC',
                            5:'DLOG',
                            6:'LOG',
                            7:'MC',
                            8:'INC',
                            None:None}

    model.translate_constr = {0:'UB',
                              1:'LB',
                              2:'EQ',
                              None:None}

    model.repn_DQ = repn_DQ
    model.constr_DQ = constr_DQ

    # Number of Breakpoints
    def shift_bit_length(x):
        """Computes the next power of two::
        
            (2^n)
        """  
        return (1<<(x-1).bit_length())

    if repn_DQ == 5 or repn_DQ == 6:
        PieceCnt_DQ = shift_bit_length(PieceCnt_DQ)-1  # -1 for next use
        print("PieceCnt_DQ has been shifted to next (2^n)+1:", PieceCnt_DQ+2)

    model.PieceCnt_DQ = PieceCnt_DQ


    ###########################################################################
    # Mengen (list([index, ]))
    model.I = pe.Set()
    model.J = pe.Set()
    model.G = pe.Set()
    model.H = pe.Set()
    model.S = pe.Set()
    model.V = pe.Set()
    model.P = pe.Set()
    model.Q = pe.Set()
    model.Q_bild = pe.Set()


    ###########################################################################

    ############
    # Parameter
    ############

    # Einstellungen
    #model.epsilon           = pe.Param(default=1, domain=pe.NonNegativeReals, mutable=True)
    model.u                 = pe.Param(default=0, domain=pe.NonNegativeReals, mutable=True)
    model.n                 = pe.Param(default=1, domain=pe.NonNegativeIntegers, mutable=True)
    model.kappa_sum_zul     = pe.Param(default=1, domain=pe.NonNegativeIntegers, mutable=True)
    model.beta_sum_zul      = pe.Param(default=1, domain=pe.NonNegativeIntegers, mutable=True)
    model.delta_sum_zul     = pe.Param(default=1, domain=pe.NonNegativeIntegers, mutable=True)
    model.gamma_sum_zul     = pe.Param(default=1, domain=pe.NonNegativeIntegers, mutable=True)
    model.tau_sum_zul       = pe.Param(default=1, domain=pe.NonNegativeIntegers, mutable=True)
    model.DQ_ges_max        = pe.Param(default=float('inf'), domain=pe.NonNegativeReals, mutable=True)# TODO
    model.Costs_ges_max     = pe.Param(default=float('inf'), domain=pe.NonNegativeReals, mutable=True)  # TODO

    # Kombinationen
    model.kappa_ub          = pe.Param(model.G, model.J, default=0, domain=pe.Binary)
    model.beta_ub           = pe.Param(model.J, model.H, default=0, domain=pe.Binary)
    model.gamma_ub          = pe.Param(model.H, model.S, default=0, domain=pe.Binary)
    model.delta_ub          = pe.Param(model.J, model.V, default=0, domain=pe.Binary)
    model.tau_ub            = pe.Param(model.J, model.P, default=0, domain=pe.Binary)

    # Zuordnungen
    model.upsilon_tilde     = pe.Param(model.G, model.Q, default=0, domain=pe.Binary)
    model.lambda_tilde      = pe.Param(model.I, model.G, default=0, domain=pe.Binary)

    # Richtung der Güte
    model.ny                = pe.Param(model.Q, default=0, domain=pe.Binary)
    model.ny_invers         = pe.Param(model.Q, default=0, domain=pe.Binary)

    def upsilon_ny_init(model, g, q):
        return model.upsilon_tilde[g, q] * model.ny[q]
    model.upsilon_ny        = pe.Param(model.G, model.Q, default=0, initialize=upsilon_ny_init,  domain=pe.Binary)
    def upsilon_ny_invers_init(model, g, q):
        return model.upsilon_tilde[g, q] * model.ny_invers[q]
    model.upsilon_ny_invers        = pe.Param(model.G, model.Q, default=0, initialize=upsilon_ny_invers_init,  domain=pe.Binary)
    # Kostensätze

    model.K_hat_J           = pe.Param(model.J, default=0, domain=pe.NonNegativeReals)
    model.k_hat_J           = pe.Param(model.J, default=0, domain=pe.NonNegativeReals)
    model.k_var_J           = pe.Param(model.G, model.J, default=0, domain=pe.NonNegativeReals)                # (z_g)


    model.K_hat_H           = pe.Param(model.H, default=0, domain=pe.NonNegativeReals)
    model.k_hat_H           = pe.Param(model.H, default=0, domain=pe.NonNegativeReals)
    model.k_var_H           = pe.Param(model.G, model.H, default=0, domain=pe.NonNegativeReals)                # time active

    model.K_hat_S           = pe.Param(model.S, default=0, domain=pe.NonNegativeReals)
    model.k_hat_S           = pe.Param(model.S, default=0, domain=pe.NonNegativeReals)
    model.k_var_S           = pe.Param(model.G, model.S, default=0, domain=pe.NonNegativeReals)                # time active

    model.k_var_P           = pe.Param(model.P, default=0, domain=pe.NonNegativeReals)

    # DataQuality Beschränkungen
    model.z_ub              = pe.Param(model.G,      default=0, domain=pe.NonNegativeReals)
    model.z_lb              = pe.Param(model.G,      default=0, domain=pe.NonNegativeReals)
    model.z_J_lb            = pe.Param(model.G, model.J, default=0, domain=pe.NonNegativeReals)
    model.z_J_ub            = pe.Param(model.G, model.J, default=0, domain=pe.NonNegativeReals)
    def z_DQ_ub_init(model, g):
        return min(model.z_ub[g],max(model.z_J_ub[g, j] for j in model.J))
    model.z_DQ_ub           = pe.Param(model.G, default=0, initialize=z_DQ_ub_init, domain=pe.NonNegativeReals)
    def z_DQ_lb_init(model, g):
        return max(model.z_lb[g],min(model.z_J_lb[g, j] for j in model.J))
    model.z_DQ_lb           = pe.Param(model.G, default=0, initialize=z_DQ_lb_init, domain=pe.NonNegativeReals)

    # Gewichte
    model.w                 = pe.Param(model.G,      default=0, domain=pe.NonNegativeReals)
    def w_init(model, g):  # Normierung
        return model.w[g] / sum(model.w[g] for g in model.G)
    model.w_norm               = pe.Param(model.G, default=0, initialize=w_init, domain=pe.NonNegativeReals)

    # Arbeitszeit
    model.t_tilde           = pe.Param(model.J, model.P, default=0, domain=pe.NonNegativeReals)    # min
    model.t_ub              = pe.Param(model.P, default=8*60, domain=pe.NonNegativeReals)      # min

    # Temperaturen
    model.theta_V_ub        = pe.Param(model.V, default=30, domain=pe.Integers)  # constr. if not None
    model.theta_V_lb        = pe.Param(model.V, default=20, domain=pe.Integers)
    model.theta_J_ub        = pe.Param(model.J, default=1000000, domain=pe.Integers)
    model.theta_J_lb        = pe.Param(model.J, default=-275, domain=pe.Integers)

    # Bandbreiten
    model.r                 = pe.Param(model.Q, default=1, domain=pe.Reals)
    model.Hz = pe.Param(default=100)
    model.sec = pe.Param(default=101)
    def upsilon_r_init(model, g):
        return sum(model.upsilon_tilde[g, q] * model.r[q] for q in model.Q_bild)
    model.upsilon_r         = pe.Param(model.G, domain=pe.Reals, initialize=upsilon_r_init)
    model.R_J_base          = pe.Param(model.J, default=0, domain=pe.NonNegativeIntegers)
    model.D_J_base          = pe.Param(model.J, default=0, domain=pe.NonNegativeIntegers)

    model.R_H_ub   = pe.Param(model.H,      default=10000000000, domain=pe.NonNegativeReals)  #10GBit/s
    model.R_S_ub   = pe.Param(model.S,      default=16000000000, domain=pe.NonNegativeReals)  #2000MB/s





    ###########################################################################

    ############
    # Variablen
    ############
    # Base Binary

    def z_bounds(model, g):
        return (model.z_lb[g], model.z_ub[g])
    model.z                 = pe.Var(model.G,           domain=pe.NonNegativeReals, bounds=z_bounds)  # ???

    model.alpha             = pe.Var(model.I, model.J,  domain=pe.Binary)
    model.kappa             = pe.Var(model.G, model.J,  domain=pe.Binary)
    model.beta              = pe.Var(model.J, model.H,  domain=pe.Binary)
    model.gamma             = pe.Var(model.H, model.S,  domain=pe.Binary)
    model.delta             = pe.Var(model.J, model.V,  domain=pe.Binary)
    model.tau               = pe.Var(model.J, model.P,  domain=pe.Binary)
    model.a                 = pe.Var(model.J,           domain=pe.Binary)
    model.b                 = pe.Var(model.H,           domain=pe.Binary)
    model.c                 = pe.Var(model.S,           domain=pe.Binary)
    model.d                 = pe.Var(model.P,           domain=pe.Binary)

    # Help Binary
    model.a_beta            = pe.Var(model.J, model.H,  domain=pe.Binary)
    model.b_gamma           = pe.Var(model.H, model.S,  domain=pe.Binary)
    model.a_tau             = pe.Var(model.J, model.P,  domain=pe.Binary)
    model.xi                = pe.Var(model.J,           domain=pe.Binary)
    model.a_xi              = pe.Var(model.J,           domain=pe.Binary)

    # Help Binary/Real
    model.z_kappa           = pe.Var(model.G, model.J,  domain=pe.NonNegativeReals)
    model.R_J_a_beta        = pe.Var(model.J, model.H,  domain=pe.NonNegativeReals)
    model.R_H_b_gamma       = pe.Var(model.H, model.S,  domain=pe.NonNegativeReals)

    # Help Others
    model.R_J               = pe.Var(model.J,           domain=pe.NonNegativeReals)
    model.R_H               = pe.Var(model.H,           domain=pe.NonNegativeReals)
    model.R_S               = pe.Var(model.S,           domain=pe.NonNegativeReals)
    model.K_J               = pe.Var(model.J,           domain=pe.NonNegativeReals)
    model.K_H               = pe.Var(model.H,           domain=pe.NonNegativeReals)
    model.K_S               = pe.Var(model.S,           domain=pe.NonNegativeReals)
    model.K_P               = pe.Var(model.P,           domain=pe.NonNegativeReals)
    model.Costs_J           = pe.Var(                   domain=pe.NonNegativeReals)
    model.Costs_H           = pe.Var(                   domain=pe.NonNegativeReals)
    model.Costs_S           = pe.Var(                   domain=pe.NonNegativeReals)
    model.Costs_P           = pe.Var(                   domain=pe.NonNegativeReals)
    model.Costs_ges         = pe.Var(                   domain=pe.NonNegativeReals)
    model.DQ                = pe.Var(model.G,           domain=pe.NonNegativeReals)
    model.DQ_ges            = pe.Var(                   domain=pe.NonNegativeReals)
    model.K_J_a             = pe.Var(model.J,           domain=pe.NonNegativeReals)
    model.K_H_b             = pe.Var(model.H,           domain=pe.NonNegativeReals)
    model.K_S_c             = pe.Var(model.S,           domain=pe.NonNegativeReals)
    model.K_P_d             = pe.Var(model.P,           domain=pe.NonNegativeReals)
    model.t                 = pe.Var(model.P,           domain=pe.NonNegativeReals)

    def R_J_ub_init(model, j):  # !!!
        if pe.value(model.R_J_func_type) == 1:
            return (model.R_J_base[j]
                + (sum(model.z_ub[g] * model.upsilon_tilde[g, pe.value(model.Hz)] * (
                        model.D_J_base[j])
                + (1 / model.z_lb[g]) * model.upsilon_tilde[g, pe.value(model.sec)] * (
                        model.D_J_base[j])
                  for g in model.G)) / 1000 + 1)
        if pe.value(model.R_J_func_type) is 2:
            return (model.R_J_base[j]
                    + sum(model.z_J_ub[g, j] * model.upsilon_tilde[g, pe.value(model.Hz)] * (
                          model.D_J_base[j]
                          + pe.prod(model.z_J_ub[g, j]  #!!!
                                    * model.upsilon_r[g]
                                    for g in model.G))
                    + (1 / model.z_J_lb[g, j]) * model.upsilon_tilde[g, pe.value(model.sec)] * (
                          model.D_J_base[j]
                          + pe.prod(model.z_J_ub[g, j]  #!!!
                                    * model.upsilon_r[g]
                                    for g in model.G))
                          for g in model.G) / 1000 + 1)
    model.R_J_ub = pe.Param(model.J, default=1e10, initialize=R_J_ub_init)  # !!!

    def K_J_ub_init(model, j):
        return (model.K_hat_J[j] + model.n * (model.k_hat_J[j])
                + sum(model.z_ub[g] * model.k_var_J[g,j] for g in model.G))
    model.K_J_ub = pe.Param(model.J, domain=pe.PositiveReals, initialize=K_J_ub_init)

    def K_H_ub_init(model, h):
        return (model.K_hat_H[h] + model.n * model.k_hat_H[h] + 1)
    model.K_H_ub = pe.Param(model.H, domain=pe.PositiveReals, rule=K_H_ub_init)

    def K_S_ub_init(model, s):
        return (model.K_hat_S[s] + model.n * model.k_hat_S[s] + 1)
    model.K_S_ub = pe.Param(model.S, domain=pe.PositiveReals, initialize=K_S_ub_init)

    def K_P_ub_init(model, p):  # Achtung hier werden Min in Stunden umgerechnet
        return (model.n * model.t_ub[p] / 60 * model.k_var_P[p] + 1)  #!!!
    model.K_P_ub = pe.Param(model.P, domain=pe.PositiveReals, initialize=K_P_ub_init)



    # Binary Operations

    def Constrain_a_beta_1_rule(model, j, h):
        return model.a_beta[j, h] <= model.a[j]
    model.Constrain_a_beta_1 = pe.Constraint(model.J, model.H, rule=Constrain_a_beta_1_rule)

    def Constrain_a_beta_2_rule(model, j, h):
        return model.a_beta[j, h] <= model.beta[j, h]
    model.Constrain_a_beta_2 = pe.Constraint(model.J, model.H, rule=Constrain_a_beta_2_rule)

    def Constrain_a_beta_3_rule(model, j, h):
        return model.a_beta[j, h] >= (model.a[j] + model.beta[j, h] - 1)
    model.Constrain_a_beta_3 = pe.Constraint(model.J, model.H, rule=Constrain_a_beta_3_rule)


    def Constrain_b_gamma_1_rule(model, h, s):
        return model.b_gamma[h, s] <= model.b[h]
    model.Constrain_b_gamma_1 = pe.Constraint(model.H, model.S, rule=Constrain_b_gamma_1_rule)

    def Constrain_b_gamma_2_rule(model, h, s):
        return model.b_gamma[h, s] <= model.gamma[h, s]
    model.Constrain_b_gamma_2 = pe.Constraint(model.H, model.S, rule=Constrain_b_gamma_2_rule)

    def Constrain_b_gamma_3_rule(model, h, s):
        return model.b_gamma[h, s] >= model.b[h] + model.gamma[h, s] - 1
    model.Constrain_b_gamma_3 = pe.Constraint(model.H, model.S, rule=Constrain_b_gamma_3_rule)


    def Constrain_a_tau_1_rule(model, j, p):
        return model.a_tau[j, p] <= model.a[j]
    model.Constrain_a_tau_1 = pe.Constraint(model.J, model.P, rule=Constrain_a_tau_1_rule)

    def Constrain_a_tau_2_rule(model, j, p):
        return model.a_tau[j, p] <= model.tau[j, p]
    model.Constrain_a_tau_2 = pe.Constraint(model.J, model.P, rule=Constrain_a_tau_2_rule)

    def Constrain_a_tau_3_rule(model, j, p):
        return model.a_tau[j, p] >= model.a[j] + model.tau[j, p] - 1
    model.Constrain_a_tau_3 = pe.Constraint(model.J, model.P, rule=Constrain_a_tau_3_rule)


    ###########################################################################
    #
    # Constraints
    #

    #==============================================================================

    def Constrain_alpha_BigM_rule(model, i, j):
        """
        .. math::

            \\tilde{\\lambda}_{gj} \\wedge \\kappa_{gj} = 1 \\Rightarrow \\alpha_{ij} = 1
        """
        return (sum(model.lambda_tilde[i, g]
                    * model.kappa[g, j] for g in model.G)
                - model.alpha[i, j] * (len(model.I)+len(model.G)) <= 0)
    model.Constrain_alpha_BigM = pe.Constraint(model.I, model. J,
                                               rule=Constrain_alpha_BigM_rule)

    def Constrain_alpha_smallm_rule(model, i, j):
        return (sum(model.lambda_tilde[i, g]
                    * model.kappa[g, j] for g in model.G) - model.alpha[i, j] * 1 >= 0)
    model.Constrain_alpha_smallm = pe.Constraint(model.I, model.J,
                                                 rule=Constrain_alpha_smallm_rule)

    def Constrain_alpha_rule(model, i):
        return sum(model.alpha[i, j] for j in model.J) <= 1
    model.Constrain_alpha = pe.Constraint(model.I, rule=Constrain_alpha_rule)



    def Constrain_a_BigM_rule(model, j):
        return sum(model.kappa[g, j]
                   for g in model.G) - model.a[j] * (len(model.J)+len(model.G)) <= 0
    model.Constrain_a_BigM = pe.Constraint(model.J, rule=Constrain_a_BigM_rule)

    def Constrain_a_smallm_rule(model, j):
        return sum(model.kappa[g, j] for g in model.G) - model.a[j] * 1 >= 0
    model.Constrain_a_smallm = pe.Constraint(model.J, rule=Constrain_a_smallm_rule)


    def Constrain_b_BigM_rule(model, h):
        return sum(model.a_beta[j, h]
                   for j in model.J) - model.b[h] * (len(model.H)+len(model.J)) <= 0
    model.Constrain_b_BigM = pe.Constraint(model.H, rule=Constrain_b_BigM_rule)

    def Constrain_b_smallm_rule(model, h):
        return sum(model.a_beta[j, h] for j in model.J) - model.b[h] * 1 >= 0
    model.Constrain_b_smallm = pe.Constraint(model.H, rule=Constrain_b_smallm_rule)


    def Constrain_c_BigM_rule(model, s):
        return sum(model.b_gamma[h, s]
                   for h in model.H) - model.c[s] * (len(model.S)+len(model.H)) <= 0
    model.Constrain_c_BigM = pe.Constraint(model.S, rule=Constrain_c_BigM_rule)

    def Constrain_c_smallm_rule(model, s):
        return sum(model.b_gamma[h, s] for h in model.H) - model.c[s] * 1 >= 0
    model.Constrain_c_smallm = pe.Constraint(model.S, rule=Constrain_c_smallm_rule)


    def Constrain_d_BigM_rule(model, p):
        return sum(model.a_tau[j, p]
                   for j in model.J) - model.d[p] * (len(model.P)+len(model.J)) <= 0
    model.Constrain_d_BigM = pe.Constraint(model.P, rule=Constrain_d_BigM_rule)

    def Constrain_d_smallm_rule(model, p):
        return sum(model.a_tau[j, p] for j in model.J) - model.d[p] * 1 >= 0
    model.Constrain_d_smallm = pe.Constraint(model.P, rule=Constrain_d_smallm_rule)

    #==============================================================================

    #  # Max min Dataquality
    # Einführung von Hilfsvariablen zur Linearisierung

    def Constrain_z_kappa_1_rule(model, g, j):
        return model.z_kappa[g, j] <= model.kappa[g, j] * model.z_ub[g]
    model.Constrain_z_kappa_1 = pe.Constraint(model.G, model.J,
                                             rule=Constrain_z_kappa_1_rule)
    def Constrain_z_kappa_2_rule(model, g, j):
        return model.z_kappa[g, j] <= model.z[g]
    model.Constrain_z_kappa_2 = pe.Constraint(model.G, model.J,
                                             rule=Constrain_z_kappa_2_rule)
    def Constrain_z_kappa_3_rule(model, g, j):
        return model.z_kappa[g, j] >= model.z[g] - (1 - model.kappa[g, j]) * model.z_ub[g]
    model.Constrain_z_kappa_3 = pe.Constraint(model.G, model.J,
                                             rule=Constrain_z_kappa_3_rule)

    # Beschränkung Datenqualität lokal
    def ConstrainLocalDQ_ub_rule(model, g, j):  # DQ Upper bound local
        return model.z_kappa[g, j] <= model.kappa[g, j] * model.z_J_ub[g, j]
    model.ConstrainLocalDQ_ub = pe.Constraint(model.G, model.J,
                                              rule=ConstrainLocalDQ_ub_rule)
    def ConstrainLocalDQ_lb_rule(model, g, j):  # DQ Lower bound local
        return model.z_kappa[g, j] >= model.kappa[g, j] * model.z_J_lb[g, j]
    model.ConstrainLocalDQ_lb = pe.Constraint(model.G, model.J,
                                              rule=ConstrainLocalDQ_lb_rule)

    # Beschränkung Datenqualität global
    def ConstrainGlobalDQ_ub_rule(model, g):  # DQ upper bound global
        return model.z[g] <= model.z_ub[g]
    model.ConstrainGlobalDQ_ub = pe.Constraint(model.G,
                                               rule=ConstrainGlobalDQ_ub_rule)
    def ConstrainGlobalDQ_lb_rule(model, g):  # DQ lower bound global
        return model.z[g] >= model.z_lb[g]
    model.ConstrainGlobalDQ_lb = pe.Constraint(model.G,
                                               rule=ConstrainGlobalDQ_lb_rule)

    #==============================================================================

    # Gültige Kombinationen
    def ConstrainPossibleCombinations_kappa_rule(model, x, y):
        return model.kappa[x, y] <= model.kappa_ub[x, y]
    model.ConstraintPossibleCombinations_kappa = pe.Constraint(model.G, model.J,
                    rule=ConstrainPossibleCombinations_kappa_rule)

    def ConstrainPossibleCombinations_beta_rule(model, x, y):
        return model.beta[x, y] <= model.beta_ub[x, y]
    model.ConstraintPossibleCombinations_beta  = pe.Constraint(model.J, model.H,
                    rule=ConstrainPossibleCombinations_beta_rule)

    def ConstrainPossibleCombinations_gamma_rule(model, x, y):
        return model.gamma[x, y] <= model.gamma_ub[x, y]
    model.ConstraintPossibleCombinations_gamma = pe.Constraint(model.H, model.S,
                    rule=ConstrainPossibleCombinations_gamma_rule)

    def ConstrainPossibleCombinations_delta_rule(model, x, y):
        return model.delta[x, y] <= model.delta_ub[x, y]
    model.ConstraintPossibleCombinations_delta = pe.Constraint(model.J, model.V,
                    rule=ConstrainPossibleCombinations_delta_rule)

    def ConstrainPossibleCombinations_tau_rule(model, x, y):
        return model.tau[x, y] <= model.tau_ub[x, y]
    model.ConstraintPossibleCombinations_tau = pe.Constraint(model.J, model.P,
                    rule=ConstrainPossibleCombinations_tau_rule)

    #==============================================================================
    # x pro y
    # def ConstrainMethodsPerDQ_rule(model, g):
    #         return sum(model.kappa[g, j] for j in model.J) == 1  #model.kappa_sum_zul  # ???
    # model.ConstrainMethodsPerDQ = pe.Constraint(
    #         model.G, rule=ConstrainMethodsPerDQ_rule)
    # def ConstrainMethodsPerDQ_rule(model, g):
    #         return sum(pe.sumproduct(model.lambda_tilde,model.kappa) for j in model.J == 1  #model.kappa_sum_zul  # ???
    # model.ConstrainMethodsPerDQ = pe.Constraint(
    #         model.G, rule=ConstrainMethodsPerDQ_rule)

    def ConstrainMethodsPerDQ_smallm_rule(model, g):
            return sum(model.kappa[g, j] for j in model.J) >= 0.5  # ???
    model.ConstrainMethodsPerDQ_smallm = pe.Constraint(
            model.G, rule=ConstrainMethodsPerDQ_smallm_rule)
    #def ConstrainMethodsPerDQ_smallm_rule(model):
    #        return sum(sum(model.kappa[g, j] for j in model.J) for g in model.G) >= 1  # ???
    #model.ConstrainMethodsPerDQ_smallm = pe.Constraint(
    #       rule=ConstrainMethodsPerDQ_smallm_rule)

    def ConstrainTransfersPerMethod_rule(model, j):
            return sum(model.beta[j, h] for h in model.H) <= model.beta_sum_zul
    model.ConstrainTransfersPerMethod = pe.Constraint(
            model.J, rule=ConstrainTransfersPerMethod_rule)

    def ConstrainMinTransfersPerMethod_rule(model, j):
            return sum(model.beta[j, h] for h in model.H) - 1 >= (model.a[j] - 1)
    model.ConstrainMinTransfersPerMethod = pe.Constraint(
            model.J, rule=ConstrainMinTransfersPerMethod_rule)

    def ConstrainStoragesPerTransfers_rule(model, h):
            return sum(model.gamma[h, s] for s in model.S) <= model.gamma_sum_zul
    model.ConstrainStoragesPerTransfers = pe.Constraint(
            model.H, rule=ConstrainStoragesPerTransfers_rule)

    def ConstrainMinStoragesPerTransfers_rule(model, h):
            return sum(model.gamma[h, s] for s in model.S) - 1 >= (model.b[h] - 1)
    model.ConstrainMinStoragesPerTransfers = pe.Constraint(
            model.H, rule=ConstrainMinStoragesPerTransfers_rule)

    def ConstrainMethodPerPosition_rule(model, v):
        return sum(model.delta[j, v] for j in model.J) <= model.delta_sum_zul
    model.ConstrainMethodPerPosition = pe.Constraint(
            model.V, rule=ConstrainMethodPerPosition_rule)

    #def ConstrainMinMethodPerPosition_rule(model, v):
    #    return sum(model.delta[j, v] for j in model.J) - 1 >= (model.d[v] - 1)
    #model.ConstrainMinMethodPerPosition = pe.Constraint(
    #        model.V, rule=ConstrainMinMethodPerPosition_rule)

    def ConstrainPersonPerMethod_rule(model, j):
        return sum(model.tau[j, p] for p in model.P) <= model.tau_sum_zul
    model.ConstrainPersonPerMethod = pe.Constraint(
            model.J, rule=ConstrainPersonPerMethod_rule)
    #==============================================================================

    # Temperatures
    def ConstrainMaxTheta_rule(model, j, v):
        return model.delta[j, v] * model.theta_J_ub[j] >= model.delta[j, v] * model.theta_V_ub[v]
    model.ConstrainMaxTheta = pe.Constraint(model.J, model.V,
                                            rule=ConstrainMaxTheta_rule)
    def ConstrainMinTheta_rule(model, j, v):
        return model.delta[j, v] * model.theta_J_lb[j] <= model.delta[j, v] * model.theta_V_lb[v]
    model.ConstrainMinTheta = pe.Constraint(model.J, model.V,
                                            rule=ConstrainMinTheta_rule)

    #==============================================================================

    # Personalzeiten
    # Berechnung xi "Sollte eine Person benötigt werden?"
    def Constrain_bigM_xi_rule(model, j):
        return sum(model.tau_ub[j, p] for p in model.P) - len(model.P) * model.xi[j] <= 0
    model.Constrain_bigM_xi = pe.Constraint(model.J, rule=Constrain_bigM_xi_rule)

    def Constrain_smallm_xi_rule(model, j):
        return sum(model.tau_ub[j, p] for p in model.P) - 1 * model.xi[j] >= 0
    model.Constrain_smallm_xi = pe.Constraint(model.J, rule=Constrain_smallm_xi_rule)

    # Berechnung a_xi "Wird eine Person wirklich benötigt?"
    def Constrain_a_xi_1_rule(model, j):
        return model.a_xi[j] <= model.a[j]
    model.Constrain_a_xi_1 = pe.Constraint(model.J, rule=Constrain_a_xi_1_rule)

    def Constrain_a_xi_2_rule(model, j):
        return model.a_xi[j] <= model.xi[j]
    model.Constrain_a_xi_2 = pe.Constraint(model.J, rule=Constrain_a_xi_2_rule)

    def Constrain_a_xi_3_rule(model, j):
        return model.a_xi[j] >= model.a[j] + model.xi[j] - 1
    model.Constrain_a_xi_3 = pe.Constraint(model.J, rule=Constrain_a_xi_3_rule)


    # wenn a_xi = 1 -> sum(tau[j, p] for p in model.P) >= 1 "Min 1 Person muss ran"
    def Constrain_sumtau_smallm_rule(model, j):
        return sum(model.a_tau[j, p] for p in model.P) - model.a_xi[j] * 1 >= 0
    model.Constrain_sumtau_smallm = pe.Constraint(model.J, rule=Constrain_sumtau_smallm_rule)


    def Compute_t_rule(model, p):  # Berechne Zeit pro Person
        return model.t[p] - sum(model.a_tau[j, p] * model.t_tilde[j, p] for j in model.J) == 0
    model.Compute_t = pe.Constraint(model.P, rule=Compute_t_rule)

    def ConstrainPersonalWorkTimes_rule(model, p):  # Zeitrestriktion pro Person
        return model.t[p] <= model.t_ub[p]
    model.ConstrainPersonalWorkTimes = pe.Constraint(model.P,
                                            rule=ConstrainPersonalWorkTimes_rule)

    #==============================================================================

    # Bitrates

    def Compute_R_J_rule(model, j):
        if pe.value(model.R_J_func_type) == 1:
            return (model.R_J[j] == model.R_J_base[j]
                + (sum(model.z_kappa[g, j] * model.upsilon_tilde[g, pe.value(model.Hz)] * (
                        model.D_J_base[j])
                + (1 / model.z_kappa[g, j]) * model.upsilon_tilde[g, pe.value(model.sec)] * (
                        model.D_J_base[j])
                  for g in model.G)) / 1000)
        if pe.value(model.R_J_func_type) is 2:
            return (model.R_J[j] ==
                    model.R_J_base[j]
                    + sum(model.z_kappa[g, j] * model.upsilon_tilde[g, pe.value(model.Hz)] * (
                          model.D_J_base[j]
                          + pe.prod(model.z_kappa[g, j]  #!!!
                                    * model.upsilon_r[g]
                                    for g in model.G))
                    + (1 / model.z_kappa[g, j]) * model.upsilon_tilde[g, pe.value(model.sec)] * (
                          model.D_J_base[j]
                          + pe.prod(model.z_kappa[g, j]  #!!!
                                    * model.upsilon_r[g]
                                    for g in model.G))
                          for g in model.G) / 1000)
    model.Compute_R_J = pe.Constraint(model.J, rule=Compute_R_J_rule)

    # Linearisierung R_J_a_beta = a_beta * R_J
    def Constrain_R_J_a_beta_1_rule(model, j, h):
        return model.R_J_a_beta[j, h] <= model.a_beta[j, h] * model.R_J_ub[j]  # !!!
    model.Constrain_R_J_a_beta_1 = pe.Constraint(model.J, model.H,
                                             rule=Constrain_R_J_a_beta_1_rule)
    def Constrain_R_J_a_beta_2_rule(model, j, h):
        return model.R_J_a_beta[j, h] <= model.R_J[j]
    model.Constrain_R_J_a_beta_2 = pe.Constraint(model.J, model.H,
                                             rule=Constrain_R_J_a_beta_2_rule)
    def Constrain_R_J_a_beta_3_rule(model, j, h):
        return model.R_J_a_beta[j, h] >= model.R_J[j] - (1 - model.a_beta[j, h]) * model.R_J_ub[j]  # !!!
    model.Constrain_R_J_a_beta_3 = pe.Constraint(model.J, model.H,
                                             rule=Constrain_R_J_a_beta_3_rule)


    # Constraint for Transfer Rates
    def Compute_R_H_rule(model, h):
        return model.R_H[h] - sum(model.R_J_a_beta[j, h] for j in model.J) == 0
    model.Compute_R_H = pe.Constraint(model.H, rule=Compute_R_H_rule)

    def Constrain_R_H_rule(model, j, h):
        return model.R_H[h] <= model.R_H_ub[h]
    model.Constrain_R_H = pe.Constraint(model.J, model.H, rule=Constrain_R_H_rule)

    # Linearisierung R_H_b_gamma = b_gamma * R_H
    def Constrain_R_H_b_gamma_1_rule(model, h, s):
        return model.R_H_b_gamma[h, s] <= model.b_gamma[h, s] * model.R_H_ub[h]  # !!!
    model.Constrain_R_H_b_gamma_1 = pe.Constraint(model.H, model.S,
                                             rule=Constrain_R_H_b_gamma_1_rule)
    def Constrain_R_H_b_gamma_2_rule(model, h, s):
        return model.R_H_b_gamma[h, s] <= model.R_H[h]
    model.Constrain_R_H_b_gamma_2 = pe.Constraint(model.H, model.S,
                                             rule=Constrain_R_H_b_gamma_2_rule)
    def Constrain_R_H_b_gamma_3_rule(model, h, s):
        return model.R_H_b_gamma[h, s] >= model.R_H[h] - (1 - model.b_gamma[h, s]) * model.R_H_ub[h]  # !!!
    model.Constrain_R_H_b_gamma_3 = pe.Constraint(model.H, model.S,
                                             rule=Constrain_R_H_b_gamma_3_rule)

    # Constraint for Storage Rates
    def Compute_R_S_rule(model, s):
        return model.R_S[s] - sum(model.R_H_b_gamma[h, s] for h in model.H) == 0
    model.Compute_R_S = pe.Constraint(model.S, rule=Compute_R_S_rule)

    def Constrain_R_S_rule(model, s):
        return model.R_S[s] <= model.R_S_ub[s]
    model.Constrain_R_S = pe.Constraint(model.S, rule=Constrain_R_S_rule)


    ###########################################################################


    # Funktionen Kosten

    # Kostenberechnungen

    def Compute_K_J_rule(model, j):
        return model.K_J[j] == (model.K_hat_J[j] + model.n * (model.k_hat_J[j])
                 # + sum(model.z[g] * model.k_var_J[g,j] for g in model.G)
                )
    model.Compute_K_J = pe.Constraint(model.J, rule=Compute_K_J_rule)

    def Constrain_K_J_a_1_rule(model, j):
        return model.K_J_a[j] <= model.a[j] * model.K_J_ub[j]  # !!!
    model.Constrain_K_J_a_1 = pe.Constraint(model.J,
                                             rule=Constrain_K_J_a_1_rule)
    def Constrain_K_J_a_2_rule(model, j):
        return model.K_J_a[j] <= model.K_J[j]
    model.Constrain_K_J_a_2 = pe.Constraint(model.J,
                                             rule=Constrain_K_J_a_2_rule)
    def Constrain_K_J_a_3_rule(model, j):
        return model.K_J_a[j] >= model.K_J[j] - (1 - model.a[j]) * model.K_J_ub[j]  # !!!
    model.Constrain_K_J_a_3 = pe.Constraint(model.J,
                                             rule=Constrain_K_J_a_3_rule)

    # =============================================================================
    def Compute_K_H_rule(model, h):
        return model.K_H[h] == (model.K_hat_H[h] + model.n * model.k_hat_H[h])
    model.Compute_K_H = pe.Constraint(model.H, rule=Compute_K_H_rule)

    def Constrain_K_H_b_1_rule(model, h):
        return model.K_H_b[h] <= model.b[h] * model.K_H_ub[h]
    model.Constrain_K_H_b_1 = pe.Constraint(model.H,
                                             rule=Constrain_K_H_b_1_rule)
    def Constrain_K_H_b_2_rule(model, h):
        return model.K_H_b[h] <= model.K_H[h]
    model.Constrain_K_H_b_2 = pe.Constraint(model.H,
                                             rule=Constrain_K_H_b_2_rule)
    def Constrain_K_H_b_3_rule(model, h):
        return model.K_H_b[h] >= model.K_H[h] - (1 - model.b[h]) * model.K_H_ub[h]
    model.Constrain_K_H_b_3 = pe.Constraint(model.H,
                                             rule=Constrain_K_H_b_3_rule)

    # =============================================================================
    def Compute_K_S_rule(model, s):
        return model.K_S[s] == (model.K_hat_S[s]
                                + model.n * model.k_hat_S[s])
    model.Compute_K_S = pe.Constraint(model.S, rule=Compute_K_S_rule)

    def Constrain_K_S_c_1_rule(model, s):
        return model.K_S_c[s] <= model.c[s] * model.K_S_ub[s]
    model.Constrain_K_S_c_1 = pe.Constraint(model.S,
                                             rule=Constrain_K_S_c_1_rule)
    def Constrain_K_S_c_2_rule(model, s):
        return model.K_S_c[s] <= model.K_S[s]
    model.Constrain_K_S_c_2 = pe.Constraint(model.S,
                                             rule=Constrain_K_S_c_2_rule)
    def Constrain_K_S_c_3_rule(model, s):
        return model.K_S_c[s] >= model.K_S[s] - (1 - model.c[s]) * model.K_S_ub[s]
    model.Constrain_K_S_c_3 = pe.Constraint(model.S,
                                             rule=Constrain_K_S_c_3_rule)

    # =============================================================================
    def Compute_K_P_rule(model, p):  # Achtung hier werden Min in Stunden umgerechnet
        return model.K_P[p] == (model.n * model.t[p] / 60
                                * model.k_var_P[p])  #!!!
    model.Compute_K_P = pe.Constraint(model.P, rule=Compute_K_P_rule)



    def Constrain_K_P_d_1_rule(model, p):
        return model.K_P_d[p] <= model.d[p] * model.K_P_ub[p]
    model.Constrain_K_P_d_1 = pe.Constraint(model.P, rule=Constrain_K_P_d_1_rule)

    def Constrain_K_P_d_2_rule(model, p):
        return model.K_P_d[p] <= model.K_P[p]
    model.Constrain_K_P_d_2 = pe.Constraint(model.P, rule=Constrain_K_P_d_2_rule)

    def Constrain_K_P_d_3_rule(model, p):
        return model.K_P_d[p] >= model.K_P[p] - (1 - model.d[p]) * model.K_P_ub[p]
    model.Constrain_K_P_d_3 = pe.Constraint(model.P, rule=Constrain_K_P_d_3_rule)

    #==============================================================================

    # Zielfunktion Kosten Sammler
    def ComputeCosts_J_rule(model):
        expcost = sum(model.K_J_a[j] for j in model.J)
        return model.Costs_J - expcost == 0.0
    model.ComputeCosts_J = pe.Constraint(rule=ComputeCosts_J_rule)

    def ComputeCosts_H_rule(model):
        expcost = sum(model.K_H_b[h] for h in model.H)
        return model.Costs_H - expcost == 0.0
    model.ComputeCosts_H = pe.Constraint(rule=ComputeCosts_H_rule)

    def ComputeCosts_S_rule(model):
        expcost = sum(model.K_S_c[s] for s in model.S)
        return model.Costs_S - expcost == 0.0
    model.ComputeCosts_S = pe.Constraint(rule=ComputeCosts_S_rule)

    def ComputeCosts_P_rule(model):
        expcost = sum(model.K_P_d[p] for p in model.P)
        return model.Costs_P - expcost == 0.0
    model.ComputeCosts_P = pe.Constraint(rule=ComputeCosts_P_rule)

    def ComputeCosts_ges_rule(model):
        return model.Costs_ges - (model.Costs_J + model.Costs_H + model.Costs_S + model.Costs_P) == 0
    model.ComputeCosts_ges = pe.Constraint(rule=ComputeCosts_ges_rule)


    ###########################################################################

    # Funktionen Datenqualität

    model.bpts_DQ = {}
    def bpts_DQ_build(model, g):
        model.bpts_DQ[g] = []
        for i in range(int(model.PieceCnt_DQ+2)):
                (model.bpts_DQ[g].append((i**(pe.value(model.func_factor_DQ))
                / (model.PieceCnt_DQ+1)**(pe.value(model.func_factor_DQ)))
                * (model.z_ub[g]-model.z_lb[g]) + model.z_lb[g]))
    # The object model.BuildBpts_DQ is not refered to again;
    # the only goal is to trigger the action at build time
    model.BuildBpts_DQ = pe.BuildAction(model.G, rule=bpts_DQ_build)

    def Compute_DQ_sqrt_rule(model, g, x):
            return (sum(model.upsilon_ny[g, q] for q in model.Q)
                    * ((x - pe.value(model.z_DQ_lb[g]))**(
                            1/pe.value(model.func_factor_DQ)))
                    / ((pe.value(model.z_DQ_ub[g])
                    - pe.value(model.z_DQ_lb[g]))**(
                            1/pe.value(model.func_factor_DQ)))
                    + sum(model.upsilon_ny_invers[g, q] for q in model.Q)
                    * (1-((x - pe.value(model.z_DQ_lb[g]))**(
                            1/pe.value(model.func_factor_DQ)))
                    / ((pe.value(model.z_DQ_ub[g])
                    - pe.value(model.z_DQ_lb[g]))**(
                            1/pe.value(model.func_factor_DQ)))) )
    model.ComputePieces = pe.Piecewise(model.G, model.DQ, model.z,
                                       pw_pts=model.bpts_DQ,
                                       pw_repn=
                                       model.translate_repn[model.repn_DQ],
                                       pw_constr_type=
                                       model.translate_constr[model.constr_DQ],
                                       f_rule=Compute_DQ_sqrt_rule)



    # Zielfunktion Datenqualität Sammler
    def Compute_DQ_ges_rule(model):
        return model.DQ_ges == (sum(model.w_norm[g] * model.DQ[g] for g in model.G))
    model.Compute_DQ_ges = pe.Constraint(rule=Compute_DQ_ges_rule)

    ###########################################################################


    # Zielfunktion Kosten Formel
    def Constrain_DQ_ges(model):
        return model.DQ_ges <= model.DQ_ges_max
    model.Constrain_DQ = pe.Constraint(rule=Constrain_DQ_ges)
    #
    #def CostValue_rule(model):
    #    return model.Costs_ges
    #model.CostValue = pe.Objective(rule=CostValue_rule, sense=1)


    def Constrain_Costs_ges(model):
        return model.Costs_ges <= model.Costs_ges_max
    model.Constrain_Costs = pe.Constraint(rule=Constrain_Costs_ges)
    #
    ## Zielfunktion Datenqualität Formel
    #def DQValue_rule(model):
    #    return model.DQ_ges     # Maximizing
    #model.DQValue = pe.Objective(rule=DQValue_rule, sense=-1)

    def obj_Combined_rule(model):
        return model.u * model.Costs_ges - (1-model.u) * model.DQ_ges
    #    return model.u * (model.Costs_J + model.Costs_H + model.Costs_S + model.Costs_P) - (1-model.u) * model.DQ_ges
    model.CombinedValue = pe.Objective(rule=obj_Combined_rule, sense=1)

    return model