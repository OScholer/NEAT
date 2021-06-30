#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import scipy.constants
from scipy import optimize
import pandas as pd
from matplotlib.lines import Line2D
#import seaborn as sns

import PSFclasses
from scipy import integrate

from NMEs import Load_NMEs

pc = scipy.constants.physical_constants
u_to_GeV = pc["atomic mass constant energy equivalent in MeV"][0]/1000


# In[2]:


SMEFT_WCs = {#dim5                    #Wilson Coefficients of SMEFT
             "LLHH"      : 0,         #up to dimension 7. We only 
             #dim7                    #list the operators violating
             "LH(7)"     : 0,         #lepton number by 2 units.
             "LHD1(7)"   : 0,
             "LHD2(7)"   : 0,
             "LHDe(7)"   : 0,
             #"LHB(7)"    : 0,
             "LHW(7)"    : 0,
             "LLduD1(7)" : 0,
             #"LLeH(7)"   : 0,
             "LLQdH1(7)" : 0,
             "LLQdH2(7)" : 0,
             "LLQuH(7)" : 0,
             "LeudH(7)"  : 0, 
             #dim9
             #  -6-fermi
             "ddueue(9)"    : 0,
             "dQdueL1(9)"   : 0,
             "dQdueL2(9)"   : 0,
             "QudueL1(9)"   : 0,
             "QudueL2(9)"   : 0,
             "dQQuLL1(9)"   : 0,
             "dQQuLL2(9)"   : 0,
             "QuQuLL1(9)"   : 0,
             "QuQuLL2(9)"   : 0,
             "dQdQLL1(9)"   : 0,
             "dQdQLL2(9)"   : 0,
             #  -other
             "LLH4W1(9)"    : 0,
             "deueH2D(9)"   : 0,
             "dLuLH2D2(9)"  : 0,
             "dQLeH2D2(9)"  : 0,
             "dLQeH2D1(9)"  : 0,
             "deQLH2D(9)"   : 0,
             "QueLH2D2(9)"  : 0,
             "QeuLH2D2(9)"  : 0,
             "QLQLH2D2(9)"  : 0,
             "QLQLH2D5(9)"  : 0,
             "QQLLH2D2(9)"  : 0,
             "eeH4D2(9)"    : 0,
             "LLH4D23(9)"   : 0,
             "LLH4D24(9)"   : 0
             }


# In[3]:


LEFT_WCs = {#dim3
            "m_bb"       : 0, 
            #dim6
            "SL(6)"      : 0, 
            "SR(6)"      : 0, 
            "T(6)"       : 0, 
            "VL(6)"      : 0, 
            "VR(6)"      : 0, 
            #dim7
            "VL(7)"      : 0,
            "VR(7)"      : 0, 
            #dim9
            "1L(9)"      : 0, 
            "1R(9)"      : 0, 
            "1L(9)prime" : 0, 
            "1R(9)prime" : 0, 
            "2L(9)"      : 0, 
            "2R(9)"      : 0, 
            "2L(9)prime" : 0, 
            "2R(9)prime" : 0, 
            "3L(9)"      : 0, 
            "3R(9)"      : 0, 
            "3L(9)prime" : 0, 
            "3R(9)prime" : 0, 
            "4L(9)"      : 0, 
            "4R(9)"      : 0, 
            "5L(9)"      : 0, 
            "5R(9)"      : 0, 
            "6(9)"       : 0,
            "6(9)prime"  : 0,
            "7(9)"       : 0,
            "7(9)prime"  : 0,
            "8(9)"       : 0,
            "8(9)prime"  : 0,
            "9(9)"       : 0,
            "9(9)prime"  : 0}


# In[4]:


'''
    
'''
#generate element classes to calculate PSF observables which need the electron wave functions
#elements without NMEs are commented
m_e = pc["electron mass energy equivalent in MeV"][0]

U238 =  PSFclasses.element(92, 238, 1.144154 + 2*m_e)
Th232 = PSFclasses.element(90, 232, 0.837879 + 2*m_e)
#Hg204 = PSFclasses.element(80, 204, 0.419154 + 2*m_e)
Pt198 = PSFclasses.element(78, 198, 1.049142 + 2*m_e)
#Os192 = PSFclasses.element(76, 192, 0.408274 + 2*m_e)
#W186 =  PSFclasses.element(74, 186, 0.491643 + 2*m_e)
#Yb176 = PSFclasses.element(70, 176, 1.088730 + 2*m_e)
#Er170 = PSFclasses.element(68, 170, 0.655586 + 2*m_e)
Gd160 = PSFclasses.element(64, 160, 1.730530 + 2*m_e)
Sm154 = PSFclasses.element(62, 154, 1.250810 + 2*m_e)
Nd150 = PSFclasses.element(60, 150, 3.371357 + 2*m_e)
Nd148 = PSFclasses.element(60, 148, 1.928286 + 2*m_e)
#Nd146 = PSFclasses.element(60, 146, 0.070421 + 2*m_e)
#Ce142 = PSFclasses.element(58, 142, 1.417175 + 2*m_e)
Xe136 = PSFclasses.element(54, 136, 2.457984 + 2*m_e)
Xe134 = PSFclasses.element(54, 134, 0.825751 + 2*m_e)
Te130 = PSFclasses.element(52, 130, 2.527515 + 2*m_e)
Te128 = PSFclasses.element(52, 128, 0.866550 + 2*m_e)
Sn124 = PSFclasses.element(50, 124, 2.291010 + 2*m_e)
#Sn122 = PSFclasses.element(50, 122, 0.372877 + 2*m_e)
Cd116 = PSFclasses.element(48, 116, 2.813438 + 2*m_e)
#Cd114 = PSFclasses.element(48, 114, 0.542493 + 2*m_e)
Pd110 = PSFclasses.element(46, 110, 2.017234 + 2*m_e)
#Ru104 = PSFclasses.element(44, 104, 1.301297 + 2*m_e)
Mo100 = PSFclasses.element(42, 100, 3.034342 + 2*m_e)
#Mo98 =  PSFclasses.element(42, 98, 0.109935 + 2*m_e)
Zr96 =  PSFclasses.element(40, 96, 3.348982 + 2*m_e)
#Zr94 =  PSFclasses.element(40, 94, 1.141919 + 2*m_e)
#Kr86 =  PSFclasses.element(36, 86, 1.257542 + 2*m_e)
Se82 =  PSFclasses.element(34, 82, 2.996402 + 2*m_e)
#Se80 =  PSFclasses.element(34, 80, 0.133874 + 2*m_e)
Ge76 =  PSFclasses.element(32, 76, 2.039061 + 2*m_e)
#Zn70 =  PSFclasses.element(30, 70, 0.997118 + 2*m_e)
Ca48 =  PSFclasses.element(20, 48, 4.266970 + 2*m_e)
#Ca46 =  PSFclasses.element(20, 46, 0.988576 + 2*m_e)


'''
    ##########################################################

    Download atomic weights to calculate avrg nucleon mass M_N

    ##########################################################
'''


#Get masses from all known isotopes
#data = []

#import atomic masses from list
#print("__________________")
#print("")
#print("Importing atomic weights of different isotopes:")
#for i in range(3352):
#    data.append(pd.read_csv("Atomic_Weights.txt", header=None, sep="=", skiprows=i*8, nrows=7))
#summarize all data
#masses = np.zeros((295, 118))

#iterate over all isotopes and get Z, A, and the mass
#for i in range(len(data)):
#    Z = int(data[i][1][0])
#    A = int(data[i][1][2])
#    mass = data[i][1][3]
#    midx = len(mass)

#    #format string to float
#    for idx in range(len(mass)):
#        if mass[idx] =="(":
#            midx=idx
#            break
#    mass = float(mass[:midx])
#
#    #put mass in mass matrix
#    masses[A-1,Z-1] = mass
#
##average nucleon mass
#M_N = {#"46Ca" : masses[45, 19] / 46*u_to_GeV, 
#       #"48Ca" : masses[47, 19] / 48*u_to_GeV, 
#       #"70Zn" : masses[69, 29] / 70*u_to_GeV, 
#       "76Ge" : masses[75, 31] / 76*u_to_GeV, 
#       #"80Se" : masses[79, 33] / 80*u_to_GeV, 
#       "82Se" : masses[81, 33] / 82*u_to_GeV,
#       #"86Kr" : masses[85, 35] / 86*u_to_GeV,
#       #"94Zr" : masses[93, 39] / 94*u_to_GeV,
#       "96Zr" : masses[95, 39] / 96*u_to_GeV,
#       #"98Mo" : masses[97, 41] / 98*u_to_GeV,
#       "100Mo" : masses[99, 41] / 100*u_to_GeV, 
#       #"104Ru" : masses[103, 43] / 104*u_to_GeV, 
#       "110Pd" : masses[109, 45] / 110*u_to_GeV,
#       #"114Cd" : masses[113, 47] / 114*u_to_GeV,
#       "116Cd" : masses[115, 47] / 116*u_to_GeV,
#       #"122Sn" : masses[121, 49] / 122*u_to_GeV,
#       "124Sn" : masses[123, 49] / 124*u_to_GeV,
#       "128Te" : masses[127, 51] / 128*u_to_GeV,
#       "130Te" : masses[129, 51] / 130*u_to_GeV,  
#       "136Xe" : masses[135, 53] / 136*u_to_GeV}


#list of element classes
elements = {"238U" : U238,
            "232Th" : Th232,
             #"204Hg" : Hg204,
            "198Pt" : Pt198,
             #"192Os" : Os192,
             #"186W" : W186,
             #"176Yb" : Yb176,
             #"170Er" : Er170,
            "160Gd" : Gd160,
            "154Sm" : Sm154,
            "150Nd" : Nd150,
            "148Nd" : Nd148,
             #"146Nd" : Nd146,
             #"142Ce" : Ce142,
            "136Xe" : Xe136,
            "134Xe" : Xe134,
            "130Te" : Te130,
            "128Te" : Te128,
            "124Sn" : Sn124,
             #"122Sn" : Sn122,
            "116Cd" : Cd116,
             #"114Cd" : Cd114,
            "110Pd" : Pd110,
             #"104Ru" : Ru104,
            "100Mo" : Mo100,
             #"98Mo" : Mo98,
            "96Zr" : Zr96,
             #"94Zr" : Zr94,
             #"86Kr" : Kr86,
            "82Se" : Se82,
             #"80Se" : Se80,
            "76Ge" : Ge76,
             #"70Zn" : Zn70,
             "48Ca" : Ca48,
             #"46Ca" : Ca46
            }

#list of corresponding names
#I don't know why I didn't make a dict here...
element_names = list(elements.keys())

#GENERATE THE RUNNING MATRIX FOR LEFT

def alpha_s(mu, mu_0=91, alpha_0=0.12):
    m_t = 175
    m_b = 4.8
    m_c = 1.4
    m_s = 0#0.96
    m_u = 0#0.022
    m_d = 0#0.047

    masses = np.array([m_u, m_d, m_s, m_c, m_b, m_t])

    f = np.searchsorted(2*masses, mu)

    return (alpha_0/(1 + (33 - 2*f)*alpha_0/(6*np.pi)*np.log(mu/mu_0)))



#running couplings from m_W down to 2GeV
#define ODEs to solve i.e. define
# dC / dln(mu)

def RGEC6_S(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    C_F = (n_c**2 - 1) / (2*n_c)

    return (-6*C_F * alpha_s(mu)/(4*np.pi) * C)

def RGEC6_T(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    C_F = (n_c**2 - 1) / (2*n_c)

    return (2*C_F * alpha_s(mu)/(4*np.pi) * C)

def RGEC9_1(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    return 6*(1-1/n_c)*alpha_s(mu)/(4*np.pi) * C

def RGEC9_23(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    M = np.array([[8 + 2/n_c - 6*n_c, -4-8/n_c + 4*n_c], [4 - 8/n_c, 4 + 2/n_c + 2*n_c]])
    return alpha_s(mu)/(4*np.pi) *(np.dot(M, C))

def RGEC9_45(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    C_F = (n_c**2 - 1) / (2*n_c)
    M = np.array([[6/n_c, 0],[-6, -12*C_F]])
    return alpha_s(mu)/(4*np.pi) * np.dot(M, C)

def RGEC9_67_89(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    C_F = (n_c**2 - 1) / (2*n_c)
    M = np.array([[-2*C_F*(3*n_c - 4)/n_c, 2*C_F * (n_c + 2)*(n_c - 1) / n_c**2], 
                  [4*(n_c-2)/n_c, (4 - n_c + 2*n_c**2 + n_c**3)/n_c**2]])
    return alpha_s(mu)/(4*np.pi) *(np.dot(M, C))

#the primed operators as well as the LR run the same



def run_LEFT(WC, initial_scale=80, final_scale = 2):
######################################################################################
#
#Define RGEs as differential equations and solve them numerically for the given scales
#
######################################################################################

    #chi_scale = low_scale
    #print("Running operators from m_W to chiPT")

    #m_W = 80#GeV
    m_t = 175


    C6_SL_sol = integrate.solve_ivp(RGEC6_S, [np.log(initial_scale), np.log(final_scale)], [WC["SL(6)"]])
    C6_SR_sol = integrate.solve_ivp(RGEC6_S, [np.log(initial_scale), np.log(final_scale)], [WC["SR(6)"]])
    C6_T_sol = integrate.solve_ivp(RGEC6_T, [np.log(initial_scale), np.log(final_scale)], [WC["T(6)"]])

    WC["SL(6)"] = C6_SL_sol.y[0][-1]
    WC["SR(6)"] = C6_SR_sol.y[0][-1]
    WC["T(6)"] = C6_T_sol.y[0][-1]

    C9_1_sol_L = integrate.solve_ivp(RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [WC["1L(9)"]])
    C9_1_prime_sol_L = integrate.solve_ivp(RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [WC["1L(9)prime"]])

    WC["1L(9)"] = C9_1_sol_L.y[0][-1]
    WC["1L(9)prime"] = C9_1_prime_sol_L.y[0][-1]

    C9_1_sol_R = integrate.solve_ivp(RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [WC["1R(9)"]])
    C9_1_prime_sol_R = integrate.solve_ivp(RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [WC["1R(9)prime"]])

    WC["1R(9)"] = C9_1_sol_R.y[0][-1]
    WC["1R(9)prime"] = C9_1_prime_sol_R.y[0][-1]

    C9_23_sol_L = integrate.solve_ivp(RGEC9_23, [np.log(initial_scale), np.log(final_scale)], [WC["2L(9)"], WC["3L(9)"]])
    C9_23_prime_sol_L = integrate.solve_ivp(RGEC9_23, [np.log(initial_scale), np.log(final_scale)], [WC["2L(9)prime"], WC["3L(9)prime"]])

    WC["2L(9)"] = C9_23_sol_L.y[0][-1]
    WC["2L(9)prime"] = C9_23_prime_sol_L.y[0][-1]
    WC["3L(9)"] = C9_23_sol_L.y[1][-1]
    WC["3L(9)prime"] = C9_23_prime_sol_L.y[1][-1]

    C9_23_sol_R = integrate.solve_ivp(RGEC9_23, [np.log(initial_scale), np.log(final_scale)], [WC["2R(9)"], WC["3R(9)"]])
    C9_23_prime_sol_R = integrate.solve_ivp(RGEC9_23, [np.log(initial_scale), np.log(final_scale)], [WC["2R(9)prime"], WC["3R(9)prime"]])

    WC["2R(9)"] = C9_23_sol_R.y[0][-1]
    WC["2R(9)prime"] = C9_23_prime_sol_R.y[0][-1]
    WC["3R(9)"] = C9_23_sol_R.y[1][-1]
    WC["3R(9)prime"] = C9_23_prime_sol_R.y[1][-1]

    C9_45_sol_L = integrate.solve_ivp(RGEC9_45, [np.log(initial_scale), np.log(final_scale)], [WC["4L(9)"], WC["5L(9)"]])

    WC["4L(9)"] = C9_45_sol_L.y[0][-1]
    WC["5L(9)"] = C9_45_sol_L.y[1][-1]

    C9_45_sol_R = integrate.solve_ivp(RGEC9_45, [np.log(initial_scale), np.log(final_scale)], [WC["4R(9)"], WC["5R(9)"]])

    WC["4R(9)"] = C9_45_sol_R.y[0][-1]
    WC["5R(9)"] = C9_45_sol_R.y[1][-1]

    C9_67_sol = integrate.solve_ivp(RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], [WC["6(9)"], WC["7(9)"]])
    C9_67_prime_sol = integrate.solve_ivp(RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], [WC["6(9)prime"], WC["7(9)prime"]])

    WC["6(9)"] = C9_67_sol.y[0][-1]
    WC["6(9)prime"] = C9_67_prime_sol.y[0][-1]
    WC["7(9)"] = C9_67_sol.y[1][-1]
    WC["7(9)prime"] = C9_67_prime_sol.y[1][-1]

    C9_89_sol = integrate.solve_ivp(RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], [WC["8(9)"], WC["9(9)"]])
    C9_89_prime_sol = integrate.solve_ivp(RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], [WC["8(9)prime"], WC["9(9)prime"]])

    WC["8(9)"] = C9_89_sol.y[0][-1]
    WC["8(9)prime"] = C9_89_prime_sol.y[0][-1]
    WC["9(9)"] = C9_89_sol.y[1][-1]
    WC["9(9)prime"] = C9_89_prime_sol.y[1][-1]

    return WC





matrix = np.zeros((len(LEFT_WCs), len(LEFT_WCs)))
idx = 0
for operator in LEFT_WCs:
    LEFT_WCs[operator] = 1
    a = np.array(list(run_LEFT(LEFT_WCs).values()))
    b = np.array(list(LEFT_WCs.values()))
    matrix[idx] = a
    idx += 1
    for operators in LEFT_WCs:
        LEFT_WCs[operators] = 0
matrix = matrix.T


# In[11]:

#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#                                            LEFT                                              #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################


class LEFT(object):
    ################################################################
    # this class generates LEFT models with given Wilson coefficients
    # the WCs are entered at the scale of M_W=80GeV
    # it can calculate the low energy observables of 0nuBB decay
    ################################################################
    def __init__(self, WC, name = None, use_unknown_LECs = False, method = "SM", basis = "C"):
        
        self.method = method
        self.basis = basis
        
        # physical constands
        
        self.m_N = 0.93
        self.m_e = pc["electron mass energy equivalent in MeV"][0] * 1e-3
        self.m_e_MEV = pc["electron mass energy equivalent in MeV"][0]
        self.vev = 246
        
        #store wilson coefficients
        #there are two possible choices of a LEFT WC basis
        #the C-basis is the preferred one and all calculations are done within this basis
        #however, also the epsilon-basis can be chosen and is translated to the C-basis internally
        self.CWC = {#dim3
              "m_bb":0, 
              #dim6
              "SL(6)": 0, "SR(6)": 0, 
              "T(6)":0, 
              "VL(6)":0, "VR(6)":0, 
              #dim7
              "VL(7)":0, "VR(7)":0, 
              #dim9
              "1L(9)":0, "1R(9)":0, 
              "1L(9)prime":0, "1R(9)prime":0, 
              "2L(9)":0, "2R(9)":0, 
              "2L(9)prime":0, "2R(9)prime":0, 
              "3L(9)":0, "3R(9)":0, 
              "3L(9)prime":0, "3R(9)prime":0, 
              "4L(9)":0, "4R(9)":0, 
              "5L(9)":0, "5R(9)":0, 
              "6(9)":0,
              "6(9)prime":0,
              "7(9)":0,
              "7(9)prime":0,
              "8(9)":0,
              "8(9)prime":0,
              "9(9)":0,
              "9(9)prime":0}
        
        self.EpsilonWC = {#dim3
                          "m_bb":0, 
                          #dim6
                          "V+AV+A": 0, "V+AV-A": 0, 
                          "TRTR":0, 
                          "S+PS+P":0, "S+PS-P":0,
                          #dim7
                          "VL(7)":0, "VR(7)":0, #copied from C basis
                          #dim9
                          "1LLL":0, "1LLR":0,
                          "1RRL":0, "1RRR":0,
                          "1RLL":0, "1RLR":0,
                          "2LLL":0, "2LLR":0,
                          "2RRL":0, "2RRR":0,
                          "3LLL":0, "3LLR":0,
                          "3RRL":0, "3RRR":0,
                          "3RLL":0, "3RLR":0,
                          "4LLR":0, "4LRR":0,
                          "4RRR":0, "4RLR":0,
                          "5LLR":0, "5LRR":0,
                          "5RRR":0, "5RLR":0,
                          #redundant operators
                          "1LRL":0, "1LRR":0, 
                          "3LRL":0, "3LRR":0,
                          "4LLL":0, "4LRL":0,
                          "4RRL":0, "4RLL":0,
                          "TLTL":0,
                          "5LLL":0, "5LRL":0,
                          "5RRL":0, "5RLL":0,
                          #vanishing operators
                          "2LRL":0, "2LRR":0, 
                          "2RLL":0, "2RLR":0, 
                          "TRTL":0, "TLTR":0, 
                          #operators not contributing directly
                          "V-AV+A": 0, "V-AV-A": 0, 
                          "S-PS+P":0, "S-PS-P":0,
                         }
        #get the WCs right
        if basis == "C" or basis == "c":
            for operator in WC:
                self.CWC[operator] = WC[operator]
            self.EpsilonWC = self.change_basis(basis=self.basis, inplace = False)
        elif basis == "E" or basis == "e" or basis == "epsilon" or basis == "Epsilon":
            for operator in WC:
                self.EpsilonWC[operator] = WC[operator]
            self.CWC = self.change_basis(basis=self.basis, inplace = False)
        else:
            print("Basis",basis,'is not defined. Choose either "C" for the Grasser basis used in the master formula, or "epsilon" for the old standard basis by Päs et al. Setting the basis to C...')
        
        
        #WC Dict with WCs @ chiPT scale used for calculations
        self.WC = self.CWC.copy()
        
        '''
            Run WCs down to chiPT
        '''

        #self.WC = self.running_mW_to_chiPT(self.WC)
        self.WC = self.run(WC = self.WC, updown="down")#ning_mW_to_chiPT(self.WC)
        
        #store model name
        if name == None:
            self.name = "Model"
        else:
            self.name = name
        
        
        #Import PSFs
        self.PSFpanda = pd.read_csv("PSFs/PSFs.csv")
        self.PSFpanda.set_index("PSFs", inplace = True)
        
        #Import NMEs
        self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        
        #Store the Low Energy Constants (LECs) required
        self.use_unknown_LECs = use_unknown_LECs
        if use_unknown_LECs == True:
            self.LEC = {"A":1.271, "S":0.97, "M":4.7, "T":0.99, "B":2.7, "1pipi":0.36, 
                       "2pipi":2.0, "3pipi":-0.62, "4pipi":-1.9, "5pipi":-8, 
                       # all the below are expected to be order 1 in absolute magnitude
                       "Tprime":1, "Tpipi":1, "1piN":1, "6piN":1, "7piN":1, "8piN":1, "9piN":1, "VLpiN":1, "TpiN":1, 
                       "1NN":1, "6NN":1, "7NN":1, "VLNN":1, "TNN": 1, "VLE":1, "VLme":1, "VRE":1, "VRme":1, 
                       # all the below are expected to be order (4pi)**2 in absolute magnitude
                       "2NN":(4*np.pi)**2, "3NN":(4*np.pi)**2, "4NN":(4*np.pi)**2, "5NN":(4*np.pi)**2, 
                       # expected to be 1/F_pipi**2 pion decay constant
                       "nuNN": -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
                      }
            
            self.LEC["VpiN"] = self.LEC["6piN"] + self.LEC["8piN"]
            self.LEC["tildeVpiN"] = self.LEC["7piN"] + self.LEC["9piN"]
        
        else:
            self.LEC = {"A":1.271, "S":0.97, "M":4.7, "T":0.99, "B":2.7, "1pipi":0.36, 
                       "2pipi":2.0, "3pipi":-0.62, "4pipi":-1.9, "5pipi":-8, 
                       # all the below are expected to be order 1 in absolute magnitude
                       "Tprime":1, "Tpipi":0, "1piN":1, "6piN":0, "7piN":0, "8piN":0, "9piN":0, "VLpiN":0, "TpiN":0, 
                       "1NN":1, "6NN":1, "7NN":1, "VLNN":0, "TNN": 0, "VLE":0, "VLme":0, "VRE":0, "VRme":0, 
                       # all the below are expected to be order (4pi)**2 in absolute magnitude
                       "2NN":0, "3NN":0, "4NN":0, "5NN":0, 
                       # expected to be 1/F_pipi**2 pion decay constant
                       "nuNN": -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
                      }
            self.LEC["VpiN"] = 1#LEC["6piN"] + LEC["8piN"]
            self.LEC["tildeVpiN"] = 1#LEC["7piN"] + LEC["9piN"]
        
        
        
        #generate element classes to calculate PSF observables
        #these contain the electron wavefunctions etc.
        #elements without NMEs are commented out to speed up the code
        #note that the Q-values here are in MeV!!!
        #m_e = pc["electron mass energy equivalent in MeV"][0]

        #U238 =  PSFclasses.element(92, 238, 1.144154 + 2*m_e)
        #Th232 = PSFclasses.element(90, 232, 0.837879 + 2*m_e)
        #Hg204 = PSFclasses.element(80, 204, 0.419154 + 2*m_e)
        #Pt198 = PSFclasses.element(78, 198, 1.049142 + 2*m_e)
        #Os192 = PSFclasses.element(76, 192, 0.408274 + 2*m_e)
        #W186 =  PSFclasses.element(74, 186, 0.491643 + 2*m_e)
        #Yb176 = PSFclasses.element(70, 176, 1.088730 + 2*m_e)
        #Er170 = PSFclasses.element(68, 170, 0.655586 + 2*m_e)
        #Gd160 = PSFclasses.element(64, 160, 1.730530 + 2*m_e)
        #Sm154 = PSFclasses.element(62, 154, 1.250810 + 2*m_e)
        #Nd150 = PSFclasses.element(60, 150, 3.371357 + 2*m_e)
        #Nd148 = PSFclasses.element(60, 148, 1.928286 + 2*m_e)
        #Nd146 = PSFclasses.element(60, 146, 0.070421 + 2*m_e)
        #Ce142 = PSFclasses.element(58, 142, 1.417175 + 2*m_e)
        #Xe136 = PSFclasses.element(54, 136, 2.457984 + 2*m_e)
        #Xe134 = PSFclasses.element(54, 134, 0.825751 + 2*m_e)
        #Te130 = PSFclasses.element(52, 130, 2.527515 + 2*m_e)
        #Te128 = PSFclasses.element(52, 128, 0.866550 + 2*m_e)
        #Sn124 = PSFclasses.element(50, 124, 2.291010 + 2*m_e)
        #Sn122 = PSFclasses.element(50, 122, 0.372877 + 2*m_e)
        #Cd116 = PSFclasses.element(48, 116, 2.813438 + 2*m_e)
        #Cd114 = PSFclasses.element(48, 114, 0.542493 + 2*m_e)
        #Pd110 = PSFclasses.element(46, 110, 2.017234 + 2*m_e)
        #Ru104 = PSFclasses.element(44, 104, 1.301297 + 2*m_e)
        #Mo100 = PSFclasses.element(42, 100, 3.034342 + 2*m_e)
        #Mo98 =  PSFclasses.element(42, 98, 0.109935 + 2*m_e)
        #Zr96 =  PSFclasses.element(40, 96, 3.348982 + 2*m_e)
        #Zr94 =  PSFclasses.element(40, 94, 1.141919 + 2*m_e)
        #Kr86 =  PSFclasses.element(36, 86, 1.257542 + 2*m_e)
        #Se82 =  PSFclasses.element(34, 82, 2.996402 + 2*m_e)
        #Se80 =  PSFclasses.element(34, 80, 0.133874 + 2*m_e)
        #Ge76 =  PSFclasses.element(32, 76, 2.039061 + 2*m_e)
        #Zn70 =  PSFclasses.element(30, 70, 0.997118 + 2*m_e)
        #Ca48 =  PSFclasses.element(20, 48, 4.266970 + 2*m_e)
        #Ca46 =  PSFclasses.element(20, 46, 0.988576 + 2*m_e)

        #list of element classes
        self.elements = elements

        #list of corresponding names
        #I don't know why I didn't make a dict here...
        #self.element_names = list(self.elements.keys())
        self.element_names = np.flip(list(self.NMEs.keys()))
        
        
        

    def set_LECs(self, use_unknown_LECs):
        self.use_unknown_LECs = use_unknown_LECs
        if use_unknown_LECs == True:
            self.LEC = {"A":1.271, "S":0.97, "M":4.7, "T":0.99, "B":2.7, "1pipi":0.36, 
                       "2pipi":2.0, "3pipi":-0.62, "4pipi":-1.9, "5pipi":-8, 
                       # all the below are expected to be order 1 in absolute magnitude
                       "Tprime":1, "Tpipi":1, "1piN":1, "6piN":1, "7piN":1, "8piN":1, "9piN":1, "VLpiN":1, "TpiN":1, 
                       "1NN":1, "6NN":1, "7NN":1, "VLNN":1, "TNN": 1, "VLE":1, "VLme":1, "VRE":1, "VRme":1, 
                       # all the below are expected to be order (4pi)**2 in absolute magnitude
                       "2NN":(4*np.pi)**2, "3NN":(4*np.pi)**2, "4NN":(4*np.pi)**2, "5NN":(4*np.pi)**2, 
                       # expected to be 1/F_pipi**2 pion decay constant
                       "nuNN": -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
                      }
            
            self.LEC["VpiN"] = self.LEC["6piN"] + self.LEC["8piN"]
            self.LEC["tildeVpiN"] = self.LEC["7piN"] + self.LEC["9piN"]
            
        else:
            self.LEC = {"A":1.271, "S":0.97, "M":4.7, "T":0.99, "B":2.7, "1pipi":0.36, 
                       "2pipi":2.0, "3pipi":-0.62, "4pipi":-1.9, "5pipi":-8, 
                       # all the below are expected to be order 1 in absolute magnitude
                       "Tprime":1, "Tpipi":0, "1piN":1, "6piN":0, "7piN":0, "8piN":0, "9piN":0, "VLpiN":0, "TpiN":0, 
                       "1NN":1, "6NN":1, "7NN":1, "VLNN":0, "TNN": 0, "VLE":0, "VLme":0, "VRE":0, "VRme":0, 
                       # all the below are expected to be order (4pi)**2 in absolute magnitude
                       "2NN":0, "3NN":0, "4NN":0, "5NN":0, 
                       # expected to be 1/F_pipi**2 pion decay constant
                       "nuNN": -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
                      }
            self.LEC["VpiN"] = 1#LEC["6piN"] + LEC["8piN"]
            self.LEC["tildeVpiN"] = 1#LEC["7piN"] + LEC["9piN"]


    #define the running of the strong coupling constant
    #should be made continuous
    #right now it has steps at 2*m_quarks
    def alpha_s(self, mu, mu_0=91, alpha_0=0.12):
        m_t = 175
        m_b = 4.8
        m_c = 1.4
        m_s = 0#0.96
        m_u = 0#0.022
        m_d = 0#0.047

        masses = np.array([m_u, m_d, m_s, m_c, m_b, m_t])

        f = np.searchsorted(2*masses, mu)

        return (alpha_0/(1 + (33 - 2*f)*alpha_0/(6*np.pi)*np.log(mu/mu_0)))
    
    
    def change_basis(self, WC = None, basis = None, inplace = True):
    #this functions lets you switch between the C and the epsilon basis
    #if you only want to see the translation and dont want the change to be saved you can set inplace=False
        if basis == None:
            basis = self.basis
        #m_N = 0.93
        #vev = 246
        if basis in ["C" ,"c"]:
            if WC == None:
                WC = self.CWC
            else:
                for operator in self.CWC:
                    if operator not in WC:
                        WC[operator] = 0
            New_WCs = self.EpsilonWC.copy()
            for operator in New_WCs:
                New_WCs[operator] = 0
                
            New_WCs["m_bb"]   = WC["m_bb"]
                
            New_WCs["V+AV+A"] = 1/2 * WC["VR(6)"]
            New_WCs["V+AV-A"] = 1/2 * WC["VL(6)"]
            New_WCs["S+PS+P"] = 1/2 * WC["SR(6)"]
            New_WCs["S+PS-P"] = 1/2 * WC["SL(6)"]
            New_WCs["TRTR"]   = 1/2 * WC["T(6)"]
            
            New_WCs["1LLL"]   =  self.m_N/self.vev * (1/2*WC["2L(9)"] - 1/4*WC["3L(9)"])
            New_WCs["1LLR"]   =  self.m_N/self.vev * (1/2*WC["2R(9)"] - 1/4*WC["3R(9)"])
            New_WCs["1RRR"]   =  self.m_N/self.vev * (1/2*WC["2R(9)prime"] - 1/4*WC["3R(9)prime"])
            New_WCs["1RRL"]   =  self.m_N/self.vev * (1/2*WC["2L(9)prime"] - 1/4*WC["3L(9)prime"])
            New_WCs["1RLL"]   = -self.m_N/self.vev * WC["5L(9)"]
            New_WCs["1RLR"]   = -self.m_N/self.vev * WC["5R(9)"]
            
            New_WCs["2LLL"]   = -self.m_N/(16*self.vev) * WC["3L(9)"]
            New_WCs["2RRL"]   = -self.m_N/(16*self.vev) * WC["3L(9)prime"]
            New_WCs["2LLR"]   = -self.m_N/(16*self.vev) * WC["3R(9)"]
            New_WCs["2RRR"]   = -self.m_N/(16*self.vev) * WC["3R(9)prime"]
            
            New_WCs["3LLL"]   =  self.m_N/(2*self.vev) * WC["1L(9)"]
            New_WCs["3LLR"]   =  self.m_N/(2*self.vev) * WC["1R(9)"]
            New_WCs["3RRL"]   =  self.m_N/(2*self.vev) * WC["1R(9)prime"]
            New_WCs["3RRR"]   =  self.m_N/(2*self.vev) * WC["1R(9)prime"]
            
            New_WCs["4LLR"]   = 1j*self.m_N/self.vev * WC["9(9)"]
            New_WCs["4RRR"]   = 1j*self.m_N/self.vev * WC["9(9)prime"]
            New_WCs["4LRR"]   = -1j*self.m_N/self.vev * WC["7(9)"]
            New_WCs["4RLR"]   = -1j*self.m_N/self.vev * WC["7(9)prime"]
            
            New_WCs["5LRR"]   =  self.m_N/self.vev * (WC["6(9)"] - 5/3*WC["7(9)"])
            New_WCs["5RLR"]   =  self.m_N/self.vev * (WC["6(9)prime"] - 5/3*WC["7(9)prime"])
            New_WCs["5LLR"]   =  self.m_N/self.vev * (WC["8(9)"] - 5/3*WC["9(9)"])
            New_WCs["5RRR"]   =  self.m_N/self.vev * (WC["8(9)prime"] - 5/3*WC["9(9)prime"])
            
            New_WCs["VL(7)"]  = WC["VL(7)"]
            New_WCs["VR(7)"]  = WC["VR(7)"]
        
        elif basis in ["Epsilon", "epsilon", "E", "e"]:
            if WC == None:
                WC = self.EpsilonWC
            else:
                for operator in self.EpsilonWC:
                    if operator not in WC:
                        WC[operator] = 0
            New_WCs = self.CWC.copy()
            for operator in New_WCs:
                New_WCs[operator] = 0
                
            #long-range matching
            New_WCs["m_bb"]  = WC["m_bb"]
            New_WCs["SL(6)"] = 2*WC["S+PS-P"]
            New_WCs["SR(6)"] = 2*WC["S+PS+P"]
            New_WCs["VL(6)"] = 2*WC["V+AV-A"]
            New_WCs["VR(6)"] = 2*WC["V+AV+A"]
            New_WCs["T(6)"]  = 2*WC["TRTR"]            
            New_WCs["VL(7)"]  = WC["VL(7)"]
            New_WCs["VR(7)"]  = WC["VR(7)"]
            
            #short-range matching
            New_WCs["1L(9)"]      = 2*self.vev/self.m_N * WC["3LLL"]
            New_WCs["1R(9)"]      = 2*self.vev/self.m_N * WC["3LLR"]
            New_WCs["1L(9)prime"] = 2*self.vev/self.m_N * WC["3RRL"]
            New_WCs["1R(9)prime"] = 2*self.vev/self.m_N * WC["3RRR"]
            
            New_WCs["2L(9)"]      = 2*self.vev/self.m_N * (WC["1LLL"] - 4*WC["2LLL"])
            New_WCs["2R(9)"]      = 2*self.vev/self.m_N * (WC["1LLR"] - 4*WC["2LLR"])
            New_WCs["2L(9)prime"] = 2*self.vev/self.m_N * (WC["1RRL"] - 4*WC["2RRL"])
            New_WCs["2R(9)prime"] = 2*self.vev/self.m_N * (WC["1RRR"] - 4*WC["2RRR"])
            
            New_WCs["3L(9)"]      = -16*self.vev/self.m_N * WC["2LLL"]
            New_WCs["3R(9)"]      = -16*self.vev/self.m_N * WC["2LLR"]
            New_WCs["3L(9)prime"] = -16*self.vev/self.m_N * WC["2RRL"]
            New_WCs["3R(9)prime"] = -16*self.vev/self.m_N * WC["2RRR"]
            
            New_WCs["4L(9)"]      = 2*self.vev/self.m_N * (WC["3RLL"] + WC["3LRL"]) #including redundancies
            New_WCs["4R(9)"]      = 2*self.vev/self.m_N * (WC["3RLR"] + WC["3LRR"]) #including redundancies
            
            New_WCs["5L(9)"]      = -self.vev/self.m_N * (WC["1RLL"] + WC["1LRL"]) #including redundancies
            New_WCs["5R(9)"]      = -self.vev/self.m_N * (WC["1RLR"] + WC["1LRR"]) #including redundancies
            
            New_WCs["6(9)"]       = self.vev/self.m_N *(   WC["5LRR"] + 1j*5/3 * WC["4LRR"] 
                                              -( WC["5LRL"] + 1j*5/3 * WC["4LRL"])) #including redundancies
            New_WCs["6(9)prime"]  = self.vev/self.m_N *(   WC["5RLR"] + 1j*5/3 * WC["4RLR"]
                                              -( WC["5RLL"] + 1j*5/3 * WC["4RLL"])) #including redundancies
            
            New_WCs["7(9)"]       = 1j*self.vev/self.m_N * (WC["4LRR"] - WC["4LRL"]) #including redundancies
            New_WCs["7(9)prime"]  = 1j*self.vev/self.m_N * (WC["4RLR"] - WC["4RLL"]) #including redundancies
            
            New_WCs["8(9)"]       = self.vev/self.m_N *(   WC["5LLR"] - 1j*5/3 * WC["4LLR"]
                                              -( WC["5LLL"] - 1j*5/3 * WC["4LLL"])) #including redundancies
            New_WCs["8(9)prime"]  = self.vev/self.m_N *(   WC["5RRR"] - 1j*5/3 * WC["4RRR"]
                                              -( WC["5RRL"] - 1j*5/3 * WC["4RRL"])) #including redundancies
            
            New_WCs["9(9)"]       = -1j*self.vev/self.m_N * (WC["4LLR"] - WC["4LLL"]) #including redundancies
            New_WCs["9(9)prime"]  = -1j*self.vev/self.m_N * (WC["4RRR"] - WC["4RRL"]) #including redundancies
            
            
        else:
            print("Unknown basis",basis)
            return
        
        if inplace:
            if basis in ["C" ,"c"]:
                self.basis     = "e"
                self.EpsilonWC = New_WCs
                self.CWC       = WC
                self.WC = self.CWC.copy()
                #self.WC = self.run()
            else:
                self.basis     = "C"
                self.CWC       = New_WCs
                self.EpsilonWC = WC
                self.WC = self.CWC.copy()
                #self.WC = self.run()
                
        return(New_WCs)
                
    
    
    '''
        ################################################################################################
        
        Define RGEs from 1806.02780 and define a "running" 
        function to run WCs from m_W down to chiPT scale.
        
        Note that "scale" in the RGEs refers to log(mu)
        i.e. "scale" = log(mu), while scale in the final 
        functions refers to the actual energy scale i.e. 
        "scale" = mu
        
        ################################################################################################
    '''

    #running couplings from m_W down to 2GeV
    #define ODEs to solve i.e. define
    # dC / dln(mu)
    
    def RGEC6_S(self, ln_mu, C):
        mu = np.exp(ln_mu)
        n_c = 3
        C_F = (n_c**2 - 1) / (2*n_c)

        return (-6*C_F * self.alpha_s(mu)/(4*np.pi) * C)

    def RGEC6_T(self, ln_mu, C):
        mu = np.exp(ln_mu)
        n_c = 3
        C_F = (n_c**2 - 1) / (2*n_c)

        return (2*C_F * self.alpha_s(mu)/(4*np.pi) * C)

    def RGEC9_1(self, ln_mu, C):
        mu = np.exp(ln_mu)
        n_c = 3
        return 6*(1-1/n_c)*self.alpha_s(mu)/(4*np.pi) * C

    def RGEC9_23(self, ln_mu, C):
        mu = np.exp(ln_mu)
        n_c = 3
        M = np.array([[8 + 2/n_c - 6*n_c, -4-8/n_c + 4*n_c], [4 - 8/n_c, 4 + 2/n_c + 2*n_c]])
        return self.alpha_s(mu)/(4*np.pi) *(np.dot(M, C))

    def RGEC9_45(self, ln_mu, C):
        mu = np.exp(ln_mu)
        n_c = 3
        C_F = (n_c**2 - 1) / (2*n_c)
        M = np.array([[6/n_c, 0],[-6, -12*C_F]])
        return self.alpha_s(mu)/(4*np.pi) * np.dot(M, C)

    def RGEC9_67_89(self, ln_mu, C):
        mu = np.exp(ln_mu)
        n_c = 3
        C_F = (n_c**2 - 1) / (2*n_c)
        M = np.array([[-2*C_F*(3*n_c - 4)/n_c, 2*C_F * (n_c + 2)*(n_c - 1) / n_c**2], 
                      [4*(n_c-2)/n_c, (4 - n_c + 2*n_c**2 + n_c**3)/n_c**2]])
        return self.alpha_s(mu)/(4*np.pi) *(np.dot(M, C))

    #the primed operators as well as the LR run the same
    def run(self, WC = None, updown = "down"):
        if WC == None:
            WC = self.WC.copy()
        else:
            WCcopy = WC.copy()
            WC = self.WC.copy()
            
            for operator in WCcopy:
                WC[operator] = WCcopy[operator]
        
        if updown == "down":
            new_WC_values = matrix@np.array(list(WC.values()))
        else:
            new_WC_values = np.linalg.inv(matrix)@np.array(list(WC.values()))
        new_WC = {}
        for idx in range(len(WC)):
            operator = list(WC.keys())[idx]
            new_WC[operator] = new_WC_values[idx]
            
        return(new_WC)
        
        
    def _run(self, WC = None, initial_scale=80, final_scale = 2):
    ######################################################################################
    #
    #Define RGEs as differential equations and solve them numerically for the given scales
    #
    ######################################################################################
        if WC == None:
            pass
        else:
            self.WC = WC
        #chi_scale = low_scale
        #print("Running operators from m_W to chiPT")

        #m_W = 80#GeV
        m_t = 175
        
        
        C6_SL_sol = integrate.solve_ivp(self.RGEC6_S, [np.log(initial_scale), np.log(final_scale)], [self.WC["SL(6)"]])
        C6_SR_sol = integrate.solve_ivp(self.RGEC6_S, [np.log(initial_scale), np.log(final_scale)], [self.WC["SR(6)"]])
        C6_T_sol = integrate.solve_ivp(self.RGEC6_T, [np.log(initial_scale), np.log(final_scale)], [self.WC["T(6)"]])
        
        self.WC["SL(6)"] = C6_SL_sol.y[0][-1]
        self.WC["SR(6)"] = C6_SR_sol.y[0][-1]
        self.WC["T(6)"] = C6_T_sol.y[0][-1]

        C9_1_sol_L = integrate.solve_ivp(self.RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [self.WC["1L(9)"]])
        C9_1_prime_sol_L = integrate.solve_ivp(self.RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [self.WC["1L(9)prime"]])

        self.WC["1L(9)"] = C9_1_sol_L.y[0][-1]
        self.WC["1L(9)prime"] = C9_1_prime_sol_L.y[0][-1]

        C9_1_sol_R = integrate.solve_ivp(self.RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [self.WC["1R(9)"]])
        C9_1_prime_sol_R = integrate.solve_ivp(self.RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [self.WC["1R(9)prime"]])

        self.WC["1R(9)"] = C9_1_sol_R.y[0][-1]
        self.WC["1R(9)prime"] = C9_1_prime_sol_R.y[0][-1]

        C9_23_sol_L = integrate.solve_ivp(self.RGEC9_23, [np.log(initial_scale), np.log(final_scale)], [self.WC["2L(9)"], self.WC["3L(9)"]])
        C9_23_prime_sol_L = integrate.solve_ivp(self.RGEC9_23, [np.log(initial_scale), np.log(final_scale)], [self.WC["2L(9)prime"], self.WC["3L(9)prime"]])

        self.WC["2L(9)"] = C9_23_sol_L.y[0][-1]
        self.WC["2L(9)prime"] = C9_23_prime_sol_L.y[0][-1]
        self.WC["3L(9)"] = C9_23_sol_L.y[1][-1]
        self.WC["3L(9)prime"] = C9_23_prime_sol_L.y[1][-1]

        C9_23_sol_R = integrate.solve_ivp(self.RGEC9_23, [np.log(initial_scale), np.log(final_scale)], [self.WC["2R(9)"], self.WC["3R(9)"]])
        C9_23_prime_sol_R = integrate.solve_ivp(self.RGEC9_23, [np.log(initial_scale), np.log(final_scale)], [self.WC["2R(9)prime"], self.WC["3R(9)prime"]])

        self.WC["2R(9)"] = C9_23_sol_R.y[0][-1]
        self.WC["2R(9)prime"] = C9_23_prime_sol_R.y[0][-1]
        self.WC["3R(9)"] = C9_23_sol_R.y[1][-1]
        self.WC["3R(9)prime"] = C9_23_prime_sol_R.y[1][-1]

        C9_45_sol_L = integrate.solve_ivp(self.RGEC9_45, [np.log(initial_scale), np.log(final_scale)], [self.WC["4L(9)"], self.WC["5L(9)"]])

        self.WC["4L(9)"] = C9_45_sol_L.y[0][-1]
        self.WC["5L(9)"] = C9_45_sol_L.y[1][-1]

        C9_45_sol_R = integrate.solve_ivp(self.RGEC9_45, [np.log(initial_scale), np.log(final_scale)], [self.WC["4R(9)"], self.WC["5R(9)"]])

        self.WC["4R(9)"] = C9_45_sol_R.y[0][-1]
        self.WC["5R(9)"] = C9_45_sol_R.y[1][-1]

        C9_67_sol = integrate.solve_ivp(self.RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], [self.WC["6(9)"], self.WC["7(9)"]])
        C9_67_prime_sol = integrate.solve_ivp(self.RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], [self.WC["6(9)prime"], self.WC["7(9)prime"]])

        self.WC["6(9)"] = C9_67_sol.y[0][-1]
        self.WC["6(9)prime"] = C9_67_prime_sol.y[0][-1]
        self.WC["7(9)"] = C9_67_sol.y[1][-1]
        self.WC["7(9)prime"] = C9_67_prime_sol.y[1][-1]

        C9_89_sol = integrate.solve_ivp(self.RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], [self.WC["8(9)"], self.WC["9(9)"]])
        C9_89_prime_sol = integrate.solve_ivp(self.RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], [self.WC["8(9)prime"], self.WC["9(9)prime"]])

        self.WC["8(9)"] = C9_89_sol.y[0][-1]
        self.WC["8(9)prime"] = C9_89_prime_sol.y[0][-1]
        self.WC["9(9)"] = C9_89_sol.y[1][-1]
        self.WC["9(9)prime"] = C9_89_prime_sol.y[1][-1]

        return self.WC
    
    
    '''
        ####################################################################################################
        
        Define necessary functions to calculate observables
        1. half_lives
        2. hl_ratios
        3. PSF observables
        
        ####################################################################################################
    '''
    
    
    ####################################################################################################
    #                                                                                                  #
    #                                   Half-live calculations                                         #
    #                                                                                                  #
    ####################################################################################################
        
    def t_half(self, element_name, WC = None, method = None):
        #set the method and import NMEs if necessary
        if method == None:
            method = self.method
            pass
        elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            print("Changing method to",method)
            self.method = method
            self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        method = self.method


        if WC == None:
            WC = self.WC.copy()
        #scale all operators to order something around order 1
        
        #for operator in WC:
        #    WC[operator] *= 1e+6
        #Calculates the half-live for a given element and WCs
        amp, M = self.amplitudes(element_name, WC)
        element = self.elements[element_name]
        
        g_A=self.LEC["A"]
        
        #G = PSFs[element]
        G = self.to_G(element_name)
        #m_e = pc["electron mass energy equivalent in MeV"][0]

        #Some PSFs need a rescaling due to different definitions in DOIs paper and 1806...
        g_06_rescaling = self.m_e_MEV*element.R/2
        g_09_rescaling = g_06_rescaling**2
        g_04_rescaling = 9/2
        G["06"] *= g_06_rescaling
        G["04"] *= g_04_rescaling
        G["09"] *= g_09_rescaling

        #Calculate half-life following eq 38. in 1806.02780
        inverse_result = g_A**4*(G["01"] * (np.absolute(amp["nu"])**2 + np.absolute(amp["R"])**2)
                          - 2 * (G["01"] - G["04"])*(np.conj(amp["nu"])*amp["R"]).real
                          + 4 *  G["02"]* np.absolute(amp["E"])**2
                          + 2 *  G["04"]*(np.absolute(amp["me"])**2 + (np.conj(amp["me"])*(amp["nu"]+amp["R"])).real)
                          - 2 *  G["03"]*((amp["nu"]+amp["R"])*np.conj(amp["E"]) + 2*amp["me"]*np.conj(amp["E"])).real
                          + G["09"] * np.absolute(amp["M"])**2
                          + G["06"] * ((amp["nu"]-amp["R"])*np.conj(amp["M"])).real)
        #for operator in WC:
        #    WC[operator] *= 1e-6
        #return(1e+12*1/inverse_result)
        return(1/inverse_result)
    
    def amplitudes(self, element_name, WC, method=None):
    #calculate transition amplitudes as given in 1806.02780
        #set the method and import NMEs if necessary
        if method == None:
            pass
        elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            print("Changing method to",method)
            self.method = method
            self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        method = self.method
        #method = self.method
        C = self.WC.copy()
        for x in C:
            C[x] = 0
        for x in WC:
            C[x] = WC[x]
        LEC = self.LEC.copy()

        #Constants: all in GeV
        #vev = 246
        V_ud = 0.97417
        #m_e = 5.10998e-4
        g_A=LEC["A"]
        g_V=1
        m_pi = 0.13957
        #m_N = 0.93



        #generate constants that enter the ME calculation

        #right below eq 21 on p.14
        C["V(9)"]      = C["6(9)"] + C["8(9)"] + C["6(9)prime"] + C["8(9)prime"]
        C["tildeV(9)"] = C["7(9)"] + C["9(9)"] + C["7(9)prime"] + C["9(9)prime"]

        #eq.25
        C["pipiL(9)"]  = (  LEC["2pipi"]*(C["2L(9)"] + C["2L(9)prime"]) 
                          + LEC["3pipi"]*(C["3L(9)"] + C["3L(9)prime"])
                          - LEC["4pipi"]*C["4L(9)"]
                          - LEC["5pipi"]*C["5L(9)"]
                          - 5/3*(m_pi**2) * LEC["1pipi"]*(  C["1L(9)"] 
                                                       + C["1L(9)prime"]))

        C["pipiR(9)"]  = (  LEC["2pipi"]*(C["2R(9)"] + C["2R(9)prime"]) 
                          + LEC["3pipi"]*(C["3R(9)"] + C["3R(9)prime"])
                          - LEC["4pipi"]*C["4R(9)"]
                          - LEC["5pipi"]*C["5R(9)"]
                          - 5/3*(m_pi**2) * LEC["1pipi"]*(C["1R(9)"] + C["1R(9)prime"]))

        C["piNL(9)"]   = (LEC["1piN"] - 5/6*LEC["1pipi"])*(C["1L(9)"] + C["1L(9)prime"])
        C["piNR(9)"]   = (LEC["1piN"] - 5/6*LEC["1pipi"])*(C["1R(9)"] + C["1R(9)prime"])

        C["NNL(9)"]    = (  LEC["1NN"]*(C["1L(9)"] + C["1L(9)prime"]) 
                          + LEC["2NN"]*(C["2L(9)"] + C["2L(9)prime"])
                          + LEC["3NN"]*(C["3L(9)"] + C["3L(9)prime"])
                          + LEC["4NN"]*C["4L(9)"] 
                          + LEC["5NN"]*C["5L(9)"])
        C["NNR(9)"]    = (  LEC["1NN"]*(C["1R(9)"] + C["1R(9)prime"]) 
                          + LEC["2NN"]*(C["2R(9)"] + C["2R(9)prime"])
                          + LEC["3NN"]*(C["3R(9)"] + C["3R(9)prime"])
                          + LEC["4NN"]*C["4R(9)"] 
                          + LEC["5NN"]*C["5R(9)"])
        #Matrix Elements




        #generate NMEs that directly enter ME calculations
        NME = self.NMEs[element_name][method].copy()


        #eq. 33
        NME["T"] = NME["TAP"] + NME["TPP"] + NME["TMM"]
        NME["GT"] = NME["GTAA"] + NME["GTAP"] + NME["GTPP"] + NME["GTMM"]
        NME["PS"] = 0.5*NME["GTAP"] + NME["GTPP"] + 0.5*NME["TAP"] + NME["TPP"]
        NME["T6"] = (2 *(LEC["Tprime"] - LEC["TNN"])/LEC["A"]**2 * m_pi**2/self.m_N**2*NME["F,sd"] 
                     - 8*LEC["T"]/LEC["M"] * (NME["GTMM"] + NME["TMM"])
                     + LEC["TpiN"] * m_pi**2/(4*self.m_N**2)*(NME["GTAP,sd"] + NME["TAP,sd"])
                     + LEC["Tpipi"] * m_pi**2/(4*self.m_N**2)*(NME["GTPP,sd"] + NME["TPP,sd"]))


        #store MEs in dictionary to return
        M= {}

        #Matrix Elements

        #eq. 30
        M["nu(3)"] = -V_ud**2*(-                       1/g_A**2 * NME["F"] 
                               +                                  NME["GT"] 
                               +                                  NME["T"] 
                               + 2*m_pi**2 * LEC["nuNN"]/g_A**2 * NME["F,sd"])


        #eq. 31
        M["nu(6)"] = (  V_ud * (         LEC["B"]/self.m_N * (C["SL(6)"] - C["SR(6)"]) 
                                + m_pi**2/(self.m_N * self.vev) * (C["VL(7)"] - C["VR(7)"]))*NME["PS"] 
                      + V_ud * C["T(6)"] * NME["T6"])

        #eq. 32
        M["nu(9)"] = (      (-1/(2*self.m_N**2) * C["pipiL(9)"]) * (  1/2 * NME["GTAP,sd"] 
                                                               +       NME["GTPP,sd"] 
                                                               + 1/2 * NME["TAP,sd"] 
                                                               +       NME["TPP,sd"]) 
                      + m_pi**2/(2*self.m_N**2) * C["piNL(9)"]   * (        NME["GTAP,sd"] 
                                                               +       NME["TAP,sd"])
                      - 2/g_A**2 * m_pi**2/self.m_N**2 * C["NNL(9)"]      * NME["F,sd"])

        #equal to eq. 32 but L --> R see eq.34
        M["R(9)"] = (       (-1/(2*self.m_N**2) * C["pipiR(9)"]) * (  1/2 * NME["GTAP,sd"] 
                                                    +                  NME["GTPP,sd"] 
                                                    +            1/2 * NME["TAP,sd"] 
                                                    +                  NME["TPP,sd"]) 
                      + m_pi**2/(2*self.m_N**2) * C["piNR(9)"]   * (        NME["GTAP,sd"] 
                                                               +       NME["TAP,sd"])
                      - 2/g_A**2 * m_pi**2/self.m_N**2 * C["NNR(9)"]      * NME["F,sd"])

        #eq. 35
        M["EL(6)"] = -V_ud * C["VL(6)"]/3 * (  g_V**2/g_A**2 *       NME["F"] 
                                             +           1/3 * ( 2 * NME["GTAA"] 
                                                                +    NME["TAA"]) 
                                             + 6*LEC["VLE"]/g_A**2 * NME["F,sd"])

        M["ER(6)"] = -V_ud * C["VR(6)"]/3 * (  g_V**2/g_A**2 *       NME["F"] 
                                             - 1/3           * ( 2 * NME["GTAA"] 
                                                                +    NME["TAA"]) 
                                             + 6*LEC["VRE"]/g_A**2 * NME["F,sd"])

        M["meL(6)"] = V_ud*C["VL(6)"]/6 * (  g_V**2/g_A**2   *       NME["F"] 
                                           -           1/3   * (     NME["GTAA"] 
                                                                - 4* NME["TAA"]) 
                                           -             3   * (     NME["GTAP"] 
                                                                +    NME["GTPP"] 
                                                                +    NME["TAP"] 
                                                                +    NME["TPP"])
                                           - 12*LEC["VLme"]/g_A**2 * NME["F,sd"])

        M["meR(6)"] = V_ud*C["VR(6)"]/6 * (  g_V**2/g_A**2 *        NME["F"] 
                                           +           1/3 * (      NME["GTAA"] 
                                                              - 4 * NME["TAA"]) 
                                           +             3 * (      NME["GTAP"] 
                                                              +     NME["GTPP"] 
                                                              +     NME["TAP"] 
                                                              +     NME["TPP"])
                                           - 12*LEC["VRme"]/g_A**2 *NME["F,sd"])

        #eq. 36
        M["M(6)"] = V_ud*C["VL(6)"] * (  2*g_A/LEC["M"] * (                           NME["GTMM"] 
                                                           +                          NME["TMM"]) 
                                       + m_pi**2/self.m_N**2 * (- 2/g_A**2*LEC["VLNN"] *   NME["F,sd"]
                                                           + 1/2*LEC["VLpiN"]     *(  NME["GTAP,sd"] 
                                                                                    + NME["TAP,sd"])))

        M["M(9)"] = m_pi**2/self.m_N**2 * (-2/g_A**2*(LEC["6NN"]*C["V(9)"] + LEC["7NN"]*C["tildeV(9)"])    *    NME["F,sd"] 
                                      + 1/2*(LEC["VpiN"]*C["V(9)"] + LEC["tildeVpiN"]*C["tildeV(9)"]) * (  NME["GTAP,sd"] 
                                                                                                         + NME["TAP,sd"]))





        #generate subamplitudes
        #eq. 29
        A={}
        A["nu"] = (C["m_bb"]/self.m_e * M["nu(3)"] 
                   + self.m_N/self.m_e * M["nu(6)"] 
                   + self.m_N**2/(self.m_e*self.vev) * M["nu(9)"])
        
        A["R"] = self.m_N**2/(self.m_e*self.vev) * M["R(9)"]
        
        A["E"] = M["EL(6)"] + M["ER(6)"]
        
        A["me"] = M["meL(6)"] + M["meR(6)"]
        
        A["M"] = self.m_N/self.m_e * M["M(6)"] + self.m_N**2/(self.m_e*self.vev) * M["M(9)"]


        #return subamplitudes and MEs
        return (A, M)

    def to_G(self, element_name):
        #transform imported PSFs into the necessary dict
        #format, which is used in the amplitudes function.

        #bring element_name of the type 76Ge into the PSFpanda type ^{76}Ge
        try:
            PSFs = self.PSFpanda[element_name]
        except:
            if element_name[-1] == "U" or element_name[-1] == "W":
                element_name = r"$^{"+element_name[:-1]+ "}$" + element_name[-1:]
            else:
                element_name = r"$^{"+element_name[:-2]+ "}$" + element_name[-2:]
        #G = {element_name : {}}
        G = {}
        for key in self.PSFpanda[element_name].keys():
            G[key[-4:-2]] = self.PSFpanda[element_name][key]
        return(G)
    
    ####################################################################################################
    #                                                                                                  #
    #                                 PSF observables calculations                                     #
    #                                                                                                  #
    ####################################################################################################
    
    def spectra(self, Ebar, amp=None, element_name = "76Ge", method=None):
        
        #set the method and import NMEs if necessary
        if method == None:
            pass
        elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            print("Changing method to",method)
            self.method = method
            self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        method = self.method
        #calculates the single electron spectrum
        
        #get element class
        element = self.elements[element_name]
        
        #if not initialized calculate amplitudes
        if amp == None:
            amp = self.amplitudes(element_name, self.WC)[0]
        
        #electron mass in MeV
        #m_e = pc["electron mass energy equivalent in MeV"][0]
        
        #Mass difference between mother and daughter nuclei in MeV
        Delta_M = element.Delta_M
        
        #Energy from normalized Energy scale
        E = Ebar * (Delta_M - 2*self.m_e_MEV) + self.m_e_MEV
        
        #rescale PSFs due to different definitions in 1806.02780
        g_06_rescaling = self.m_e_MEV*element.R/2
        g_09_rescaling = g_06_rescaling**2
        g_04_rescaling = 9/2

        def p(E, m = self.m_e_MEV):
            return(np.sqrt(E**2 - m**2))

        result = (element.g_01(E, Delta_M - E) * (np.absolute(amp["nu"])**2 + np.absolute(amp["R"])**2)
                          - 2 * (element.g_01(E, Delta_M - E) - g_04_rescaling * element.g_04(E, Delta_M - E))*(np.conj(amp["nu"])*amp["R"]).real
                          + 4 *  element.g_02(E, Delta_M - E)* np.absolute(amp["E"])**2
                          + 2 *  g_04_rescaling * element.g_04(E, Delta_M - E)*(np.absolute(amp["me"])**2 + (np.conj(amp["me"])*(amp["nu"]+amp["R"])).real)
                          - 2 *  element.g_03(E, Delta_M - E)*((amp["nu"]+amp["R"])*np.conj(amp["E"]) + 2*amp["me"]*np.conj(amp["E"]))
                          + g_09_rescaling * element.g_09(E, Delta_M - E) * np.absolute(amp["M"])**2
                          + g_06_rescaling * element.g_06(E, Delta_M - E) * ((amp["nu"]-amp["R"])*np.conj(amp["M"])).real)* p(E)*p(Delta_M-E)* E * (Delta_M - E)
        return(result)

    def angular_corr(self, Ebar, amp = None, element_name = "76Ge", method=None):
        
        #set the method and import NMEs if necessary
        if method == None:
            pass
        elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            print("Changing method to",method)
            self.method = method
            self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        method = self.method
        #calculates the angular correlation coefficient for a given normalized energy Ebar
        
        #get element class
        element = self.elements[element_name]
        
        #if not initialized calculate amplitudes
        if amp == None:
            amp = self.amplitudes(element_name, self.WC)[0]
            
        #electron mass in MeV
        #m_e = pc["electron mass energy equivalent in MeV"][0]
        
        #Mass difference between mother and daughter nuclei in MeV
        Delta_M = element.Delta_M
        
        #Energy from normalized Energy scale
        E = Ebar * (Delta_M - 2*self.m_e_MEV) + self.m_e_MEV
        
        #rescale PSFs due to different definitions in 1806.02780
        g_06_rescaling = self.m_e_MEV*element.R/2
        g_09_rescaling = g_06_rescaling**2
        g_04_rescaling = 9/2

        hs = (element.h_01(E, Delta_M - E) * (np.absolute(amp["nu"])**2 + np.absolute(amp["R"])**2)
                          - 2 * (element.h_01(E, Delta_M - E) - g_04_rescaling * element.h_04(E, Delta_M - E))*(np.conj(amp["nu"])*amp["R"]).real
                          + 4 *  element.h_02(E, Delta_M - E)* np.absolute(amp["E"])**2
                          + 2 *  g_04_rescaling * element.h_04(E, Delta_M - E)*(np.absolute(amp["me"])**2 + (np.conj(amp["me"])*(amp["nu"]+amp["R"])).real)
                          - 2 *  element.h_03(E, Delta_M - E)*((amp["nu"]+amp["R"])*np.conj(amp["E"]) + 2*amp["me"]*np.conj(amp["E"]))
                          + g_09_rescaling * element.h_09(E, Delta_M - E) * np.absolute(amp["M"])**2
                          + g_06_rescaling * element.h_06(E, Delta_M - E) * ((amp["nu"]-amp["R"])*np.conj(amp["M"])).real)

        gs = (element.g_01(E, Delta_M - E) * (np.absolute(amp["nu"])**2 + np.absolute(amp["R"])**2)
                          - 2 * (element.g_01(E, Delta_M - E) - g_04_rescaling * element.g_04(E, Delta_M - E))*(np.conj(amp["nu"])*amp["R"]).real
                          + 4 *  element.g_02(E, Delta_M - E)* np.absolute(amp["E"])**2
                          + 2 *  g_04_rescaling * element.g_04(E, Delta_M - E)*(np.absolute(amp["me"])**2 + (np.conj(amp["me"])*(amp["nu"]+amp["R"])).real)
                          - 2 *  element.g_03(E, Delta_M - E)*((amp["nu"]+amp["R"])*np.conj(amp["E"]) + 2*amp["me"]*np.conj(amp["E"]))
                          + g_09_rescaling * element.g_09(E, Delta_M - E) * np.absolute(amp["M"])**2
                          + g_06_rescaling * element.g_06(E, Delta_M - E) * ((amp["nu"]-amp["R"])*np.conj(amp["M"])).real)

        return (hs/gs)
    
    
    '''
        ####################################################################################################
        
        Define outputting functions for
        1. half_lives
        2. hl_ratios
        3. PSF observables
        
        ####################################################################################################
    '''
    def half_lives(self, method = None, use_unknown_LECs = None):
    #returns a pandas.DataFrame with all half-lives of the available isotopes for the considered NME method
        if use_unknown_LECs != None and use_unknown_LECs != self.use_unknown_LECs:
            self.set_LECs(use_unknown_LECs)
        if method == None:
            pass
        elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            print("Changing method to",method)
            self.method = method
            self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass

        hl = pd.DataFrame([], [r"$y$"])
        for element_name in list(self.NMEs.keys()):
            element = self.elements[element_name]
            hl[element_name] = [self.t_half(element_name, self.WC)]
        
        return(hl)
    
    
    def ratios(self, save = False, plot = False, reference_isotope = "76Ge", normalized = True, method=None, vary = False, n_points = 100, addgrid = True):
    #returns the half-live ratios compared to the standard mass mechanism based on the chosen reference isotope
    #can optionally also generate a plot of the ratios and save them as "ratios_"+self.name+".png"
    
        
        #set the method and import NMEs if necessary
        if method == None:
            pass
        elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            print("Changing method to",method)
            self.method = method
            self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        method = self.method

        #generate WC dict for mass mechanism
        WC_mbb = self.WC.copy()
        for operator in WC_mbb:
            WC_mbb[operator] = 0
        WC_mbb["m_bb"] = 1e-9#m_bb * 1e-9
        NMEs = self.NMEs.copy()

        #store ratios in a pd.DataFrame
        self.ratio_values = pd.DataFrame()

        #Generate a column for each isotope
        for element_name in NMEs.keys():
            self.ratio_values[element_name] = []

        #Generate a column that contains the operator names -> set to index later
        self.ratio_values["Model"] = [self.name]
        self.ratio_values.set_index("Model", inplace = True)
        
        if vary:
            self.ratio_values_varried = {}
            for element_name in NMEs.keys():
                self.ratio_values_varried[element_name] = []

        #generate figure
        if plot:
            fig = plt.figure(figsize=(6.4*1.85, 4.8*2))

            plt.xlabel(r"$\frac{R^{\mathcal{O}_i}-R^{m_\nu}}{R^{m_\nu}}$", fontsize=30)#, x = 0.45,y=1.05)

        #generate a list to store labels in, 1st label has to be empty. It corresponds to the axvline
        labels = [""]

        #iterate over all elements
        for element_name in NMEs.keys():#["76Ge", "82Se" , "130Te", "136Xe"]:

            ratio_list = []

            #match element_name to element class
            element = self.elements[element_name]

            #get A from the element_name for the plot labels
            if element_name[0] != "1" and element_name[0] != "2":
                element_name_plot = element_name[0:2]
            else:
                element_name_plot = element_name[0:3]

            #set the color
            c = "b"
            
            #set the marker
            m = "x"

            #Calculate Rates for Current element and the reference element 76Ge
            #This could be done more efficiently by storing the GE amplitudes 
            #one loop earlier, but I was too lazy and including the amps in the
            #t_half function is cleaner.

            #Store half lives for light nu exchange model
            t_half_m_bb = self.t_half(element_name, WC_mbb)
            t_half_m_bb_Ge = self.t_half(reference_isotope, WC_mbb)

            #calculate half lives for current model
            t_half_model = self.t_half(element_name, self.WC)
            t_half_model_Ge = self.t_half(reference_isotope, self.WC)

            if normalized:
                #take ratio normalized to the standard mass mechanism
                ratio = ((t_half_model/t_half_model_Ge) / (t_half_m_bb/t_half_m_bb_Ge))
            else:
                ratio = t_half_model/t_half_model_Ge


            if plot:
                plt.scatter(np.log10(ratio), element_name_plot, marker = m, color = c, s=150)

            self.ratio_values[element_name] = [ratio]
            
            if vary:
                #define unknown LECs to be varied over
                LECs = {'Tprime': 1,
                         'Tpipi': 1,
                         '1piN': 1,
                         '6piN': 1,
                         '7piN': 1,
                         '8piN': 1,
                         '9piN': 1,
                         'VLpiN': 1,
                         'TpiN': 1,
                         '1NN': 1,
                         '6NN': 1,
                         '7NN': 1,
                         'VLNN': 1,
                         'TNN': 1,
                         'VLE': 1,
                         'VLme': 1,
                         'VRE': 1,
                         'VRme': 1,
                         '2NN': 157.91367041742973,
                         '3NN': 157.91367041742973,
                         '4NN': 157.91367041742973,
                         '5NN': 157.91367041742973,
                         'nuNN': -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
                       }
                LEC_backup = self.LEC.copy()
                n_points = 500
                varried_ratios = pd.DataFrame()
                for idx in range(n_points):
                    for LEC in LECs:
                        if LEC == "nuNN":
                            random_LEC = (np.random.rand()+0.5)*LECs[LEC]
                            self.LEC[LEC] = random_LEC
                        else:
                            #random_LEC = variation_range*2*(np.random.rand()-0.5)*LECs[LEC]
                            random_LEC = np.random.choice([1,-1])*((np.log10(10)-
                                                                    np.log10(1))*np.random.rand()+np.log10(1))*LECs[LEC]
                            self.LEC[LEC] = random_LEC
                    #set LECs that depend on others
                    self.LEC["VpiN"] = self.LEC["6piN"] + self.LEC["8piN"]
                    self.LEC["tildeVpiN"] = self.LEC["7piN"] + self.LEC["9piN"]
                    
                    #Store half lives for light nu exchange model
                    t_half_m_bb = self.t_half(element_name, WC_mbb)
                    t_half_m_bb_Ge = self.t_half(reference_isotope, WC_mbb)

                    #calculate half lives for current model
                    t_half_model = self.t_half(element_name, self.WC)
                    t_half_model_Ge = self.t_half(reference_isotope, self.WC)

                    if normalized:
                        #take ratio normalized to the standard mass mechanism
                        ratio = ((t_half_model/t_half_model_Ge) / (t_half_m_bb/t_half_m_bb_Ge))
                    else:
                        ratio = t_half_model/t_half_model_Ge
                        
                    
                    self.ratio_values_varried[element_name].append(ratio)
                    if plot:
                        plt.scatter(np.log10(ratio), element_name_plot, marker = ".", color = c, s=20, alpha = 0.25)
                    
                #restore standard LECs
                self.LEC = LEC_backup.copy()
        if plot:
            #plt.legend([r"$m_{\beta\beta}$", self.name], loc="upper right", ncol=1, fontsize = 20)
            if normalized:
                plt.axvline(0, color="k", linewidth=1, label="")
                plt.legend([r"$m_{\beta\beta}$", self.name], loc="upper right", ncol=1, fontsize = 20)
                #plt.axvline(0, color="k", linewidth=1, label="")
                plt.xlabel(r"$\log_{10}\frac{R^{\mathcal{O}_i}}{R^{m_{\beta\beta}}}$", fontsize=30)
            else:
                plt.legend([self.name], loc="upper right", ncol=1, fontsize = 20)
                plt.xlabel(r"$\log_{10}R^{\mathcal{O}_i}$", fontsize=30)
            plt.rc("ytick", labelsize = 20)
            plt.rc("xtick", labelsize = 20)
            plt.tight_layout()
            if addgrid:
                plt.grid(linestyle = "--")
        if save and plot:
            plt.savefig("ratios_"+self.name+".png", dpi=300)
        if plot:
            return(fig)

        return(self.ratio_values)
    
    
    def PSF_plot(self, element_name="76Ge", save=False, method=None):
        #generates plots of the PSF observables
        #i.e. angular corr. and single e spectrum
        
        
        #set the method and import NMEs if necessary
        if method == None:
            pass
        elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            print("Changing method to",method)
            self.method = method
            self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        method = self.method
        
        #generate WC dict for mass mechanism
        WC_mbb = self.WC.copy()
        for operator in WC_mbb:
            WC_mbb[operator] = 0
        WC_mbb["m_bb"] = 1
        
        #m_e = pc["electron mass energy equivalent in MeV"][0]
        
        #necessary to avoid pole
        epsilon = 1e-6
        
        #get element class
        element = self.elements[element_name]
        
        #energy range for spectrum
        E = np.linspace(self.m_e_MEV+epsilon, element.Delta_M-self.m_e_MEV-epsilon, 1000)
        
        #normalized energy
        Ebar = (E-self.m_e_MEV)/(element.Delta_M-2*self.m_e_MEV)
        
        #calculate amplitudes for a model and mass mechanism
        Amplitude, _ = self.amplitudes(element_name, self.WC)
        Amplitude_mbb, _ = self.amplitudes(element_name, WC_mbb)
        
        #normalization factors for single electron spectra
        integral = integrate.quad(lambda E: self.spectra(E, Amplitude, element_name = element_name), 0, 1)
        integral_mbb = integrate.quad(lambda E: self.spectra(E, Amplitude_mbb, element_name = element_name), 0, 1)
        
        #generate figures
        plt.figure()
        plt.title("Single Electron Spectrum")
        plt.plot(Ebar, self.spectra(Ebar, amp = Amplitude, element_name = element_name)/integral[0], "b", label = r"$m_{\beta\beta}$")
        plt.plot(Ebar, self.spectra(Ebar, amp = Amplitude_mbb, element_name = element_name)/integral_mbb[0], "r", label = self.name)
        plt.legend(fontsize = 20)
        if save:
            plt.savefig("spectra_"+element_name+"_"+self.name+".png", dpi = 300)
        
        plt.figure()
        plt.title("Angular Correlation")
        plt.plot(Ebar, self.angular_corr(Ebar, amp = Amplitude_mbb, element_name = element_name), "b", label = r"$m_{\beta\beta}$")
        plt.plot(Ebar, self.angular_corr(Ebar, amp = Amplitude, element_name = element_name), "r", label = self.name)
        plt.legend()
        
        if save:
            plt.savefig("angular_correlation_"+element_name+"_"+self.name+".png", dpi = 300)
            
    def plot_spec(self, element_name="76Ge", save=False, method=None, print_title = False, addgrid = True, show_mbb = True):
        #generates a plot of the single electron spectrum
        
        #set the method and import NMEs if necessary
        if method == None:
            pass
        elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            print("Changing method to",method)
            self.method = method
            self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        method = self.method
        
        #generate WC dict for mass mechanism
        if show_mbb:
            WC_mbb = self.WC.copy()
            for operator in WC_mbb:
                WC_mbb[operator] = 0
            WC_mbb["m_bb"] = 1
        
        #m_e = pc["electron mass energy equivalent in MeV"][0]
        
        #necessary to avoid pole
        epsilon = 1e-6
        
        #get element class
        element = self.elements[element_name]
        
        #energy range for spectrum
        E = np.linspace(self.m_e_MEV+epsilon, element.Delta_M-self.m_e_MEV-epsilon, 1000)
        
        #normalized energy
        Ebar = (E-self.m_e_MEV)/(element.Delta_M-2*self.m_e_MEV)
        
        #calculate amplitudes for a model and mass mechanism
        Amplitude, _ = self.amplitudes(element_name, self.WC)
        if show_mbb:
            Amplitude_mbb, _ = self.amplitudes(element_name, WC_mbb)
        
        #normalization factors for single electron spectra
        integral = integrate.quad(lambda E: self.spectra(E, Amplitude, element_name = element_name), 0, 1)
        if show_mbb:
            integral_mbb = integrate.quad(lambda E: self.spectra(E, Amplitude_mbb, element_name = element_name), 0, 1)
        
        #generate figures
        fig = plt.figure(figsize=(6.4*1.85, 4.8*2))
        if print_title:
            plt.title("Single Electron Spectrum")
        spec = self.spectra(Ebar, amp = Amplitude, element_name = element_name)/integral[0]
        if show_mbb:
            spec_mbb = self.spectra(Ebar, amp = Amplitude_mbb, element_name = element_name)/integral_mbb[0]
        if show_mbb:
            plt.plot(Ebar, spec_mbb, "r", label = r"$m_{\beta\beta}$")
        plt.plot(Ebar, spec, "b", label = self.name)
        plt.legend(fontsize = 20, loc="upper right")
        plt.xlim(0,1)
        if show_mbb:
            plt.ylim(0, 1.05*max(max(spec),max(spec_mbb)))
        else:
            plt.ylim(0, 1.05*max(spec))
        plt.rc("ytick", labelsize = 20)
        plt.rc("xtick", labelsize = 20)
        if addgrid:
            plt.grid(linestyle = "--")
        if save:
            plt.savefig("spectra_"+element_name+"_"+self.name+".png", dpi = 300)
        return(fig)
            
    def plot_corr(self, element_name="76Ge", save=False, method=None, print_title = False, addgrid = True, show_mbb = True):
        #generates a plot of the angular correlation coefficient
        
        
        #set the method and import NMEs if necessary
        if method == None:
            pass
        elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            print("Changing method to",method)
            self.method = method
            self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        method = self.method
        
        #generate WC dict for mass mechanism
        if show_mbb:
            WC_mbb = self.WC.copy()
            for operator in WC_mbb:
                WC_mbb[operator] = 0
            WC_mbb["m_bb"] = 1
        
        #m_e = pc["electron mass energy equivalent in MeV"][0]
        
        #necessary to avoid pole
        epsilon = 1e-6
        
        #get element class
        element = self.elements[element_name]
        
        #energy range for spectrum
        E = np.linspace(self.m_e_MEV+epsilon, element.Delta_M-self.m_e_MEV-epsilon, 1000)
        
        #normalized energy
        Ebar = (E-self.m_e_MEV)/(element.Delta_M-2*self.m_e_MEV)
        
        #calculate amplitudes for a model and mass mechanism
        Amplitude, _ = self.amplitudes(element_name, self.WC)
        if show_mbb:
            Amplitude_mbb, _ = self.amplitudes(element_name, WC_mbb)
        
        #generate figures            
        fig = plt.figure(figsize=(6.4*1.85, 4.8*2))
        if print_title:
            plt.title("Angular Correlation")
        if show_mbb:
            a_corr_mbb = self.angular_corr(Ebar, amp = Amplitude_mbb, element_name = element_name)
        a_corr = self.angular_corr(Ebar, amp = Amplitude, element_name = element_name)
        if show_mbb:
            plt.plot(Ebar, a_corr_mbb, "r", label = r"$m_{\beta\beta}$")
        plt.plot(Ebar, a_corr, "b", label = self.name)
        plt.legend(fontsize=20, loc="upper right")
        plt.xlim(0,1)
        plt.ylim(-1,1)
        plt.rc("ytick", labelsize = 20)
        plt.rc("xtick", labelsize = 20)
        if addgrid:
            plt.grid(linestyle = "--")
        if save:
            plt.savefig("angular_correlation_"+element_name+"_"+self.name+".png", dpi = 300)
        return(fig)
    
    def get_limits2(self, half_live, element_name= "76Ge", basis = None, method = None, onlygroups = False):
    #this function can calculate the limits on the different LEFT coefficients for a given experimental half_live and isotope
    #the limits are calculate at the scale "scale" and for the chosen basis
        
        
        #set the method and import NMEs if necessary
        if method == None:
            method = self.method
            pass
        elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            print("Changing method to",method)
            self.method = method
            self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
            method = self.method
        else:
            pass
            
        #method = self.method
        
        #vev = 246
        result_2GeV = {}
        scales = {}
        #result_80GeV = {}
        #scale_80GeV = {}

        #make a backup so you can overwrite the running afterwards
        WC_backup = self.WC.copy()

        #calculate the limits on the WCs at the scale of 2GeV
        if onlygroups:
            WCgroups = ["m_bb" , "VL(6)", 
                        "VR(6)", "T(6)", 
                        "SL(6)", "VL(7)", "1L(9)", 
                        "2L(9)", "3L(9)", 
                        "4L(9)", "5L(9)", 
                        "6(9)", "7(9)"]

            #define labels for plots
            WC_names = {"m_bb"  : "m_bb", 
                        "VL(6)" : "VL(6)", 
                        "VR(6)" : "VR(6)", 
                        "T(6)"  : "T(6)" , 
                        "SL(6)" : "S(6)", 
                        "VL(7)" : "V(7)"     , 
                        "1L(9)" : "S1(9)",
                        "2L(9)" : "S2(9)", 
                        "3L(9)" : "S3(9)", 
                        "4L(9)" : "S4(9)", 
                        "5L(9)" : "S5(9)",
                        "6(9)"  : "V(9)", 
                        "7(9)"  : "Vtilde(9)"}

            
            for WC_name in WCgroups:
                hl = self.t_half(WC = {WC_name:1}, method = method, element_name = element_name)
                result_2GeV[WC_name] = np.sqrt(hl/half_live)
        else:
            for WC_name in self.WC:
                hl = self.t_half(WC = {WC_name:1}, method = method, element_name = element_name)
                result_2GeV[WC_name] = np.sqrt(hl/half_live)
#run results up to the desired scale
        #results = self.run(result_2GeV, initial_scale = 2, final_scale = scale)
        results = self.run(WC = result_2GeV, updown="up")
        if onlygroups:
            res = {}
            for WC in WCgroups:
                WC_name = WC_names[WC]
                res[WC_name] = results[WC]
            results = res.copy()
        
        
        #take abs value to get positive results in case the numerical estimate returned a negative
        for result in results:
            results[result] = np.absolute(results[result])

        self.WC = WC_backup.copy()

        #calculate the corresponding scales of new physics assuming naturalness
        if onlygroups:
            for WC in WCgroups:
                WC_name = WC_names[WC]
                if WC_name == "m_bb":
                    scales[WC_name] = np.absolute(results[WC_name])*1e+9
                elif WC_name in ["SR(6)", "SL(6)", "T(6)", "VL(6)", "VR(6)", "VL(7)", "VR(7)", "1L(9)", "4L(9)", "5L(9)"]:
                    scales[WC_name] = self.vev/(results[WC_name]**(1/3))/1000
                else:
                    scales[WC_name] = self.vev/(results[WC_name]**(1/5))/1000
        else:
            for WC_name in self.WC:
                if WC_name == "m_bb":
                    scales[WC_name] = np.absolute(results[WC_name])*1e+9
                elif WC_name in ["SR(6)", "SL(6)", "T(6)", "VL(6)", "VR(6)", "VL(7)", "VR(7)", "1L(9)", "4L(9)", "5L(9)"]:
                    scales[WC_name] = self.vev/(results[WC_name]**(1/3))/1000
                else:
                    scales[WC_name] = self.vev/(results[WC_name]**(1/5))/1000
        return(results, scales)

    def get_limits(self, half_live, element_name="76Ge", scale = 80, basis = None, method=None):
    #this function can calculate the limits on the different LEFT coefficients for a given experimental half_live and isotope
    #the limits are calculate at the scale "scale" and for the chosen basis
        
        
        #set the method and import NMEs if necessary
        if method == None:
            pass
        elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            print("Changing method to",method)
            self.method = method
            self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        method = self.method

        def t_half_optimize(WC_value, WC_name, element_name, run = False):
            #helper function to find the root
            WC = self.WC.copy()
            for operator in WC:
                WC[operator]=0

            WC[WC_name]=WC_value
            #Calculates the half-live for a given element and WCs
            amp, M = self.amplitudes(element_name, WC)
            element = self.elements[element_name]
            
            g_A=self.LEC["A"]
            
            #G = PSFs[element]
            G = self.to_G(element_name)
            #m_e = pc["electron mass energy equivalent in MeV"][0]
    
            #Some PSFs need a rescaling due to different definitions in DOIs paper and 1806...
            g_06_rescaling = self.m_e_MEV*element.R/2
            g_09_rescaling = g_06_rescaling**2
            g_04_rescaling = 9/2
            G["06"] *= g_06_rescaling
            G["04"] *= g_04_rescaling
            G["09"] *= g_09_rescaling

            #Calculate half-life following eq 38. in 1806.02780
            inverse_result = g_A**4*(G["01"] * (np.absolute(amp["nu"])**2 + np.absolute(amp["R"])**2)
                              - 2 * (G["01"] - G["04"])*(np.conj(amp["nu"])*amp["R"]).real
                              + 4 *  G["02"]* np.absolute(amp["E"])**2
                              + 2 *  G["04"]*(np.absolute(amp["me"])**2 + (np.conj(amp["me"])*(amp["nu"]+amp["R"])).real)
                              - 2 *  G["03"]*((amp["nu"]+amp["R"])*np.conj(amp["E"]) + 2*amp["me"]*np.conj(amp["E"])).real
                              + G["09"] * np.absolute(amp["M"])**2
                              + G["06"] * ((amp["nu"]-amp["R"])*np.conj(amp["M"])).real)
            #return expected hl - experimental hl to search for root
            return(1/inverse_result-half_live)
        
        #vev = 246
        result_2GeV = {}
        scales = {}
        #result_80GeV = {}
        #scale_80GeV = {}

        #make a backup so you can overwrite the running afterwards
        WC_backup = self.WC.copy()

        #calculate the limits on the WCs at the scale of 2GeV
        for WC_name in self.WC:
            limit_2GeV = optimize.root(t_half_optimize, args=(WC_name, element_name, False), x0=1e-15).x[0]
            result_2GeV[WC_name] = limit_2GeV

        #run results up to the desired scale
        #results = self.run(result_2GeV, initial_scale = 2, final_scale = scale)
        results = self.run(WC = result_2GeV, updown="up")
        
        
        #take abs value to get positive results in case the numerical estimate returned a negative
        for result in results:
            results[result] = np.absolute(results[result])

        self.WC = WC_backup.copy()

        #calculate the corresponding scales of new physics assuming naturalness
        for WC_name in self.WC:
            if WC_name == "m_bb":
                scales[WC_name] = np.absolute(results[WC_name])*1e+9
            elif WC_name in ["SR(6)", "SL(6)", "T(6)", "VL(6)", "VR(6)", "VL(7)", "VR(7)", "1L(9)", "4L(9)", "5L(9)"]:
                scales[WC_name] = self.vev/(results[WC_name]**(1/3))/1000
            else:
                scales[WC_name] = self.vev/(results[WC_name]**(1/5))/1000
        return(results, scales)
    
    #fancy plots:
    def _m_bb(self, alpha, m_min=1, ordering="NO", dcp=1.36):
        #majorana phases
        alpha1=alpha[0]
        alpha2=alpha[1]

        #squared mass differences
        m21 = 7.53e-5
        m32 = 2.453e-3
        m32IO = -2.546e-3

        #get mass eigenvalues from minimal neutrino mass
        m = m_min
        m1 = m
        m2 = np.sqrt(m1**2+m21)
        m3 = np.sqrt(m2**2+m32)

        m3IO = m
        m2IO = np.sqrt(m3IO**2-m32IO)
        m1IO = np.sqrt(m2IO**2-m21)

        #create diagonal mass matrices
        M_nu_NO = np.diag([m1,m2,m3])
        M_nu_IO = np.diag([m1IO,m2IO,m3IO])

        #mixing angles
        s12 = np.sqrt(0.307)
        s23 = np.sqrt(0.545)
        s13 = np.sqrt(2.18e-2)

        c12 = np.cos(np.arcsin(s12))
        c23 = np.cos(np.arcsin(s23))
        c13 = np.cos(np.arcsin(s13))

        #mixing marix
        U = np.array([[c12*c13, s12*c13, s13*np.exp(-1j*dcp)], 
                       [-s12*c23-c12*s23*s13*np.exp(1j*dcp), c12*c23-s12*s23*s13*np.exp(1j*dcp), s23*c13], 
                       [s12*s23-c12*c23*s13*np.exp(1j*dcp), -c12*s23-s12*c23*s13*np.exp(1j*dcp), c23*c13]])

        majorana = np.diag([1, np.exp(1j*alpha1), np.exp(1j*alpha2)])

        UPMNS = U@majorana

        #create non-diagonal mass matrix
        m_BB_NO = np.abs(UPMNS[0,0]**2*m1+UPMNS[0,1]**2*m2+UPMNS[0,2]**2*m3)
        m_BB_IO = np.abs(UPMNS[0,0]**2*m1IO+UPMNS[0,1]**2*m2IO+UPMNS[0,2]**2*m3IO)

        if ordering == "NO":
            return(m_BB_NO)
        elif ordering =="IO":
            return(m_BB_IO)
        else:
            return(m_BB_NO,m_BB_IO)
    def _m_eff(self, alpha, m_min=1, ordering="both", dcp=1.36, 
               element_name = "76Ge", normalize_to_mass = False, vary_WC  = "m_min"):
        m_bb_backup = self.WC["m_bb"]
        WC_backup = self.WC.copy()
        #majorana phases
        if vary_WC  == "m_min":
            #for m_min you have two majorana phases in the mixing matrix
            #for all other WCs you just need to vary the corresponding phase of the WC
            alpha1=alpha[0]
            alpha2=alpha[1]
        #if len(alpha)>2:
        #    LEC_backup = self.LEC.copy()
        #    LEC = alpha[2:]
        #    LECranges = { 'Tprime': 1,
        #         'Tpipi': 1,
        #         '1piN': 1,
        #         '6piN': 1,
        #         '7piN': 1,
        #         '8piN': 1,
        #         '9piN': 1,
        #         'VLpiN': 1,
        #         'TpiN': 1,
        #         '1NN': 1,
        #         '6NN': 1,
        #         '7NN': 1,
        #         'VLNN': 1,
        #         'TNN': 1,
        #         'VLE': 1,
        #         'VLme': 1,
        #         'VRE': 1,
        #         'VRme': 1,
        #         '2NN': 157.91367041742973,
        #         '3NN': 157.91367041742973,
        #         '4NN': 157.91367041742973,
        #         '5NN': 157.91367041742973,
        #         'nuNN': -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
        #       }
        #    for idx in range(len(LEC)):
        #        key = list(LECranges.keys())[idx]
        #        if key == "nuNN":
        #            
        #            self.LEC[key] = np.sign(LEC[idx])*np.min([np.max([0.5*LECranges[key], np.abs(LEC[idx])]), 1.5*LECranges[key]])
        #        else:
        #            self.LEC[key] = np.sign(LEC[idx])*np.min([np.max([1/np.sqrt(10)*LECranges[key], np.abs(LEC[idx])]), np.sqrt(10)])
            

        #squared mass differences [eV]
            m21 = 7.53e-5
            m32 = 2.453e-3
            m32IO = -2.546e-3

            #get mass eigenvalues from minimal neutrino mass in [eV]
            m = m_min
            m1 = m
            m2 = np.sqrt(m1**2+m21)
            m3 = np.sqrt(m2**2+m32)

            m3IO = m
            m2IO = np.sqrt(m3IO**2-m32IO)
            m1IO = np.sqrt(m2IO**2-m21)

            #create diagonal mass matrices
            M_nu_NO = np.diag([m1,m2,m3])
            M_nu_IO = np.diag([m1IO,m2IO,m3IO])

            #mixing angles
            s12 = np.sqrt(0.307)
            s23 = np.sqrt(0.545)
            s13 = np.sqrt(2.18e-2)

            c12 = np.cos(np.arcsin(s12))
            c23 = np.cos(np.arcsin(s23))
            c13 = np.cos(np.arcsin(s13))

            #mixing marix
            U = np.array([[c12*c13, s12*c13, s13*np.exp(-1j*dcp)], 
                           [-s12*c23-c12*s23*s13*np.exp(1j*dcp), c12*c23-s12*s23*s13*np.exp(1j*dcp), s23*c13], 
                           [s12*s23-c12*c23*s13*np.exp(1j*dcp), -c12*s23-s12*c23*s13*np.exp(1j*dcp), c23*c13]])

            majorana = np.diag([1, np.exp(1j*alpha1), np.exp(1j*alpha2)])

            UPMNS = U@majorana

            #create non-diagonal mass matrix
            m_BB_NO = np.abs(UPMNS[0,0]**2*m1+UPMNS[0,1]**2*m2+UPMNS[0,2]**2*m3)
            m_BB_IO = np.abs(UPMNS[0,0]**2*m1IO+UPMNS[0,1]**2*m2IO+UPMNS[0,2]**2*m3IO)

            self.WC["m_bb"] = m_BB_NO*1e-9 #[GeV]
        else:
            if vary_WC == "m_bb":
                factor = 1e-9
            else:
                factor = 1
            self.WC[vary_WC] = np.exp(1j*alpha)*m_min*factor
        g_A = self.LEC["A"]
        V_ud = 0.97417


        G01 = self.to_G(element_name)["01"]
        M3 = self.amplitudes(element_name, self.WC)[1]["nu(3)"]
        #NO_min_eff = self.m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)
        NO_eff = self.m_e / (g_A**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)
        
        

        
        if vary_WC == "m_min":
            if normalize_to_mass:
                NO_eff /= self.WC["m_bb"]
            self.WC["m_bb"] = m_BB_IO*1e-9
            #IO_min_eff = self.m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)
            IO_eff = self.m_e / (g_A**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)

            if normalize_to_mass:
                IO_eff /= self.WC["m_bb"]

            self.WC["m_bb"] = m_bb_backup
            self.WC = WC_backup.copy()
            if len(alpha)>2:
                self.LEC = LEC_backup.copy()
            if ordering == "NO":
                return(NO_eff)
            elif ordering =="IO":
                return(IO_eff)
            else:
                return(NO_eff,IO_eff)
        else:
            self.WC = WC_backup.copy()
            return(NO_eff)
    
    
    
    def _m_eff_minus(self, alpha, m_min=1, ordering="both", dcp=1.36, 
                     element_name = "76Ge", normalize_to_mass = False, vary_WC  = "m_min"):
        m_bb_backup = self.WC["m_bb"]
        WC_backup = self.WC.copy()
        #majorana phases
        if vary_WC  == "m_min":
            #for m_min you have two majorana phases in the mixing matrix
            #for all other WCs you just need to vary the corresponding phase of the WC
            alpha1=alpha[0]
            alpha2=alpha[1]
            

        #squared mass differences [eV]
            m21 = 7.53e-5
            m32 = 2.453e-3
            m32IO = -2.546e-3

            #get mass eigenvalues from minimal neutrino mass in [eV]
            m = m_min
            m1 = m
            m2 = np.sqrt(m1**2+m21)
            m3 = np.sqrt(m2**2+m32)

            m3IO = m
            m2IO = np.sqrt(m3IO**2-m32IO)
            m1IO = np.sqrt(m2IO**2-m21)

            #create diagonal mass matrices
            M_nu_NO = np.diag([m1,m2,m3])
            M_nu_IO = np.diag([m1IO,m2IO,m3IO])

            #mixing angles
            s12 = np.sqrt(0.307)
            s23 = np.sqrt(0.545)
            s13 = np.sqrt(2.18e-2)

            c12 = np.cos(np.arcsin(s12))
            c23 = np.cos(np.arcsin(s23))
            c13 = np.cos(np.arcsin(s13))

            #mixing marix
            U = np.array([[c12*c13, s12*c13, s13*np.exp(-1j*dcp)], 
                           [-s12*c23-c12*s23*s13*np.exp(1j*dcp), c12*c23-s12*s23*s13*np.exp(1j*dcp), s23*c13], 
                           [s12*s23-c12*c23*s13*np.exp(1j*dcp), -c12*s23-s12*c23*s13*np.exp(1j*dcp), c23*c13]])

            majorana = np.diag([1, np.exp(1j*alpha1), np.exp(1j*alpha2)])

            UPMNS = U@majorana

            #create non-diagonal mass matrix
            m_BB_NO = np.abs(UPMNS[0,0]**2*m1+UPMNS[0,1]**2*m2+UPMNS[0,2]**2*m3)
            m_BB_IO = np.abs(UPMNS[0,0]**2*m1IO+UPMNS[0,1]**2*m2IO+UPMNS[0,2]**2*m3IO)

            self.WC["m_bb"] = m_BB_NO*1e-9 #[GeV]
        else:
            if vary_WC == "m_bb":
                factor = 1e-9
            else:
                factor = 1
            self.WC[vary_WC] = np.exp(1j*alpha)*m_min*factor
        g_A = self.LEC["A"]
        V_ud = 0.97417


        G01 = self.to_G(element_name)["01"]
        M3 = self.amplitudes(element_name, self.WC)[1]["nu(3)"]
        #NO_min_eff = self.m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)
        NO_eff = self.m_e / (g_A**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)
        
        

        
        if vary_WC == "m_min":
            if normalize_to_mass:
                NO_eff /= self.WC["m_bb"]
            self.WC["m_bb"] = m_BB_IO*1e-9
            #IO_min_eff = self.m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)
            IO_eff = self.m_e / (g_A**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)

            if normalize_to_mass:
                IO_eff /= self.WC["m_bb"]

            self.WC["m_bb"] = m_bb_backup
            self.WC = WC_backup.copy()
            #if len(alpha)>2:
            #    self.LEC = LEC_backup.copy()
            if ordering == "NO":
                return(-NO_eff)
            elif ordering =="IO":
                return(-IO_eff)
            else:
                return(-NO_eff,-IO_eff)
        else:
            self.WC = WC_backup.copy()
            return(-NO_eff)
    
    
    def _m_bb_minus(self, alpha, m_min=1, ordering="both", dcp=1.36):
        #majorana phases
        alpha1=alpha[0]
        alpha2=alpha[1]

        #squared mass differences
        m21 = 7.53e-5
        m32 = 2.453e-3
        m32IO = -2.546e-3

        #get mass eigenvalues from minimal neutrino mass
        m = m_min
        m1 = m
        m2 = np.sqrt(m1**2+m21)
        m3 = np.sqrt(m2**2+m32)

        m3IO = m
        m2IO = np.sqrt(m3IO**2-m32IO)
        m1IO = np.sqrt(m2IO**2-m21)

        #create diagonal mass matrices
        M_nu_NO = np.diag([m1,m2,m3])
        M_nu_IO = np.diag([m1IO,m2IO,m3IO])

        #mixing angles
        s12 = np.sqrt(0.307)
        s23 = np.sqrt(0.545)
        s13 = np.sqrt(2.18e-2)

        c12 = np.cos(np.arcsin(s12))
        c23 = np.cos(np.arcsin(s23))
        c13 = np.cos(np.arcsin(s13))

        #mixing marix
        U = np.array([[c12*c13, s12*c13, s13*np.exp(-1j*dcp)], 
                       [-s12*c23-c12*s23*s13*np.exp(1j*dcp), c12*c23-s12*s23*s13*np.exp(1j*dcp), s23*c13], 
                       [s12*s23-c12*c23*s13*np.exp(1j*dcp), -c12*s23-s12*c23*s13*np.exp(1j*dcp), c23*c13]])

        majorana = np.diag([1, np.exp(1j*alpha1), np.exp(1j*alpha2)])

        UPMNS = U@majorana

        #create non-diagonal mass matrix
        m_BB_NO = np.abs(UPMNS[0,0]**2*m1+UPMNS[0,1]**2*m2+UPMNS[0,2]**2*m3)
        m_BB_IO = np.abs(UPMNS[0,0]**2*m1IO+UPMNS[0,1]**2*m2IO+UPMNS[0,2]**2*m3IO)

        if ordering == "NO":
            return(-m_BB_NO)
        elif ordering =="IO":
            return(-m_BB_IO)
        else:
            return(-m_BB_NO,-m_BB_IO)
        
       
    def _m_eff_minmax(self, m_min, element_name = "76Ge", ordering="both", dcp=1.36, 
                      numerical_method="Powell", varyLECs = False, normalize_to_mass = False, vary_WC  = "m_min"):
        #this function returns the effective majorana mass m_bb_eff
        #m_bb_eff reflects the majorana mass m_bb necessary to generate the same half-live as the input model does
        

        #the neutrino mass from the model needs to be overwritten to be able to produce the plot
        #this is because for the plot we wanto to vary m_min!
        m_bb_backup = self.WC["m_bb"]
        WC_backup = self.WC.copy()
        if vary_WC  == "m_min":
            self.WC["m_bb"] = 0
        else:
            self.WC[vary_WC ] = 0

        #some parameters
        #m_e = 5.10998e-4
        g_A = self.LEC["A"]
        V_ud = 0.97417


        G01 = self.to_G(element_name)["01"]
        M3 = self.amplitudes(element_name, self.WC)[1]["nu(3)"]

        if ordering == "NO":
            if vary_WC == "m_min":
                pre_alpha = [1,0]
            else:
                pre_alpha = 1
            #get minimal and maximal m_bb by varying phases
            NO_min = (scipy.optimize.minimize(self._m_bb, x0=pre_alpha,args=(m_min, "NO", dcp, element_name, normalize_to_mass, vary_WC), method=numerical_method)["fun"])
            NO_max = (-scipy.optimize.minimize(self._m_bb_minus, x0=pre_alpha,args=(m_min, "NO", dcp, element_name, normalize_to_mass, vary_WC), method=numerical_method)["fun"])

            self.WC["m_bb"] = NO_min*1e-9
            #NO_min_eff = self.m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)
            NO_min_eff = self.m_e / (g_A**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)

            self.WC["m_bb"] = NO_max*1e-9
            #NO_max_eff = self.m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)
            NO_max_eff = self.m_e / (g_A**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)
            
            
            self.WC["m_bb"] = m_bb_backup
            self.WC = WC_backup.copy()
            return ([NO_min_eff*1e+9, NO_max_eff*1e+9])

        elif ordering == "IO":
            if vary_WC == "m_min":
                pre_alpha = [1,0]
            else:
                pre_alpha = 1
            #get minimal and maximal m_bb by varying phases
            IO_min = (scipy.optimize.minimize(m_bb, x0=pre_alpha,args=(m_min, "IO", dcp, element_name, normalize_to_mass, vary_WC), method=numerical_method)["fun"])
            IO_max = (-scipy.optimize.minimize(m_bb_minus, x0=pre_alpha,args=(m_min, "IO", dcp, element_name, normalize_to_mass, vary_WC), method=numerical_method)["fun"])

            self.WC["m_bb"] = IO_min*1e-9
            #IO_min_eff = self.m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)

            IO_min_eff = self.m_e / (g_A**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)


            self.WC["m_bb"] = IO_max*1e-9
            #IO_max_eff = self.m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)
            IO_max_eff = self.m_e / (g_A**2*M3*G01**(1/2)) * self.t_half(element_name)**(-1/2)
            
            
            self.WC["m_bb"] = m_bb_backup
            self.WC = WC_backup.copy()
            return ([IO_min_eff*1e+9, IO_max_eff*1e+9])

        else:
            #get minimal and maximal m_bb by varying phases
            if vary_WC == "m_min":
                pre_alpha = [1,0]
            else:
                pre_alpha = 1
            NO_min_eff = (scipy.optimize.minimize(self._m_eff, x0=pre_alpha,args=(m_min, "NO", dcp, element_name, normalize_to_mass, vary_WC), method=numerical_method)["fun"])
            NO_max_eff = (-scipy.optimize.minimize(self._m_eff_minus, x0=pre_alpha,args=(m_min, "NO", dcp, element_name, normalize_to_mass, vary_WC), method=numerical_method)["fun"])
            if vary_WC  == "m_min":
                IO_min_eff = (scipy.optimize.minimize(self._m_eff, x0=pre_alpha,args=(m_min, "IO", dcp, element_name, normalize_to_mass, vary_WC), method=numerical_method)["fun"])
                IO_max_eff = (-scipy.optimize.minimize(self._m_eff_minus, x0=pre_alpha,args=(m_min, "IO", dcp, element_name, normalize_to_mass, vary_WC), method=numerical_method)["fun"])
            
            
            self.WC["m_bb"] = m_bb_backup
            self.WC = WC_backup.copy()
            
            if normalize_to_mass:
                if vary_WC  == "m_min":
                    return ([NO_min_eff, NO_max_eff], [IO_min_eff, IO_max_eff]) 
                else:
                    return ([NO_min_eff, NO_max_eff]) 
                    
            else:
                #return in eV
                if vary_WC  == "m_min":
                    return ([NO_min_eff*1e+9, NO_max_eff*1e+9], [IO_min_eff*1e+9, IO_max_eff*1e+9]) 
                else:
                    return ([NO_min_eff*1e+9, NO_max_eff*1e+9])
        
    def plot_m_eff(self, m_cosmo = 0.15, x_min = 1e-4, x_max = 1e+0, y_min=None, y_max=None, n_dots = 100, 
                   element_name = "76Ge", cosmo = True, experiments = None, ordering="both", savefig=False, 
                   numerical_method="Powell", compare_to_mass = False, normalize_to_mass = False, vary_WC  = "m_min"):#, varyLECs = False):
        
        
        M = np.logspace(int(np.log10(x_min)),int(np.log10(x_max)), n_dots)

        NO_min = np.zeros(n_dots)
        NO_max = np.zeros(n_dots)
        IO_min = np.zeros(n_dots)
        IO_max = np.zeros(n_dots)
        
        if vary_WC not in ["m_bb", "m_min"] and compare_to_mass:
            print("comparing to mass mechanism only makes sense if you put either the minimal neutrino mass or m_bb on the x axis. Setting compare_to_mass = False")
            compare_to_mass = False
        if compare_to_mass or normalize_to_mass:
            NO_min_mbb = np.zeros(n_dots)
            NO_max_mbb = np.zeros(n_dots)
            IO_min_mbb = np.zeros(n_dots)
            IO_max_mbb = np.zeros(n_dots)
            

        for idx in range(n_dots):
            m_min = M[idx]
            if vary_WC  == "m_min":
                [NO_min[idx], NO_max[idx]], [IO_min[idx], IO_max[idx]] = self._m_eff_minmax(m_min, 
                                                                                        element_name, 
                                                                                        ordering=ordering, 
                                                                                        normalize_to_mass = normalize_to_mass, 
                                                                                        vary_WC  = vary_WC )#, 
                                                                                        #varyLECs = varyLECs)
                if compare_to_mass or normalize_to_mass:
                    WCbackup = self.WC.copy()
                    for operator in self.WC:
                        self.WC[operator]=0
                    [NO_min_mbb[idx], NO_max_mbb[idx]], [IO_min_mbb[idx], IO_max_mbb[idx]] = self._m_eff_minmax(m_min, 
                                                                                        element_name, 
                                                                                        normalize_to_mass = normalize_to_mass, 
                                                                                        vary_WC  = vary_WC )#, varyLECs = varyLECs)
                    self.WC = WCbackup.copy()
                    
            else:
                
                [NO_min[idx], NO_max[idx]] = self._m_eff_minmax(m_min, 
                                                                element_name, 
                                                                ordering=ordering, 
                                                                normalize_to_mass = normalize_to_mass,
                                                                vary_WC  = vary_WC )#, 
                                                                                        #varyLECs = varyLECs)
                if compare_to_mass or normalize_to_mass:
                    WCbackup = self.WC.copy()
                    for operator in self.WC:
                        self.WC[operator]=0
                    [NO_min_mbb[idx], NO_max_mbb[idx]] = self._m_eff_minmax(m_min, 
                                                                            element_name, 
                                                                            normalize_to_mass = normalize_to_mass, 
                                                                            vary_WC  = vary_WC )#, varyLECs = varyLECs)
                    self.WC = WCbackup.copy()
        
        print(NO_min)
        print(NO_max)
        print(IO_min)
        print(IO_max)
        NO_min = np.absolute(NO_min)
        NO_max = np.absolute(NO_max)
        IO_min = np.absolute(IO_min)
        IO_max = np.absolute(IO_max)
            
        if compare_to_mass or normalize_to_mass:
            NO_min_mbb = np.absolute(NO_min_mbb)
            NO_max_mbb = np.absolute(NO_max_mbb)
            IO_min_mbb = np.absolute(IO_min_mbb)
            IO_max_mbb = np.absolute(IO_max_mbb)
            #print(NO_min_mbb)
            #if normalize_to_mass:
                
                #NO_min_mbb -=1
                #NO_max_mbb -=1
                #IO_min_mbb -=1
                #IO_max_mbb -=1
                #NO_min -=1
                #NO_max -=1
                #IO_min -=1
                #IO_max -=1
                #print(NO_min)
                #print(IO_min)

        fig = plt.figure(figsize=(9,8))
        if vary_WC == "m_min":
            NO_label = "NO"
            IO_label =  "IO"
        else:
            NO_label = None
            IO_label =  None
        if ordering == "NO":
            plt.plot(M,NO_min, "b")
            plt.plot(M,NO_max, "b")
            plt.fill_between(M, NO_max, NO_min, color="b", alpha=0.5, label = NO_label)
        elif ordering == "IO" and vary_WC == "m_min":
            plt.plot(M,IO_min, "r")
            plt.plot(M,IO_max, "r")
            plt.fill_between(M, IO_max, IO_min, color="r", alpha=0.5, label = IO_label)
        else:
            plt.plot(M,NO_min, "b")
            plt.plot(M,NO_max, "b")
            plt.plot(M,IO_min, "r")
            plt.plot(M,IO_max, "r")
            plt.fill_between(M, NO_max, NO_min, color="b", alpha=0.5, label = NO_label)
            if vary_WC == "m_min":
                plt.fill_between(M, IO_max, IO_min, color="r", alpha=0.5, label = IO_label)
        
        if compare_to_mass:
            if ordering == "NO":
                plt.plot(M,NO_min_mbb, "grey")
                plt.plot(M,NO_max_mbb, "grey")
                plt.fill_between(M, NO_max_mbb, NO_min_mbb, color="k", alpha=0.1, label = r"$m_{\beta\beta}$")
                
            elif ordering == "IO" and vary_WC == "m_min":
                plt.plot(M,IO_min_mbb, "grey")
                plt.plot(M,IO_max_mbb, "grey")
                plt.fill_between(M, IO_max_mbb, IO_min_mbb, color="k", alpha=0.1, label=r"$m_{\beta\beta}$")
            else:
                plt.plot(M,NO_min_mbb, "grey")
                plt.plot(M,NO_max_mbb, "grey")
                plt.plot(M,IO_min_mbb, "grey")
                plt.plot(M,IO_max_mbb, "grey")
                plt.fill_between(M, IO_max_mbb, IO_min_mbb, color="k", alpha=0.1, label=r"$m_{\beta\beta}$")
                plt.fill_between(M, NO_max_mbb, NO_min_mbb, color="k", alpha=0.1)

        
        if y_max == None:
            if vary_WC == "m_min":
                y_max = np.max([np.max(IO_min), np.max(NO_min)])
            else:
                y_max = np.max(NO_min)
            
            if normalize_to_mass:
                y_min = 1e-4
                y_max = 1e+4
                
            y_max = 10**np.ceil(np.log10(y_max))
            print("setting ymax")
        if y_min == None:
            y_min = 1e-4*y_max
            #y_min = 10**np.round(np.log10(y_min))
    
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(y_min,y_max)
        plt.xlim(x_min,x_max)
        if vary_WC == "m_min":
            plt.legend(fontsize=20)
        if normalize_to_mass:
            plt.ylabel(r"$\left|\frac{m_{\beta\beta}^{eff}}{m_{\beta\beta}}\right|$", fontsize=20)
        else:
            plt.ylabel(r"$|m_{\beta\beta}^{eff}|$ [eV]", fontsize=20)
        if vary_WC == "m_min":
            plt.xlabel(r"$m_{min}$ [eV]", fontsize=20)
        else:
            if vary_WC == "m_bb":
                plt.xlabel(r"$|m_{\beta\beta}|$ [eV]", fontsize=20)
            else:
                plt.xlabel(r"$|C_{"+vary_WC[:-3]+"}^{"+vary_WC[-3:]+"}|$", fontsize=20)
        plt.tight_layout()
        if cosmo:
            def m_sum(m_min):
                m21 = 7.53e-5
                m32 = 2.453e-3
                msum = m_min + np.sqrt(m_min**2+m21) + np.sqrt(m_min**2 + m21 + m32)
                return(msum-m_cosmo)
            cosmo_limit = scipy.optimize.root(m_sum, x0 = 0.05).x[0]
            #print(cosmo_limit)
            m_cosmo
            plt.fill_betweenx([y_min,y_max], [1], [cosmo_limit], alpha=0.2, color="k")
        if experiments != None:
            for experiment in experiments:
                plt.axhline(experiments[experiment], linewidth = 1)
                plt.text(x = 1e-3, y = experiments[experiment], s = experiment)
        if savefig:
            plt.savefig("m_eff.png", dpi=300)
        return(fig)
    
    def _t_half(self, alpha, m_min, ordering="NO", dcp=1.36, 
              element_name="76Ge", normalize_to_mass = False):
        m_bb_backup = self.WC["m_bb"]
        self.WC["m_bb"] = self._m_bb(alpha=alpha, m_min=m_min, ordering=ordering, dcp=dcp)*1e-9
        t_half = self.t_half(element_name=element_name)
        if normalize_to_mass:
            WCbackup = self.WC.copy()
            for operator in self.WC:
                if operator != "m_bb":
                    self.WC[operator] = 0
            t_half_mbb = self.t_half(element_name=element_name)
            t_half/=t_half_mbb
            self.WC = WCbackup.copy()
        self.WC["m_bb"]=m_bb_backup
        return(t_half)
    def _t_half_minus(self, alpha, m_min, ordering="NO", dcp=1.36, 
              element_name="76Ge", normalize_to_mass = False):
        m_bb_backup = self.WC["m_bb"]
        self.WC["m_bb"] =self._m_bb(alpha=alpha, m_min=m_min, ordering=ordering, dcp=dcp)*1e-9
        t_half = self.t_half(element_name=element_name)
        if normalize_to_mass:
            WCbackup = self.WC.copy()
            for operator in self.WC:
                if operator != "m_bb":
                    self.WC[operator] = 0
            t_half_mbb = self.t_half(element_name=element_name)
            t_half/=t_half_mbb
            self.WC = WCbackup.copy()
        self.WC["m_bb"]=m_bb_backup
        return(-t_half)
    def _t_half_minmax(self, m_min, ordering="both", dcp=1.36, element_name="76Ge", numerical_method="powell", tol=None, normalize_to_mass = False):
        if ordering == "NO":
            t_half_min_NO = (scipy.optimize.minimize(self._t_half, x0=[1,0], args=(m_min, ordering, dcp, element_name, normalize_to_mass), method=numerical_method, tol=tol)["fun"])
            t_half_max_NO = (scipy.optimize.minimize(self._t_half_minus, x0=[1,0], args=(m_min, ordering, dcp, element_name, normalize_to_mass), method=numerical_method, tol=tol)["fun"])
            return([t_half_min_NO, t_half_max_NO])

        elif ordering == "IO":
            t_half_min_IO = (scipy.optimize.minimize(self._t_half, x0=[1,0], args=(m_min, ordering, dcp, element_name, normalize_to_mass), method=numerical_method, tol=tol)["fun"])
            t_half_max_IO = (scipy.optimize.minimize(self._t_half_minus, x0=[1,0], args=(m_min, ordering, dcp, element_name, normalize_to_mass), method=numerical_method, tol=tol)["fun"])
            return([t_half_min_IO, t_half_max_IO])

        else:
            t_half_min_NO = (scipy.optimize.minimize(self._t_half, x0=[1,0], args=(m_min, "NO", dcp, element_name, normalize_to_mass), method=numerical_method, tol=tol)["fun"])
            t_half_max_NO = (scipy.optimize.minimize(self._t_half_minus, x0=[1,0], args=(m_min, "NO", dcp, element_name, normalize_to_mass), method=numerical_method, tol=tol)["fun"])

            t_half_min_IO = (scipy.optimize.minimize(self._t_half, x0=[1,0], args=(m_min, "IO", dcp, element_name, normalize_to_mass), method=numerical_method, tol=tol)["fun"])
            t_half_max_IO = (scipy.optimize.minimize(self._t_half_minus, x0=[1,0], args=(m_min, "IO", dcp, element_name, normalize_to_mass), method=numerical_method, tol=tol)["fun"])
            return([t_half_min_NO, t_half_max_NO], [t_half_min_IO, t_half_max_IO])
        
    def plot_t_half_inv(self, m_cosmo = 0.15, x_min = 1e-4, x_max = 1e+0, y_min=None, y_max=None, n_dots = 100, 
                   element_name = "76Ge", cosmo = True, experiments = None, ordering="both", savefig=True, dcp=1.36,
                    numerical_method="Powell", compare_to_mass = False, normalize_to_mass = False, vary_WC = "m_min"):
        #model=EFT.LEFT(WC)
        M = np.logspace(int(np.log10(x_min)),int(np.log10(x_max)), n_dots)

        NO_min = np.zeros(n_dots)
        NO_max = np.zeros(n_dots)
        IO_min = np.zeros(n_dots)
        IO_max = np.zeros(n_dots)
        if vary_WC not in ["m_bb", "m_min"] and compare_to_mass:
            print("comparing to mass mechanism only makes sense if you put either the minimal neutrino mass or m_bb on the x axis. Setting compare_to_mass = False")
            compare_to_mass = False
        
        if compare_to_mass or normalize_to_mass:
            NO_min_mbb = np.zeros(n_dots)
            NO_max_mbb = np.zeros(n_dots)
            IO_min_mbb = np.zeros(n_dots)
            IO_max_mbb = np.zeros(n_dots)

        for idx in range(n_dots):
            m_min = M[idx]
            [NO_min[idx], NO_max[idx]], [IO_min[idx], IO_max[idx]] = self._t_half_minmax(m_min=m_min, 
                                                                                  element_name=element_name, ordering=ordering, 
                                                                                   dcp=dcp, numerical_method=numerical_method, 
                                                                                         normalize_to_mass = normalize_to_mass)
            if compare_to_mass:
                WCbackup = self.WC.copy()
                for operator in self.WC:
                    self.WC[operator] = 0
                    
                [NO_min_mbb[idx], NO_max_mbb[idx]], [IO_min_mbb[idx], IO_max_mbb[idx]] = self._t_half_minmax(m_min=m_min, 
                                                                                      element_name=element_name, 
                                                                                                             ordering=ordering, 
                                                                                                             dcp=dcp,
                                                                                                              numerical_method=numerical_method, 
                                                                                         normalize_to_mass = normalize_to_mass)
                
                self.WC = WCbackup.copy()

        NO_min = 1/np.absolute(NO_min)
        NO_max = 1/np.absolute(NO_max)
        IO_min = 1/np.absolute(IO_min)
        IO_max = 1/np.absolute(IO_max)
            
        if compare_to_mass or normalize_to_mass:
            NO_min_mbb = 1/np.absolute(NO_min_mbb)
            NO_max_mbb = 1/np.absolute(NO_max_mbb)
            IO_min_mbb = 1/np.absolute(IO_min_mbb)
            IO_max_mbb = 1/np.absolute(IO_max_mbb)
            #if normalize_to_mass:
            #    
            #    NO_min_mbb -=1
            #    NO_max_mbb -=1
            #    IO_min_mbb -=1
            #    IO_max_mbb -=1
            #    NO_min -=1
            #    NO_max -=1
            #    IO_min -=1
            #    IO_max -=1
        if y_max == None:
            y_max = np.max([np.max(IO_max), np.max(NO_max)])
            y_max = 10**np.ceil(np.log10(y_max))
        if y_min == None:
            y_min = 1e-7*y_max
            #y_min = 10**np.round(np.log10(y_min))

        fig = plt.figure(figsize=(9,8))
        plt.plot(M,NO_min, "b")
        plt.plot(M,NO_max, "b")

        plt.plot(M,IO_min, "r")
        plt.plot(M,IO_max, "r")

        plt.fill_between(M, IO_max, IO_min, color="r", alpha=0.5, label="IO")
        plt.fill_between(M, NO_max, NO_min, color="b", alpha=0.5, label="NO")
        
        if compare_to_mass:
            
            plt.plot(M,NO_min_mbb, "grey")
            plt.plot(M,NO_max_mbb, "grey")

            plt.plot(M,IO_min_mbb, "grey")
            plt.plot(M,IO_max_mbb, "grey")

            plt.fill_between(M, IO_max_mbb, IO_min_mbb, color="k", alpha=0.1, label=r"$m_{\beta\beta}$")
            plt.fill_between(M, NO_max_mbb, NO_min_mbb, color="k", alpha=0.1)
        
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(y_min,y_max)
        plt.xlim(x_min,x_max)
        plt.legend(fontsize=20)
        if normalize_to_mass:
            plt.ylabel(r"$\frac{t_{1/2, m_{\beta\beta}}}{t_{1/2}}$", fontsize=20)
        else:
            plt.ylabel(r"$t_{1/2}^{-1}$ [yr$^{-1}$]", fontsize=20)
        plt.xlabel(r"$m_{min}$ [eV]", fontsize=20)
        plt.tight_layout()
        if cosmo:
            def m_sum(m_min):
                m21 = 7.53e-5
                m32 = 2.453e-3
                msum = m_min + np.sqrt(m_min**2+m21) + np.sqrt(m_min**2 + m21 + m32)
                return(msum-m_cosmo)
            cosmo_limit = scipy.optimize.root(m_sum, x0 = 0.05).x[0]
            #print(cosmo_limit)
            m_cosmo
            plt.fill_betweenx([y_min, y_max], [1], [cosmo_limit], alpha=0.2, color="k")
        #if experiments != None:
        #    for experiment in experiments:
        #        plt.axhline(experiments[experiment], linewidth = 1)
        #        plt.text(x = 1e-3, y = experiments[experiment], s = experiment)
        if savefig:
            plt.savefig("t_half_inv.png", dpi=300)
        return(fig)
    
    def plot_t_half(self, m_cosmo = 0.15, x_min = 1e-4, x_max = 1e+0, y_min=None, y_max=None, n_dots = 100, 
                   element_name = "76Ge", cosmo = True, experiments = None, ordering="both", savefig=False, dcp=1.36,
                    numerical_method="Powell", compare_to_mass = False, normalize_to_mass = False, vary_WC = "m_min"):

        M = np.logspace(int(np.log10(x_min)),int(np.log10(x_max)), n_dots)

        NO_min = np.zeros(n_dots)
        NO_max = np.zeros(n_dots)
        IO_min = np.zeros(n_dots)
        IO_max = np.zeros(n_dots)
        
        if vary_WC not in ["m_bb", "m_min"] and compare_to_mass:
            print("comparing to mass mechanism only makes sense if you put either the minimal neutrino mass or m_bb on the x axis. Setting compare_to_mass = False")
            compare_to_mass = False
        if compare_to_mass or normalize_to_mass:
            NO_min_mbb = np.zeros(n_dots)
            NO_max_mbb = np.zeros(n_dots)
            IO_min_mbb = np.zeros(n_dots)
            IO_max_mbb = np.zeros(n_dots)

        for idx in range(n_dots):
            m_min = M[idx]
            [NO_min[idx], NO_max[idx]], [IO_min[idx], IO_max[idx]] = self._t_half_minmax(m_min=m_min, 
                                                                                  element_name=element_name, ordering=ordering, 
                                                                                   dcp=dcp, numerical_method=numerical_method,
                                                                                         normalize_to_mass = normalize_to_mass)
            if compare_to_mass:
                WCbackup = self.WC.copy()
                for operator in self.WC:
                    self.WC[operator] = 0
                    
                [NO_min_mbb[idx], NO_max_mbb[idx]], [IO_min_mbb[idx], IO_max_mbb[idx]] = self._t_half_minmax(m_min=m_min, 
                                                                                      element_name=element_name, ordering=ordering, 
                                                                                       dcp=dcp, numerical_method=numerical_method, normalize_to_mass = normalize_to_mass)
                
                self.WC = WCbackup.copy()

        NO_min = np.absolute(NO_min)
        NO_max = np.absolute(NO_max)
        IO_min = np.absolute(IO_min)
        IO_max = np.absolute(IO_max)
            
        if compare_to_mass or normalize_to_mass:
            NO_min_mbb = np.absolute(NO_min_mbb)
            NO_max_mbb = np.absolute(NO_max_mbb)
            IO_min_mbb = np.absolute(IO_min_mbb)
            IO_max_mbb = np.absolute(IO_max_mbb)
            #if normalize_to_mass:
            #    
            #    NO_min_mbb -=1
            #    NO_max_mbb -=1
            #    IO_min_mbb -=1
            #    IO_max_mbb -=1
            #    NO_min -=1
            #    NO_max -=1
            #    IO_min -=1
            #    IO_max -=1
        if y_min == None:
            y_min = np.min([np.min(IO_max), np.min(NO_max)])
            #y_min = 1e-7*y_max
            y_min = 10**np.floor(np.log10(y_min))
        if y_max == None:
            #y_max = np.max([np.max(IO_max), np.max(NO_max)])
            #y_max = 10**np.ceil(np.log10(y_max))
            y_max = 1e+7*y_min

        fig = plt.figure(figsize=(9,8))
        plt.plot(M,NO_min, "b")
        plt.plot(M,NO_max, "b")

        plt.plot(M,IO_min, "r")
        plt.plot(M,IO_max, "r")

        plt.fill_between(M, IO_max, IO_min, color="r", alpha=0.5, label="IO")
        plt.fill_between(M, NO_max, NO_min, color="b", alpha=0.5, label="NO")
        
        
        if compare_to_mass:
            
            plt.plot(M,NO_min_mbb, "grey")
            plt.plot(M,NO_max_mbb, "grey")

            plt.plot(M,IO_min_mbb, "grey")
            plt.plot(M,IO_max_mbb, "grey")

            plt.fill_between(M, IO_max_mbb, IO_min_mbb, color="k", alpha=0.1, label=r"$m_{\beta\beta}$")
            plt.fill_between(M, NO_max_mbb, NO_min_mbb, color="k", alpha=0.1)

        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(y_min,y_max)
        plt.xlim(x_min,x_max)
        plt.legend(fontsize=20)
        if normalize_to_mass:
            plt.ylabel(r"$\frac{t_{1/2}}{t_{1/2, m_{\beta\beta}}}$", fontsize=20)
        else:
            plt.ylabel(r"$t_{1/2}$ [yr]", fontsize=20)
        plt.xlabel(r"$m_{min}$ [eV]", fontsize=20)
        plt.tight_layout()
        if cosmo:
            def m_sum(m_min):
                m21 = 7.53e-5
                m32 = 2.453e-3
                msum = m_min + np.sqrt(m_min**2+m21) + np.sqrt(m_min**2 + m21 + m32)
                return(msum-m_cosmo)
            cosmo_limit = scipy.optimize.root(m_sum, x0 = 0.05).x[0]
            #print(cosmo_limit)
            #m_cosmo
            plt.fill_betweenx([y_min, y_max], [1], [cosmo_limit], alpha=0.2, color="k")
        #if experiments != None:
        #    for experiment in experiments:
        #        plt.axhline(experiments[experiment], linewidth = 1)
        #        plt.text(x = 1e-3, y = experiments[experiment], s = experiment)
        if savefig:
            plt.savefig("t_half.png", dpi=300)
        return(fig)
    
    def plot_limits(self, experiments, method="IBM2", plot_groups=True, savefig=False, plottype="scales"):
        #model = EFT.LEFT({}, method=method)
        limits = {}
        scales = {}
        for element in experiments:
            limits[element], scales[element] = self.get_limits(experiments[element], element_name = element)
                    

        scales = pd.DataFrame(scales)
        limits = pd.DataFrame(limits)
        #for element in experiments:
        #   plt.bar(list(scales[element].keys()), scales[element].values())
        #plt.figure(figsize=(16,8))
        if plot_groups:
            if plottype == "scales":
                #some dim9 operators scale differently because they can be generated at smeft dim7
                WC_operator_groups = ["m_bb" , "VL(6)", 
                                      "VR(6)", "T(6)", 
                                      "SL(6)", "VL(7)", 
                                      "1L(9)", "1R(9)",
                                      "2L(9)", "3L(9)", 
                                      "4L(9)", "4R(9)",
                                      "5L(9)", "5R(9)", 
                                      "6(9)", "7(9)"]
                
                #define labels for plots
                WC_group_names = [r"$m_{\beta\beta}$"      , r"$C_{VL}^{(6)}$", 
                                  r"$C_{VR}^{(6)}$", r"$C_{T}^{(6)}$" , 
                                  r"$C_{S6}}$", r"$C_{V7}$"     , r"$C_{1L}^{(9)}$", r"$C_{S1}^{(9)}$", 
                                  r"$C_{S2}^{(9)}$", r"$C_{S3}^{(9)}$", r"$C_{4L}^{(9)}$",
                                  r"$C_{S4}^{(9)}$", r"$C_{5L}^{(9)}$", r"$C_{S5}^{(9)}$",
                                  r"$C_{V}^{(9)}$", r"$C_{\tilde{V}}^{(9)}$"]
            else:
                
                WC_operator_groups = ["m_bb" , "VL(6)", 
                                      "VR(6)", "T(6)", 
                                      "SL(6)", "VL(7)", "1L(9)", 
                                      "2L(9)", "3L(9)", 
                                      "4L(9)", "5L(9)", 
                                      "6(9)", "7(9)"]

                #define labels for plots
                WC_group_names = [r"$m_{\beta\beta}$"      , r"$C_{VL}^{(6)}$", 
                                  r"$C_{VR}^{(6)}$", r"$C_{T}^{(6)}$" , 
                                  r"$C_{S6}}$", r"$C_{V7}$"     , r"$C_{S1}^{(9)}$", 
                                  r"$C_{S2}^{(9)}$", r"$C_{S3}^{(9)}$", 
                                  r"$C_{S4}^{(9)}$", r"$C_{S5}^{(9)}$",
                                  r"$C_{V}^{(9)}$", r"$C_{\tilde{V}}^{(9)}$"]

            idx = 0
            for operator in scales.T:
                if operator not in WC_operator_groups or operator == "m_bb":
                    scales.drop(operator, axis=0, inplace=True)
                    limits.drop(operator, axis=0, inplace=True)
                else:
                    idx+=1
                    scales.rename(index={operator:WC_group_names[idx]}, inplace=True)
                    limits.rename(index={operator:WC_group_names[idx]}, inplace=True)
        
        
        if plottype == "scales":
            fig = scales.plot.bar(figsize=(16,6))
            fig.set_ylabel(r"$\Lambda$ [TeV]", fontsize =14)
        else:
            fig = limits.plot.bar(figsize=(16,6))
            fig.set_ylabel(r"$C_X$", fontsize =14)
        fig.set_yscale("log")
        fig.set_xticklabels(WC_group_names[1:], fontsize=14)
        fig.grid(linestyle="--")
        if len(experiments)>10:
            ncol = int(len(experiments)/2)
        else:
            ncol = len(experiments)
        fig.legend(fontsize=12, loc = (0,1.02), ncol=ncol)
        #fig.figsize*16,9
        #return(pd.DataFrame(scales))
        if savefig:
            if plottype == "scales":
                fig.get_figure().savefig("scale_limits.png")
            elif plottype == "limits":
                fig.get_figure().savefig("WC_limits.png")
                
        return(fig.get_figure())
    
    def plot_m_eff_scatter(self, vary_WC = "m_min", vary_phases = True, element_name="76Ge", 
                           vary_LECs=False, n_dots=10000, ordering = "both", 
                           save=False, file="m_eff_scatter.png", alpha_plot=1, 
                           x_min=1e-4, x_max = 1, y_min = None, y_max = None, 
                           cosmo = True, m_cosmo = 0.15, compare_to_mass = False, normalize_to_mass = False):
        #model = EFT.LEFT(WC)
        #m_N = 0.93
        LECs = { 'Tprime': 1,
                 'Tpipi': 1,
                 '1piN': 1,
                 '6piN': 1,
                 '7piN': 1,
                 '8piN': 1,
                 '9piN': 1,
                 'VLpiN': 1,
                 'TpiN': 1,
                 '1NN': 1,
                 '6NN': 1,
                 '7NN': 1,
                 'VLNN': 1,
                 'TNN': 1,
                 'VLE': 1,
                 'VLme': 1,
                 'VRE': 1,
                 'VRme': 1,
                 '2NN': 157.91367041742973,
                 '3NN': 157.91367041742973,
                 '4NN': 157.91367041742973,
                 '5NN': 157.91367041742973,
                 'nuNN': -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
               }
        
        if vary_WC not in ["m_bb", "m_min"] and compare_to_mass:
            print("comparing to mass mechanism only makes sense if you put either the minimal neutrino mass or m_bb on the x axis. Setting compare_to_mass = False")
            compare_to_mass = False
            
        if compare_to_mass:
            M_mbb = np.logspace(int(np.log10(x_min)),int(np.log10(x_max)), 100)
            NO_min_mbb = np.zeros(100)
            NO_max_mbb = np.zeros(100)
            IO_min_mbb = np.zeros(100)
            IO_max_mbb = np.zeros(100)
            WCbackup = self.WC.copy()
            for operator in self.WC:
                self.WC[operator]=0
            for idx in range(100):
                m_min = M_mbb[idx]
                [NO_min_mbb[idx], NO_max_mbb[idx]], [IO_min_mbb[idx], IO_max_mbb[idx]] = self._m_eff_minmax(m_min, 
                                                                              element_name, normalize_to_mass = normalize_to_mass)#, varyLECs = varyLECs)
            self.WC = WCbackup.copy()
            
            NO_min_mbb = np.absolute(NO_min_mbb)
            NO_max_mbb = np.absolute(NO_max_mbb)
            IO_min_mbb = np.absolute(IO_min_mbb)
            IO_max_mbb = np.absolute(IO_max_mbb)
            
            
            
            
        G01 = self.to_G(element_name)["01"]
        #m_e = 5.10998e-4
        g_A = self.LEC["A"]
        V_ud = 0.97417
        #mspace = np.logspace(-4, 1, 1000000)
        fig = plt.figure(figsize=(9, 8))
        points = np.zeros((n_dots,2))
        pointsIO = np.zeros((n_dots,2))
        mspace = np.logspace(np.log10(x_min), np.log10(x_max), int(10*n_dots))
        forbidden_LECsNO = []
        forbidden_LECsNOm = []
        forbidden_LECsIO = []
        forbidden_LECsIOm = []
        if vary_WC != "m_min":
            WC_backup = self.WC[vary_WC]
        m_backup = self.WC["m_bb"]
        LEC_backup = self.LEC.copy()
        for x in range(n_dots):
            if vary_WC == "m_min":
                m_min = np.random.choice(mspace)
            #for n_m_min in range(10):
            #    for n_LECs in range(10):
            #        for n_alpha in range(10):
                alpha = np.pi*np.random.random(2)
                m = self._m_bb(alpha, m_min, "NO")*1e-9
                mIO = self._m_bb(alpha, m_min, "IO")*1e-9
                #model_standard.WC["m_bb"] = m
                self.WC["m_bb"] = m
            else:
                if vary_WC == "m_bb":
                    self.WC[vary_WC] = np.random.choice(mspace)*1e-9
                else:
                    self.WC[vary_WC] = np.random.choice(mspace)
                if vary_phases:
                    alpha = np.pi*np.random.rand()
                    self.WC[vary_WC] *= np.exp(1j*alpha)
            if vary_LECs:
                for LEC in LECs:
                    if LEC == "nuNN":
                        random_LEC = (np.random.rand()+0.5)*LECs[LEC]
                        self.LEC[LEC] = random_LEC
                    else:
                        #random_LEC = variation_range*2*(np.random.rand()-0.5)*LECs[LEC]
                        random_LEC = np.random.choice([1,-1])*((np.sqrt(10)-1/np.sqrt(10))*np.random.rand()+1/np.sqrt(10))*LECs[LEC]
                        self.LEC[LEC] = random_LEC

            t = self.t_half(element_name)
            M3 = self.amplitudes(element_name, self.WC)[1]["nu(3)"]
            #m_eff = np.absolute(self.m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * t**(-1/2))*1e+9
            m_eff = np.absolute(self.m_e / (g_A**2*M3*G01**(1/2)) * t**(-1/2))*1e+9
            #if t < 1e+26:
            #    forbidden_LECsNO.append(model.LEC["nuNN"])
            #    forbidden_LECsNOm.append(m_min)
            if vary_WC == "m_min":
                self.WC["m_bb"] = mIO
                #for LEC in LECs:
                #    random_LEC = 3*2*(np.random.rand()-0.5)*LECs[LEC]
                #    model.LEC[LEC] = random_LEC


                t = self.t_half(element_name)
                #if t < 1e+26:
                #    forbidden_LECsIO.append(model.LEC["nuNN"])
                #    forbidden_LECsIOm.append(m_min)
                M3 = self.amplitudes(element_name, self.WC)[1]["nu(3)"]
                #m_effIO = np.absolute(self.m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * t**(-1/2))*1e+9
                m_effIO = np.absolute(self.m_e / (g_A**2*M3*G01**(1/2)) * t**(-1/2))*1e+9
                pointsIO[x][0] = m_min
                pointsIO[x][1] = m_effIO
                points[x][0] = m_min
                if normalize_to_mass:
                    points[x][1]/=self.WC["m_bb"]
                    pointsIO[x][1] /= self.WC["m_bb"]
                    
            else:
                if vary_WC == "m_bb":
                    points[x][0] = np.absolute(self.WC[vary_WC])*1e+9
                else:
                    points[x][0] = np.absolute(self.WC[vary_WC])
            points[x][1] = m_eff
            
            

        self.WC["m_bb"] = m_backup
        self.LEC = LEC_backup.copy()
        if vary_WC != "m_min":
            self.WC[vary_WC] = WC_backup
        if y_min == None:
            y_min = np.min([np.min(points[:,1]), np.min(pointsIO[:,1])])
        if y_max == None:
            y_max = np.max([np.max(points[:,1]), np.max(pointsIO[:,1])])
        if cosmo:
            def m_sum(m_min):
                m21 = 7.53e-5
                m32 = 2.453e-3
                msum = m_min + np.sqrt(m_min**2+m21) + np.sqrt(m_min**2 + m21 + m32)
                return(msum-m_cosmo)
            cosmo_limit = scipy.optimize.root(m_sum, x0 = 0.05).x[0]
            print(cosmo_limit)
            
            #m_cosmo
            plt.fill_betweenx([y_min, y_max], [1], [cosmo_limit], alpha=0.1, color="k")
        if vary_WC == "m_min":   
            plt.plot(points[:,0],points[:,1], "b.", alpha = alpha_plot, label="NO", markersize = 0.15)
            plt.plot(pointsIO[:,0],pointsIO[:,1], "r.", alpha = alpha_plot, label="IO", markersize = 0.15)
        else:
            plt.plot(points[:,0],points[:,1], "b,", alpha = alpha_plot, label=vary_WC)
        plt.yscale("log")
        plt.xscale("log")
        plt.ylabel(r"$|m_{\beta\beta}^{eff}|$ [eV]", fontsize=20)
        if vary_WC == "m_min":
            plt.xlabel(r"$m_{min}$ [eV]", fontsize=20)
        else:
            if vary_WC == "m_bb":
                plt.xlabel(r"$|m_{\beta\beta}|$ [eV]", fontsize=20)
            else:
                plt.xlabel(r"$|C_{"+vary_WC[:-3]+"}^{"+vary_WC[-3:]+"}|$", fontsize=20)
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        if compare_to_mass:
            if ordering == "NO":
                plt.plot(M_mbb,NO_min_mbb, "grey")
                plt.plot(M_mbb,NO_max_mbb, "grey")
                plt.fill_between(M_mbb, NO_max_mbb, NO_min_mbb, color="k", alpha=0.1, label = r"$m_{\beta\beta}$")
                
            elif ordering == "IO":
                plt.plot(M_mbb,IO_min_mbb, "grey")
                plt.plot(M_mbb,IO_max_mbb, "grey")
                plt.fill_between(M_mbb, IO_max_mbb, IO_min_mbb, color="k", alpha=0.1, label=r"$m_{\beta\beta}$")
            else:
                plt.plot(M_mbb,NO_min_mbb, "grey")
                plt.plot(M_mbb,NO_max_mbb, "grey")
                plt.plot(M_mbb,IO_min_mbb, "grey")
                plt.plot(M_mbb,IO_max_mbb, "grey")
                plt.fill_between(M_mbb, IO_max_mbb, IO_min_mbb, color="k", alpha=0.1, label=r"$m_{\beta\beta}$")
                plt.fill_between(M_mbb, NO_max_mbb, NO_min_mbb, color="k", alpha=0.1)
        if vary_WC == "m_min":
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='NO',
                                  markerfacecolor='b', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='IO',
                                  markerfacecolor='r', markersize=10)]
            plt.legend(handles = legend_elements, fontsize=20)
        if save:
            plt.savefig(file)
        return(fig)
        #return(forbidden_LECsNO, forbidden_LECsNOm, forbidden_LECsIO, forbidden_LECsIOm)
        
    def plot_t_half_scatter(self, vary_WC = "m_min", vary_phases = True, vary_LECs=False, experiments=None, n_points=10000, 
                            save = False, file="t_half_scatter.png", alpha_plot=1, element_name = "76Ge", ordering = None, 
                            x_min=1e-4, x_max = 1, y_min = None, y_max = None, cosmo=True, m_cosmo  = 0.15, 
                            compare_to_mass = False, normalize_to_mass = False):
        #model = EFT.LEFT(WC)
        #m_N = 0.93
        #element_name = "76Ge"
        if vary_WC not in ["m_bb", "m_min"] and compare_to_mass:
            print("comparing to mass mechanism only makes sense if you put either the minimal neutrino mass or m_bb on the x axis. Setting compare_to_mass = False")
            compare_to_mass = False
        LECs = { 'Tprime': 1,
                 'Tpipi': 1,
                 '1piN': 1,
                 '6piN': 1,
                 '7piN': 1,
                 '8piN': 1,
                 '9piN': 1,
                 'VLpiN': 1,
                 'TpiN': 1,
                 '1NN': 1,
                 '6NN': 1,
                 '7NN': 1,
                 'VLNN': 1,
                 'TNN': 1,
                 'VLE': 1,
                 'VLme': 1,
                 'VRE': 1,
                 'VRme': 1,
                 '2NN': 157.91367041742973,
                 '3NN': 157.91367041742973,
                 '4NN': 157.91367041742973,
                 '5NN': 157.91367041742973,
                 'nuNN': -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
               }
        
        if compare_to_mass:
            M_mbb = np.logspace(int(np.log10(x_min)),int(np.log10(x_max)), 100)
            NO_min_mbb = np.zeros(100)
            NO_max_mbb = np.zeros(100)
            IO_min_mbb = np.zeros(100)
            IO_max_mbb = np.zeros(100)
            WCbackup = self.WC.copy()
            for operator in self.WC:
                self.WC[operator]=0
            for idx in range(100):
                m_min = M_mbb[idx]
                [NO_min_mbb[idx], NO_max_mbb[idx]], [IO_min_mbb[idx], IO_max_mbb[idx]] = self._t_half_minmax(m_min, 
                                                                              element_name, normalize_to_mass = normalize_to_mass)#, varyLECs = varyLECs)
            self.WC = WCbackup.copy()
            
            NO_min_mbb = np.absolute(NO_min_mbb)
            NO_max_mbb = np.absolute(NO_max_mbb)
            IO_min_mbb = np.absolute(IO_min_mbb)
            IO_max_mbb = np.absolute(IO_max_mbb)
        G01 = self.to_G(element_name)["01"]
        #m_e = 5.10998e-4
        g_A = self.LEC["A"]
        V_ud = 0.97417
        #mspace = np.logspace(-4, 1, 1000000)
        fig = plt.figure(figsize=(9, 8))
        points = np.zeros((n_points,2))
        pointsIO = np.zeros((n_points,2))
        mspace = np.logspace(np.log10(x_min), np.log10(x_max), int(10*n_points))
        #forbidden_LECsNO = []
        #forbidden_LECsNOm = []
        #forbidden_LECsIO = []
        #forbidden_LECsIOm = []
        if vary_WC != "m_min":
            WC_backup = self.WC[vary_WC]
        m_backup = self.WC["m_bb"]
        LEC_backup = self.LEC.copy()
        for x in range(n_points):
            if vary_WC == "m_min":
                m_min = np.random.choice(mspace)
            #for n_m_min in range(10):
            #    for n_LECs in range(10):
            #        for n_alpha in range(10):
                alpha = np.pi*np.random.random(2)
                m = self._m_bb(alpha, m_min, "NO")*1e-9
                mIO = self._m_bb(alpha, m_min, "IO")*1e-9
                #model_standard.WC["m_bb"] = m
                self.WC["m_bb"] = m
            else:
                if vary_WC == "m_bb":
                    self.WC[vary_WC] = np.random.choice(mspace)*1e-9
                else:
                    self.WC[vary_WC] = np.random.choice(mspace)
                if vary_phases:
                    alpha = np.pi*np.random.rand()
                    self.WC[vary_WC] *= np.exp(1j*alpha)
            if vary_LECs == True:
                for LEC in LECs:
                    if LEC == "nuNN":
                        random_LEC = (np.random.rand()+0.5)*LECs[LEC]
                        self.LEC[LEC] = random_LEC
                    else:
                        #random_LEC = variation_range*2*(np.random.rand()-0.5)*LECs[LEC]
                        random_LEC = np.random.choice([1,-1])*((np.sqrt(10)-1/np.sqrt(10))*np.random.rand()+1/np.sqrt(10))*LECs[LEC]
                        self.LEC[LEC] = random_LEC

            t = self.t_half(element_name)
            #M3 = model.amplitudes(element_name, model.WC)[1]["nu(3)"]
            #m_eff = np.absolute(m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * t**(-1/2))*1e+9
            #if t < 1e+26:
            #    forbidden_LECsNO.append(model.LEC["nuNN"])
            #    forbidden_LECsNOm.append(m_min)
            if vary_WC == "m_min":
                self.WC["m_bb"] = mIO
                #for LEC in LECs:
                #    random_LEC = 3*2*(np.random.rand()-0.5)*LECs[LEC]
                #    model.LEC[LEC] = random_LEC


                tIO = self.t_half(element_name)
                #if t < 1e+26:
                #    forbidden_LECsIO.append(model.LEC["nuNN"])
                #    forbidden_LECsIOm.append(m_min)
                #M3 = model.amplitudes(element_name, model.WC)[1]["nu(3)"]
                #m_effIO = np.absolute(m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * t**(-1/2))*1e+9
                pointsIO[x][0] = m_min
                pointsIO[x][1] = tIO
                points[x][0] = m_min
            else:
                if vary_WC == "m_bb":
                    points[x][0] = np.absolute(self.WC[vary_WC])*1e+9
                else:
                    points[x][0] = np.absolute(self.WC[vary_WC])
            points[x][1] = t

        self.WC["m_bb"] = m_backup
        self.LEC = LEC_backup.copy()
        if vary_WC != "m_min":
            self.WC[vary_WC] = WC_backup
        if y_min == None:
            y_min = np.min([np.min(points[:,1]), np.min(pointsIO[:,1])])
        if y_max == None:
            y_max = np.max([np.max(points[:,1]), np.max(pointsIO[:,1])])
        if cosmo:
            #find the minimal neutrino mass in normal ordering that corresponds to the cosmology limit on the sum of neutrino masses
            def m_sum(m_min):
                m21 = 7.53e-5
                m32 = 2.453e-3
                msum = m_min + np.sqrt(m_min**2+m21) + np.sqrt(m_min**2 + m21 + m32)
                return(msum-m_cosmo)
            cosmo_limit = scipy.optimize.root(m_sum, x0 = 0.05).x[0]
            #print(cosmo_limit)
            #m_cosmo
            plt.fill_betweenx([y_min, y_max], [1], [cosmo_limit], alpha=0.1, color="k")
            
        if compare_to_mass:
            if ordering == "NO":
                plt.plot(M_mbb,NO_min_mbb, "grey")
                plt.plot(M_mbb,NO_max_mbb, "grey")
                plt.fill_between(M_mbb, NO_max_mbb, NO_min_mbb, color="k", alpha=0.1, label = r"$m_{\beta\beta}$")
                
            elif ordering == "IO":
                plt.plot(M_mbb,IO_min_mbb, "grey")
                plt.plot(M_mbb,IO_max_mbb, "grey")
                plt.fill_between(M_mbb, IO_max_mbb, IO_min_mbb, color="k", alpha=0.1, label=r"$m_{\beta\beta}$")
            else:
                plt.plot(M_mbb,NO_min_mbb, "grey")
                plt.plot(M_mbb,NO_max_mbb, "grey")
                plt.plot(M_mbb,IO_min_mbb, "grey")
                plt.plot(M_mbb,IO_max_mbb, "grey")
                plt.fill_between(M_mbb, IO_max_mbb, IO_min_mbb, color="k", alpha=0.1, label=r"$m_{\beta\beta}$")
                plt.fill_between(M_mbb, NO_max_mbb, NO_min_mbb, color="k", alpha=0.1)
        plt.plot(points[:,0],points[:,1], "b.", alpha = alpha_plot, markersize = 0.15)
        plt.plot(pointsIO[:,0],pointsIO[:,1], "r.", alpha = alpha_plot, markersize = 0.15)
        plt.yscale("log")
        plt.xscale("log")
        plt.ylabel(r"$t_{1/2}^{-1} [yr]$ ", fontsize=20)
        if vary_WC == "m_min":
            plt.xlabel(r"$m_{min}$ [eV]", fontsize=20)
        else:
            if vary_WC == "m_bb":
                plt.xlabel(r"$|m_{\beta\beta}|$ [eV]", fontsize=20)
            else:
                plt.xlabel(r"$|C_{"+vary_WC[:-3]+"}^{"+vary_WC[-3:]+"}|$", fontsize=20)
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        if vary_WC == "m_min":
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='NO',
                                  markerfacecolor='b', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='IO',
                                  markerfacecolor='r', markersize=10)]
            plt.legend(handles = legend_elements, fontsize=20)
        plt.tight_layout()
        if experiments != None:
            for experiment in experiments:
                plt.axhline(experiments[experiment], label )
                plt.text(x = 1e-3, y = experiments[experiment], s = experiment)
        #return(forbidden_LECsNO, forbidden_LECsNOm, forbidden_LECsIO, forbidden_LECsIOm)
        if save == True:
            plt.savefig(file)
        return(fig)
    def plot_t_half_inv_scatter(self, vary_WC = "m_min", vary_phases = True, vary_LECs=False, experiments=None, n_points=10000, 
                                save = False, file="t_half_scatter.png", alpha_plot=1, element_name = "76Ge", 
                                x_min=1e-4, x_max = 1, y_min = None, y_max = None, cosmo=True, m_cosmo = 0.15, 
                                compare_to_mass = False, normalize_to_mass = False):
        #m_N = 0.93
        if vary_WC not in ["m_bb", "m_min"] and compare_to_mass:
            print("comparing to mass mechanism only makes sense if you put either the minimal neutrino mass or m_bb on the x axis. Setting compare_to_mass = False")
            compare_to_mass = False
        #element_name = "76Ge"
        LECs = { 'Tprime': 1,
                 'Tpipi': 1,
                 '1piN': 1,
                 '6piN': 1,
                 '7piN': 1,
                 '8piN': 1,
                 '9piN': 1,
                 'VLpiN': 1,
                 'TpiN': 1,
                 '1NN': 1,
                 '6NN': 1,
                 '7NN': 1,
                 'VLNN': 1,
                 'TNN': 1,
                 'VLE': 1,
                 'VLme': 1,
                 'VRE': 1,
                 'VRme': 1,
                 '2NN': 157.91367041742973,
                 '3NN': 157.91367041742973,
                 '4NN': 157.91367041742973,
                 '5NN': 157.91367041742973,
                 'nuNN': -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6}
        
        
        if compare_to_mass:
            M_mbb = np.logspace(int(np.log10(x_min)),int(np.log10(x_max)), 100)
            NO_min_mbb = np.zeros(100)
            NO_max_mbb = np.zeros(100)
            IO_min_mbb = np.zeros(100)
            IO_max_mbb = np.zeros(100)
            WCbackup = self.WC.copy()
            for operator in self.WC:
                self.WC[operator]=0
            for idx in range(100):
                m_min = M_mbb[idx]
                [NO_min_mbb[idx], NO_max_mbb[idx]], [IO_min_mbb[idx], IO_max_mbb[idx]] = self._t_half_minmax(m_min, 
                                                                              element_name, normalize_to_mass = normalize_to_mass)#, varyLECs = varyLECs)
            self.WC = WCbackup.copy()
            
            NO_min_mbb = np.absolute(1/NO_min_mbb)
            NO_max_mbb = np.absolute(1/NO_max_mbb)
            IO_min_mbb = np.absolute(1/IO_min_mbb)
            IO_max_mbb = np.absolute(1/IO_max_mbb)
        
        
        G01 = self.to_G(element_name)["01"]
        m_backup = self.WC["m_bb"]
        LEC_backup = self.LEC.copy()
        #m_e = 5.10998e-4
        g_A = self.LEC["A"]
        V_ud = 0.97417
        #mspace = np.logspace(-4, 1, 1000000)
        fig = plt.figure(figsize=(9, 8))
        points = np.zeros((n_points,2))
        pointsIO = np.zeros((n_points,2))
        mspace = np.logspace(np.log10(x_min), np.log10(x_max), int(10*n_points))
        #forbidden_LECsNO = []
        #forbidden_LECsNOm = []
        #forbidden_LECsIO = []
        #forbidden_LECsIOm = []
        if vary_WC != "m_min":
            WC_backup = self.WC[vary_WC]
        m_backup = self.WC["m_bb"]
        LEC_backup = self.LEC.copy()
        for x in range(n_points):
            if vary_WC == "m_min":
                m_min = np.random.choice(mspace)
            #for n_m_min in range(10):
            #    for n_LECs in range(10):
            #        for n_alpha in range(10):
                alpha = np.pi*np.random.random(2)
                m = self._m_bb(alpha, m_min, "NO")*1e-9
                mIO = self._m_bb(alpha, m_min, "IO")*1e-9
                #model_standard.WC["m_bb"] = m
                self.WC["m_bb"] = m
            else:
                if vary_WC == "m_bb":
                    self.WC[vary_WC] = np.random.choice(mspace)*1e-9
                else:
                    self.WC[vary_WC] = np.random.choice(mspace)
                if vary_phases:
                    alpha = np.pi*np.random.rand()
                    self.WC[vary_WC] *= np.exp(1j*alpha)
            if vary_LECs == True:
                for LEC in LECs:
                    if LEC == "nuNN":
                        random_LEC = (np.random.rand()+0.5)*LECs[LEC]
                        self.LEC[LEC] = random_LEC
                    else:
                        #random_LEC = variation_range*2*(np.random.rand()-0.5)*LECs[LEC]
                        random_LEC = np.random.choice([1,-1])*((np.sqrt(10)-1/np.sqrt(10))*np.random.rand()+1/np.sqrt(10))*LECs[LEC]
                        self.LEC[LEC] = random_LEC

            t = self.t_half(element_name)
            #M3 = model.amplitudes(element_name, model.WC)[1]["nu(3)"]
            #m_eff = np.absolute(m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * t**(-1/2))*1e+9
            #if t < 1e+26:
            #    forbidden_LECsNO.append(model.LEC["nuNN"])
            #    forbidden_LECsNOm.append(m_min)

            if vary_WC == "m_min":
                self.WC["m_bb"] = mIO
                #for LEC in LECs:
                #    random_LEC = 3*2*(np.random.rand()-0.5)*LECs[LEC]
                #    model.LEC[LEC] = random_LEC


                tIO = self.t_half(element_name)
                #if t < 1e+26:
                #    forbidden_LECsIO.append(model.LEC["nuNN"])
                #    forbidden_LECsIOm.append(m_min)
                #M3 = model.amplitudes(element_name, model.WC)[1]["nu(3)"]
                #m_effIO = np.absolute(m_e / (g_A**2*V_ud**2*M3*G01**(1/2)) * t**(-1/2))*1e+9
                pointsIO[x][0] = m_min
                pointsIO[x][1] = 1/tIO
                points[x][0] = m_min
            else:
                if vary_WC == "m_bb":
                    points[x][0] = np.absolute(self.WC[vary_WC])*1e+9
                else:
                    points[x][0] = np.absolute(self.WC[vary_WC])
            points[x][1] = 1/t
        
        self.WC["m_bb"] = m_backup
        self.LEC = LEC_backup.copy()
        if vary_WC == "m_min":
            plt.xlabel(r"$m_{min}$ [eV]", fontsize=20)
        else:
            if vary_WC == "m_bb":
                plt.xlabel(r"$|m_{\beta\beta}|$ [eV]", fontsize=20)
            else:
                plt.xlabel(r"$|C_{"+vary_WC[:-3]+"}^{"+vary_WC[-3:]+"}|$", fontsize=20)
        if y_min == None:
            y_min = np.min([np.min(points[:,1]), np.min(pointsIO[:,1])])
        if y_max == None:
            y_max = np.max([np.max(points[:,1]), np.max(pointsIO[:,1])])
        if cosmo:
            def m_sum(m_min):
                m21 = 7.53e-5
                m32 = 2.453e-3
                msum = m_min + np.sqrt(m_min**2+m21) + np.sqrt(m_min**2 + m21 + m32)
                return(msum-m_cosmo)
            cosmo_limit = scipy.optimize.root(m_sum, x0 = 0.05).x[0]
            #print(cosmo_limit)
            m_cosmo
            plt.fill_betweenx([y_min, y_max], [1], [cosmo_limit], alpha=0.1, color="k")
            
            
            
        if compare_to_mass:
            if ordering == "NO":
                plt.plot(M_mbb,NO_min_mbb, "grey")
                plt.plot(M_mbb,NO_max_mbb, "grey")
                plt.fill_between(M_mbb, NO_max_mbb, NO_min_mbb, color="k", alpha=0.1, label = r"$m_{\beta\beta}$")
                
            elif ordering == "IO":
                plt.plot(M_mbb,IO_min_mbb, "grey")
                plt.plot(M_mbb,IO_max_mbb, "grey")
                plt.fill_between(M_mbb, IO_max_mbb, IO_min_mbb, color="k", alpha=0.1, label=r"$m_{\beta\beta}$")
            else:
                plt.plot(M_mbb,NO_min_mbb, "grey")
                plt.plot(M_mbb,NO_max_mbb, "grey")
                plt.plot(M_mbb,IO_min_mbb, "grey")
                plt.plot(M_mbb,IO_max_mbb, "grey")
                plt.fill_between(M_mbb, IO_max_mbb, IO_min_mbb, color="k", alpha=0.1, label=r"$m_{\beta\beta}$")
                plt.fill_between(M_mbb, NO_max_mbb, NO_min_mbb, color="k", alpha=0.1)
        plt.plot(points[:,0],points[:,1], "b.", alpha = alpha_plot, markersize = 0.15)
        plt.plot(pointsIO[:,0],pointsIO[:,1], "r.", alpha = alpha_plot, markersize = 0.15)
        plt.yscale("log")
        plt.xscale("log")
        plt.ylabel(r"$t_{1/2}^{-1} [yr^{-1}]$ ", fontsize=20)
        plt.xlabel(r"$m_{min}$ [eV]", fontsize=20)
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        if vary_WC == "m_min":
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='NO',
                                  markerfacecolor='b', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='IO',
                                  markerfacecolor='r', markersize=10)]
            plt.legend(handles = legend_elements, fontsize=20)
        plt.tight_layout()
        if experiments != None:
            for experiment in experiments:
                plt.axhline(experiments[experiment], label )
                plt.text(x = 1e-3, y = experiments[experiment], s = experiment)
        #return(forbidden_LECsNO, forbidden_LECsNOm, forbidden_LECsIO, forbidden_LECsIOm)
        if save == True:
            plt.savefig(file)
        return(fig)

'''
#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#                                            SMEFT MODEL                                            #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################
'''

class SMEFT(object):
    def __init__(self, WC, scale, name = None, method = "IBM2", use_unknown_LECs = False):
        self.method = method                       #NME method
        self.name = name                           #Model name
        self.m_Z = 91                              #Z-Boson Mass in GeV
        self.scale = scale                         #Matching scale BSM -> SMEFT
        self.use_unknown_LECs = use_unknown_LECs   #Use unknown LECs or not
        
        self.SMEFT_WCs = {#dim5                    #Wilson Coefficients of SMEFT
                          "LLHH"      : 0,         #up to dimension 9. We only 
                          #dim7                    #list the operators violating
                          "LH(7)"     : 0,         #lepton number by 2 units.
                          "LHD1(7)"   : 0,         #at dim9 we only list operators
                          "LHD2(7)"   : 0,         #that contribute to LEFT dim9
                          "LHDe(7)"   : 0,         #as others will be suppressed
                          #"LHB(7)"    : 0,        #by an additional factor of G_F
                          "LHW(7)"    : 0,
                          "LLduD1(7)" : 0,
                          #"LLeH(7)"   : 0,
                          "LLQdH1(7)" : 0,
                          "LLQdH2(7)" : 0,
                          "LLQuH(7)" : 0,
                          "LeudH(7)"  : 0, 
                          #dim9
                          #  -6-fermi
                          "ddueue(9)"    : 0,
                          "dQdueL1(9)"   : 0,
                          "dQdueL2(9)"   : 0,
                          "QudueL1(9)"   : 0,
                          "QudueL2(9)"   : 0,
                          "dQQuLL1(9)"   : 0,
                          "dQQuLL2(9)"   : 0,
                          "QuQuLL1(9)"   : 0,
                          "QuQuLL2(9)"   : 0,
                          "dQdQLL1(9)"   : 0,
                          "dQdQLL2(9)"   : 0,
                          #  -other
                          "LLH4W1(9)"    : 0,
                          "deueH2D(9)"   : 0,
                          "dLuLH2D2(9)"  : 0,
                          "duLLH2D(9)"   : 0,
                          "dQLeH2D2(9)"  : 0,
                          "dLQeH2D1(9)"  : 0,
                          "deQLH2D(9)"   : 0,
                          "QueLH2D2(9)"  : 0,
                          "QeuLH2D2(9)"  : 0,
                          "QLQLH2D2(9)"  : 0,
                          "QLQLH2D5(9)"  : 0,
                          "QQLLH2D2(9)"  : 0,
                          "eeH4D2(9)"    : 0,
                          "LLH4D23(9)"   : 0,
                          "LLH4D24(9)"   : 0}
        
        for operator in WC:
            #store SMEFT operators
            #need to be conjugated to have d -> u transitions
            self.SMEFT_WCs[operator] = WC[operator].conjugate()

        self.WC_input = self.SMEFT_WCs.copy()
        self.WC = self.WC_input.copy()
        #'''
        #self.WC = self.run() #generates SMEFT WCs at electroweak scale
        #''''
    #import NMEs for method
    #self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
            
    '''
        ################################################################################################
        
        Define RGEs computed in 1901.10302
        
        Note that "scale" in the RGEs refers to log(mu)
        i.e. "scale" = log(mu), while scale in the final 
        functions refers to the actual energy scale i.e. 
        "scale" = mu
        
        ################################################################################################
    '''
    
    def RGEalpha(self, scale, alpha, n_g = 3):
        M1 = np.zeros((5,5))
        M1[0,0]      = 1/(2*np.pi)*(1/10 + 4/3 * n_g)
        M1[0,1]      = 0
        M1[0,2]      = 0
        M1[0,3]      = 0
        M1[0,4]      = 0
        
        
        M1[1,0]      = 0
        M1[1,1]      = 1/(2*np.pi)*(-43/6 + 4/3*n_g)
        M1[1,2]      = 0
        M1[1,3]      = 0
        M1[1,4]      = 0
        
        
        M1[2,0]      = 0
        M1[2,1]      = 0
        M1[2,2]      = 1/(2*np.pi)*(-11 + 4/3*n_g)
        M1[2,3]      = 0
        M1[2,4]      = 0
        
        
        M1[3,0]      = 1/(2*np.pi)*(-17/20)
        M1[3,1]      = 1/(2*np.pi)*(-  9/4)
        M1[3,2]      = 1/(2*np.pi)*(-    8)
        M1[3,3]      = 1/(2*np.pi)*(+  9/2)
        M1[3,4]      = 0
        
        
        M1[4,0]      = 1/(4*np.pi) * (- 9/5)
        M1[4,1]      = 1/(4*np.pi) * (-   9)
        M1[4,2]      = 0
        M1[4,3]      = 1/(4*np.pi) * (+  12)
        M1[4,4]      = 1/(4*np.pi) * (+  24)
        
        
        M2 = np.zeros(5)
        alpha12 = np.array(alpha[0:2])
        M2[-1] = 1/(8*np.pi) * np.dot(alpha12, np.dot(np.array([[27/100, 9/10], [0, 9/4]]), alpha12))
        
        return(np.dot(M1, alpha)*alpha + M2)
    
    def alpha(self, scale, alpha0 = [0.0169225, 0.033735, 0.1173, 0.07514, 0.13/(4*np.pi)], n_g = 3, mu0 = 91):
        alpha_new = integrate.solve_ivp(self.RGEalpha, [np.log(mu0), np.log(scale)], alpha0).y[:,-1]
        return(alpha_new)
    
    def alpha1(self, scale, alpha0 = [0.0169225, 0.033735, 0.1173, 0.07514, 0.13/(4*np.pi)], n_g = 3, mu0 = 91):
        return(self.alpha(scale, alpha0)[0])
    
    def alpha2(self, scale, alpha0 = [0.0169225, 0.033735, 0.1173, 0.07514, 0.13/(4*np.pi)], n_g = 3, mu0 = 91):
        return(self.alpha(scale, alpha0)[1])
    
    def alpha3(self, scale, alpha0 = [0.0169225, 0.033735, 0.1173, 0.07514, 0.13/(4*np.pi)], n_g = 3, mu0 = 91):
        return(self.alpha(scale, alpha0)[2])
    
    def alpha_t(self, scale, alpha0 = [0.0169225, 0.033735, 0.1173, 0.07514, 0.13/(4*np.pi)], n_g = 3, mu0 = 91):
        return(self.alpha(scale, alpha0)[3])
    
    def alpha_lambda(self, scale, alpha0 = [0.0169225, 0.033735, 0.1173, 0.07514, 0.13/(4*np.pi)], n_g = 3, mu0 = 91):
        return(self.alpha(scale, alpha0)[4])
    
    ####################################################################################################
        
    def RGELLduD1(self, scale, C):
        scale = np.exp(scale)
        rge = 1/(4*np.pi) * (  1/10 * self.alpha1(scale) 
                             -  1/2 * self.alpha2(scale))*C
        return(rge)
    
    def C_LLduD1(self, final_scale = 80, initial_scale = None):
        if initial_scale == None:
            initial_scale = self.scale
        C = integrate.solve_ivp(self.RGELLduD1, [np.log(initial_scale), np.log(final_scale)], [self.SMEFT_WCs["LLduD1(7)"]]).y[0][-1]
        return(C)
        
    ####################################################################################################
    
    def RGELHDe(self, scale, C):
        scale = np.exp(scale)
        rge = 1/(4*np.pi) * (- 9/10 * self.alpha1(scale) 
                             +    6 * self.alpha_lambda(scale) 
                             +    9 * self.alpha_t(scale))*C
        return(rge)
    
    def C_LHDe(self, final_scale = 80, initial_scale = None):
        if initial_scale == None:
            initial_scale = self.scale
        C = integrate.solve_ivp(self.RGELHDe, [np.log(initial_scale), np.log(final_scale)], [self.SMEFT_WCs["LHDe(7)"]]).y[0][-1]
        return(C)
    
    ####################################################################################################
    
    def RGELeudH(self, scale, C):
        scale = np.exp(scale)
        rge = 1/(4*np.pi) * (- 69/20 * self.alpha1(scale) 
                             -   9/4 * self.alpha2(scale) 
                             +     9 * self.alpha_t(scale))*C
        return(rge)
    
    def C_LeudH(self, final_scale = 80, initial_scale = None):
        if initial_scale == None:
            initial_scale = self.scale
        C = integrate.solve_ivp(self.RGELeudH, [np.log(initial_scale), np.log(final_scale)], [self.SMEFT_WCs["LeudH(7)"]]).y[0][-1]
        return(C)
    
    ####################################################################################################
    
    def RGELLQuH(self, scale, C):
        scale = np.exp(scale)
        rge = 1/(4*np.pi) * (  1/20 * self.alpha1(scale) 
                             -  3/4 * self.alpha2(scale) 
                             -    8 * self.alpha3(scale)
                             +    3 * self.alpha_t(scale))*C
        return(rge)
    
    def C_LLQuH(self, final_scale = 80, initial_scale = None):
        if initial_scale == None:
            initial_scale = self.scale
        C = integrate.solve_ivp(self.RGELLQuH, [np.log(initial_scale), np.log(final_scale)], [self.SMEFT_WCs["LLQuH(7)"]]).y[0][-1]
        return(C)  
    
    ####################################################################################################
    
    def RGELLQdH12(self, scale, C):
        scale = np.exp(scale)
        LLQdH11 = (  13/20 * self.alpha1(scale) 
                   +   9/4 * self.alpha2(scale) 
                   -     8 * self.alpha3(scale) 
                   +     3 * self.alpha_t(scale))
        
        LLQdH12 = (      6 * self.alpha2(scale))
        
        LLQdH21 = (-    4/3 * self.alpha1(scale) 
                   +   16/3 * self.alpha3(scale))
        
        LLQdH22 = (- 121/60 * self.alpha1(scale) 
                   -   15/4 * self.alpha2(scale) 
                   +    8/3 * self.alpha3(scale) 
                   +      3 * self.alpha_t(scale))
        
        rge = np.array([[LLQdH11, LLQdH12], 
                        [LLQdH21, LLQdH22]])
        
        return(np.dot(rge, C))
    
    def C_LLQdH1(self, final_scale = 80, initial_scale = None):
        if initial_scale == None:
            initial_scale = self.scale
        C0 = [self.SMEFT_WCs["LLQdH1(7)"], self.SMEFT_WCs["LLQdH2(7)"]]
        C = integrate.solve_ivp(self.RGELLQdH12, [np.log(initial_scale), np.log(final_scale)], C0).y[0][-1]
        return(C) 
    
    def C_LLQdH2(self, final_scale = 80, initial_scale = None):
        if initial_scale == None:
            initial_scale = self.scale
        C0 = [self.SMEFT_WCs["LLQdH1(7)"], self.SMEFT_WCs["LLQdH2(7)"]]
        C = integrate.solve_ivp(self.RGELLQdH12, [np.log(initial_scale), np.log(final_scale)], C0).y[1][-1]
        return(C) 
    
    ####################################################################################################
    
    def RGE_LHD1_LHD2_LHW(self, scale, C):
        scale = np.exp(scale)
        LHD1_LHD1 = (-  9/20 * self.alpha1(scale) 
                     +  11/2 * self.alpha2(scale) 
                     +     6 * self.alpha_t(scale))
        
        LHD1_LHD2 = (- 33/20 * self.alpha1(scale) 
                     -  19/2 * self.alpha2(scale) 
                     -     2 * self.alpha_lambda(scale))
        LHD1_LHW  = 0
        
        LHD2_LHD1 = (-     8 * self.alpha2(scale))
        
        LHD2_LHD2 = (   12/5 * self.alpha1(scale) 
                     +     3 * self.alpha2(scale) 
                     +     4 * self.alpha_lambda(scale)
                     +     6 * self.alpha_t(scale))
        
        LHD2_LHW  = 0
        
        LHW_LHD1  = (    5/8 * self.alpha2(scale))
        
        LHW_LHD2  = (   9/80 * self.alpha1(scale)
                     + 11/16 * self.alpha2(scale))
         
        LHW_LHW   = (-   6/5 * self.alpha1(scale) 
                     -  13/2 * self.alpha2(scale) 
                     +     4 * self.alpha_lambda(scale) 
                     +     6 * self.alpha_t(scale))
        
        
        rge = np.array([[LHD1_LHD1, LHD1_LHD2, LHD1_LHW], 
                        [LHD2_LHD1, LHD2_LHD2, LHD2_LHW], 
                        [ LHW_LHD1,  LHW_LHD2,  LHW_LHW]])
        
        return(np.dot(rge, C))
    
    def C_LHD1(self, final_scale = 80, initial_scale = None):
        if initial_scale == None:
            initial_scale = self.scale
        C0 = [self.SMEFT_WCs["LHD1(7)"], self.SMEFT_WCs["LHD2(7)"], self.SMEFT_WCs["LHW(7)"]]
        C = integrate.solve_ivp(self.RGE_LHD1_LHD2_LHW, [np.log(initial_scale), np.log(final_scale)], C0).y[0][-1]
        return(C)
    
    def C_LHD2(self, final_scale = 80, initial_scale = None):
        if initial_scale == None:
            initial_scale = self.scale
        C0 = [self.SMEFT_WCs["LHD1(7)"], self.SMEFT_WCs["LHD2(7)"], self.SMEFT_WCs["LHW(7)"]]
        C = integrate.solve_ivp(self.RGE_LHD1_LHD2_LHW, [np.log(initial_scale), np.log(final_scale)], C0).y[1][-1]
        return(C)
    
    def C_LHW(self, final_scale = 80, initial_scale = None):
        if initial_scale == None:
            initial_scale = self.scale
        C0 = [self.SMEFT_WCs["LHD1(7)"], self.SMEFT_WCs["LHD2(7)"], self.SMEFT_WCs["LHW(7)"]]
        #print(C0)
        C = integrate.solve_ivp(self.RGE_LHD1_LHD2_LHW, [np.log(initial_scale), np.log(final_scale)], C0).y[2][-1]
        #print(C)
        return(C)
    
    def run(self, final_scale = 80, initial_scale = None, WC = None, inplace = False):
        if initial_scale == None:
            initial_scale = self.scale
        if WC == None:
            WC = self.WC_input.copy()
            
        final_WC = WC.copy()
        final_WC["LHDe(7)"]   = self.C_LHDe(final_scale)
        final_WC["LHW(7)"]    = self.C_LHW(final_scale)
        final_WC["LHD1(7)"]   = self.C_LHD1(final_scale)
        final_WC["LHD2(7)"]   = self.C_LHD2(final_scale)
        final_WC["LeudH(7)"]  = self.C_LeudH(final_scale)
        final_WC["LLQdH1(7)"] = self.C_LLQdH1(final_scale)
        final_WC["LLQdH2(7)"] = self.C_LLQdH2(final_scale)
        final_WC["LLQuH(7)"] = self.C_LLQuH(final_scale)
        final_WC["LLduD1(7)"] = self.C_LLduD1(final_scale)
        
        if inplace:
            self.WC = final_WC.copy()
        
        return(final_WC)
        
        
    
    ####################################################################################################
        
        
        
    '''
        Match SMEFT Operators to LEFT at 80GeV = M_W scale
    '''
    def LEFT_matching(self, WC = None):#, scale = 80):
        #match SMEFT WCs onto LEFT WCs at EW scale
        #this script takes some time because it needs to solve the different RGEs
        if WC == None:
            WC = self.WC.copy()
            #print(WC)
            
        else:
            C = self.SMEFT_WCs.copy()
            for operator in C:
                C[operator]=0
            for operator in WC:
                C[operator] = WC[operator]
            WC = C.copy()
        m_d   = 4.7e-3
        m_u   = 2.2e-3
        m_e   = pc["electron mass energy equivalent in MeV"][0]*1e-3
        vev   = 246
        V_ud  = 0.97417
        
        LEFT_WCs = {"m_bb":0, "SL(6)": 0, "SR(6)": 0, 
                    "T(6)":0, "VL(6)":0, "VR(6)":0, 
                    "VL(7)":0, "VR(7)":0, 
                    "1L(9)":0, "1R(9)":0, 
                    "1L(9)prime":0, "1R(9)prime":0,
                    "2L(9)":0, "2R(9)":0, 
                    "2L(9)prime":0, "2R(9)prime":0, 
                    "3L(9)":0, "3R(9)":0, 
                    "3L(9)prime":0, "3R(9)prime":0,
                    "4L(9)":0, "4R(9)":0, 
                    "5L(9)":0, "5R(9)":0, 
                    "6(9)":0,"6(9)prime":0,
                    "7(9)":0,"7(9)prime":0,
                    "8(9)":0,"8(9)prime":0,
                    "9(9)":0,"9(9)prime":0}
        
        #dim 5 matching
        LEFT_WCs["m_bb"] = -vev**2 * WC["LLHH"] - vev**4/2 * WC["LH(7)"]
        
        #LEFT_WCs["VL(6)"] = -vev**3*V_ud*(  1j/np.sqrt(2) * self.C_LHDe(scale) 
        #                                  + 4*m_e/vev     * self.C_LHW(scale))
        
        LEFT_WCs["VL(6)"] = (-vev**3*V_ud*(  1j/np.sqrt(2) * WC["LHDe(7)"]
                                          + 4*m_e/vev     * WC["LHW(7)"])
                             +vev**4*(-m_e*V_ud*WC["LLH4W1(9)"] 
                                  +m_d/4 * WC["deQLH2D(9)"]))
        
        #LEFT_WCs["VR(6)"] = vev**3/np.sqrt(2) * self.C_LeudH(scale)
        LEFT_WCs["VR(6)"] = vev**3/np.sqrt(2) * WC["LeudH(7)"]
        
        #LEFT_WCs["SR(6)"] = vev**3*( 1/(2*np.sqrt(2)) * self.C_LLQdH1(scale)
        #                            -V_ud/2*m_d/vev   * self.C_LHD2(scale))
        LEFT_WCs["SR(6)"] = (vev**3*( 1/(2*np.sqrt(2)) * WC["LLQdH1(7)"]
                                    -V_ud/2*m_d/vev   * WC["LHD2(7)"])
                             +vev**4*(m_d*V_ud/8 * WC["LLH4D24(9)"]
                                      -m_d*V_ud/2 * WC["LLH4D23(9)"]
                                      +m_d/4 * WC["QQLLH2D2(9)"]))
        
        #LEFT_WCs["SL(6)"] = vev**3*(    1/(np.sqrt(2)) * self.C_LLQuH1(scale)
        #                            + V_ud/2 * m_u/vev * self.C_LHD2(scale))
        LEFT_WCs["SL(6)"] = (vev**3*(    1/(np.sqrt(2)) * WC["LLQuH(7)"]
                                    + V_ud/2 * m_u/vev * WC["LHD2(7)"])
                             +vev**4*(-m_u*V_ud/8 * WC["LLH4D24(9)"]
                                      +m_u*V_ud/2 * WC["LLH4D23(9)"]
                                      -m_u/4 * WC["QQLLH2D2(9)"]
                                      -1/4*m_d * WC["duLLH2D(9)"]))
        
        #LEFT_WCs["T(6)"]  = vev**3/(8*np.sqrt(2)) * (2*self.C_LLQdH2(scale)
        #                                             + self.C_LLQdH1(scale))
        LEFT_WCs["T(6)"]  = (vev**3/(8*np.sqrt(2)) * (2*WC["LLQdH2(7)"]
                                                     + WC["LLQdH1(7)"])
                             +vev**4*(m_e/16*WC["deQLH2D(9)"]))
        
        #LEFT_WCs["VL(7)"] = -vev**3*V_ud/2 * (  2 * self.C_LHD1(scale)
        #                                      +     self.C_LHD2(scale)
        #                                      + 8 * self.C_LHW(scale))
        LEFT_WCs["VL(7)"] = (-vev**3*V_ud/2 * (  2 * WC["LHD1(7)"]
                                              +     WC["LHD2(7)"]
                                              + 8 * WC["LHW(7)"])
                             +vev**5*(-V_ud*WC["LLH4W1(9)"]
                                      -V_ud/8 * WC["LLH4D24(9)"]
                                      -V_ud/2 * WC["LLH4D23(9)"]
                                      +1/4 * WC["QQLLH2D2(9)"]))
        
        #LEFT_WCs["VR(7)"] = -1j*vev**3/2 * (2*self.C_LLduD1(scale))
        LEFT_WCs["VR(7)"] = (-1j*vev**3/2 * (2*WC["LLduD1(7)"])
                             +vev**5*(1j/4 * WC["duLLH2D(9)"]))
        
        #LEFT_WCs["1L(9)"] = -vev**3*2*V_ud**2*(self.C_LHD1(scale) + 4*self.C_LHW(scale))
        LEFT_WCs["1L(9)"] = (-vev**3*2*V_ud**2*(WC["LHD1(7)"] + 4*WC["LHW(7)"])
                             +vev**5*(-2*V_ud**2 * WC["LLH4W1(9)"]
                                      +V_ud**2/2 * WC["LLH4D24(9)"]
                                      -V_ud**2 * WC["LLH4D23(9)"]
                                      +V_ud * WC["QQLLH2D2(9)"]
                                      +V_ud/2 * WC["QLQLH2D5(9)"]
                                      +V_ud/2 * WC["QLQLH2D2(9)"]))
        
        LEFT_WCs["1R(9)"] = (vev**5*(V_ud**2 * WC["eeH4D2(9)"]))
        
        LEFT_WCs["1R(9)prime"] = vev**5*(-1/4*WC["ddueue(9)"])
        
        LEFT_WCs["2L(9)"] = vev**5*(-WC["QuQuLL1(9)"])
        
        LEFT_WCs["3L(9)"] = vev**5*(-WC["QuQuLL2(9)"])
        
        LEFT_WCs["2L(9)prime"] = vev**5*(-WC["dQdQLL1(9)"])
        
        LEFT_WCs["3L(9)prime"] = vev**5*(-WC["dQdQLL2(9)"])
        
        LEFT_WCs["4L(9)"] = (-vev**3*2j*V_ud*WC["LLduD1(7)"]
                             +vev**5*(-V_ud * WC["duLLH2D(9)"]
                                      +V_ud/2 * WC["dLuLH2D2(9)"]
                                      -WC["dQQuLL2(9)"]))
        
        LEFT_WCs["5L(9)"] = vev**5*(-WC["dQQuLL1(9)"])
        
        LEFT_WCs["4R(9)"] = (vev**5*(V_ud/2 * WC["deueH2D(9)"]))
        
        
        LEFT_WCs["6(9)"]  = (vev**5*(V_ud/4 * WC["dLQeH2D1(9)"]
                                     -V_ud/2 * WC["dQLeH2D2(9)"] 
                                     -2 * V_ud * WC["deQLH2D(9)"]))
        
        LEFT_WCs["6(9)prime"]  = (vev**5*(1/6 * WC["QudueL2(9)"]
                                          + 1/2 * WC["QudueL1(9)"]))
        LEFT_WCs["7(9)"] = vev**5*(-7/12*V_ud*WC["deQLH2D(9)"])
        
        LEFT_WCs["7(9)prime"]  = (vev**5*(WC["QudueL2(9)"]))
        
        LEFT_WCs["8(9)"] = (vev**5*(-V_ud * WC["QueLH2D2(9)"]))
        
        LEFT_WCs["8(9)prime"] = (vev**5*(V_ud/2 * WC["QeuLH2D2(9)"]
                                         +1/6 * WC["dQdueL2(9)"] 
                                         +1/2 * WC["dQdueL1(9)"]))
        
        LEFT_WCs["9(9)"] = (vev**5*(V_ud*WC["QeuLH2D2(9)"]))
        
        LEFT_WCs["9(9)prime"] = vev**5*WC["dQdueL2(9)"]
        
        #LEFT_WCs["5L(9)"] = only contributions from running in LEFT
        
        return(LEFT_WCs)

    def set_LECs(self, use_unknown_LECs):
        self.use_unknown_LECs = unknown_LECs
        
    def half_lives(self, WC = None, use_unknown_LECs = None, method = None, printing = True):
        if WC == None:
            WC = self.WC.copy()
        if use_unknown_LECs == None:
            use_unknown_LECs = self.use_unknown_LECs
        if method == None:
            method = self.method
        #elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            #print("Changing method to",method)
        #    self.method = method
            #self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        #if printing:
        #    print("... solving RGEs ...")
        LEFT_WCs = self.LEFT_matching(WC)
        if printing:
            print("... matching onto LEFT ...")
        model = LEFT(LEFT_WCs, use_unknown_LECs = use_unknown_LECs, method = method)
        return(model.half_lives())
    
    def spectra(self, Ebar, WC = None, use_unknown_LECs = None, method = None, printing = True):
        if WC == None:
            WC = self.WC.copy()
        if use_unknown_LECs == None:
            use_unknown_LECs = self.use_unknown_LECs
        if method == None:
            method = self.method
        #elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            #print("Changing method to",method)
        #    self.method = method
            #self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        #if printing:
        #    print("... solving RGEs ...")
        LEFT_WCs = self.LEFT_matching(WC)
        if printing:
            print("... matching onto LEFT ...")
        model = LEFT(LEFT_WCs, use_unknown_LECs = use_unknown_LECs, method = method)
        return(model.spectra(Ebar))
    
    def angular_corr(self, Ebar, WC = None, use_unknown_LECs = None, method = None, printing = True):
        if WC == None:
            WC = self.WC.copy()
        if use_unknown_LECs == None:
            use_unknown_LECs = self.use_unknown_LECs
        if method == None:
            method = self.method
        #elif method != self.method and method in ["IBM2", "QRPA", "SM"]:
            #print("Changing method to",method)
        #    self.method = method
            #self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        elif method not in ["IBM2", "QRPA", "SM"]:
            print("Method",method,"is unavailable. Keeping current method",self.method)
        else:
            pass
            
        #if printing:
        #    print("... solving RGEs ...")
        LEFT_WCs = self.LEFT_matching(WC)
        if printing:
            print("... matching onto LEFT ...")
        model = LEFT(LEFT_WCs, use_unknown_LECs = use_unknown_LECs, method = method)
        return(model.angular_corr(Ebar))

    def get_limits2(self, hl, use_unknown_LECs = False, method = None, element_name = "76Ge", onlygroups = False):
        if method == None:
            method = self.method
        vev = 246
        #make a backup of WCs
        WC = self.SMEFT_WCs.copy()
        half_live = hl
        result = {}
        scales = {}
        #scale = {}
        for WC_name in self.SMEFT_WCs:
            print(WC_name)
            limit = 1
            LEFT_WCs = self.LEFT_matching({WC_name : limit})
            LEFT_model = LEFT(LEFT_WCs, method=method)
            hl = LEFT_model.t_half(element_name=element_name)
            limit = np.sqrt(hl/half_live)
            #if limit != 1:
            result[WC_name] = limit
            if WC_name == "LLHH":
                dimension = 5
            else:
                dimension = int(WC_name[-2])
            scale = (1/np.absolute(limit))**(1/(dimension -4))
            scales[WC_name] = scale
        
        return(result, scales)#, scale)
    
    def get_limits(self, hl, use_unknown_LECs = False, method = None, element_name = "76Ge"):
        if method == None:
            method = self.method
        vev = 246
        #make a backup of WCs
        WC = self.SMEFT_WCs.copy()
        half_live = hl
        #define function to optimize to fit observed half-life
        #print("... solving a lot of RGEs ...")
        #print("This is going to take some time")
        def t_half_optimize(WC_value, WC_name, element_name):
            #overwrite SMEFT WCs for the calculation and afterwards use backup
            for operator in WC:
                WC[operator] = 0
            WC[WC_name] = WC_value[0]
            #print(WC_value)
            LEFT_WCs = self.LEFT_matching(WC=WC)
            #print(LEFT_WCs)
            LEFT_model = LEFT(LEFT_WCs, method = method, use_unknown_LECs = use_unknown_LECs)
            #print(LEFT_WCs)
            #print(SMEFT_WCs)
            result = LEFT_model.t_half(element_name) - half_live
            ##print(result)
            return(result)

        result = {}
        scales = {}
        #scale = {}
        for WC_name in self.SMEFT_WCs:
            print(WC_name)
            limit = np.absolute(optimize.root(t_half_optimize, args=(WC_name, element_name), x0=1e-15).x[0])
            if limit != 1e-15:
                result[WC_name] = limit
                if WC_name == "LLHH":
                    dimension = 5
                else:
                    dimension = int(WC_name[-2])
                scale = (1/np.absolute(limit))**(1/(dimension -4))
                scales[WC_name] = scale
            #else:
            #    result[WC_name] = 0
            #    scales[WC_name] = np.inf
            #print(limit)
            #result[WC_name] = limit
            #if WC_name == "LLHH":
            #    scale[WC_name] = limit*1e+9
            #    #scale[WC_name] = limit
            #elif "(7)" in WC_name:
            #    scale[WC_name] = limit**(-1/3)

        
        return(result, scales)#, scale)
    
    def ratios(self, save = False, plot = False, reference_isotope = "76Ge", normalized = True, method=None, vary = False, n_points = 100, addgrid = True):
        if method == None:
            method = self.method
        LEFT_model = LEFT(self.LEFT_matching(), method = method)
        return(LEFT_model.ratios(save = save, plot = plot, 
                                 reference_isotope = reference_isotope, 
                                 normalized = normalized, method=method, 
                                 vary = vary, n_points = n_points))




    #def get_limits(self, hl, use_unknown_LECs = False, method = "IBM2", element_name = "76Ge", scale = None):
    #    #get scale to run limits from 80GeV up to relevant scale
    #    if scale == None:
    #        scale = self.scale
    #    # calculate limits on smeft WCs
    #    #first get limits on LEFT_WCs
    #    LEFT_model = LEFT({}, method = method, use_unknown_LECs =use_unknown_LECs)
    #    LEFT_limits = LEFT_model.get_limits(hl, element_name = element_name, method = method, scale = scale)
    #
    #    #now get limits on SMEFT_WCs by matching them onto 
    #    def match_limits(value, operator, limits):
    #        WC = {operator, value}
    #        #match operator to LEFT
    #        LEFT_WCs = self.LEFT_matching(WC)
    #        delta = np.zeros(len(LEFT_WCs))
    #        for idx in range(len(LEFT_WCs)):
    #            operator = LEFT_WCs.keys()[idx]
    #            delta[idx] = np.absolute(limits[operator]-LEFT_WCs[operator])
    #        result = np.sum(delta)
    #        return(result)
    #        
    #    for operator in self.SMEFT_WC:
    #        #get root of match_limits(function)
    #        limits = 0
    #    return(limits)
            
