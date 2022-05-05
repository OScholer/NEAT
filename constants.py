import numpy as np
import scipy.constants

#helpful stuff from scipy
pc = scipy.constants.physical_constants

#conversion factors
u_to_GeV = pc["atomic mass constant energy equivalent in MeV"][0] * 1e-3

#######################################################################
#                                                                     #
#                 Quark and Neutrino Mixing Parameters                #
#                                                                     #
#######################################################################

#quark mixing angles
V_ud = 0.97417

#neutrino mixing angles (s = sin, c=cos)
s12 = np.sqrt(0.307)
s23 = np.sqrt(0.546)
s13 = np.sqrt(2.2e-2)

c12 = np.cos(np.arcsin(s12))
c23 = np.cos(np.arcsin(s23))
c13 = np.cos(np.arcsin(s13))

#squared mass differences [eV]
m21 = 7.53e-5
m32 = 2.453e-3
m32IO = -2.546e-3

#######################################################################
#                                                                     #
# masses and scales (all in GeV if not explicitly stated differently) #
#                                                                     #
#######################################################################

vev = 246         #Higgs vev
lambda_chi = 2    #chiPT scale
m_W = 80          #W-boson mass (matching scale of SMEFT->LEFT)

#heavy quarks
m_t = 175         #top-quark
m_b = 4.8         #bottom-quark
m_c = 1.4         #charm-quark

#light quarks are set to 0 except for the matching procedures. If you want to set these to 0 too you need to do so in the EFT.py
m_s = 0 #0.96     #strange-quark
m_u = 0 #0.022    #up-quark
m_d = 0 #0.047    #down-quark

m_pi = 0.13957    #pion mass
m_N = 0.93        #nucleon mass scale in GeV


m_e = pc["electron mass energy equivalent in MeV"][0] * 1e-3 #electron mass in GeV
m_e_MeV = pc["electron mass energy equivalent in MeV"][0]    #electron mass in MeV (used for electron wave functions)


#######################################################################
#                                                                     #
#                        Low-Energy Constants                         #
#                                                                     #
#######################################################################

g_A = 1.271
g_V = 1

F_pi = 0.0922

#all LECs at known values or NDA estimates
LECs = {"A":g_A, 
        "S":0.97, 
        "M":4.7, 
        "T":0.99, 
        "B":2.7, 
        "1pipi":0.36, 
        "2pipi":2.0, 
        "3pipi":-0.62, 
        "4pipi":-1.9, 
        "5pipi":-8, 
        # all the below are expected to be order 1 in absolute magnitude
        "Tprime":1, 
        "Tpipi":1, 
        "1piN":1, 
        "6piN":1, 
        "7piN":1, 
        "8piN":1, 
        "9piN":1, 
        "VLpiN":1,
        "TpiN":1, 
        "1NN":1, 
        "6NN":1, 
        "7NN":1, 
        "VLNN":1, 
        "TNN": 1, 
        "VLE":1, 
        "VLme":1,
        "VRE":1, 
        "VRme":1, 
        # all the below are expected to be order (4pi)**2 in absolute magnitude
        "2NN":(4*np.pi)**2, 
        "3NN":(4*np.pi)**2, 
        "4NN":(4*np.pi)**2,
        "5NN":(4*np.pi)**2, 
        # expected to be 1/F_pipi**2 pion decay constant
        "nuNN": -1/(4*np.pi) * (m_N*g_A**2/(4*F_pi**2))**2*0.6
       }

#known LECs
LECs_known = {"A":g_A, 
             "S":0.97, 
             "M":4.7, 
             "T":0.99, 
             "B":2.7, 
             "1pipi":0.36, 
             "2pipi":2.0, 
             "3pipi":-0.62, 
             "4pipi":-1.9, 
             "5pipi":-8
             }

#Unknown Low-Energy Constants order of magnitude estimate
LECs_unknown = { 'Tprime': 1,
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
                 'nuNN': -1/(4*np.pi) * (m_N*g_A**2/(4*F_pi**2))**2*0.6
               }



#######################################################################
#                                                                     #
#                         Wilson Coefficients                         #
#                                                                     #
#######################################################################



SMEFT_WCs = {#dim5                    #Wilson Coefficients of SMEFT
             "LH(5)"      : 0,        #up to dimension 7. We only 
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
            "9(9)prime"  : 0
            }


LEFT_WCs_epsilon = {#dim3
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
                    "S-PS+P":0, "S-PS-P":0
                    }