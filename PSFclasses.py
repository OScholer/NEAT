#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.special import gamma as Gamma
from scipy.constants import physical_constants as pc
from scipy import integrate
import pandas as pd


# In[2]:


class wavefunction(object):
    def __init__(self):#, S_beta):
        #constants
        self.m_e = pc["electron mass energy equivalent in MeV"][0]
        self.alpha = pc["fine-structure constant"][0]
        #self.S_beta = S_beta
        
        
        
    '''Define Helper Functions'''
    def gamma(self,k, Z):
        return(np.sqrt(k**2 - (self.alpha*Z)**2))

    def y(self, E, Z):
        p = np.sqrt(E**2 - self.m_e**2)
        return(self.alpha*Z*(E/p))

    def F(self, k, E, r, Z): #note that F_k = F(k+1)
        p = np.sqrt(E**2 - self.m_e**2)
        return(((Gamma(2*k+1)/(Gamma(k)*Gamma(1+2*self.gamma(k, Z))))**2
               * (2*p*r)**(2*(self.gamma(k, Z)-1))
               * np.exp(np.pi*self.y(E, Z))#*self.S_beta)
               * np.absolute(Gamma(self.gamma(k, Z)+1j*self.y(E, Z)))**2
              ))



    '''Define Wavefunctions'''
    #S-Wave
    def g_m(self, E, r, Z):
        return(np.sqrt(self.F(1, E, r, Z)*((E+self.m_e)/(2*E))))

    def f_p(self, E, r, Z):            
        return(np.sqrt(self.F(1, E, r, Z)*((E-self.m_e)/(2*E))))

    #P-Wave
    def g_p(self, E, r, Z):
        return(np.sqrt(self.F(1, E, r, Z)*((E-self.m_e)/(2*E)))
               * (self.alpha*Z/2 + (E+self.m_e)*r/3))

    def f_m(self, E, r, Z):
        return(-np.sqrt(self.F(1, E, r, Z)*((E+self.m_e)/(2*E)))
               * (self.alpha*Z/2 + (E-self.m_e)*r/3))
    


# In[3]:


class element(object):
    def __init__(self, Z, A, Delta_M):#, g_p, g_m, f_p, f_m):
        self.Z = Z + 2 # We need Z after the decay hence +2
        self.A = A
        self.Delta_M = Delta_M
        self.E_max = Delta_M
        e = self.wavefunction()
        self.g_m = e.g_m
        self.g_p = e.g_p
        self.f_m = e.f_m
        self.f_p = e.f_p
        self.m_e = pc["electron mass energy equivalent in MeV"][0]
        fm_to_inverse_MeV = 1/197.3
        self.R = 1.2 * A**(1/3) * fm_to_inverse_MeV
        
        
    '''_______________________________________'''
    '''                                       '''
    ''' Define Wave Functions to compute PSFs '''
    '''_______________________________________'''  
    
    class wavefunction(object):
        def __init__(self):#, S_beta):
            #constants
            self.m_e = pc["electron mass energy equivalent in MeV"][0]
            self.alpha = pc["fine-structure constant"][0]
            #self.S_beta = S_beta



        '''Define Helper Functions'''
        def gamma(self,k, Z):
            return(np.sqrt(k**2 - (self.alpha*Z)**2))

        def y(self, E, Z):
            p = np.sqrt(E**2 - self.m_e**2)
            return(self.alpha*Z*(E/p))

        def F(self, k, E, r, Z): #note that F_k = F(k+1)
            p = np.sqrt(E**2 - self.m_e**2)
            return(((Gamma(2*k+1)/(Gamma(k)*Gamma(1+2*self.gamma(k, Z))))**2
                   * (2*p*r)**(2*(self.gamma(k, Z)-1))
                   * np.exp(np.pi*self.y(E, Z))#*self.S_beta)
                   * np.absolute(Gamma(self.gamma(k, Z)+1j*self.y(E, Z)))**2
                  ))



        '''Define Wavefunctions'''
        #S-Wave
        def g_m(self, E, r, Z):
            return(np.sqrt(self.F(1, E, r, Z)*((E+self.m_e)/(2*E))))

        def f_p(self, E, r, Z):            
            return(np.sqrt(self.F(1, E, r, Z)*((E-self.m_e)/(2*E))))

        #P-Wave
        def g_p(self, E, r, Z):
            return(np.sqrt(self.F(1, E, r, Z)*((E-self.m_e)/(2*E)))
                   * (self.alpha*Z/2 + (E+self.m_e)*r/3))

        def f_m(self, E, r, Z):
            return(-np.sqrt(self.F(1, E, r, Z)*((E+self.m_e)/(2*E)))
                   * (self.alpha*Z/2 + (E-self.m_e)*r/3))
        
        
        
    '''_______________________________________'''
    '''                                       '''
    '''Define Helper Functions to compute PSFs'''
    '''_______________________________________'''
    
    
    '''Wavefunction Combinations'''
    def Css(self, E):
        return (self.g_m(E, self.R, self.Z) * self.f_p(E, self.R, self.Z))
    
    def Css_m(self, E):
        return (self.g_m(E, self.R, self.Z)**2 - self.f_p(E, self.R, self.Z)**2)
    
    def Css_p(self, E):
        return (self.g_m(E, self.R, self.Z)**2 + self.f_p(E, self.R, self.Z)**2)
    
    def Csp_f(self, E):
        return (self.f_m(E, self.R, self.Z) * self.f_p(E, self.R, self.Z))
    
    def Csp_m(self, E):
        return (self.g_m(E, self.R, self.Z)*self.f_m(E, self.R, self.Z) 
                - self.g_p(E, self.R, self.Z)*self.f_p(E, self.R, self.Z))
    
    def Csp_p(self, E):
        return (self.g_m(E, self.R, self.Z)*self.f_m(E, self.R, self.Z) 
                + self.g_p(E, self.R, self.Z)*self.f_p(E, self.R, self.Z))
    
    def Cpp(self, E):
        return (self.g_p(E, self.R, self.Z)*self.f_m(E, self.R, self.Z))
    
    def Csp_g(self, E):
        return (self.g_m(E, self.R, self.Z)*self.g_p(E, self.R, self.Z))
    
    def Cpp_m(self, E):
        return (self.g_p(E, self.R, self.Z)**2 - self.f_m(E, self.R, self.Z)**2)
    
    def Cpp_p(self, E):
        return (self.g_p(E, self.R, self.Z)**2 + self.f_m(E, self.R, self.Z)**2)
    
    
    '''Angular Distribution Functions'''
    
    def h_01(self, E_1, E_2):
        return (-4*self.Css(E_1)*self.Css(E_2))
    
    def h_02(self, E_1, E_2):
        return (2*(E_1-E_2)**2/(self.m_e**2)*self.Css(E_1)*self.Css(E_2))
    
    def h_03(self, E_1, E_2):
        return 0
    
    def h_04(self, E_1, E_2):
        return (-2/(3*self.m_e*self.R)*(self.Csp_f(E_1)*self.Css(E_2) 
                                      + self.Csp_f(E_2)*self.Css(E_1)
                                      + self.Csp_g(E_2)*self.Css(E_1) 
                                      + self.Csp_g(E_1)*self.Css(E_2)))
    
    def h_05(self, E_1, E_2):
        return (4/(self.m_e*self.R) * (self.Csp_f(E_1)*self.Css(E_2) 
                                       + self.Csp_f(E_2)*self.Css(E_1) 
                                       + self.Csp_g(E_2)*self.Css(E_1) 
                                       + self.Csp_g(E_1)*self.Css(E_2)))
    
    def h_06(self, E_1, E_2):
        return 0
    
    def h_07(self, E_1, E_2):
        return (-16/(self.m_e*self.R)**2 * (self.Csp_f(E_1)*self.Css(E_2) 
                                       + self.Csp_f(E_2)*self.Css(E_1) 
                                       - self.Csp_g(E_2)*self.Css(E_1) 
                                       - self.Csp_g(E_1)*self.Css(E_2)))
    
    def h_08(self, E_1, E_2):
        return (-8/(self.m_e*self.R)**2 * (self.Csp_f(E_1)*self.Csp_g(E_2) 
                                       + self.Csp_f(E_2)*self.Csp_g(E_1) 
                                       + self.Css(E_1)*self.Cpp(E_2) 
                                       + self.Css(E_2)*self.Cpp(E_1)))
    
    def h_09(self, E_1, E_2):
        return (32/(self.m_e*self.R)**2 *self.Css(E_1)*self.Css(E_2))
    
    def h_010(self, E_1, E_2):
        return (-9/2*self.h_010tilde(E_1, E_2) - self.h_02(E_1, E_2))
    
    def h_011(self, E_1, E_2):
        return (9*self.h_011tilde(E_1, E_2) + 1/9*self.h_02(E_1, E_2) + self.h_010tilde(E_1, E_2))
    
    '''with'''
    
    def h_010tilde(self, E_1, E_2):
        return (2*(E_1-E_2)/(3*self.m_e**2*self.R) * (self.Csp_f(E_1)*self.Css(E_2) 
                                                     - self.Csp_f(E_2)*self.Css(E_1) 
                                                     + self.Csp_g(E_2)*self.Css(E_1) 
                                                     - self.Csp_g(E_1)*self.Css(E_2)))
    
    def h_011tilde(self, E_1, E_2):
        return (-2/(3*self.m_e*self.R)**2 * (self.Csp_f(E_1)*self.Csp_f(E_2) 
                                             + self.Csp_g(E_2)*self.Csp_g(E_1)
                                             + self.Css(E_1)*self.Cpp(E_2) 
                                             + self.Css(E_2)*self.Cpp(E_1)))
    
    '''Components of Phase Space Factors'''
    def g_01(self, E_1, E_2):
        return (self.Css_p(E_1)*self.Css_p(E_2))
    
    def g_11(self, E_1, E_2):
        return self.g_01(E_1, E_2)
    
    def g_02(self, E_1, E_2):
        return ((E_1 - E_2)**2/(2*self.m_e**2) * (self.Css_p(E_1)*self.Css_p(E_2) 
                                               - self.Css_m(E_1)*self.Css_m(E_2)))
    
    def g_03(self, E_1, E_2):
        return ((E_1-E_2)/self.m_e * (self.Css_p(E_1)*self.Css_m(E_2) 
                                      - self.Css_p(E_2)*self.Css_m(E_1)))
    
    def g_04(self, E_1, E_2):
        return (1/(3*self.m_e*self.R) * (-self.Css_m(E_1)*self.Csp_m(E_2) 
                                         - self.Css_m(E_2)*self.Csp_m(E_1) 
                                         + self.Css_p(E_1)*self.Csp_p(E_2) 
                                         + self.Css_p(E_2)*self.Csp_p(E_1)) - self.g_03(E_1, E_2)/9)
    
    def g_05(self, E_1, E_2):
        return (-2/(self.m_e*self.R) * (self.Css_m(E_1)*self.Csp_m(E_2) 
                                         + self.Css_m(E_2)*self.Csp_m(E_1) 
                                         + self.Css_p(E_1)*self.Csp_p(E_2) 
                                         + self.Css_p(E_2)*self.Csp_p(E_1)))
    
    def g_06(self, E_1, E_2):
        return (4/(self.m_e*self.R) * (self.Css_p(E_1)*self.Css_m(E_2) 
                                       + self.Css_p(E_2)*self.Css_m(E_1)))
    
    def g_07(self, E_1, E_2):
        return (-8/(self.m_e*self.R)**2 * (self.Css_p(E_1)*self.Csp_m(E_2) 
                                           + self.Css_p(E_2)*self.Csp_m(E_1) 
                                           + self.Css_m(E_1)*self.Csp_p(E_2) 
                                           + self.Css_m(E_2)*self.Csp_p(E_1)))
    
    def g_08(self, E_1, E_2):
        return (2/(self.m_e*self.R)**2 * (-self.Cpp_m(E_1)*self.Css_m(E_2) 
                                          - self.Cpp_m(E_2)*self.Css_m(E_1) 
                                          + self.Cpp_p(E_1)*self.Css_p(E_2)
                                          + self.Cpp_p(E_2)*self.Css_p(E_1) 
                                          + 2*self.Csp_m(E_1)*self.Csp_m(E_2) 
                                          + 2*self.Csp_p(E_1)*self.Csp_p(E_2)))
    
    def g_09(self, E_1, E_2):
        return (8/(self.m_e*self.R)**2 * (self.Css_p(E_1)*self.Css_p(E_2) 
                                          + self.Css_m(E_1)*self.Css_m(E_2)))
    
    def g_010(self, E_1, E_2):
        return (-9/2*self.g_010tilde(E_1, E_2) - self.g_02(E_1, E_2))
    
    def g_011(self, E_1, E_2):
        return (9*self.g_011tilde(E_1, E_2) + 1/9 * self.g_02(E_1, E_2) + self.g_010tilde(E_1, E_2))
    
    '''with'''
    
    def g_010tilde(self, E_1, E_2):
        return ((E_1 - E_2)/(3*self.m_e**2*self.R) * (-self.Css_p(E_1)*self.Csp_m(E_2) 
                                                      + self.Css_p(E_2)*self.Csp_m(E_1) 
                                                      + self.Css_m(E_1)*self.Csp_p(E_2) 
                                                      - self.Css_m(E_2)*self.Csp_p(E_1)))
    
    def g_011tilde(self, E_1, E_2):
        return (1/(18*self.m_e**2*self.R**2) * (self.Cpp_m(E_1)*self.Css_m(E_2) 
                                             + self.Cpp_m(E_2)*self.Css_m(E_1)
                                             + self.Cpp_p(E_1)*self.Css_p(E_2) 
                                             + self.Cpp_p(E_2)*self.Css_p(E_1) 
                                             - 2*self.Csp_m(E_1)*self.Csp_m(E_2) 
                                             + 2*self.Csp_p(E_1)*self.Csp_p(E_2)))
    
    '''________________________________________'''
    '''                                        '''
    '''   Calculation of Phase Space Factors   '''
    '''________________________________________'''
    
    def PSFs(self):
        g = [self.g_01, self.g_02, self.g_03, 
             self.g_04, self.g_05, self.g_06, 
             self.g_07, self.g_08, self.g_09, 
             self.g_010, self.g_011]
        
        h = [self.h_01, self.h_02, self.h_03, 
             self.h_04, self.h_05, self.h_06, 
             self.h_07, self.h_08, self.h_09, 
             self.h_010, self.h_011]
        
        PSFs = []
        
        '''Constants'''
        V_ud =0.97427
        G_F = pc["Fermi coupling constant"][0] * 1e-6
        G_beta = G_F*V_ud 
        
        
        prefactor = G_beta**4*self.m_e**2 / (64 * np.pi**5 * np.log(2) * self.R**2)
        
        def p(E, m = self.m_e):
            return (np.sqrt(E**2-m**2))
        
        G_theta_0k = []
        G_0k = []
        
        for k in range(11):
            G_theta = integrate.quad(lambda E_1: (h[k](E_1, self.Delta_M - E_1) 
                                                              * p(E_1)*p(self.Delta_M-E_1)
                                                              * E_1 * (self.Delta_M - E_1)), self.m_e, self.Delta_M-self.m_e)
            G = integrate.quad(lambda E_1: (g[k](E_1, self.Delta_M - E_1) 
                                                              * p(E_1)*p(self.Delta_M-E_1)
                                                              * E_1 * (self.Delta_M - E_1)), self.m_e, self.Delta_M-self.m_e)
            
            G_theta_0k.append(G_theta)
            G_0k.append(G)
            
            
            #G_0k = integrate.quad(lambda costheta: G_theta/np.log(2)*costheta + G/2, -1, 1)
            
        return (G_0k, G_theta_0k, 2*prefactor, np.log(2)*prefactor) #G_theta = G_theta[0]*prefactor *log(2), G = G * prefactor * 2
    

    #elements = {"Nd150" : Nd150, 
    #            "Xe136" : Xe136, 
    #            "Te130" : Te130, 
    #            "Ge76"  : Ge76}
    #
    #PSFs = []
    #for element in elements:
    #    PSFs.append(elements[element].PSFs())
    #
    #print("__________________________")
    #print("")
    #for i in range(len(PSFs)):
    #    print(names[i])
    #    print(PSFs[i][0][0][0] * MeV_to_inverseyear * PSFs[i][2],"1/y")
    #    print("__________________________")
    #    print("")


# In[4]:


def spectra(Ebar, g, Delta_M):
    E_1 = Ebar*(Delta_M - 2*m_e) + m_e
    def p(E, m = m_e):
        return (np.sqrt(E**2-m**2))
    return ((g(E_1, Delta_M - E_1) * p(E_1)*p(Delta_M-E_1)* E_1 * (Delta_M - E_1)))

def angular_corr(E, g, h, Delta_M):
    return (h(E, Delta_M-E)/g(E, Delta_M-E))


# In[ ]:




