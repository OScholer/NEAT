#!/usr/bin/env python
# coding: utf-8

# In[12]:


m_pi = 0.13957
m_e = 5.10998e-4
m_p = 0.938272088


# In[16]:


import pandas as pd


# In[93]:


def Load_NMEs(method):
    NMEpanda = pd.read_csv("NMEs/NMEs_"+method+".csv")
    NMEpanda.set_index("NME", inplace = True)
    for i in NMEpanda:
        for j in range(len(NMEpanda[i])):
            if j == 5:
                pass
            else:
                NMEpanda[i][j] = float(NMEpanda[i][j])
                if j >= 9:
                    NMEpanda[i][j] *= m_e*m_p/(m_pi**2)
    #NMEpanda.reset_index("NME", inplace = True)
    NMEpanda
    NMEnames = ["F", 
            "GTAA", "GTAP", "GTPP", "GTMM", 
            "TAA", "TAP", "TPP", "TMM" , 
            "F,sd", 
            "GTAA,sd", "GTAP,sd", "GTPP,sd" , 
            "TAP,sd" , "TPP,sd"]
    NMEs = {}
    for column in NMEpanda:
        element = column[1:]
        #print(element)
        #method = "QRPA"
        NME = {}
        for idx in range(len(NMEnames)):
            try:
                NME[NMEnames[idx]] = float(NMEpanda[column][idx])
            except:
                NME[NMEnames[idx]] = 0
        try:
            NMEs[element][method] = NME
        except:
            NMEs[element] = {method : NME}
            
    return (NMEs, NMEpanda, NMEnames)

