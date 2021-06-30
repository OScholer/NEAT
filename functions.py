#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary stuff
import EFT
from matplotlib import pyplot as plt
import scipy.optimize
import numpy as np
import pandas as pd
from scipy import integrate
from matplotlib.lines import Line2D


# In[2]:


def plot_contours(WCx, WCy, experiments = {"GERDA": [5e+25, "76Ge"]}, method = "IBM2", 
                  numerical_method="lm", n_dots=600, linewidth=0, x_min=None, x_max=None, 
                  savefig=False, phase=3/4*np.pi, varyphases = False, n_vary=5):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model=EFT.LEFT({}, method=method)
    if (x_min == None or x_max == None) and x_min != x_max:
        #print("You need to set both x_min and x_max or let the code chose both!")
        return()
    
    for operator in model.WC:
        model.WC[operator] = 0
    def t_half_optimize(WC_valuey, WC_valuex, WCx, WCy, element_name, phase):
        #WC_value1, WC_value2 = WC
        #print(WC_value1)
        #print(WC_value2)
        model.WC[WCx] = WC_valuex#[0]
        model.WC[WCy] = WC_valuey*np.exp(1j*phase)#[0]
        #print(model.WC)
        hl = model.t_half(element_name, method = method)
        model.WC[WCx] = 0
        model.WC[WCy] = 0
        #print(hl)
        return ((hl-limit))
    
    exp=experiments.copy()
    experiments={}
    #sort experiments by hl
    for key in sorted(exp, key=exp.get):
        experiments[key]=exp[key]
    #generate contour points
    fig = plt.figure(figsize=(9, 8))
    
    #iterate over experiments
    exp_idx = 0
    radius = {}
    for experiment in experiments:
        limit = experiments[experiment][0]
        element_name = experiments[experiment][1]

        
        #if experiment == list(experiments.keys())[0]:
        #find the range of the curve by finding the maximal value 
        radius[experiment] = 1.5*scipy.optimize.root(t_half_optimize, args=(0, WCy, WCx, element_name, phase), x0 = 1e-15).x[0]
        #radius2 = 1.2*scipy.optimize.fsolve(t_half_optimize, args=(0, WCy, WCx, element_name), x0 = 1e-15, maxfev=10000)[0]

        if x_min == None:
            x_min = -radius[experiment]
            x_max =  radius[experiment]
        elif radius[experiment] > x_max:
            x_min = -radius[experiment]
            x_max =  radius[experiment]
            #print("radius",radius)
            #print("radius2",radius2)
    for experiment in experiments:
        limit = experiments[experiment][0]
        element_name = experiments[experiment][1]
        if varyphases:
            phases = np.linspace(0, np.pi, n_vary)
            for idx in range(n_vary):

                #lists to store contour points
                contour = []
                contour2 = []
                #contour3 = []
                #contour4 = []
                xplot = []
                xplot2 = []
                phase = phases[idx]
                
                #for x in np.linspace(-radius[experiment],radius[experiment], n_dots):
                for x in np.linspace(x_min,x_max, n_dots):
                    #find the scale at which to search for the minima
                    x0 = 1000*scipy.optimize.root(t_half_optimize, args=(0, WCx, WCy, element_name, phase), 
                                                  x0 = 1e-15).x[0]
                    #x0=1
                    model.WC[WCx] = x
                    optim1 = scipy.optimize.root(t_half_optimize, args = (x, WCx, WCy, element_name, phase) , 
                                                 x0 = x0, method = numerical_method)
                    result1 = optim1.x[0]
                    #model.WC[WCy] = result1

                    #look for multiple roots
                    #optim3 = scipy.optimize.root(t_half_optimize, args = (x, WCx, WCy, element_name) , x0 = -result1, method = numerical_method)
                    #result3 = optim3.x[0]
                    #model.WC[WCy] = result1


                    optim2 = scipy.optimize.root(t_half_optimize, args = (x, WCx, WCy, element_name, phase) , 
                                                 x0 = -x0, method = numerical_method)
                    result2 = optim2.x[0]
                    #model.WC[WCy] = result2

                    #look for multiple roots
                    #optim4 = scipy.optimize.root(t_half_optimize, args = (x, WCx, WCy, element_name) , x0 = -result2, method = numerical_method)
                    #result4 = optim4.x[0]


                    #model.WC[WCy] = result1
                    #if model.t_half(element_name) >= limit:
                    #if np.absolute((result1 - result2)/(result1+result2))>1e-3:
                    if optim1.status == 2 and optim2.status == 2:
                        xplot.append(x)
                        contour.append(result1)
                        contour2.append(result2)
                        #if result1 != result3 and result2 != result4:
                        #    xplot2.append(x)
                        #    contour3.append(result3)
                        #    contour4.append(result4)
                        model.WC[WCy] = contour[-1]
                        model.WC[WCx] = xplot[-1]
                    #print(model.t_half(element_name))

                    #set WCs to 0 just to be save
                    model.WC[WCx] = 0
                    model.WC[WCy] = 0



                #print(x0)

                #plt.plot(np.array(xplot)*1e+9, np.array(contour), "b-", linewidth = 2)
                #plt.plot(np.array(xplot)*1e+9, np.array(contour2), "r-", linewidth = 2)
                #print(len(contour))
                if WCx == "m_bb":
                    plt.xlabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                    plt.ylabel(WCy, fontsize = 20)
                    #plt.plot(np.array(xplot)*1e+9, np.array(contour), "b-", linewidth = linewidth)
                    #plt.plot(np.array(xplot)*1e+9, np.array(contour2), "b-", linewidth = linewidth)
                    plt.fill_between(np.array(xplot)*1e+9, np.array(contour), np.array(contour2), #label=experiment, 
                                     alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]), color=colors[exp_idx])
                    #plt.fill_between(np.array(xplot2)*1e+9, np.array(contour3), np.array(contour4), #color = "c", 
                    #                 label=experiment)
                elif WCy == "m_bb":
                    plt.ylabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                    plt.xlabel(WCx, fontsize = 20)
                    #plt.plot(np.array(xplot), np.array(contour)*1e+9, "b-", linewidth = linewidth)
                    #plt.plot(np.array(xplot), np.array(contour2)*1e+9, "b-", linewidth = linewidth)
                    plt.fill_between(np.array(xplot), np.array(contour)*1e+9, np.array(contour2)*1e+9, #label=experiment, 
                                     alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]), color=colors[exp_idx])
                    #plt.fill_between(np.array(xplot2), np.array(contour3)*1e+9, np.array(contour4)*1e+9, color = "c", label=experiment)
                else:
                    plt.xlabel(WCx, fontsize = 20)
                    plt.ylabel(WCy, fontsize = 20)
                    #plt.plot(np.array(xplot), np.array(contour), "b-", linewidth = linewidth)
                    #plt.plot(np.array(xplot), np.array(contour2), "b-", linewidth = linewidth)
                    plt.fill_between(np.array(xplot), np.array(contour), np.array(contour2), #label=experiment, 
                                     alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]), color=colors[exp_idx])
                    #plt.fill_between(np.array(xplot2), np.array(contour3), np.array(contour4), color = "c", 
                    #                 label=experiment)
            exp_idx += 1
        else:
            
            #lists to store contour points
            contour = []
            contour2 = []
            #contour3 = []
            #contour4 = []
            xplot = []
            xplot2 = []
            #for x in np.linspace(-radius[experiment],radius[experiment], n_dots):
            for x in np.linspace(x_min,x_max, n_dots):
                #find the scale at which to search for the minima
                x0 = 1000*scipy.optimize.root(t_half_optimize, args=(0, WCx, WCy, element_name, phase), 
                                              x0 = 1e-15).x[0]
                #x0=1
                model.WC[WCx] = x
                optim1 = scipy.optimize.root(t_half_optimize, args = (x, WCx, WCy, element_name, phase) , 
                                             x0 = x0, method = numerical_method)
                result1 = optim1.x[0]
                #model.WC[WCy] = result1

                #look for multiple roots
                #optim3 = scipy.optimize.root(t_half_optimize, args = (x, WCx, WCy, element_name) , x0 = -result1, method = numerical_method)
                #result3 = optim3.x[0]
                #model.WC[WCy] = result1


                optim2 = scipy.optimize.root(t_half_optimize, args = (x, WCx, WCy, element_name, phase) , 
                                             x0 = -x0, method = numerical_method)
                result2 = optim2.x[0]
                #model.WC[WCy] = result2

                #look for multiple roots
                #optim4 = scipy.optimize.root(t_half_optimize, args = (x, WCx, WCy, element_name) , x0 = -result2, method = numerical_method)
                #result4 = optim4.x[0]


                #model.WC[WCy] = result1
                #if model.t_half(element_name) >= limit:
                #if np.absolute((result1 - result2)/(result1+result2))>1e-3:
                if optim1.status == 2 and optim2.status == 2:
                    xplot.append(x)
                    contour.append(result1)
                    contour2.append(result2)
                    #if result1 != result3 and result2 != result4:
                    #    xplot2.append(x)
                    #    contour3.append(result3)
                    #    contour4.append(result4)
                    model.WC[WCy] = contour[-1]
                    model.WC[WCx] = xplot[-1]
                #print(model.t_half(element_name))

                #set WCs to 0 just to be save
                model.WC[WCx] = 0
                model.WC[WCy] = 0



            #print(x0)

            #plt.plot(np.array(xplot)*1e+9, np.array(contour), "b-", linewidth = 2)
            #plt.plot(np.array(xplot)*1e+9, np.array(contour2), "r-", linewidth = 2)
            print(len(contour))
            if WCx == "m_bb":
                plt.xlabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                plt.ylabel(WCy, fontsize = 20)
                #plt.plot(np.array(xplot)*1e+9, np.array(contour), "b-", linewidth = linewidth)
                #plt.plot(np.array(xplot)*1e+9, np.array(contour2), "b-", linewidth = linewidth)
                plt.fill_between(np.array(xplot)*1e+9, np.array(contour), np.array(contour2), #label=experiment, 
                                 alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]))#, color="b")
                #plt.fill_between(np.array(xplot2)*1e+9, np.array(contour3), np.array(contour4), #color = "c", 
                #                 label=experiment)
            elif WCy == "m_bb":
                plt.ylabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                plt.xlabel(WCx, fontsize = 20)
                #plt.plot(np.array(xplot), np.array(contour)*1e+9, "b-", linewidth = linewidth)
                #plt.plot(np.array(xplot), np.array(contour2)*1e+9, "b-", linewidth = linewidth)
                plt.fill_between(np.array(xplot), np.array(contour)*1e+9, np.array(contour2)*1e+9, #label=experiment, 
                                 alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]))#, color="b")
                #plt.fill_between(np.array(xplot2), np.array(contour3)*1e+9, np.array(contour4)*1e+9, color = "c", label=experiment)
            else:
                plt.xlabel(WCx, fontsize = 20)
                plt.ylabel(WCy, fontsize = 20)
                #plt.plot(np.array(xplot), np.array(contour), "b-", linewidth = linewidth)
                #plt.plot(np.array(xplot), np.array(contour2), "b-", linewidth = linewidth)
                plt.fill_between(np.array(xplot), np.array(contour), np.array(contour2), #label=experiment, 
                                 alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]))#, color="b")
                #plt.fill_between(np.array(xplot2), np.array(contour3), np.array(contour4), color = "c", 
                #                
    if WCx == "m_bb":
        plt.xlim([x_min*1e+9, x_max*1e+9])
    else:
        plt.xlim([x_min, x_max])
       
    legend_elements = []
    for exp_idx in range(len(experiments)):
        legend_elements.append(Line2D([0], [0], color = colors[exp_idx], 
                                      label=list(experiments.keys())[exp_idx],
                                      markerfacecolor=colors[exp_idx], markersize=10))
    plt.legend(handles = legend_elements, fontsize=20)
    
    plt.rc("ytick", labelsize = 15)
    plt.rc("xtick", labelsize = 15)
    if savefig:
        if file == None:
            file = "contours_"+WCx+"_"+WCy+".png"
        plt.savefig(file)
        

    return(fig)

