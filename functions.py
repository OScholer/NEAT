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
                  numerical_method="lm", n_dots=1000, linewidth=0, x_min=None, x_max=None, 
                  savefig=False, phase=3/4*np.pi, varyphases = False, n_vary=5):#, eft = "LEFT"):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #if eft == "LEFT":
    SMEFT_WCs = EFT.SMEFT_WCs
    g_A = 1.27
    def get_constants(element_name):
        model=EFT.LEFT({}, method=method)
        is_SMEFT = False
        if WCx in SMEFT_WCs:
            is_SMEFT = True
            SMEFT_model = EFT.SMEFT({}, scale = 1e+3)
            WCsx = SMEFT_model.LEFT_matching(WC={WCx:1})
            WCsy = SMEFT_model.LEFT_matching(WC={WCy:1})
            WC = {}
            WC[WCx] = WCsx
            WC[WCy] = WCsy
            for operator in model.WC:
                model.WC[operator] = WC[WCx][operator]
            Ax, Mx = model.amplitudes(element_name = element_name, WC = model.WC)
            tx = model.t_half(element_name = element_name)
            for operator in model.WC:
                model.WC[operator] = 0
            for operator in model.WC:
                model.WC[operator] = WC[WCy][operator]
            Ay, My = model.amplitudes(element_name = element_name, WC = model.WC)
            ty = model.t_half(element_name = element_name)
            for operator in model.WC:
                model.WC[operator] = 0
        else:
            for operator in model.WC:
                model.WC[operator] = 0
            model.WC[WCx] = 1
            Ax, Mx = model.amplitudes(element_name = element_name, WC = model.WC)
            tx = model.t_half(element_name = element_name)
            model.WC[WCx] = 0
            model.WC[WCy] = 1
            Ay, My = model.amplitudes(element_name = element_name, WC = model.WC)
            ty = model.t_half(element_name = element_name)
            model.WC[WCy] = 0
        element = model.elements[element_name]
        G = model.to_G(element_name)
        #m_e = pc["electron mass energy equivalent in MeV"][0]

        #Some PSFs need a rescaling due to different definitions in DOIs paper and 1806...
        g_06_rescaling = model.m_e_MEV*element.R/2
        g_09_rescaling = g_06_rescaling**2
        g_04_rescaling = 9/2
        G["06"] *= g_06_rescaling
        G["04"] *= g_04_rescaling
        G["09"] *= g_09_rescaling
        return(model, tx, Ax, ty, Ay, G, is_SMEFT)
    
    #model=EFT.LEFT({}, method=method)
    #else:
    #    model = EFT.SMEFT({}, method = method, scale = 1e+3)
    

    if (x_min == None or x_max == None) and x_min != x_max:
        print("You need to set both x_min and x_max or let the code choose both!")
        return()

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
        model, tx, Ax, ty, Ay, G, is_SMEFT = get_constants(element_name)
        if is_SMEFT:
            fac = 3
        else:
            fac = 1.5
        #if experiment == list(experiments.keys())[0]:
        #find the range of the curve by finding the maximal value 
        #radius[experiment] = 1.5*scipy.optimize.root(t_half_optimize, args=(0, WCx, WCy, element_name, phase), x0 = 1e-15).x[0]
        radius[experiment] = fac * np.sqrt(tx/limit)
        #radius2 = 1.2*scipy.optimize.fsolve(t_half_optimize, args=(0, WCy, WCx, element_name), x0 = 1e-15, maxfev=10000)[0]

        if x_min == None:
            x_min = -radius[experiment]
            x_max =  radius[experiment]
        elif radius[experiment] > x_max:
            x_min = -radius[experiment]
            x_max =  radius[experiment]
            #print("radius",radius)
            #print("radius2",radius2)
    print(x_min, x_max)
    for experiment in experiments:
        limit = experiments[experiment][0]
        element_name = experiments[experiment][1]
        model, tx, Ax, ty, Ay, G, is_SMEFT = get_constants(element_name)
        
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

                    #Calculate half-life following eq 38. in 1806.02780
                    a = 1/ty
                    b = 2*np.cos(phase)*x*g_A**4*(G["01"] * (Ax["nu"]*Ay["nu"]
                                                             + Ax["R"]*Ay["R"])
                                                  - 2 * (G["01"] - G["04"])*(Ax["nu"]*Ay["R"]
                                                                             +Ay["nu"]*Ax["R"])
                                                  + 4 *  G["02"]* (Ax["E"]*Ay["E"])
                                                  + 2 *  G["04"]*(Ax["me"]*Ay["me"] 
                                                                  + (Ax["me"]*(Ay["nu"]+Ay["R"]))
                                                                  + (Ay["me"]*(Ax["nu"]+Ax["R"])))
                                                  - 2 *  G["03"]*((Ax["nu"]+Ax["R"])*(Ay["E"]) 
                                                                  + (Ay["nu"]+Ay["R"])*(Ax["E"])
                                                                  + 2*Ax["me"]*Ay["E"]
                                                                  + 2*Ay["me"]*Ax["E"])
                                                  + G["09"] * Ax["M"]*Ay["M"]
                                                  + G["06"] * ((Ax["nu"]-Ax["R"])*(Ay["M"]) 
                                                               + (Ay["nu"]-Ay["R"])*(Ax["M"])))
                    c = 1/tx*x**2 - 1/limit
                    
                    
                    y = (-b + np.sqrt(b**2-4*a*c+0*1j))/(2*a)
                    y2 = (-b - np.sqrt(b**2-4*a*c+0*1j))/(2*a)


                    if y.imag == 0 and y2.imag == 0:
                        xplot.append(x)
                        contour.append(y)
                        contour2.append(y2)



                if WCx == "m_bb":
                    plt.xlabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                    plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$", fontsize = 20)
                    #plt.plot(np.array(xplot)*1e+9, np.array(contour), "b-", linewidth = linewidth)
                    #plt.plot(np.array(xplot)*1e+9, np.array(contour2), "b-", linewidth = linewidth)
                    plt.fill_between(np.array(xplot)*1e+9, np.array(contour), np.array(contour2), #label=experiment, 
                                     alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]), color=colors[exp_idx])
                    #plt.fill_between(np.array(xplot2)*1e+9, np.array(contour3), np.array(contour4), #color = "c", 
                    #                 label=experiment)
                elif WCy == "m_bb":
                    plt.ylabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                    plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$", fontsize = 20)
                    #plt.plot(np.array(xplot), np.array(contour)*1e+9, "b-", linewidth = linewidth)
                    #plt.plot(np.array(xplot), np.array(contour2)*1e+9, "b-", linewidth = linewidth)
                    plt.fill_between(np.array(xplot), np.array(contour)*1e+9, np.array(contour2)*1e+9, #label=experiment, 
                                     alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]), color=colors[exp_idx])
                    #plt.fill_between(np.array(xplot2), np.array(contour3)*1e+9, np.array(contour4)*1e+9, color = "c", label=experiment)
                else:
                    if is_SMEFT:
                        xdimension = int(WCx[-2])
                        ydimension = int(WCy[-2])
                        plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$ [TeV$^-"+str(xdimension-4)+"$]", fontsize = 20)
                        plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$ [TeV$^-"+str(ydimension-4)+"$]", fontsize = 20)
                        plt.fill_between((1e+3)**(xdimension-4)*np.array(xplot), 
                                         (1e+3)**(ydimension-4)*np.array(contour), 
                                         (1e+3)**(ydimension-4)*np.array(contour2), #label=experiment, 
                                         alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]), color=colors[exp_idx])
                    
                    else:
                        plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$", fontsize = 20)
                        plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$", fontsize = 20)
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
                
            #for x in np.linspace(-radius[experiment],radius[experiment], n_dots):
            for x in np.linspace(x_min,x_max, n_dots):
                #find the scale at which to search for the minima

                #Calculate half-life following eq 38. in 1806.02780
                a = 1/ty
                b = 2*np.cos(phase)*x*g_A**4*(G["01"] * (Ax["nu"]*Ay["nu"]
                                                         + Ax["R"]*Ay["R"])
                                              - 2 * (G["01"] - G["04"])*(Ax["nu"]*Ay["R"]
                                                                         +Ay["nu"]*Ax["R"])
                                              + 4 *  G["02"]* (Ax["E"]*Ay["E"])
                                              + 2 *  G["04"]*(Ax["me"]*Ay["me"] 
                                                              + (Ax["me"]*(Ay["nu"]+Ay["R"]))
                                                              + (Ay["me"]*(Ax["nu"]+Ax["R"])))
                                              - 2 *  G["03"]*((Ax["nu"]+Ax["R"])*(Ay["E"]) 
                                                              + (Ay["nu"]+Ay["R"])*(Ax["E"])
                                                              + 2*Ax["me"]*Ay["E"]
                                                              + 2*Ay["me"]*Ax["E"])
                                              + G["09"] * Ax["M"]*Ay["M"]
                                              + G["06"] * ((Ax["nu"]-Ax["R"])*(Ay["M"]) 
                                                           + (Ay["nu"]-Ay["R"])*(Ax["M"])))
                c = 1/tx*x**2 - 1/limit


                y = (-b + np.sqrt(b**2-4*a*c+0*1j))/(2*a)
                y2 = (-b - np.sqrt(b**2-4*a*c+0*1j))/(2*a)


                if y.imag == 0 and y2.imag == 0:
                    xplot.append(x)
                    contour.append(y)
                    contour2.append(y2)



            #print(x0)

            #plt.plot(np.array(xplot)*1e+9, np.array(contour), "b-", linewidth = 2)
            #plt.plot(np.array(xplot)*1e+9, np.array(contour2), "r-", linewidth = 2)
            print(len(contour))
            
            if WCx == "m_bb":
                plt.xlabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$", fontsize = 20)
                #plt.plot(np.array(xplot)*1e+9, np.array(contour), "b-", linewidth = linewidth)
                #plt.plot(np.array(xplot)*1e+9, np.array(contour2), "b-", linewidth = linewidth)
                plt.fill_between(np.array(xplot)*1e+9, np.array(contour), np.array(contour2), #label=experiment, 
                                 alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]), color=colors[exp_idx])
                #plt.fill_between(np.array(xplot2)*1e+9, np.array(contour3), np.array(contour4), #color = "c", 
                #                 label=experiment)
            elif WCy == "m_bb":
                plt.ylabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$", fontsize = 20)
                #plt.plot(np.array(xplot), np.array(contour)*1e+9, "b-", linewidth = linewidth)
                #plt.plot(np.array(xplot), np.array(contour2)*1e+9, "b-", linewidth = linewidth)
                plt.fill_between(np.array(xplot), np.array(contour)*1e+9, np.array(contour2)*1e+9, #label=experiment, 
                                 alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]), color=colors[exp_idx])
                #plt.fill_between(np.array(xplot2), np.array(contour3)*1e+9, np.array(contour4)*1e+9, color = "c", label=experiment)
            else:
                if is_SMEFT:
                    xdimension = int(WCx[-2])
                    ydimension = int(WCy[-2])
                    plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$ [TeV$^-"+str(xdimension-4)+"$]", fontsize = 20)
                    plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$ [TeV$^-"+str(ydimension-4)+"$]", fontsize = 20)
                    plt.fill_between((1e+3)**(xdimension-4)*np.array(xplot), 
                                     (1e+3)**(ydimension-4)*np.array(contour), 
                                     (1e+3)**(ydimension-4)*np.array(contour2), #label=experiment, 
                                     alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]), color=colors[exp_idx])
                else:
                    plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$", fontsize = 20)
                    plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$", fontsize = 20)
                    #plt.plot(np.array(xplot), np.array(contour), "b-", linewidth = linewidth)
                    #plt.plot(np.array(xplot), np.array(contour2), "b-", linewidth = linewidth)
                    plt.fill_between(np.array(xplot), np.array(contour), np.array(contour2), #label=experiment, 
                                     alpha=np.min([1/(2*n_vary), 1/(2*len(experiments))]), color=colors[exp_idx])
                    #plt.fill_between(np.array(xplot2), np.array(contour3), np.array(contour4), color = "c", 
                    #                 label=experiment)              
    if WCx == "m_bb":
        plt.xlim([x_min*1e+9, x_max*1e+9])
    elif is_SMEFT:
        plt.xlim([x_min*1e+3**(xdimension-4), x_max*1e+3**(xdimension-4)])
        
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
    plt.tight_layout()
    if savefig:
        if file == None:
            file = "contours_"+WCx+"_"+WCy+".png"
        plt.savefig(file)
        

    #print(optim1)
    return(fig)


