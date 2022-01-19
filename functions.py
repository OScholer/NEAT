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
                  numerical_method="lm", n_dots=5000, linewidth=0, x_min=None, x_max=None, 
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
            #print(len(contour))
            #print(xplot)
            #print(contour)
            #print(contour2)
            if WCx == "m_bb":
                plt.xlabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$", fontsize = 20)
                #plt.plot(np.array(xplot)*1e+9, np.array(contour), "b-", linewidth = linewidth)
                #plt.plot(np.array(xplot)*1e+9, np.array(contour2), "b-", linewidth = linewidth)
                plt.fill_between(np.array(xplot)*1e+9, np.array(contour), np.array(contour2), #label=experiment, 
                                 alpha=np.min([1/(2*1), 1/(2*len(experiments))]), color=colors[exp_idx])
                #plt.fill_between(np.array(xplot2)*1e+9, np.array(contour3), np.array(contour4), #color = "c", 
                #                 label=experiment)
            elif WCy == "m_bb":
                plt.ylabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$", fontsize = 20)
                #plt.plot(np.array(xplot), np.array(contour)*1e+9, "b-", linewidth = linewidth)
                #plt.plot(np.array(xplot), np.array(contour2)*1e+9, "b-", linewidth = linewidth)
                plt.fill_between(np.array(xplot), np.array(contour)*1e+9, np.array(contour2)*1e+9, #label=experiment, 
                                 alpha=np.min([1/(2*1), 1/(2*len(experiments))]), color=colors[exp_idx])
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
                                     alpha=np.min([1/(2*1), 1/(2*len(experiments))]), color=colors[exp_idx])
                else:
                    plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$", fontsize = 20)
                    plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$", fontsize = 20)
                    #plt.plot(np.array(xplot), np.array(contour), "b-", linewidth = linewidth)
                    #plt.plot(np.array(xplot), np.array(contour2), "b-", linewidth = linewidth)
                    plt.fill_between(np.array(xplot), np.array(contour), np.array(contour2), #label=experiment, 
                                     alpha=np.min([1/(2*1), 1/(2*len(experiments))]), color=colors[exp_idx])
                    #plt.fill_between(np.array(xplot2), np.array(contour3), np.array(contour4), color = "c", 
                    #                 label=experiment)   
        exp_idx += 1
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



#################################################################################################################################
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                         Generate analytical expression for the decay rate                                     #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################
def generate_formula_coefficients(WC, method = "IBM2"):
    C = {}

    for WC1 in WC:
        model1 = EFT.LEFT({WC1:1}, method=method)
        thalf1 = model1.half_lives()
        C[WC1] = 1/thalf1
        for WC2 in WC:
            if WC2 != WC1:
                model2 = EFT.LEFT({WC2:1}, method=method)
                model3 = EFT.LEFT({WC1:1, 
                               WC2:1}, method=method)
                thalf2 = model2.half_lives()
                thalf3 = model3.half_lives()
                C[WC2] = 1/thalf2
                C[WC1+WC2] = 1/thalf3-(1/thalf2+1/thalf1)
    return(C)

def generate_terms(WC, isotope = "76Ge", output = "latex", method = "IBM2", decimal = 2):
    C = generate_formula_coefficients(WC, method)
    if output not in ["latex", "html"]:
        raise ValueError("output must be either 'latex' or 'html'")
    terms = {}
    for WC1 in WC:
        exponent = int(np.floor(np.log10(C[WC1][isotope][0])))
        prefactor = np.round(C[WC1][isotope][0]*10**(-exponent), decimal)
        if WC1 == "m_bb":
            if output == "latex":
                WC1string = "\\frac{m_{\\beta\\beta}}{1\mathrm{GeV}}"
            elif output == "html":
                WC1string = "m<sub>&beta;&beta;</sub>"
                
        elif WC1[-5:] == "prime":
            if output == "latex":
                WC1string = "C_{"+WC1[:-8]+"}^{"+WC1[-8:-5]+"}`"
            elif output == "html":
                WC1string = "C<sub>"+WC1[:-8]+"</sub><sup>"+WC1[-8:-5]+"</sup>'"
        else:
            if output == "latex":
                WC1string = "C_{"+WC1[:-3]+"}^{"+WC1[-3:]+"}"
            elif output == "html":
                WC1string = "C<sub>"+WC1[:-3]+"</sub><sup>"+WC1[-3:]+"</sup>"
        if output == "latex":
            terms[WC1] = "$"+str(prefactor)+"\\times 10^{"+str(exponent)+"}|"+WC1string+"|^2$"
        elif output == "html":
            terms[WC1] = str(prefactor)+"&times;10<sup>"+str(exponent)+"</sup>|"+WC1string+"|<sup>2</sup>"
        for WC2 in WC:
            if WC2 not in terms:
                #add second WC
                exponent = int(np.floor(np.log10(C[WC2][isotope][0])))
                prefactor = np.round(C[WC2][isotope][0]*10**(-exponent), decimal)
                if WC2 == "m_bb":
                    if output == "latex":
                        WC2string = "\\frac{m_{\\beta\\beta}}{1\mathrm{GeV}}"
                    elif output == "html":
                        WC2string = "m<sub>&beta;&beta;</sub>"
                elif WC2[-5:] == "prime":
                    if output == "latex":
                        WC2string = "C_{"+WC2[:-8]+"}^{"+WC2[-8:-5]+"}`"
                    elif output == "html":
                        WC2string = "C<sub>"+WC2[:-8]+"</sub><sup>"+WC2[-8:-5]+"</sup>'"
                else:
                    if output == "latex":
                        WC2string = "C_{"+WC2[:-3]+"}^{"+WC2[-3:]+"}"
                    elif output == "html":
                        WC2string = "C<sub>"+WC2[:-3]+"</sub><sup>"+WC2[-3:]+"</sup>"
                if output == "latex":
                    terms[WC2] = "$"+str(prefactor)+"\\times 10^{"+str(exponent)+"}|"+WC2string+"|^2$"
                elif output == "html":
                    terms[WC2] = str(prefactor)+"&times;10<sup>"+str(exponent)+"</sup>|"+WC2string+"|<sup>2</sup>"
                    
            
            if WC2+WC1 not in terms and WC1 != WC2:
                #add interference terms
                if C[WC1+WC2][isotope][0] != 0:
                    exponent = int(np.floor(np.log10(np.abs(C[WC1+WC2][isotope][0]))))
                    prefactor = np.round(C[WC1+WC2][isotope][0]*10**(-exponent), decimal)
                    if output == "latex":
                        terms[WC1+WC2] = ("$"+str(prefactor)+"\\times 10^{"+str(exponent)+"} \\mathrm{Re}["+WC1string+"({"+WC2string+"})^*]$")
                    if output == "html":
                        terms[WC1+WC2] = (str(prefactor)+"&times; 10<sup>"+str(exponent)+"</sup> Re&#91;"+WC1string+"("+WC2string+")<sup>&#42;</sup>&#93;")
                    
    return(terms)

def generate_formula(WC, isotope = "76Ge", output = "latex", method = "IBM2", decimal = 2):
    terms = generate_terms(WC, isotope, output, method, decimal)
    if output == "latex":
        formula = r"$T_{1/2}^{-1} = "
    elif output == "html":
        formula = "T<sub>1/2</sub><sup>-1</sup> = "
    for WC in WC:
        if output == "latex":
            formula+="+"+terms[WC][1:-1]
        elif output == "html":
            formula+=" +"+terms[WC]
    for WC1 in WC:
        for WC2 in WC:
            if WC2 != WC1 and WC1+WC2 in terms:
                if output == "latex":
                    if terms[WC1+WC2][1] != "-":
                        formula += "+"
                    formula+=terms[WC1+WC2][1:-1]
                elif output == "html":
                    if terms[WC1+WC2][0] != "-":
                        formula += " +"
                    else:
                        formula += " "
                    formula+=terms[WC1+WC2]
    if output == "latex":
        formula+="$"
    return(formula)
#################################################################################################################################
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                         Generate matrix for decay rate C^dagger M C = T^-1                                    #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################

def generate_matrix_coefficients(WC, isotope = "76Ge", method = "IBM2"):
    C = {}

    for WC1 in WC:
        #if WC1 == "m_bb":
        #    factor1 = 1e-9
        #else:
        #    factor1 = 1
        model1 = EFT.LEFT({WC1:1}, method=method)#*factor1})
        thalf1 = model1.half_lives()[isotope][0]
        #C[WC1] = 1/thalf1
        for WC2 in WC:
            #if WC2 = WC1:
            #if WC2 == "m_bb":
            #    factor2 = 1e-9
            #else:
            #    factor2 = 1
            model2 = EFT.LEFT({WC2:1}, method=method)
            model3 = EFT.LEFT({WC1:1,
                           WC2:1}, method=method)
            thalf2 = model2.half_lives()[isotope][0]
            thalf3 = model3.half_lives()[isotope][0]
            #C[WC2] = 1/thalf2
            if WC1 == WC2:
                C[WC1+WC2] = 1/thalf1
            else:
                C[WC1+WC2] = 1/2*(1/thalf3-(1/thalf2+1/thalf1))
    return(C)

def generate_matrix(WC, isotope = "76Ge", method = "IBM2"):
    C = generate_matrix_coefficients(WC, isotope, method)
    
    M = np.zeros([len(WC), len(WC)])
    for idx1 in range(len(WC)):
        try:
            WC1 = list(WC.keys())[idx1]
        except:
            WC1 = WC[idx1]
        for idx2 in range(len(WC)):
            try:
                WC2 = list(WC.keys())[idx2]
            except:
                WC2 = WC[idx2]
            value = C[WC1+WC2]
            M[idx1, idx2] = value
    return(M)



#################################################################################################################################
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                         Neutrino Physics Formulae (mixing and masses).                                        #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################


def m_bb(alpha, m_min=1, ordering="NO", dcp=1.36):
    #this function returns m_bb from m_min and the majorana mixing matrix

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
    s23 = np.sqrt(0.546)
    s13 = np.sqrt(2.2e-2)

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
    
    
    
#translate the sum of neutrino masses to the minimal neutrino mass
def m_sum_to_m_min(m_sum):
    def m_sum_NO(m_min, m_sum):
        m21 = 7.53e-5
        m32 = 2.453e-3
        m32IO = -2.546e-3
        
        m1 = m_min
        m2 = np.sqrt(m1**2+m21)
        m3 = np.sqrt(m2**2+m32)
        
        msum = m1+m2+m3
        return(msum-m_sum)
    
    def m_sum_IO(m_min, m_sum):
        m21 = 7.53e-5
        m32 = 2.453e-3
        m32IO = -2.546e-3

        m3IO = m_min
        m2IO = np.sqrt(m3IO**2-m32IO)
        m1IO = np.sqrt(m2IO**2-m21)
        
        msum = m1IO+m2IO+m3IO
        return(msum-m_sum)
    
    m_min_NO = scipy.optimize.root(m_sum_NO, x0 = 0.05, args = [m_sum]).x[0]
    m_min_IO = scipy.optimize.root(m_sum_IO, x0 = 0.05, args = [m_sum]).x[0]
    return({"NO" : m_min_NO, "IO" : m_min_IO})


#translate the minimal neutrino mass to the sum of neutrino masses
def m_min_to_m_sum(m_min):
    def m_sum_NO(m_min):
        m21 = 7.53e-5
        m32 = 2.453e-3
        m32IO = -2.546e-3
        
        m1 = m_min
        m2 = np.sqrt(m1**2+m21)
        m3 = np.sqrt(m2**2+m32)
        
        msum = m1+m2+m3
        return(msum)
    
    def m_sum_IO(m_min):
        m21 = 7.53e-5
        m32 = 2.453e-3
        m32IO = -2.546e-3

        m3IO = m_min
        m2IO = np.sqrt(m3IO**2-m32IO)
        m1IO = np.sqrt(m2IO**2-m21)
        
        msum = m1IO+m2IO+m3IO
        return(msum)
    
    return({"NO" : m_sum_NO(m_min), "IO" : m_sum_IO(m_min)})
