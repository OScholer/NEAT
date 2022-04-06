import streamlit as st
import EFT
from math import floor, log10
import numpy as np
from scipy import integrate
import pandas as pd
import base64
import functions as f
#st.write("Hello")

st.title("The tool which must not be named")
st.markdown('If you use this tool please cite **arxiv:xxxx.xxxx**. <br>You can download the full code from <a href = "https://github.com">gitHub</a> for advanced use cases.<br> You have any suggestions/comments on how to improve the tool?<br> => Contact:<br>Lukas Graf: lukas.graf@mpi-hd.mpg.de<br> Oliver Scholer: scholer@mpi-hd.mpg.de<br> Jordy de Vries:', unsafe_allow_html=True)
path_option = st.selectbox("Choose what you want to do:", options = ["-", "Define a model", "Study operator limits"])



####################################################################################################
#                                                                                                  #
#                                                                                                  #
#                                                                                                  #
#                                 DEFINE A MODEL IN SMEFT OR LEFT                                  #
#                                                                                                  #
#                                                                                                  #
#                                                                                                  #
####################################################################################################

if path_option == "Define a model":
    name = st.text_input("If you want you can give your model name. This name will be displayed in all plots", value="Model")
    method = st.sidebar.selectbox("Which NME approximation do you want to use?", options = ["IBM2", "QRPA", "SM"], help = "Currently we allow for 3 different sets of nuclear matrix elements (NMEs): IBM2: F. Deppisch et al., 2020, arxiv:2009.10119 | QRPA: J. Hyvärinen and J. Suhonen, 2015, Phys. Rev. C 91, 024613 | Shell Model (SM): J. Menéndez, 2018, arXiv:1804.02105")
    model_option = st.sidebar.selectbox("Do you want to define Wilson coefficients at LEFT or SMEFT?", options = ["-","LEFT", "SMEFT"])
    if model_option == "LEFT":
        phases = st.sidebar.checkbox("Allow for complex phases?", help = "If you check this box you can set complex phases for each Wilson coefficient.")
        LEFT_WCs = {}
        cols = {}
        st.sidebar.subheader("Effective Neutrino Mass")
        
        LEFT_WCs["m_bb"] = st.sidebar.number_input("m_bb [meV]", value = 100.)*1e-12
        if phases:
            LEFT_WCs["m_bb"] *= np.exp(1j*st.sidebar.number_input("m_bb phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        st.sidebar.subheader("Dimension 6")
        ##############
        LEFT_WCs["SL(6)"] = st.sidebar.number_input("SL(6) [10^-9]")*1e-9
        if phases:
            LEFT_WCs["SL(6)"] *= np.exp(1j*st.sidebar.number_input("SL(6) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["SR(6)"] = st.sidebar.number_input("SR(6) [10^-9]")*1e-9
        if phases:
            LEFT_WCs["SR(6)"] *= np.exp(1j*st.sidebar.number_input("SR(6) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["VL(6)"] = st.sidebar.number_input("VL(6) [10^-9]")*1e-9
        if phases:
            LEFT_WCs["VL(6)"] *= np.exp(1j*st.sidebar.number_input("VL(6) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["VR(6)"] = st.sidebar.number_input("VR(6) [10^-9]")*1e-9
        if phases:
            LEFT_WCs["VR(6)"] *= np.exp(1j*st.sidebar.number_input("VR(6) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["T(6)"] = st.sidebar.number_input("T(6) [10^-9]")*1e-9
        if phases:
            LEFT_WCs["T(6)"] *= np.exp(1j*st.sidebar.number_input("T(6) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        st.sidebar.subheader("Dimension 7")
        LEFT_WCs["VL(7)"] = st.sidebar.number_input("VL(7) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["VL(7)"] *= np.exp(1j*st.sidebar.number_input("VL(7) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["VR(7)"] = st.sidebar.number_input("VR(7) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["VR(7)"] *= np.exp(1j*st.sidebar.number_input("VR(7) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        st.sidebar.subheader("Dimension 9")
        LEFT_WCs["1L(9)"] = st.sidebar.number_input("1L(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["1L(9)"] *= np.exp(1j*st.sidebar.number_input("1L(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["1L(9)prime"] = st.sidebar.number_input("1L(9)' [10^-6]")*1e-6
        if phases:
            LEFT_WCs["1L(9)prime"] *= np.exp(1j*st.sidebar.number_input("1L(9)' phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["1R(9)"] = st.sidebar.number_input("1R(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["1R(9)"] *= np.exp(1j*st.sidebar.number_input("1R(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["1R(9)prime"] = st.sidebar.number_input("1R(9)' [10^-6]")*1e-6
        if phases:
            LEFT_WCs["1R(9)prime"] *= np.exp(1j*st.sidebar.number_input("1R(9)' phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["2L(9)"] = st.sidebar.number_input("2L(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["2L(9)"] *= np.exp(1j*st.sidebar.number_input("2L(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["2L(9)prime"] = st.sidebar.number_input("2L(9)' [10^-6]")*1e-6
        if phases:
            LEFT_WCs["2L(9)prime"] *= np.exp(1j*st.sidebar.number_input("2L(9)' phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["2R(9)"] = st.sidebar.number_input("2R(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["2R(9)"] *= np.exp(1j*st.sidebar.number_input("2R(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["2R(9)prime"] = st.sidebar.number_input("2R(9)' [10^-6]")*1e-6
        if phases:
            LEFT_WCs["2R(9)prime"] *= np.exp(1j*st.sidebar.number_input("2R(9)' phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["3L(9)"] = st.sidebar.number_input("3L(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["3L(9)"] *= np.exp(1j*st.sidebar.number_input("3L(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["3L(9)prime"] = st.sidebar.number_input("3L(9)' [10^-6]")*1e-6
        if phases:
            LEFT_WCs["3L(9)prime"] *= np.exp(1j*st.sidebar.number_input("3L(9)' phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["3R(9)"] = st.sidebar.number_input("3R(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["3R(9)"] *= np.exp(1j*st.sidebar.number_input("3R(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["3R(9)prime"] = st.sidebar.number_input("3R(9)' [10^-6]")*1e-6
        if phases:
            LEFT_WCs["3R(9)prime"] *= np.exp(1j*st.sidebar.number_input("3R(9)' phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["4L(9)"] = st.sidebar.number_input("4L(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["4L(9)"] *= np.exp(1j*st.sidebar.number_input("4L(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["4R(9)"] = st.sidebar.number_input("4R(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["4R(9)"] *= np.exp(1j*st.sidebar.number_input("4R(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["5L(9)"] = st.sidebar.number_input("5L(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["5L(9)"] *= np.exp(1j*st.sidebar.number_input("5L(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["5R(9)"] = st.sidebar.number_input("5R(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["5R(9)"] *= np.exp(1j*st.sidebar.number_input("5R(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["6(9)"] = st.sidebar.number_input("6(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["6(9)"] *= np.exp(1j*st.sidebar.number_input("6(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["6(9)prime"] = st.sidebar.number_input("6(9)' [10^-6]")*1e-6
        if phases:
            LEFT_WCs["6(9)prime"] *= np.exp(1j*st.sidebar.number_input("6(9)' phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["7(9)"] = st.sidebar.number_input("7(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["7(9)"] *= np.exp(1j*st.sidebar.number_input("7(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["7(9)prime"] = st.sidebar.number_input("7(9)' [10^-6]")*1e-6
        if phases:
            LEFT_WCs["7(9)prime"] *= np.exp(1j*st.sidebar.number_input("7(9)' phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["8(9)"] = st.sidebar.number_input("8(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["8(9)"] *= np.exp(1j*st.sidebar.number_input("8(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["8(9)prime"] = st.sidebar.number_input("8(9)' [10^-6]")*1e-6
        if phases:
            LEFT_WCs["8(9)prime"] *= np.exp(1j*st.sidebar.number_input("8(9)' phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["9(9)"] = st.sidebar.number_input("9(9) [10^-6]")*1e-6
        if phases:
            LEFT_WCs["9(9)"] *= np.exp(1j*st.sidebar.number_input("9(9) phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        LEFT_WCs["9(9)prime"] = st.sidebar.number_input("9(9)' [10^-6]")*1e-6
        if phases:
            LEFT_WCs["9(9)prime"] *= np.exp(1j*st.sidebar.number_input("9(9)' phase [pi]")*np.pi)
            st.sidebar.write("________________________________")
        ##############
        
        

        LEFT_model = EFT.LEFT(LEFT_WCs, method=method, name = name)
        #st.dataframe(LEFT_model.PSFpanda.T)
        #st.table(LEFT_model.PSFpanda)
        st.subheader("Half-lives")
        hl = LEFT_model.half_lives()
        hl.rename(index = {"$y$": "years"}, inplace = True)
        if np.inf not in hl.values:
            hl = hl.applymap(lambda x: round(x, 2 - int(floor(log10(abs(x))))))
        def get_table_download_link_csv(df):
            #csv = df.to_csv(index=False)
            csv = df.to_csv().encode()
            latex = df.to_latex().encode()
            #b64 = base64.b64encode(csv.encode()).decode() 
            b64 = base64.b64encode(csv).decode()
            href = f'Download half-lives as <a href="data:file/csv;base64,{b64}" download="LEFT_model_half_lives.csv" target="_blank">.csv</a> or as <a href="data:file/latex;base64,{b64}" download="LEFT_model_half_lives.tex" target="_blank">.tex</a> file.'
            return href
        st.markdown(get_table_download_link_csv(hl.T), unsafe_allow_html=True)
        st.table(hl.T)
        st.subheader("Angular correlation")
        ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
        plot_isotope = st.selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, 
                                    key = "angularcorrisotope")
        #st.line_chart({name: np.real(LEFT_model.angular_corr(np.linspace(1e-5,1-1e-5, 1000)))})
        show_mbb1 = st.checkbox("Compare to mass mechanism?", key="show_mbb1")
        fig_angular_corr = LEFT_model.plot_corr(show_mbb=show_mbb1, isotope = plot_isotope)
        st.pyplot(fig_angular_corr)
        st.subheader("Normalized single electron spectrum")
        ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
        plot_isotope2 = st.selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, 
                                     key = "spectraisotope")
        integral = integrate.quad(lambda E: LEFT_model.spectrum(E), 0, 1)[0]
        #st.line_chart({name: np.real(LEFT_model.spectra(np.linspace(1e-5,1-1e-5, 1000))/integral)})
        show_mbb2 = st.checkbox("Compare to mass mechanism?", key="show_mbb2")
        fig_spec = LEFT_model.plot_spec(show_mbb=show_mbb2, isotope = plot_isotope2)
        st.pyplot(fig_spec)
        st.subheader("Half-life ratios")
        reference_isotope = st.selectbox("Choose the reference isotope:", options = LEFT_model.isotope_names, index = ge_idx)
        ratio_option_cols = st.beta_columns(2)
        compare = ratio_option_cols[0].checkbox("Compare to mass mechanism?", help = "If you check this box we will normalize the ratios to the mass mechanisms ratio values")
        vary = ratio_option_cols[1].checkbox("Vary unknown LECs?", help = "If you check this box we will vary all unknown LECs around their order of magnitude estimate O (i.e. from log_10(O) to log10(O+1)) . g_nuNN will be varied 50% around it's theoretical estimate.")
        if vary:
            n_points = st.number_input("How many variations do you want to run? Remember: The higher this number the longer the calculation takes..." , value=100)
        else:
            n_points = 1
        #fig = LEFT_model.ratios(plot=True, vary = vary, n_points = n_points, 
        #                        normalized = compare, reference_isotope = reference_isotope)
        fig = LEFT_model.plot_ratios(vary = vary, n_points = n_points, 
                                normalized = compare, reference_isotope = reference_isotope)
        st.pyplot(fig)
        st.subheader("Vary single Wilson coefficients")
        def plots(plotidx):
            plotoptions = st.selectbox("Choose additional figures you want to see. These plots take a few seconds...", 
                                        options = ["-", "m_eff", "half_life", "1/half_life"], key = "chooseplottype"+str(plotidx))
            #st.write("Note: Any value of m_bb that you set for you model will be replaced for these plots according to the minimal neutrino mass. If you didn't set any value we will assume that your model is present additionally to the standard mass mechanism.")
            #if plotoptions == "m_eff":
            if plotoptions in ["m_eff", "half_life", "1/half_life"]:

                ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
                plot_cols = st.beta_columns(3)
                plot_isotope = plot_cols[0].selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, key = "isotope"+str(plotidx))
                scatter_or_line = plot_cols[1].selectbox("Choose the plot-type", options = ["Scatter", "Line"], key = "plottype"+str(plotidx), help = "Scatter plots vary all the relevant parameters and generate a number of scenarios while line plots calculate the minimum and maximum by running an optimization algorithm. If you want to vary also the LECs you will need to choose scatter plots.")
                vary_WC = plot_cols[2].selectbox("X-axis WC", options = np.append(["m_min", "m_sum"], np.array(list(LEFT_model.WC.keys()))[np.array(list(LEFT_model.WC.values()))!=0]), key = "vary"+str(plotidx), help = "Choose the Wilson coefficient you want to vary on the x-axis")
                show_cosmo = False
                m_cosmo = 0.15
                #show_cosmo = st.checkbox("Show cosmology limit?", key =plotoptions)
                if scatter_or_line == "Line":
                    xlim_cols = st.beta_columns(3)
                    if vary_WC == "m_min":
                        x_min = 10**xlim_cols[0].number_input("Minimum m_min 10^...[eV]", value = -4., key = "xmin"+str(plotidx), help = "This sets the minimum limit on the x axis as 10^a. Preset: a=-4")
                        x_max = 10**xlim_cols[1].number_input("Maximum m_min 10^...[meV]", value = 0., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                        
                    elif vary_WC == "m_bb":
                        x_min = 10**xlim_cols[0].number_input("Minimum m_bb 10^...[eV]", value = -4., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=-4")
                        x_max = 10**xlim_cols[1].number_input("Maximum m_bb 10^...[eV]", value = 0., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    elif vary_WC == "m_sum":
                        x_min = 10**xlim_cols[0].number_input("Minimum m_sum 10^...[eV]", value = -4., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=-4")
                        x_max = 10**xlim_cols[1].number_input("Maximum m_sum 10^...[eV]", value = 0., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                    elif vary_WC[-2] == "6":
                        x_min = 10**xlim_cols[0].number_input("Minimum C_"+vary_WC+" 10^...", value = -11., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        x_max = 10**xlim_cols[1].number_input("Maximum C_"+vary_WC+" 10^...", value = -5., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                    else:
                        x_min = 10**xlim_cols[0].number_input("Minimum C_"+vary_WC+" 10^...", value = -7., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        x_max = 10**xlim_cols[1].number_input("Maximum C_"+vary_WC+" 10^...", value = -2., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                    choose_ylim = xlim_cols[2].checkbox("Set y-axis limits", help = "You can either let the code choose the y-axis limits or choose them yourself by checking this box.", key = "ylim checkbox"+str(plotidx))
                    ylim_cols = st.beta_columns(2)
                    y_min =  None
                    y_max = None
                    if choose_ylim:
                        ylim_cols = st.beta_columns(3)
                        y_min = 10**ylim_cols[0].number_input("Minimum y-axis limit exponent", value = -4., key = "ymin"+str(plotidx), help = "This sets the minimum limit on the x axis as 10^a. Preset: a=-4")
                        y_max = 10**ylim_cols[1].number_input("Maximum m_min exponent [meV]", value = 0., key = "ymax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    compare_to_mass = False
                    normalize_to_mass = False
                    show_cosmo = False
                    if vary_WC == "m_min":
                        option_cols = st.beta_columns(2)
                        compare_to_mass = option_cols[0].checkbox("Compare to mass mechanism?", key =plotoptions+"3"+str(plotidx), value=False, help = "If you check this box we will plot the contribution of the standard mass mechanism for comparison.")
                        normalize_to_mass = option_cols[1].checkbox("Normalize to mass mechanism?", key =plotoptions+"3"+str(plotidx), value=False, help = "If you check this box we will normalize the y-axis with respect to the contributions of the standard mass mechanism.")
                        cosmo_options = st.beta_columns(2)
                        show_cosmo = cosmo_options[0].checkbox("Show cosmology limit?", key =plotoptions+str(plotidx), help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                        m_cosmo = 0.15
                        if show_cosmo:
                            m_cosmo = cosmo_options[1].number_input("Limit on the sum of neutrino masses [meV]", help="Preset limit: S.R. Choudhury and S. Hannestad, 2019, arxiv:1907.12598", value = 150, key = "m_cosmo"+str(plotidx))*1e-3
                    if plotoptions == "m_eff":
                        fig = LEFT_model.plot_m_eff(cosmo=show_cosmo, isotope = plot_isotope, 
                                                    compare_to_mass = compare_to_mass, m_cosmo = m_cosmo,
                                                    normalize_to_mass = normalize_to_mass, 
                                                    vary_WC = vary_WC, n_dots = 200, 
                                                    x_min = x_min, x_max = x_max, 
                                                    y_min = y_min, y_max = y_max)
                    elif plotoptions == "half_life":
                        fig = LEFT_model.plot_t_half(cosmo=show_cosmo, isotope = plot_isotope, 
                                                    compare_to_mass = compare_to_mass, m_cosmo = m_cosmo,
                                                    normalize_to_mass = normalize_to_mass, 
                                                    vary_WC = vary_WC, n_dots = 200, 
                                                    x_min = x_min, x_max = x_max, 
                                                    y_min = y_min, y_max = y_max)
                    elif plotoptions == "1/half_life":
                        fig = LEFT_model.plot_t_half_inv(cosmo=show_cosmo, isotope = plot_isotope, 
                                                    compare_to_mass = compare_to_mass, m_cosmo = m_cosmo,
                                                    normalize_to_mass = normalize_to_mass, 
                                                    vary_WC = vary_WC, n_dots = 200, 
                                                    x_min = x_min, x_max = x_max, 
                                                    y_min = y_min, y_max = y_max)
                else:
                    xlim_cols = st.beta_columns(3)
                    if vary_WC == "m_min":
                        x_min = 10**xlim_cols[0].number_input("Minimum m_min exponent [eV]", value = -4, key = "xmin"+str(plotidx))
                        x_max = 10**xlim_cols[1].number_input("Maximum m_min exponent [meV]", value = 0, key = "xmax"+str(plotidx))
                    elif vary_WC == "m_bb":
                        x_min = xlim_cols[0].number_input("Minimum m_bb [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_bb [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC == "m_sum":
                        x_min = xlim_cols[0].number_input("Minimum m_sum [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_sum [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC[-2] == "6":
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-9]", value = 0.1, key = "xmin"+str(plotidx))*1e-9
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-9]", value = 1000., key = "xmax"+str(plotidx))*1e-9
                    else:
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-6]", value = 0.1, key = "xmin"+str(plotidx))*1e-6
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-6]", value = 1000., key = "xmax"+str(plotidx))*1e-6
                    option_cols = st.beta_columns(4)
                    vary_LECs = option_cols[0].checkbox("Vary unknown LECs?", key =plotoptions+"1"+str(plotidx), help = "If you check this box we will vary all unknown LECs around their order of magnitude estimate O (i.e. from 1/sqrt(10) to sqrt(10) times the estimate . g_nuNN will be varied 50% around it's theoretical estimate.")
                    vary_phases = option_cols[1].checkbox("Vary phase?", key =plotoptions+"2"+str(plotidx), value=True, help = "If you check this box we will vary the complex phase of the operator chosen for the x-axis.")
                    n_points = xlim_cols[2].number_input("Number of points", value = 10000, key =plotoptions+"npoints"+str(plotidx))
                        
                    compare_to_mass = False
                    normalize_to_mass = False
                    show_cosmo = False
                    if vary_WC == "m_min":
                        compare_to_mass = option_cols[2].checkbox("Compare to mass mechanism?", key =plotoptions+"3"+str(plotidx), value=False, help = "If you check this box we will plot the contribution of the standard mass mechanism for comparison.")
                        normalize_to_mass = option_cols[3].checkbox("Normalize to mass mechanism?", key =plotoptions+"3"+str(plotidx), value=False, help = "If you check this box we will normalize the y-axis with respect to the contributions of the standard mass mechanism.")
                        cosmo_options = st.beta_columns(2)
                        show_cosmo = cosmo_options[0].checkbox("Show cosmology limit?", key =plotoptions+str(plotidx), help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                        if show_cosmo:
                            m_cosmo = cosmo_options[1].number_input("Limit on the sum of neutrino masses [meV]", help="Preset limit: S.R. Choudhury and S. Hannestad, 2019, arxiv:1907.12598", value = 150, key = "m_cosmo"+str(plotidx))*1e-3
                    if plotoptions == "m_eff":
                        fig = LEFT_model.plot_m_eff_scatter(vary_WC = vary_WC, vary_phases = vary_phases, 
                                                            compare_to_mass = compare_to_mass, n_dots = n_points, 
                                                            normalize_to_mass = normalize_to_mass,
                                                            cosmo=show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                            vary_LECs=vary_LECs, x_min = x_min, x_max = x_max)#, compare_to_mass = compare_to_mass, normalize_to_mass = normalize_to_mass)
                    if plotoptions == "half_life":
                        fig = LEFT_model.plot_t_half_scatter(vary_WC = vary_WC, vary_phases = vary_phases, 
                                                            compare_to_mass = compare_to_mass, n_dots = n_points, 
                                                            normalize_to_mass = normalize_to_mass,
                                                            cosmo=show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                            vary_LECs=vary_LECs, x_min = x_min, x_max = x_max)#, compare_to_mass = compare_to_mass, normalize_to_mass = normalize_to_mass)
                    if plotoptions == "1/half_life":
                        fig = LEFT_model.plot_t_half_inv_scatter(vary_WC = vary_WC, vary_phases = vary_phases, 
                                                            compare_to_mass = compare_to_mass, n_dots = n_points, 
                                                            normalize_to_mass = normalize_to_mass,
                                                            cosmo=show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                            vary_LECs=vary_LECs, x_min = x_min, x_max = x_max)#, compare_to_mass = compare_to_mass, normalize_to_mass = normalize_to_mass)

                st.pyplot(fig)
            return(plotoptions)
        plotoptions = ""
        plotidx = 0
        while plotoptions != "-":
            plotoptions = plots(plotidx)
            plotidx +=1


    elif model_option == "SMEFT":
        phases = st.sidebar.checkbox("Allow for complex phases?", help = "If you check this box you can set complex phases for each Wilson coefficient.")
        SMEFT_WCs = {#dim5                    #Wilson Coefficients of SMEFT
                "LH(5)"      : 0,         #up to dimension 7. We only 
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
                "LLQuH1(7)" : 0,
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
        scale_options = st.sidebar.selectbox("Does your model generate SMEFT operators at multiple scales? If 'Yes' you will need to define a scale for each operator.", options = ["No", "Yes"])
        if scale_options == "Yes":
            multiscales = True
        else:
            multiscales = False
            scale = st.sidebar.number_input("Set the scale at which your SMEFT model is generated [TeV].", value=10)*1e+3
        #scale = st.sidebar.number_input("Set the scale at which your SMEFT model is generated. Please be aware that we assume all operators to be generated together at a single scale. If there are several scales in your model please choose the lowest one and plug in the corresponding Wilson coefficients at this scale [TeV].", value=10)*1e+3

        st.sidebar.write("Set the dimensionless Wilson coefficients:")
        ctr = 0
        for operator in SMEFT_WCs:
            if operator[-2] == "7":
                if ctr == 0:
                    st.sidebar.subheader("Dimension 7")# [1e-6]")
                    #opfac = 1e-6
                    ctr+=1
                dimension = 7
                #steps = 0.01
            if operator[-2] == "9":
                #opfac=1
                if ctr == 1:
                    st.sidebar.subheader("Dimension 9")
                    ctr+=1
                dimension = 9
                #st.write(operator+str(dimension))
            if operator == "LH(5)":
                #opfac=1
                st.sidebar.subheader("Dimension 5")
                dimension = 5
            if multiscales:
                scale = st.sidebar.number_input("Scale [TeV].", value=10, key=operator)*1e+3
            
            if scale <=10*1e+4:
                LLHHfactor = 1e-12
                #opfac = 1e-3
            elif 10*1e+4<scale and scale <= 10*1e+7:
                LLHHfactor = 1e-9
            elif 10*1e+7<scale and scale <= 10000*1e+10:
                LLHHfactor = 1e-6
            elif 10*1e+10<scale and scale <= 10000*1e+13:
                LLHHfactor = 1e-3
            else:
                LLHHfactor = 1
                
                #st.write(operator+str(dimension))
            if operator == "LH(5)":
                #st.sidebar.subheader("Dimension 5")
                if multiscales:
                    SMEFT_WCs[operator] = st.sidebar.number_input(operator+" ["+str(LLHHfactor)+"]", value = 1.)*LLHHfactor#*opfac
                    
                    if phases:
                        SMEFT_WCs[operator] *= np.exp(1j*st.sidebar.number_input(operator+" phase [pi]")*np.pi)
                        #st.sidebar.write("________________________________")
                    ##############
                    
                    SMEFT_WCs[operator] /= scale**(dimension-4)
                    st.sidebar.write("_______________________")
                else:
                    SMEFT_WCs[operator] = st.sidebar.number_input(operator+" ["+str(LLHHfactor)+"]", value = 1.)/(scale**(dimension-4))*LLHHfactor#*opfac
                    
                    if phases:
                        SMEFT_WCs[operator] *= np.exp(1j*st.sidebar.number_input(operator+" phase [pi]")*np.pi)
                        st.sidebar.write("________________________________")
                    ##############
                #if multiscales:
                    
            else:
                if multiscales:
                    SMEFT_WCs[operator] = st.sidebar.number_input(operator)
                    if phases:
                        SMEFT_WCs[operator] *= np.exp(1j*st.sidebar.number_input(operator+" phase [pi]")*np.pi)
                        #st.sidebar.write("________________________________")
                    #scale = st.sidebar.number_input("Scale [TeV]")
                    SMEFT_WCs[operator] /= scale**(dimension-4)
                    st.sidebar.write("_______________________")
                else:
                    SMEFT_WCs[operator] = st.sidebar.number_input(operator)/(scale**(dimension-4))
                    if phases:
                        SMEFT_WCs[operator] *= np.exp(1j*st.sidebar.number_input(operator+" phase [pi]")*np.pi)
                        st.sidebar.write("________________________________")
        SMEFT_model = EFT.SMEFT(SMEFT_WCs, scale, method = method, name=name)
        LEFT_model = EFT.LEFT(SMEFT_model.LEFT_matching(), name=name, method=method)
        #hl = SMEFT_model.half_lives()
        st.subheader("Half-lives")
        hl = SMEFT_model.half_lives()
        hl.rename(index = {"$y$": "years"}, inplace = True)
        if np.inf not in hl.values:
            hl = hl.applymap(lambda x: round(x, 2 - int(floor(log10(abs(x))))))
        def get_table_download_link_csv(df):
            #csv = df.to_csv(index=False)
            csv = df.to_csv().encode()
            latex = df.to_latex().encode()
            #b64 = base64.b64encode(csv.encode()).decode() 
            b64 = base64.b64encode(csv).decode()
            href = f'Download half-lives as <a href="data:file/csv;base64,{b64}" download="SMEFT_model_half_lives.csv" target="_blank">.csv</a> or as <a href="data:file/latex;base64,{b64}" download="SMEFT_model_half_lives.tex" target="_blank">.tex</a> file.'
            return href
        st.markdown(get_table_download_link_csv(hl.T), unsafe_allow_html=True)
        st.table(hl.T)
        st.subheader("Angular correlation")
        ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
        plot_isotope = st.selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, 
                                     key = "angularcorrisotope")
        #st.line_chart({name: np.real(LEFT_model.angular_corr(np.linspace(1e-5,1-1e-5, 1000)))})
        show_mbb1 = st.checkbox("Compare to mass mechanism?", key="show_mbb1")
        fig_angular_corr = LEFT_model.plot_corr(show_mbb=show_mbb1, isotope = plot_isotope)
        st.pyplot(fig_angular_corr)
        st.subheader("Normalized single electron spectrum")
        plot_isotope2 = st.selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, 
                                     key = "spectraisotope")
        #integral = integrate.quad(lambda E: LEFT_model.spectra(E), 0, 1)[0]
        #st.line_chart({name: np.real(LEFT_model.spectra(np.linspace(1e-5,1-1e-5, 1000))/integral)})
        show_mbb2 = st.checkbox("Compare to mass mechanism?", key="show_mbb2")
        fig_spec = LEFT_model.plot_spec(show_mbb=show_mbb2, isotope = plot_isotope2)
        st.pyplot(fig_spec)
        st.subheader("Half-life ratios")
        reference_isotope = st.selectbox("Choose the reference isotope:", options = LEFT_model.isotope_names, index = ge_idx)
        ratio_option_cols = st.beta_columns(2)
        compare = ratio_option_cols[0].checkbox("Compare to mass mechanism?")
        vary = ratio_option_cols[1].checkbox("Vary unknown LECs?")
        if vary:
            n_points = st.number_input("How many variations do you want to run? Remember: The higher this number the longer the calculation takes..." , value=100)
        else:
            n_points = 1
        fig = LEFT_model.ratios(plot=True, vary = vary, n_points = n_points, 
                                normalized = compare, reference_isotope = reference_isotope)
        st.pyplot(fig)
        st.subheader("Vary single Wilson coefficients")
        def plots(plotidx):
            plotoptions = st.selectbox("Choose additional figures you want to see. These plots take a few seconds...", 
                                        options = ["-", "m_eff", "half_life", "1/half_life"], key = "chooseplottype"+str(plotidx))
            st.write("Note: Any value of m_bb that you set for you model will be replaced for these plots according to the minimal neutrino mass. If you didn't set any value we will assume that your model is present additionally to the standard mass mechanism.")
            if plotoptions == "m_eff":

                ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
                plot_cols = st.beta_columns(3)
                plot_isotope = plot_cols[0].selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, key = "isotope"+str(plotidx))
                scatter_or_line = plot_cols[1].selectbox("Choose the plot-type", options = ["Scatter", "Line"], key = "plottype"+str(plotidx))
                vary_WC = plot_cols[2].selectbox("X-axis WC", options = np.append(["m_min", "m_sum"], np.array(list(LEFT_model.WC.keys()))[np.array(list(LEFT_model.WC.values()))!=0]), key = "vary"+str(plotidx))
                show_cosmo = False
                m_cosmo = 0.15
                #show_cosmo = st.checkbox("Show cosmology limit?", key =plotoptions)
                if scatter_or_line == "Line":
                    show_cosmo = st.checkbox("Show cosmology limit?", key =plotoptions + str(plotidx), help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                    fig = LEFT_model.plot_m_eff(cosmo=show_cosmo, isotope = plot_isotope)
                else:
                    xlim_cols = st.beta_columns(2)
                    if vary_WC == "m_min":
                        x_min = xlim_cols[0].number_input("Minimum m_min [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_min [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC == "m_bb":
                        x_min = xlim_cols[0].number_input("Minimum m_bb [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_bb [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC == "m_sum":
                        x_min = xlim_cols[0].number_input("Minimum m_sum [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_sum [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC[-2] == "6":
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-9]", value = 0.1, key = "xmin"+str(plotidx))*1e-9
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-9]", value = 1000., key = "xmax"+str(plotidx))*1e-9
                    else:
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-6]", value = 0.1, key = "xmin"+str(plotidx))*1e-6
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-6]", value = 1000., key = "xmax"+str(plotidx))*1e-6
                    option_cols = st.beta_columns(2)
                    vary_LECs = option_cols[0].checkbox("Vary unknown LECs?", key =plotoptions+"1"+str(plotidx), help = "If you check this box we will vary all unknown LECs around their order of magnitude estimate O (i.e. from log_10(O) to log10(O+1)) . g_nuNN will be varied 50% around it's theoretical estimate.")
                    vary_phases = option_cols[1].checkbox("Vary phase?", key =plotoptions+"2"+str(plotidx), value=True, help = "If you check this box we will vary the complex phase of the operator chosen for the x-axis.")
                    
                        
                    compare_to_mass = False
                    normalize_to_mass = False
                    show_cosmo = False
                    if vary_WC == "m_min":
                        cosmo_options = st.beta_columns(2)
                        show_cosmo = cosmo_options[0].checkbox("Show cosmology limit?", key =plotoptions+str(plotidx), help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                        if show_cosmo:
                            m_cosmo = cosmo_options[1].number_input("Limit on the sum of neutrino masses [meV]", help="Preset limit: S.R. Choudhury and S. Hannestad, 2019, arxiv:1907.12598", value = 150, key = "m_cosmo"+str(plotidx))*1e-3
                    fig = LEFT_model.plot_m_eff_scatter(vary_WC = vary_WC, vary_phases = vary_phases,  
                                                        cosmo=show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                        vary_LECs=vary_LECs, x_min = x_min, x_max = x_max)

                st.pyplot(fig)
            elif plotoptions == "half_life":
                ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
                plot_cols = st.beta_columns(3)
                plot_isotope = plot_cols[0].selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, key = "isotope"+str(plotidx))
                scatter_or_line = plot_cols[1].selectbox("Choose the plot-type", options = ["Scatter", "Line"], key = "plottype"+str(plotidx))
                vary_WC = plot_cols[2].selectbox("X-axis WC", options = np.append(["m_min", "m_sum"], np.array(list(LEFT_model.WC.keys()))[np.array(list(LEFT_model.WC.values()))!=0]), key = "vary"+str(plotidx))
                show_cosmo = False
                m_cosmo = 0.15
                if scatter_or_line == "Line":
                    show_cosmo = st.checkbox("Show cosmology limit?", key =plotoptions+str(plotidx), help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                    fig = LEFT_model.plot_t_half(cosmo=show_cosmo, isotope = plot_isotope)
                else:
                    xlim_cols = st.beta_columns(2)
                    if vary_WC == "m_min":
                        x_min = xlim_cols[0].number_input("Minimum m_min [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_min [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC == "m_bb":
                        x_min = xlim_cols[0].number_input("Minimum m_bb [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_bb [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC == "m_sum":
                        x_min = xlim_cols[0].number_input("Minimum m_sum [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_sum [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC[-2] == "6":
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-9]", value = 0.1, key = "xmin"+str(plotidx))*1e-9
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-9]", value = 1000., key = "xmax"+str(plotidx))*1e-9
                    else:
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-6]", value = 0.1, key = "xmin"+str(plotidx))*1e-6
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-6]", value = 1000., key = "xmax"+str(plotidx))*1e-6
                    option_cols = st.beta_columns(2)
                    vary_LECs = option_cols[0].checkbox("Vary unknown LECs?", key =plotoptions+"1"+str(plotidx), help = "If you check this box we will vary all unknown LECs around their order of magnitude estimate O (i.e. from log_10(O) to log10(O+1)) . g_nuNN will be varied 50% around it's theoretical estimate.")
                    vary_phases = option_cols[1].checkbox("Vary phase?", key =plotoptions+"2"+str(plotidx), value=True, help = "If you check this box we will vary the complex phase of the operator chosen for the x-axis.")
                    
                        
                    compare_to_mass = False
                    normalize_to_mass = False
                    show_cosmo = False
                    if vary_WC == "m_min":
                        cosmo_options = st.beta_columns(2)
                        show_cosmo = cosmo_options[0].checkbox("Show cosmology limit?", key =plotoptions+str(plotidx), help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                        if show_cosmo:
                            m_cosmo = cosmo_options[1].number_input("Limit on the sum of neutrino masses [meV]", help="Preset limit: S.R. Choudhury and S. Hannestad, 2019, arxiv:1907.12598", value = 150, key = "m_cosmo"+str(plotidx))*1e-3
                    fig = LEFT_model.plot_t_half_scatter(vary_WC = vary_WC, vary_phases = vary_phases, 
                                                         cosmo=show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                         vary_LECs=vary_LECs, x_min = x_min, x_max = x_max)

                st.pyplot(fig)
            elif plotoptions == "1/half_life":
                ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
                plot_cols = st.beta_columns(3)
                plot_isotope = plot_cols[0].selectbox("Choose an isotope:", options = LEFT_model.isotope_names, 
                                                      index = ge_idx, key = "isotope"+str(plotidx))
                scatter_or_line = plot_cols[1].selectbox("Choose the plot-type", options = ["Scatter", "Line"], 
                                                         key = "plottype"+str(plotidx))
                vary_WC = plot_cols[2].selectbox("X-axis WC", options = np.append(["m_min", "m_sum"], 
                                                                                  np.array(list(LEFT_model.WC.keys()))[np.array(list(LEFT_model.WC.values()))!=0]), key = "vary"+str(plotidx))
                show_cosmo = False
                m_cosmo = 0.15
                if scatter_or_line == "Line":
                    show_cosmo = st.checkbox("Show cosmology limit?", key =plotoptions, help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                    fig = LEFT_model.plot_t_half_inv(cosmo=show_cosmo, isotope = plot_isotope)
                else:
                    xlim_cols = st.beta_columns(2)
                    if vary_WC == "m_min":
                        x_min = xlim_cols[0].number_input("Minimum m_min [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_min [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC == "m_bb":
                        x_min = xlim_cols[0].number_input("Minimum m_bb [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_bb [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC == "m_sum":
                        x_min = xlim_cols[0].number_input("Minimum m_sum [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_sum [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC[-2] == "6":
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-9]", value = 0.1, key = "xmin"+str(plotidx))*1e-9
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-9]", value = 1000., key = "xmax"+str(plotidx))*1e-9
                    else:
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-6]", value = 0.1, key = "xmin"+str(plotidx))*1e-6
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-6]", value = 1000., key = "xmax"+str(plotidx))*1e-6
                    option_cols = st.beta_columns(2)
                    vary_LECs = option_cols[0].checkbox("Vary unknown LECs?", key =plotoptions+"1"+str(plotidx), help = "If you check this box we will vary all unknown LECs around their order of magnitude estimate O (i.e. from log_10(O) to log10(O+1)) . g_nuNN will be varied 50% around it's theoretical estimate.")
                    vary_phases = option_cols[1].checkbox("Vary phase?", key =plotoptions+"2"+str(plotidx), value=True, help = "If you check this box we will vary the complex phase of the operator chosen for the x-axis.")
                        
                    compare_to_mass = False
                    normalize_to_mass = False
                    show_cosmo = False
                    if vary_WC == "m_min":
                        cosmo_options = st.beta_columns(2)
                        show_cosmo = cosmo_options[0].checkbox("Show cosmology limit?", key =plotoptions+str(plotidx), help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                        if show_cosmo:
                            m_cosmo = cosmo_options[1].number_input("Limit on the sum of neutrino masses [meV]", help="Preset limit: S.R. Choudhury and S. Hannestad, 2019, arxiv:1907.12598", value = 150, key = "m_cosmo"+str(plotidx))*1e-3
                    fig = LEFT_model.plot_t_half_inv_scatter(vary_WC = vary_WC, vary_phases = vary_phases, 
                                                             cosmo=show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                             vary_LECs=vary_LECs, x_min = x_min, x_max = x_max)
                st.pyplot(fig)
            return(plotoptions)
        plotoptions = ""
        plotidx = 0
        while plotoptions != "-":
            plotoptions = plots(plotidx)
            plotidx +=1



####################################################################################################
#                                                                                                  #
#                                                                                                  #
#                                                                                                  #
#                                    STUDY LIMITS ON OPERATORS                                     #
#                                                                                                  #
#                                                                                                  #
#                                                                                                  #
####################################################################################################

elif path_option == "Study operator limits":
    #experimental hl limits for each isotope in 10^24y
    isotope_limits = {"238U"  : 0, 
                      "232Th" : 0, 
                      "198Pt" : 0, 
                      "160Gd" : 0, 
                      "154Sm" : 0, 
                      "150Nd" : 0.18,  #arXiv:0810.0248
                      "148Nd" : 0, 
                      "136Xe" : 107.,   #arXiv:1605.02889
                      "134Xe" : 0.019,  #arXiv:1704.05042
                      "130Te" : 32.,    #arXiv:1912.10966
                      "128Te" : 0.11,  #arXiv:hep-ex/0211071
                      "124Sn" : 0, 
                      "116Cd" : 0.19,  #arXiv:1601.05578 
                      "110Pd" : 0, 
                      "100Mo" : 1.5,   #arXiv:2011.13243
                      "96Zr"  : 9.2e-3, #arXiv:0906.2694
                      #"82Se"  : 0.023        #arXiv:1806.05553
                      "82Se"  : 2.4,   #arXiv:1802.07791
                      "76Ge"  : 180,     #arXiv:2009.06079
                      "48Ca"  : 5.3e-2  #arXiv:0810.4746
                      }
    
    reference_limits = {"238U"  : None, 
                       "232Th" : None, 
                       "198Pt" : None, 
                       "160Gd" : None, 
                       "154Sm" : None, 
                       "150Nd" : "NEMO collaboration, 2008, arXiv:0810.0248", 
                       "148Nd" : None, 
                       "136Xe" : "KamLAND-Zen Collaboration, 2016, arXiv:1605.02889", 
                       "134Xe" : "EXO-200 Collaboration, 2017, arXiv:1704.05042", 
                       "130Te" : "CUORE Collaboration, 2019, arXiv:1912.10966", 
                       "128Te" : "C. Arnaboldi et al., 2002, arXiv:hep-ex/0211071", 
                       "124Sn" : None, 
                       "116Cd" : "Aurora experiment, F.A. Danevich et al., 2016, arXiv:1601.05578", 
                       "110Pd" : None, 
                       "100Mo" : "CUPID-Mo Experiment, E. Armengaud et al., 2020, arXiv:2011.13243", 
                       "96Zr"  : "NEMO-3, J.Argyriades et al., 2009, arXiv:0906.2694",
                       #"82Se"  : 0.023        #arXiv:1806.05553
                       "82Se"  : "CUPID-0 Collaboration, 2018, arXiv:1802.07791",
                       "76Ge"  : "GERDA Collaboration, 2020, arXiv:2009.06079",
                       "48Ca"  : "S.Umehara et al., 2008, arXiv:0810.4746"
                       }
    st.subheader("Limits on single Wilson coefficients:")
    st.write("The below table shows the limits assuming only one Wilson coefficient at a time to be present. The results are rounded to 3 significant digits.")
    method = st.sidebar.selectbox("Which NME approximation do you want to use?", options = ["IBM2", "QRPA", "SM"], help = "Currently we allow for 3 different sets of nuclear matrix elements (NMEs): IBM2: F. Deppisch et al., 2020, arxiv:2009.10119 | QRPA: J. Hyvärinen and J. Suhonen, 2015, Phys. Rev. C 91, 024613 | Shell Model (SM): J. Menéndez, 2018, arXiv:1804.02105")
    model_option = st.sidebar.selectbox("Do you want study limits on LEFT or SMEFT operators?", options = ["-","LEFT", "SMEFT"])
    if model_option == "LEFT":
        LEFT_model = EFT.LEFT({}, method = method)
        isotopes = LEFT_model.isotope_names
        st.sidebar.subheader("Experimental Limits")
        st.sidebar.write("Please enter the experimental limits for each isotope. The initial values represent the current experimental limits that we could find. We try to keep these limits as recent as possible. If we missed some limit please contact us. [10^24 years]")
        experiments = {}
        for isotope in isotopes:
            experiments[isotope] = st.sidebar.number_input(isotope, 
                                                           key=isotope, 
                                                           value = isotope_limits[isotope], 
                                                           step=None, help=reference_limits[isotope])*1e+24
        my_bar = st.progress(0)
        percent_complete = 0
        limits = pd.DataFrame()
        onlygroups = st.checkbox("Show only groups?", help = "Instead of showing the limits for all single Wilson coefficients you can choose to summarize those that give the same contributions.")
        for isotope in experiments:
            if experiments[isotope]>0:
                limit, scales = LEFT_model.get_limits(experiments[isotope], isotope=isotope, onlygroups = onlygroups)
                limits[isotope] = np.array(list(limit.values()))
            percent_complete += 1/(len(experiments))
            #st.write(limit)
            progress = np.round(percent_complete,2)
            my_bar.progress(progress)
        limits = limits.applymap(lambda x: round(x, 2 - int(floor(log10(abs(x))))))
        limits["Operators"] = list(limit.keys())#list(LEFT_model.WC.keys())
        limits["Operators"][0] = "m_bb [meV]"
        for idx in range(len(limits["Operators"][1:])):
            limits["Operators"][idx+1] += " [10^-9]"
        #limits["Operators"] = r"$"+limits["Operators"]
        multi = 1e+9*np.ones(len(limits["Operators"]))
        multi[0] *= 1e+3
        limits.set_index("Operators", inplace = True)
        limits = limits.multiply(multi, axis = 0)
        def get_table_download_link_csv(df):
            #csv = df.to_csv(index=False)
            csv = df.to_csv().encode()
            latex = df.to_latex().encode()
            #b64 = base64.b64encode(csv.encode()).decode() 
            b64 = base64.b64encode(csv).decode()
            href = f'Download limits as <a href="data:file/csv;base64,{b64}" download="LEFT_operator_limits.csv" target="_blank">.csv</a> or as <a href="data:file/latex;base64,{b64}" download="LEFT_operator_limits.tex" target="_blank">.tex</a> file.'
            return href
        st.markdown(get_table_download_link_csv(limits), unsafe_allow_html=True)
        st.table(limits)
        
        st.subheader("Plot limits")
        #generate plots of scales corresponding to the limits or the limits directly
        plottype = st.selectbox("You can either plot the limits directly or the corresponding high energy scale assuming naturalness.", options = ["scales", "limits"])
        plotexps = {}
        counter = 0
        checkbox_experiments = {}
        for experiment in experiments:
            if experiments[experiment] > 0:
                checkbox_experiments[experiment] = experiments[experiment]
                counter += 1
                
        cols1 = st.beta_columns(8)
        if counter > 8:
            cols2 = st.beta_columns(8)
        
        if counter > 16:
            cols3 = st.beta_columns(8)
            
        idx = 0
        for experiment in checkbox_experiments:
            if experiment in ["76Ge", "130Te", "136Xe"]:
                preset = True
            else:
                preset = False
            if idx<8:
                col = cols1[idx]
            elif idx >=8 and idx < 16:
                col = cols2[idx-8]
            else:
                col=cols3[idx-16]
            plotexp = col.checkbox(experiment, value=preset)
            idx += 1
            if plotexp:
                plotexps[experiment] = experiments[experiment]
        #st.radio("",options=list(plotexps.keys()))
        #st.checkbox("hi0")
        #st.checkbox("hi1")
        #st.checkbox("hi2")
        st.pyplot(LEFT_model.plot_limits(plotexps, plottype=plottype))
        
        st.subheader("Limits with 2 active operators")
        st.markdown("Below you can generate limit plots assuming 2 different operators at a time to be present.")
            
        def plot_contours(plotidx):
            counter = 0
            checkbox_experiments = {}
            for experiment in experiments:
                if experiments[experiment] > 0:
                    checkbox_experiments[experiment] = experiments[experiment]
                    counter += 1
            counter = 0
            for experiment in experiments:
                if experiments[experiment] > 0:
                    checkbox_experiments[experiment] = experiments[experiment]
                    counter += 1

            cols1 = st.beta_columns(8)
            if counter > 8:
                cols2 = st.beta_columns(8)

            if counter > 16:
                cols3 = st.beta_columns(8)

            for experiment in experiments:
                if experiments[experiment] > 0:
                    checkbox_experiments[experiment] = experiments[experiment]
                    counter += 1

            cols1 = st.beta_columns(8)
            if counter > 8:
                cols2 = st.beta_columns(8)

            if counter > 16:
                cols3 = st.beta_columns(8)
            idx = 0
            plotexp = {}
            plotexps_contour = {}
            for experiment in checkbox_experiments:
                if experiment in []:#, "130Te", "136Xe"]:
                    preset = True
                else:
                    preset = False
                if idx<8:
                    col = cols1[idx]
                elif idx >=8 and idx < 16:
                    col = cols2[idx-8]
                else:
                    col=cols3[idx-16]
                plotexp = col.checkbox(experiment, value=preset, key = "contour_isotope"+str(idx)+str(plotidx))
                idx += 1
                if plotexp:
                    plotexps_contour[experiment] = [experiments[experiment], experiment]

            contour_cols = st.beta_columns(2)
            #ge_idx = int(np.where(LEFT_model.element_names=="76Ge")[0][0])
            #plot_isotope = contour_cols[0].multiselect("Choose an isotope:", options = LEFT_model.element_names, #index = ge_idx, 
            #                            key = "contourrisotope")
            WCx = contour_cols[0].selectbox("X-axis WC", options = np.array(list(LEFT_model.WC.keys())), key = "WCx"+str(plotidx))
            WCy = contour_cols[1].selectbox("Y-axis WC", options = np.array(list(LEFT_model.WC.keys()))[np.array(list(LEFT_model.WC.keys()))!=WCx], key = "WCy"+str(plotidx))
            options_cols = st.beta_columns(2)
            vary_phase = options_cols[0].checkbox("Vary phase", help = "If you check this box we will vary the relative phase between the two Wilson coefficients", key = "varyphases"+str(plotidx))
            n_vary = 1
            if vary_phase:
                phase = 0
                show_variation = options_cols[1].checkbox("Show detailled variation", help = "If you check this box we will display how the variation of the relative phase deforms the contour limit.", key = "showvary"+str(plotidx))
                if show_variation:
                    n_vary = 5
                else:
                    n_vary = 2
            else:
                phase = options_cols[1].number_input("Phase [Pi]", value = 1/4, key = "contour_phase"+str(plotidx))*np.pi
            plotted = False
            #st.write(plotexps_contour)
            if plotexps_contour != {}:
                fig = f.plot_contours(WCx, WCy, experiments = plotexps_contour, method = method, 
                                      n_vary=n_vary, varyphases = vary_phase, n_dots = 400, phase=phase)
                st.pyplot(fig)
                plotted = True
            return(plotted)
        plotted = True
        plotidx = 0
        while plotted:
            plotted = plot_contours(plotidx)
            plotidx +=1
    elif model_option == "SMEFT":
        #st.write("The results are rounded to 3 significant digits.")
        SMEFT_model = EFT.SMEFT({}, method = method, scale = 82)
        LEFT_model = EFT.LEFT({}, method=method)
        #st.write(list(SMEFT_model.WC.keys()))
        isotopes = LEFT_model.isotope_names
        st.sidebar.subheader("Experimental Limits")
        st.sidebar.write("Please enter the experimental limits for each isotope. The initial values represent the current experimental limits that we could find. We try to keep these limits as recent as possible. If we missed some limit please contact us. [10^24 years]")
        experiments = {}
        for isotope in isotopes:
            experiments[isotope] = st.sidebar.number_input(isotope, 
                                                           key=isotope, 
                                                           value = isotope_limits[isotope], 
                                                           step=None, help=reference_limits[isotope])*1e+24
        my_bar = st.progress(0)
        percent_complete = 0
        limits = pd.DataFrame()
        onlygroups = st.checkbox("Show only groups?", help = "Instead of showing the limits for all single Wilson coefficients you can choose to summarize those that give the same contributions.")
        for isotope in experiments:
            if experiments[isotope]>0:
                limit, scales = SMEFT_model.get_limits(experiments[isotope], isotope_name=isotope, onlygroups = onlygroups)
                limits[isotope] = np.array(list(limit.values()))
                #for operator in limit:
                    #if limit[operator] == np.inf:
                    #    st.write(operator)
            percent_complete += 1/(len(experiments))
            #st.write(limit)
            progress = np.round(percent_complete,2)
            my_bar.progress(progress)
            #operators = list(limit.keys())
        
        #list of operators
        operators = list(limit.keys())
        #st.table(limits)
        #for x in limits:
        #    limits[x] = np.round(limits[x], 2-(np.floor(np.log10(np.abs(limits[x]))).astype(int)))
        limits.replace(np.inf, 1e+100, inplace=True)
        limits = limits.applymap(lambda x: round(x, 2 - int(floor(log10(abs(x))))))
        limits.replace(1e+100, np.inf, inplace=True)
        #multi = 1e+12*np.ones(len(limit))
        #multi[0] *= 1e+3
        #limits = limits.multiply(multi, axis = 0)
        limits["Operators"] = operators
        #st.write(operators)
        #limits.set_index("Operators", inplace = True)
        #limits["Operators"][0] = "m_bb [meV]"
        for idx in range(len(operators)):
            operator = operators[idx]
            if operator == "LH(5)":
                dimension = 5
                factor = "1e-12"
            else:
                dimension = int(operator[-2])
                if dimension == 7:
                #    if operator == "LH(7)":
                #        factor = "1e-9"
                #    else:
                    factor = "1e-9"
                else:
                    factor = "1e-3"
            limits["Operators"][idx] += " ["+factor+"/TeV^"+str(dimension-4)+"]"
        limits.set_index("Operators", inplace = True)
        for idx in range(len(operators)):
            operator = operators[idx]
            if operator == "LH(5)":
                dimension = 5
                factor = 1e+12
            else:
                dimension = int(operator[-2])
                if dimension == 7:
                    #st.write(operator)
                    #if operator == "LH(7)":
                    #    factor = 1e+9
                    #else:
                    factor = 1e+9
                else:
                    factor = 1e+3
            multi = np.ones(len(operators))
            multi[idx] *= factor*(1e+3)**(dimension-4)
            limits = limits.multiply(multi, axis = 0)


        ####################################################
        #                                                  #
        #   show limits and make them downloadable as csv  #
        #                                                  #   
        ####################################################

        def get_table_download_link_csv(df):
            #csv = df.to_csv(index=False)
            csv = df.to_csv().encode()
            latex = df.to_latex().encode()
            #b64 = base64.b64encode(csv.encode()).decode() 
            b64 = base64.b64encode(csv).decode()
            href = f'Download limits as <a href="data:file/csv;base64,{b64}" download="SMEFT_operator_limits.csv" target="_blank">.csv</a> or as <a href="data:file/latex;base64,{b64}" download="SMEFT_operator_limits.tex" target="_blank">.tex</a> file.'
            return href
        st.markdown(get_table_download_link_csv(limits), unsafe_allow_html=True)
        st.table(limits)
            
            
            
        st.subheader("Limits with 2 active operators")
        st.markdown("Below you can generate limit plots assuming 2 different operators at a time to be present.")
            
        def plot_contours(plotidx):
            counter = 0
            checkbox_experiments = {}
            for experiment in experiments:
                if experiments[experiment] > 0:
                    checkbox_experiments[experiment] = experiments[experiment]
                    counter += 1
            counter = 0
            for experiment in experiments:
                if experiments[experiment] > 0:
                    checkbox_experiments[experiment] = experiments[experiment]
                    counter += 1

            cols1 = st.beta_columns(8)
            if counter > 8:
                cols2 = st.beta_columns(8)

            if counter > 16:
                cols3 = st.beta_columns(8)

            for experiment in experiments:
                if experiments[experiment] > 0:
                    checkbox_experiments[experiment] = experiments[experiment]
                    counter += 1

            cols1 = st.beta_columns(8)
            if counter > 8:
                cols2 = st.beta_columns(8)

            if counter > 16:
                cols3 = st.beta_columns(8)
            idx = 0
            plotexp = {}
            plotexps_contour = {}
            for experiment in checkbox_experiments:
                if experiment in []:#, "130Te", "136Xe"]:
                    preset = True
                else:
                    preset = False
                if idx<8:
                    col = cols1[idx]
                elif idx >=8 and idx < 16:
                    col = cols2[idx-8]
                else:
                    col=cols3[idx-16]
                plotexp = col.checkbox(experiment, value=preset, key = "contour_isotope"+str(idx)+str(plotidx))
                idx += 1
                if plotexp:
                    plotexps_contour[experiment] = [experiments[experiment], experiment]

            contour_cols = st.beta_columns(2)
            #ge_idx = int(np.where(LEFT_model.element_names=="76Ge")[0][0])
            #plot_isotope = contour_cols[0].multiselect("Choose an isotope:", options = LEFT_model.element_names, #index = ge_idx, 
            #                            key = "contourrisotope")
            WCx = contour_cols[0].selectbox("X-axis WC", options = np.array(list(SMEFT_model.WC.keys())), key = "WCx"+str(plotidx))
            WCy = contour_cols[1].selectbox("Y-axis WC", options = np.array(list(SMEFT_model.WC.keys()))[np.array(list(SMEFT_model.WC.keys()))!=WCx], key = "WCy"+str(plotidx))
            options_cols = st.beta_columns(2)
            vary_phase = options_cols[0].checkbox("Vary phase", help = "If you check this box we will vary the relative phase between the two Wilson coefficients", key = "varyphases"+str(plotidx))
            n_vary = 1
            if vary_phase:
                phase = 0
                show_variation = options_cols[1].checkbox("Show detailled variation", help = "If you check this box we will display how the variation of the relative phase deforms the contour limit.", key = "showvary"+str(plotidx))
                if show_variation:
                    n_vary = 5
                else:
                    n_vary = 2
            else:
                phase = options_cols[1].number_input("Phase [Pi]", value = 1/4, key = "contour_phase"+str(plotidx))*np.pi
            plotted = False
            #st.write(plotexps_contour)
            if plotexps_contour != {}:
                fig = f.plot_contours(WCx, WCy, experiments = plotexps_contour, method = method, 
                                      n_vary=n_vary, varyphases = vary_phase, n_dots = 400, phase=phase)
                st.pyplot(fig)
                plotted = True
            return(plotted)
        plotted = True
        plotidx = 0
        while plotted:
            plotted = plot_contours(plotidx)
            plotidx +=1