import streamlit as st                                                                                              # Necesary for streamlit app
import pandas as pd                                                                                                 # For using of dataframes
import numpy as np                                                                                                  # For numerical calculations
import matplotlib.pyplot as plt                                                                                     # For plotting graphs
import io                                                                                                           # For handling file input/output                                          
import plotly.express as px                                                                                         # For interactive plots
from funcion_streamlit_2 import *                                                                                   # Custom functions for the app
import plotly.graph_objects as go                                                                                   # For advanced plotly graphs
from plotly.subplots import make_subplots                                                                           # For subplotting in plotly                                            
from funciones_streamlit_nb import *                                                                                # Custom functions for numerical calculations    
from mpl_toolkits.mplot3d import Axes3D                                                                             # For 3D plotting
import time                                                                                                         # For time-related functions

                                                                                                                    # Layout configuration using HTML and CSS
st.set_page_config(
    page_title="TMD 2D Analyzer",
    page_icon="⚡",
    layout="wide",
)
st.markdown(
    """
    <style>
    /*******************************
    (A) Principal colours
    *******************************/
    :root{
        --bg_app: #eae7dc;         /* <-- CAMBIA fondo principal */
        --bg_sidebar: #67e314;     /* <-- CAMBIA fondo sidebar */
        --txt: #000000;            /* <-- CAMBIA color de texto general */

        --input_bg: #ffffff;       /* <-- CAMBIA fondo de inputs */
        --input_txt: #000000;      /* <-- CAMBIA texto de inputs */
        --input_border: #00000033; /* <-- CAMBIA borde inputs */

        --stepper_bg: #ffffff;     /* <-- CAMBIA fondo del bloque +/− */
        --stepper_txt: #000000;    /* <-- CAMBIA color de +/− */
    }
    /*******************************
    (B) FONDO + TEXTO GENERAL
    *******************************/
    .stApp{
        background-color: var(--bg_app) !important;   /* <-- fondo app */
        color: var(--txt) !important;                 /* <-- texto app */
    }
    .stApp, .stApp *{
        color: var(--txt) !important;                 /* <-- fuerza texto negro */
    }
    /*******************************
    (C) SIDEBAR
    *******************************/
    [data-testid="stSidebar"]{
        background-color: var(--bg_sidebar) !important; /* <-- fondo sidebar */
        padding: 15px !important;
    }
    /*******************************
    (D) INPUTS
    *******************************/
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] textarea{
        background-color: var(--input_bg) !important;           /* <-- fondo input */
        border: 1px solid var(--input_border) !important;       /* <-- borde input */
        color: var(--input_txt) !important;                      /* <-- texto input */
    }
    input, textarea{
        background-color: var(--input_bg) !important;            /* <-- fondo input */
        color: var(--input_txt) !important;                      /* <-- texto input */
    }
    input::placeholder, textarea::placeholder{
        color: #555 !important; /* <-- CAMBIA color placeholder */
        opacity: 1 !important;
    }
    /*******************************
    (E) NUMBER INPUT: BUTTONS + / −
    *******************************/
    /* Background of the "stepper" (the area where + and − are) */
    [data-testid="stNumberInput"] div[role="group"]{
        background-color: var(--stepper_bg) !important;          /* <-- fondo +/− */
        border-radius: 8px !important;
    }
    /* Botones + y − */
    [data-testid="stNumberInput"] button{
        background-color: var(--stepper_bg) !important;          /* <-- fondo botón */
        color: var(--stepper_txt) !important;                    /* <-- color +/− */
        border: 1px solid var(--input_border) !important;        /* <-- borde */
    }
    /* Iconos dentro del botón (+/− a veces viene como svg) */
    [data-testid="stNumberInput"] button svg{
        fill: var(--stepper_txt) !important;                     /* <-- color +/− (svg) */
        color: var(--stepper_txt) !important;
    }
    /*******************************
    (F) TABS 
    *******************************/
    button[data-baseweb="tab"] *{
        color: var(--txt) !important; /* <-- color texto tabs */
    }
    [data-testid="stAlert"] *{
        color: var(--txt) !important; /* <-- texto dentro de st.info/warn/error */
    }
    /*******************************
    (G) EXPANDERS 
    *******************************/
    .streamlit-expanderHeader{
        background-color: #e6210b !important; /* <-- CAMBIA fondo header expander */
        border-radius: 5px;
    }
    .streamlit-expanderContent{
        background-color: #67e314 !important; /* <-- CAMBIA fondo contenido expander */
        border-radius: 20px;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
st.title("🏗️💻 TMD 2D Analyzer")                                                                                        # Title and subtitle text
st.markdown("""
This application allows modeling a 2D structure.
You can input data, run calculations, and view results in graphs.
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏢 General Data",
    "📈 Seismic record data",
    "⚡ Results",
    "🎞️ Animation"
])
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
with tab1:
                                                                                                                        # Sidebar for input parameters
    with st.sidebar:
        st.header("Options")
        
        mode = st.selectbox("Select study model", ["BERNOULLI-EULER MODEL (WITHOUT RIGID END ZONES)","B-E-TIMOSHENKO (WITHOUT RIGID END ZONES)"])
        if mode=="B-E-TIMOSHENKO (WITHOUT RIGID END ZONES)":
            TIMOSHENKO=1.0
        else:
           TIMOSHENKO=0.0 
        st.subheader('Geometry of the frame')                                                                           # From here onwards, input parameters
        with st. expander('Expand'):
            num_vanos = st.number_input("**Number of Spans**:",min_value=1,max_value=100,value=2)
            pisos_inicial=st.number_input("**Number of Floors (default)**:",min_value=1,max_value=100,value=2)
            Distancia_inicial=st.number_input("**Span Distance [m] (default)**:",min_value=1.00,max_value=20.00,value=2.00,step=0.05)
            Altura_inicial=st.number_input("**Height [m] (default)**:",min_value=1.00,max_value=20.00,value=3.00,step=0.05)
                                                                                                                        # Save general values in session_state if they don't exist                                      
            if "last_generales" not in st.session_state:
                st.session_state["last_generales"] = {
                    "pisos_inicial": pisos_inicial,
                    "altura_inicial": Altura_inicial,
                    "distancia_inicial": Distancia_inicial}
                                                                                                                        # Detect if general values have changed
            generales_cambiaron = (
                st.session_state["last_generales"]["pisos_inicial"] != pisos_inicial or
                st.session_state["last_generales"]["altura_inicial"] != Altura_inicial or
                st.session_state["last_generales"]["distancia_inicial"] != Distancia_inicial)
                                                                                                                        # Update session_state of individual inputs
            if generales_cambiaron:
                for i in range(1, int(num_vanos)+1):
                    st.session_state[f"vano_{i}"] = pisos_inicial
                    st.session_state[f"distancia_vano_{i}"] = Distancia_inicial

                max_pisos = max([pisos_inicial for _ in range(int(num_vanos))])
                for i in range(1, max_pisos+1):
                    st.session_state[f"altura_piso_{i}"] = Altura_inicial
                                                                                                                        # Save the new general values   
                st.session_state["last_generales"]["pisos_inicial"] = pisos_inicial
                st.session_state["last_generales"]["altura_inicial"] = Altura_inicial
                st.session_state["last_generales"]["distancia_inicial"] = Distancia_inicial
                
            st.subheader('Number of Floors per Span')                                                                   # Set number of floors per span
            with st. expander('Expand'):
                pisos = []
                cols = st.columns([0.5, 1])
                cols[0].markdown("**Span**")
                cols[1].markdown("**Number of Floors**")
                for i in range(1, int(num_vanos)+1):
                    cols = st.columns([1, 2])
                    cols[0].markdown(f"Span {i}")
                    piso = cols[1].number_input(f"", min_value=1, max_value=50, value=pisos_inicial, step=1, key=f"vano_{i}")
                    pisos.append(piso)
                    
            st.subheader('Distance between Spans [m]')
            with st. expander('Expand'):
                distancia_vanos = []
                cols = st.columns([1, 2])
                cols[0].markdown("**Span**")
                cols[1].markdown("**Distance between Span [m]**")
                for i in range(1, int(num_vanos)+1):
                    cols = st.columns([1, 2])
                    cols[0].markdown(f"Span {i}")
                    distancia = cols[1].number_input(f"", min_value=1.00, max_value=20.00, value=Distancia_inicial, step=0.05, key=f"distancia_vano_{i}")
                    distancia_vanos.append(distancia)

            st.subheader('Floor Height [m]')
            with st. expander('Expand'):
                altura_pisos = []
                cols = st.columns([1, 2])
                cols[0].markdown("**Floor**")
                cols[1].markdown("**Floor Height [m]**")
                for i in range(1, max(pisos)+1):
                    cols = st.columns([1, 2])
                    cols[0].markdown(f"Floor {i}")               
                    Altura = cols[1].number_input(f"", min_value=1.00, max_value=20.00, value=Altura_inicial, step=0.05, key=f"altura_piso_{i}")
                    altura_pisos.append(Altura)

        st.subheader('Sectional Data of the Structure (General)')
        with st. expander('Expand'):                                                                                                 # Set Geometry of structual elements
            base_col_def=st.number_input("**Base Column [m]: (All floors)**:",min_value=0.15,max_value=1.50,value=0.30,step=0.05)
            Altura_col_def=st.number_input("**Column Height [m]: (All floors)**:",min_value=0.15,max_value=1.50,value=0.30,step=0.05)
            base_vig_def=st.number_input("**Base Beam [m]: (All floors)**:",min_value=0.15,max_value=1.50,value=0.25,step=0.05)
            Altura_vig_def=st.number_input("**Beam Height [m]: (All floors)**:",min_value=0.15,max_value=1.50,value=0.25,step=0.05)

            if "last_secciones_generales" not in st.session_state:                                                                   # Save general sections if they don't exist
                st.session_state["last_secciones_generales"] = {
                    "base_col_def": base_col_def,
                    "Altura_col_def": Altura_col_def,
                    "base_vig_def": base_vig_def,
                    "Altura_vig_def": Altura_vig_def
                }
                                                                                                                                    # Detect if general sections have changed
            secciones_generales_cambiaron = (
                st.session_state["last_secciones_generales"]["base_col_def"] != base_col_def or
                st.session_state["last_secciones_generales"]["Altura_col_def"] != Altura_col_def or
                st.session_state["last_secciones_generales"]["base_vig_def"] != base_vig_def or
                st.session_state["last_secciones_generales"]["Altura_vig_def"] != Altura_vig_def
            )
                                                                                                                                    # If changed, update all floors
            if secciones_generales_cambiaron:
                for piso_num in range(1, max(pisos) + 1):
                    st.session_state[f"col_base_{piso_num}"] = base_col_def
                    st.session_state[f"col_alt_{piso_num}"] = Altura_col_def
                    st.session_state[f"viga_base_{piso_num}"] = base_vig_def
                    st.session_state[f"viga_alt_{piso_num}"] = Altura_vig_def
                                                                                                                                    # Save the new general sections
                st.session_state["last_secciones_generales"]["base_col_def"] = base_col_def
                st.session_state["last_secciones_generales"]["Altura_col_def"] = Altura_col_def
                st.session_state["last_secciones_generales"]["base_vig_def"] = base_vig_def
                st.session_state["last_secciones_generales"]["Altura_vig_def"] = Altura_vig_def

            st.subheader('Sectional Data by Floor')
            with st. expander('Expand'):                                                    
                                                                                                                    # Create table by floor
                pisos_totales = max(pisos)                                                                          # total number of floors in the entire frame
                st.write(f"Sections for {pisos_totales} floors")
                
                secciones = []                                                                                      # Initialize list to store sections
                for piso_num in range(1, pisos_totales + 1):                                                        # Loop through each floor
                    st.markdown(f"**Floor {piso_num}=====================================**")                       # Floor header
                    # Columns
                    st.markdown("Columns:")                                                                         # Columns header
                    col1, col2 = st.columns(2)                                                                      # Create two columns for input
                    base_col = col1.number_input(f"Floor {piso_num} - Column Base [m]:", min_value=0.15, max_value=1.50, value=base_col_def, step=0.05, key=f"col_base_{piso_num}")         # Column base input
                    altura_col = col2.number_input(f"Floor {piso_num} - Column Height [m]:", min_value=0.15, max_value=1.50, value=Altura_col_def, step=0.05, key=f"col_alt_{piso_num}")    # Column height input  
                    # Beams
                    st.markdown("Beams:")                                                                           # Beams header
                    col3, col4 = st.columns(2)                                                                      # Create two columns for input
                    base_viga = col3.number_input(f"Floor {piso_num} - Beam Base [m]:", min_value=0.15, max_value=1.50, value=base_vig_def, step=0.05, key=f"viga_base_{piso_num}")         # Beam base input
                    altura_viga = col4.number_input(f"Floor {piso_num} - Beam Height [m]:", min_value=0.15, max_value=1.50, value=Altura_vig_def, step=0.05, key=f"viga_alt_{piso_num}")    # Beam height input
                    
                    secciones.append({
                        "Piso": piso_num,
                        "Base Columna(m)": base_col,
                        "Altura Columna(m)": altura_col,
                        "Base Viga(m)": base_viga,
                        "Altura Viga(m)": altura_viga
                    })                                                                                              # Append section data to list
        
            df_secciones = pd.DataFrame(secciones)                                                                  # Create DataFrame from sections data
            st.markdown("### Tabla de Secciones por Piso")                                                          # Display heading
            st.dataframe(df_secciones)                                                                              # Display sections DataFrame
                                                                                                                                    # Enter MATERIAL PROPERTIES

        st.subheader('Material of the structure (general)')                                                                         # Material properties
        with st. expander('Expand'):                                                                                                # Expandable section
            Material = []
            cols = st.columns([3, 1])                                                                                               # Create columns for table
            cols[0].markdown("**Elasticity Modulus**")                                                                              # Column for Elasticity Modulus
            cols[0].markdown("**(MPa)**")                                                                                           # Unit

            cols[1].markdown("**Poisson's Ratio**")                                                                                 # Column for Poisson's Ratio
            cols[1].markdown("==")                                                                                                  # Unit
            Elasticidad = cols[0].number_input(f"", min_value=10000.00, max_value=500000.00, value=21500.00, step=1000.00)          # Elasticity Modulus input
            Material.append(Elasticidad*1000/9.8067)                                                                                # Convert Elasticity Modulus to consistent units
            Poisson = cols[1].number_input(f"", min_value=0.1, max_value=0.5, value=0.2, step=0.01)                                 # Poisson's Ratio input
            Material.append(Poisson)                                                                                                # Append Poisson's Ratio to Material list

        st.subheader('Weights of the Floors (general)')                                                                             # Weight of the Floors
        with st. expander('Expand'):                                                                                                # Expandable section
            Peso_def=st.number_input("**Weight [Tonf]: (All floors)**:",min_value=0.01,max_value=300.00,value=10.00,step=0.05)      # Weight input
            if "last_peso_general" not in st.session_state:                                                                         # Save general weight if it doesn't exist
                st.session_state["last_peso_general"] = Peso_def                                                                    # Save general weight
            peso_general_cambio = st.session_state["last_peso_general"] != Peso_def                                                 # Detect if general weight has changed
            if peso_general_cambio:                                                                                                 # If changed, update all floors
                pisos_totales = max(pisos)
                for i in range(1, pisos_totales + 1):                                                                                                                                                                                          
                    st.session_state[f"Weight of floor_{i}"] = Peso_def                                                             # If the floor already exists, update it with the general value                                                                                                                     
                st.session_state["last_peso_general"] = Peso_def                                                                    # Save the new general weight
            st.subheader('Weight per floor')                                                                                        # Set subheader 
            with st.expander('Expand'):                                                                                             # Expandable section

                                                                                                                                    # Create table per floor
                pisos_totales = max(pisos)                                                                                          # Total number of floors in the entire frame
                st.write(f"Weight for {pisos_totales} floors")                                                                      # Display total floors
                
                Masa = []                                                                                                           # Initialize list to store masses
                cols = st.columns([1, 2])                                                                                           # Create columns for table
                cols[0].markdown("**Floor**")                                                                                       # Column for floor number
                cols[1].markdown("**Weight of floor [Tonf]**")                                                                      # Column for weight input
                for i in range(1, max(pisos)+1):                                                                                    # Loop through each floor
                    cols = st.columns([1, 2])                                                                                       # Create columns for each floor
                    cols[0].markdown(f"Floor {i}")                                                                                     # Display floor number
                    Pesosi = cols[1].number_input(f"", min_value=0.01, max_value=300.00, value=Peso_def, step=0.05, key=f"Weight of floor_{i}") # Weight input
                    Masa.append(Pesosi/9.8067)                                                                                      # Convert weight to mass and store in list
        st.markdown("### 👨‍💻 Team")  
        st.image("FOTO/FOTO_BDGR.png") 
        st.markdown(
    """
    <div style="font-family: 'Times New Roman', serif; font-size:16px; font-weight:600;">
        • <a href="https://github.com/Dagu2104" target="_blank"
             style="text-decoration:none; color:#2C3E50;">
             Eng. Bryan Guzmán
          </a>
    </div>
    """,
    unsafe_allow_html=True
)
        st.markdown("""
                    [![GitHub Bryan Guzmán](https://img.shields.io/github/followers/Dagu2104?...)](https://github.com/Dagu2104)
                    """)
        st.markdown(
                    '<a href="mailto:bryan2104guzman@gmail.com">'
                    '<img alt="Email" src="https://img.shields.io/badge/Email-bryan2104guzman@gmail.com-blue?style=flat&logo=gmail">'
                    '</a>',
                        unsafe_allow_html=True
                    )
        st.image("FOTO/FOTO_FPNG.png")
        st.markdown(
    """
    <div style="font-family: 'Times New Roman', serif; font-size:16px; font-weight:600;">
        • <a href="https://github.com/FelixNavia1992" target="_blank"
             style="text-decoration:none; color:#2C3E50;">
             Eng. Félix Navia
          </a>
    </div>
    """,
    unsafe_allow_html=True
)
        st.markdown("""
                    [![GitHub Félix Navia](https://img.shields.io/github/followers/FelixNavia1992?...)](https://github.com/FelixNavia1992)
                    """)
        st.markdown(
                    '<a href="mailto:fpnavia@gmail.com">'
                    '<img alt="Email" src="https://img.shields.io/badge/Email-fpnavia@gmail.com-blue?style=flat&logo=gmail">'
                    '</a>',
                        unsafe_allow_html=True
                    )
        st.markdown("#### *Developed by Eng. Félix Patricio Navia Garcia / Eng. Bryan David Guzmán Ruano — Advisor: Eng. Carlos Andrés Celi Sánchez, M.Sc*") ## Developers credit
                            


  
                                                                                                                                    # START OF CALCULATIONS
    modelo = PorticoAnalyzer()                                                                                                      # Create model instance
    modelo.ingresar_datos(num_vanos,pisos,distancia_vanos,altura_pisos,secciones,Material)                                          # Input data into the model 
    modelo.generar_nodos()                                                                                                          # Generate nodes
    modelo.generar_elementos()                                                                                                      # Generate elements
    modelo.mostrar_portico2(escala=1.0)                                                                                             # Display frame
    modelo.obtener_nodos_base()                                                                                                     # Get base nodes
    modelo.obtener_nodos_dinamico()                                                                                                 # Get dynamic nodes
    K_global,aux_idx=modelo.Matriz_normal_rigidez(TIMOSHENKO)                                                                                 # Global stiffness matrix
    K_din3=modelo.Matriz_dinamica(K_global,aux_idx)                                                                                 # Dynamic stiffness matrix
    phi_ord ,w ,T ,M, gamma, Mei_pct, Mei=modelo.Propiedades_dinamicas_Convensional(Masa,K_din3)                                    # Modal properties calculation
    Mei_pct_acum=np.cumulative_sum(Mei_pct)                                                                                         # Cumulative mass participation
                                                                                                                                    # CALCULO DE AMORTIGUAMIENTO DE ESTRUCTURA
    zeta=0.05                                                                                                                       # DAMPING OF THE STRUCTURE
    C=modelo.Matriz_amortiguamieto_modal_inherente(phi_ord,w,M,zeta)
                                                                                                                                    # Show modal properties
    MODOS=[]
    for piso in range(max(pisos)):                                                              
        MODOS.append({
            "w(rad/seg)": w[piso],
            "T(seg)": T[piso],
            "%Participacion": Mei_pct[piso],
            "%Participacion Acumulada": Mei_pct_acum[piso]
        })
    # Mostrar tabla resumen
    df_modos = pd.DataFrame(MODOS)
    st.markdown("### Modal properties of the structure without TMD")
    st.dataframe(df_modos)
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
with tab2:
    DT_REG_SISM, dt, nombre_archivo=modelo.cargar_registro()                                                        #find name,dt and aaceleration from AT2
    REG_SISM=DT_REG_SISM.to_numpy()
    acc_filt=modelo.plot_filtered_record(REG_SISM,dt,nombre_archivo)                                                #Plot filtered record
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################      
with tab3:    

    if "REG_SISM_fft" not in st.session_state:                                                                      # Initialize seismic record in session_state                             
        st.session_state["REG_SISM_fft"] = None                                                                     # Conditional to avoid recalculation
    calcular = st.button("Run Analysis", key="btn_calcular_tab3")                                                   # Button to run analysis                          
    if calcular:
        start = time.time()                                                                                         # Start time measurement                          
        with st.spinner("Running calculation..."):                                                                   # Spinner while calculating

            (REG_SISM_fft,Respuesta_Des,Respuesta_Vel,
             Respuesta_Acc,Respuesta_Des_TMD,
             Respuesta_Vel_TMD,Respuesta_Acc_TMD,
             deriva_max,ms2,ws2,zetas2,k,R2, R_i_2,
             ws2,ms2,alturas,deriva_matrix,
             indx90)=modelo.tmd_analysis(REG_SISM,acc_filt,M,K_din3,C,w,T,Mei_pct,Mei_pct_acum,altura_pisos)        #tmd analysis parameters

        elapsed = time.time() - start                                                                               # End time measurement
        st.success(f"Calculation completed in {elapsed:.2f} s")                                                     # Display elapsed time

                                                                                                                    # Save results to session_state for use in tab 4
        st.session_state["REG_SISM_fft"] = REG_SISM_fft                                                             # Save seismic record       
        st.session_state["Respuesta_Des"] = Respuesta_Des                                                           # Save displacement response without TMD
        st.session_state["Respuesta_Vel"] = Respuesta_Vel                                                           # Save velocity response without TMD
        st.session_state["Respuesta_Acc"] = Respuesta_Acc                                                           # Save acceleration response without TMD

        st.session_state["Respuesta_Des_TMD"] = Respuesta_Des_TMD                                                   # Save displacement response with TMD
        st.session_state["Respuesta_Vel_TMD"] = Respuesta_Vel_TMD                                                   # Save velocity response with TMD
        st.session_state["Respuesta_Acc_TMD"] = Respuesta_Acc_TMD                                                   # Save acceleration response with TMD

        st.session_state["deriva_max"] = deriva_max                                                                 # Save maximum interstory drift ratio
        st.session_state["ms_opt"] = ms2                                                                            # Save optimal TMD mass
        st.session_state["ws_opt"] = ws2                                                                            # Save optimal TMD frequency
        st.session_state["zetas_opt"] = zetas2                                                                      # Save optimal TMD damping ratio
        st.session_state["ALTURAS"]=alturas                                                                         # Save floor heights
        st.session_state["mderiva"]=deriva_matrix                                                                   # Save drift calculation matrix
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
with tab4:                                                                                                          # Create tab for comparison plots
    
    if st.session_state["REG_SISM_fft"] is None:                                                                    # Check if calculations from tab 3 are available
        st.info("⚠️ Ejecute primero el cálculo en la pestaña 3.")                                                  # Inform user to run calculations first
        st.stop()                                                                                                   # Stop execution if not available

    REG_SISM_fft = st.session_state["REG_SISM_fft"]                                                                 # Retrieve seismic record from session_state
    Respuesta_Des = st.session_state["Respuesta_Des"]                                                               # Retrieve displacement response without TMD
    Respuesta_Vel = st.session_state["Respuesta_Vel"]                                                               # Retrieve velocity response without TMD
    Respuesta_Acc = st.session_state["Respuesta_Acc"]                                                               # Retrieve acceleration response without TMD

    Respuesta_Des_TMD = st.session_state["Respuesta_Des_TMD"]                                                       # Retrieve displacement response with TMD
    Respuesta_Vel_TMD = st.session_state["Respuesta_Vel_TMD"]                                                       # Retrieve velocity response with TMD
    Respuesta_Acc_TMD = st.session_state["Respuesta_Acc_TMD"]                                                       # Retrieve acceleration response with TMD
    alturas=st.session_state["ALTURAS"]                                                                             # Retrieve floor heights
  
    modelo.plot_Comparison_tmd(REG_SISM_fft,Respuesta_Des,
    Respuesta_Vel,Respuesta_Acc,Respuesta_Des_TMD, Respuesta_Vel_TMD,
    Respuesta_Acc_TMD,alturas, modelo, modelo.nice_dtick,modelo.nice_ylim)

st.markdown("---")                                                                                                  # Horizontal separator


















