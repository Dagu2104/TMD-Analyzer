from dataclasses import dataclass                                                   # Import dataclass for defining data structures
import matplotlib.pyplot as plt                                                     # Import matplotlib for plotting
import os                                                                           # Import os for clearing console       
from func_elem import *                                                             # Import functions from func_elem module
import numpy as np                                                                  # Import numpy for numerical operations
import pandas as pd                                                                 # Import pandas for data manipulation
os.system('cls' if os.name=='nt' else 'clear')                                      # Clear console based on OS
import streamlit as st                                                              # Import streamlit for web app interface
import plotly.graph_objects as go                                                   # Import plotly for interactive plotting
from plotly.subplots import make_subplots                                           # For subplotting in plotly
import plotly.express as px                                                         # For interactive plots
from funciones_streamlit_nb import *                                                # Custom functions for numerical calculations 

# 1. Define structures classes: Create objects and describes how it is organized.

@dataclass  
class Nodo:                                                                         #--- Create an object named "Nodo"---
    id: int                                                                         # Identifier of the node
    x: float                                                                        # Coordinate X        
    y: float                                                                        # Coordinate Y
    dofs: list  # [ux, uy, rz]                                                      # Degrees of freedom associated with the node

@dataclass
class Nodo_dyna:                                                                    #--- Create an object named "Nodo_dyna"---
    piso: int                                                                       # Floor to which the node belongs
    id: int                                                                         # Identifier of the node
    x: float                                                                        # Coordinate X
    y: float                                                                        # Coordinate Y
    dofs: list  # [ux, uy, rz]                                                      # Degrees of freedom associated with the node

@dataclass
class Material:                                                                     #--- Create an object named "Material"---
    E: float                                                                        # Elastic modulus
    nu: float                                                                       # Poisson's ratio

@dataclass
class Seccion:                                                                      #--- Create an object named "Sección transversal del elemento"---
    b: float                                                                        # Width
    h: float                                                                        # Height

@dataclass
class Elemento:                                                                     #--- Create an object named "Elemento estructural (viga o columna)"---
    id: int                                                                         # Identifier of the element
    tipo: str                                                                       # Type: 'beam' or 'column'
    nodo_i: int                                                                     # Initial node
    nodo_j: int                                                                     # Final node
    seccion: Seccion                                                                # Cross-sectional area
    material: Material                                                              # Material
    longitud: float                                                                 # Length of the element
    theta: float                                                                    # Angle of inclination (radians)
    piso: int                                                                       # Floor or level to which the element belongs

class PorticoAnalyzer:                              
    def __init__(self):                                                             # Initialize the attributes of the frame:
        self.num_vanos = 0                                                          # Number of spans
        self.distancias_vanos = []                                                  # Distances between spans
        self.num_pisos_por_vano = []                                                # Number of floors per span
        self.alturas_pisos = []                                                     # Heights of floors
        self.secciones_por_piso = {}                                                # Sections per floor   
        self.nodos = []                                                             # List of nodes
        self.elementos = []                                                         # List of elements
        self.node_map = {}                                                          # Map for quick access to nodes
        self.column_levels = []                                                     # Levels of columns per column
        self.nodos_base=[]                                                          # Nodes at the base
        self.material_por_piso={}                                                   # Materials per floor
        self.nodos_dinamicos_piso=[]                                                # Nodes dynamic per floor

    def ingresar_datos(self,num_vanos,pisos,distancia_vanos,altura_pisos,secciones,Material1):      # Function to store frame input data
        
        self.num_vanos = num_vanos                                                                  # Store number of spans
        self.distancias_vanos = distancia_vanos                                                     # Store distances between spans
        self.num_pisos_por_vano = pisos                                                             # Store number of floors per span
        self.alturas_pisos = altura_pisos                                                           # Store heights of floors
        Elasticidad = Material1[0]                                                                  # Store the modulus of elasticity of the material 
        poisson = Material1[1]                                                                      # Store the Poisson's ratio of the material 
        
        for i in range(1,max(pisos)+1):                                                             # Assignment of material and section properties per floor
            self.material_por_piso[i] = {"columna":Material(Elasticidad,poisson),                   # Assignment of material for columns
                                         "viga":Material(Elasticidad,poisson)}                      # Assignment of material for beams
              
            self.secciones_por_piso[i] = {"columna": Seccion(secciones[i-1]["Base Columna(m)"],     # Assignment of width of the section for columns
                                                             secciones[i-1]["Altura Columna(m)"]),  # Assignment of height of the section for columns
                                           "viga": Seccion(secciones[i-1]["Base Viga(m)"],          # Assignment of width of the section for beams
                                                           secciones[i-1]["Altura Viga(m)"])}       # Assignment of height of the section for beams


    def generar_nodos(self):                                                                        # Function to generate nodes of the frame
        self.nodos = []                                                                             # Initialize empty list of nodes             
        self.node_map = {}                                                                          # Initialize empty node map for quick access
        nodo_id = 1                                                                                 # Node identifier starts at 1
        gdl_actual = 1                                                                              # Current degree of freedom starts at 1

        num_columnas = self.num_vanos + 1                                                           # Calculate number of columns                              
        coords_x = [0.0]                                                                            # Initialize X coordinates list
        for dist in self.distancias_vanos:                                                          # Calculate X coordinates based on span distances
            coords_x.append(coords_x[-1] + dist)                                                    # Append new X coordinate

        column_levels = []                                                                          # Initialize column levels list           
        for col_idx in range(num_columnas):                                                         # Determine levels for each column
            left_vano = self.num_pisos_por_vano[col_idx - 1] if col_idx - 1 >= 0 else 0             # Floors in the left span
            right_vano = self.num_pisos_por_vano[col_idx] if col_idx < self.num_vanos else 0        # Floors in the right span
            levels = max(left_vano, right_vano)                                                     # Maximum floors for the column 
            column_levels.append(levels)                                                            # Append levels to the list    
        self.column_levels = column_levels                                                          # Store column levels in the object

        for col_idx in range(num_columnas):                                                         # Generate nodes for each column    
            x = coords_x[col_idx]                                                                   # X coordinate of the column
            n_levels = column_levels[col_idx]                                                       # Number of levels for the column   
            for nivel in range(n_levels + 1):                                                       # Generate nodes at each level
                y = sum(self.alturas_pisos[:nivel]) if nivel > 0 else 0.0                           # Y coordinate based on floor heights
                dofs = [gdl_actual, gdl_actual + 1, gdl_actual + 2]                                 # Assign degrees of freedom (ux, uy, rz)
                nodo = Nodo(nodo_id, x, y, dofs)                                                    # Create node object    
                self.nodos.append(nodo)                                                             # Append node to the list
                self.node_map[(col_idx, nivel)] = nodo                                              # Map node for quick access
                nodo_id += 1                                                                        # Increment node identifier
                gdl_actual += 3                                                                     # Increment degrees of freedom

    def generar_elementos(self):                                                                    # Function to generate elements of the frame
        self.elementos = []                                                                         # Initialize empty list of elements
        elem_id = 1                                                                                 # Element identifier starts at 1
        num_columnas = self.num_vanos + 1                                                           # Calculate number of columns                                          
                                                                                                    ####### SECTION AND MATERIALS FOR COLUMNS #######          
        for col_idx in range(num_columnas):                                                         # Generate columns for each column                                   
            n_levels = self.column_levels[col_idx]                                                  # Store the height of the column to locate the inferior and superior nodes
            for nivel in range(1, n_levels + 1):
                nodo_inf = self.node_map.get((col_idx, nivel - 1))
                nodo_sup = self.node_map.get((col_idx, nivel))
                if nodo_inf and nodo_sup:                                                           # Assign section and material properties for each column element.
                     seccion = self.secciones_por_piso.get(nivel, {}).get("columna")                # Assign section to the elments
                     material = self.material_por_piso.get(nivel, {}).get("columna")                # Assign Material to the elements
                     Longitud= ((self.nodos[nodo_sup.id-1].x - self.nodos[nodo_inf.id-1].x)**2+(self.nodos[nodo_sup.id-1].y - self.nodos[nodo_inf.id-1].y)**2)**0.5 # Calculate the lenght of the element
                     
                     if (self.nodos[nodo_sup.id-1].x - self.nodos[nodo_inf.id-1].x)< 1e-6:
                        Theta=np.pi*0.5
                     else:
                        Theta=np.atan(((self.nodos[nodo_sup.id-1].y - self.nodos[nodo_inf.id-1].y)/(self.nodos[nodo_sup.id-1].x - self.nodos[nodo_inf.id-1].x)))  

                     self.elementos.append(Elemento(elem_id, "columna", nodo_inf.id, nodo_sup.id, seccion, material, Longitud, float(Theta), nivel))
                     elem_id += 1
                                                                                                    ####### SECTION AND MATERIALS FOR BEAMS #######
        for col_idx in range(num_columnas - 1):                                                     
            max_common = min(self.column_levels[col_idx], self.column_levels[col_idx + 1])          # Determine maximum common levels between adjacent columns
            for nivel in range(1, max_common + 1):                                                  # Generate beams at each level
                nodo_i = self.node_map.get((col_idx, nivel))
                nodo_j = self.node_map.get((col_idx + 1, nivel))
                if nodo_i and nodo_j:                                                               # Assign section and material properties for each beam element.
                     seccion = self.secciones_por_piso.get(nivel, {}).get("viga")                   # Assign section to beams
                     material = self.material_por_piso.get(nivel, {}).get("viga")                   # Assign material to beams
                     Longitud= ((self.nodos[nodo_j.id-1].x - self.nodos[nodo_i.id-1].x)**2+(self.nodos[nodo_j.id-1].y - self.nodos[nodo_i.id-1].y)**2)**0.5     # Calculate the lenght of the element
                     Theta=np.atan(((self.nodos[nodo_j.id-1].y - self.nodos[nodo_i.id-1].y)/(self.nodos[nodo_j.id-1].x - self.nodos[nodo_i.id-1].x)))           # Calculate an angle in case of having inclined beams
                     self.elementos.append(Elemento(elem_id, "viga", nodo_i.id, nodo_j.id, seccion, material, Longitud, float(Theta), nivel))                   # Append properties of the beam in the list 
                     elem_id += 1                                                                   # Gives the number identifier to the element till all elements are created

    def _get_nodo_por_id(self, id):
        return next((n for n in self.nodos if n.id == id), None)                                    # Helper function to get node by ID
    
    def obtener_nodos_base(self):
        """Obtiene todos los nodos en Y=0 (base) y sus GDL"""
        self.nodos_base = []                                                                        # Create an empty list for base nodes
        for nodo in self.node_map.values():                                                         # Iterate through all nodes in the node map
            if nodo.y == 0:                                                                         # If the node is at the base (y=0)
                self.nodos_base.append(nodo)                                                        # Append the nodeto the base nodes list only if it is at y=0

    def obtener_nodos_dinamico(self):
        """Obtiene todos los nodos en Y (piso) y sus GDL"""
        self.nodos_dinamicos_piso = []
        num_pisos=max(self.num_pisos_por_vano)                                                      # Determine the maximum number of floors in the 2D frame
        h_actual=0.0                                                                                # Initialize current height to zero
        for piso in range (num_pisos):                                                              # Iterate through each floor
            h_actual+=self.alturas_pisos[piso]                                                      # Update current height by adding the height of the current floor
            for nodo in self.node_map.values():
                if abs(nodo.y - h_actual)<1e-6:                                                     # Condition to identify nodes at the current floor height
                    nodo_d=Nodo_dyna(piso+1,nodo.id, nodo.x, nodo.y, nodo.dofs)                     # Create dynamic node object for the floor
                    self.nodos_dinamicos_piso.append(nodo_d)                                        # Append dynamic node to the list

    def Matriz_normal_rigidez(self,TIMOSHENKO):
        ndof=self.nodos[-1].id*3                                                                    # Get the total number of degrees of freedom. Multiply by 3 since each node has 3 DOFs (ux, uy, rz)
        aux_idx=np.zeros(ndof,)                                                                     # Create a zero array to store DOF indices
        for i in range(ndof):                                                                       # Iterate through all DOFs
            aux_idx[i]=i                                                                            # Store the index of each DOF in the array
        
        nelem=self.elementos[-1].id                                                                 # Get the total number of elements in the frame
        K_Global=np.zeros((ndof,ndof))                                                              # Create a zero matrix with size (ndof x ndof) for the global stiffness matrix
        elem_modifier=(1,TIMOSHENKO,1)                                                                       # For this case, axial and bending are considered, shear is not
        kx=5/6                                                                                      # Shear correction factor for rectangular sections. Not useful in this case
        for elem in range(nelem):                                                                   # Iterate for each element (beams and columns)                                                        
            L=self.elementos[elem].longitud                                                         # Assign the lenght  that was stores for each element
            theta=self.elementos[elem].theta                                                        # Get the angle of  each element
            I=self.elementos[elem].seccion.b*self.elementos[elem].seccion.h**3/12                   # Get the inertia on the axis considered
            A=self.elementos[elem].seccion.b*self.elementos[elem].seccion.h                         # Get the area of each element
            E=self.elementos[elem].material.E                                                       # Get the Young's Module of each element
            nu=self.elementos[elem].material.nu                                                     # Assign the poisson coeficient to the element
            k1=local_rot_elem_normal(L,theta,I,A,kx,E,nu,elem_modifier)                             # Transform from global coordinates of the element to local coordinates
            N_i=self.elementos[elem].nodo_i                                                         # Assign the number of the initial node for each element
            N_j=self.elementos[elem].nodo_j                                                         # Assign the number of the ending node for each element
            dof_i=self.nodos[N_i-1].dofs                                                            # Get the DOFs of each node
            dof_i = [x - 1 for x in dof_i]                                                          # Adjust DOF indices to be zero-based
            dof_j=self.nodos[N_j-1].dofs                                                            # Get the DOFs of each node
            dof_j = [x - 1 for x in dof_j]                                                          # Adjust DOF indices to be zero-based
            list_dof=dof_i+dof_j                                                                    # Join the DOFs of both nodes to create a list of DOFs for the element
            list_dof=np.array(list_dof)                                                             # Convert the list of DOFs to a numpy array
            K_Global[np.ix_(list_dof,list_dof)]+=k1                                                 # Assemble the element stiffness matrix into the global stiffness matrix
        
        nnodos_restrain=len(self.nodos_base)                                                        # Get the number of restrained nodes (base nodes)
        list_dof_restrain=[]                                                                        # Create an empty list for restrained DOFs
        for n in range(nnodos_restrain):                                                            # Iterate through each restrained node
            list_dof_restrain+=self.nodos_base[n].dofs                                              # Append the DOFs of each restrained node to the list
        
        idx_list_dof_restrain = [x - 1 for x in list_dof_restrain]                                  # Adjust restrained DOF indices to be zero-based
        K_reducida=np.delete(K_Global,idx_list_dof_restrain,axis=0)                                 # Remove rows corresponding to restrained DOFs
        aux_idx_redu=np.delete( aux_idx,idx_list_dof_restrain,axis=0)                               # Remove entries from the DOF index array for restrained DOFs
        K_reducida=np.delete(K_reducida,idx_list_dof_restrain,axis=1)
        
        return K_reducida,aux_idx_redu                                                              # Gives back the reduced stiffness matrix and the array of DOF indices after removing restrained DOFs
    
    def Matriz_dinamica(self,K,aux_idx_redu):                                                       # Function to obtain the dynamic stiffness matrix    
        aux_dof_redu=aux_idx_redu+1                                                                 # Add 1 to the index of the auxiliary DOFs to be in the real poisition
        n_dof_din=len(self.nodos_dinamicos_piso)                                                    # Number of dynamic DOFs
        n_dof_total=len(K)                                                                          # Total number of DOFs 
        n_pisos=self.nodos_dinamicos_piso[-1].piso                                                  # Number of floors in the 2D frame
        list_dof=np.zeros(n_dof_total,)                                                             # Create a zero array to store DOFs
        #list_aux=[n for n in range(len(list_dof))]               
        q=0                                                                                         # Initialize a counter in zero for dynamic DOFs
        for i in range(3):                                                                          # Iterate through each DOF type (ux, uy, rz)
            for n in range(n_dof_din):                                                              # Iterate through each dynamic node
                list_dof[q]=self.nodos_dinamicos_piso[n].dofs[i]                                    # Store dynamic DOFs first
                q+=1                                                                                # Increment counter

        idx_reord=[np.where(aux_dof_redu==x)[0][0] for x in list_dof]                               # Reorder the indices to have dynamic DOFs first. Result is a Tuple.
        idx_reord=[int(x) for x in idx_reord]                                                       # Transform tuple to list of integers    
        K_din1=K[np.ix_(idx_reord,idx_reord)]                                                       # Reordered matrix with dynamic DOFs first
        a=[]                                                                                        # Create empty list a
        b=[]                                                                                        # Create empty list b
        a+=[x for x in range(n_dof_din)]                                                            # Fill list a with dynamic DOFs
        b+=[x+n_dof_din for x in range((n_dof_total-n_dof_din))]                                    # Fill list b with static DOFs
        K_din2=K_din1[np.ix_(a,a)]-K_din1[np.ix_(a,b)]@ np.linalg.inv(K_din1[np.ix_(b,b)])@ K_din1[np.ix_(b,a)]     # Condensation of static DOFs 
        x=[]                                                                                        # Create empty list x
        x+=[x.piso-1 for x in self.nodos_dinamicos_piso]                                            # Fill list x with floor numbers of dynamic nodes
        Matriz_Traspaso=np.zeros((n_dof_din,n_pisos))                                               # Create a trasnfer matrix of zeros
        for piso in range(n_pisos):                                                                 # Iterate through each floor
            for n in range(n_dof_din):                                                              # Iterate through each dynamic DOF
                if x[n]==piso:                                                                      # If the floor number matches
                    Matriz_Traspaso[n,piso]=1                                                       # Set the corresponding entry in the transfer matrix to 1

        K_din3= Matriz_Traspaso.T@K_din2@Matriz_Traspaso                                            # Final dynamic stiffness matrix reduced to number of floors
        return K_din3                                                                               # Gives back the dynamic stiffness matrix reduced to number of floors
    
    def Propiedades_dinamicas_Convensional(self,Ma,K):                                              # Function to obtain dynamic properties without TMD
        M=np.diag(Ma)                                                                               # Create mass matrix as a diagonal matrix from mass array
        M_inv = np.linalg.inv(M)                                                                    # Inverse of mass matrix
        A=M_inv@K                                                                                   # Matrix A for eigenvalue problem    
        w2, phi = np.linalg.eig(A)                                                                  # Eigenvalues and eigenvectors of matrix A to obtain natural frequencies and mode shapes
        idx = np.argsort(w2)                                                                        # Sort indices of eigenvalues in ascending order.
        w2_ord=w2[idx]                                                                              # Reordered frequencies squared from lowest to highest
        phi_ord=phi[:,idx]                                                                          # Reordered mode shapes corresponding to sorted frequencies

        w=np.sqrt(w2_ord)                                                                           # Natural frequencies (rad/s)
        T=2*np.pi/w                                                                                 # Natural periods (s)                                   
                                                                                                    ##-------- Participation factors and modal masses ------##
        n=M.shape[1]                                                                                # Number of DOFs     
        r=np.ones((n,1))                                                                            # Vector of ones for participation factor calculation
        gamma=(phi_ord.T @ M @ r).flatten()/ np.diag(phi_ord.T @ M @ phi_ord)                       # Participation factors for each mode
        Mei=(gamma**2)*np.diag(phi_ord.T @ M @ phi_ord)                                             # Modal masses for each mode
        Mei_pct=100*Mei/np.sum(Mei)                                                                 # Modal masses as percentage of total mass
        return phi_ord ,w ,T ,M, gamma, Mei_pct, Mei                                                # Return mode shapes, natural frequencies, periods, mass matrix, participation factors, modal masses percentage, and modal masses
    
    def Matriz_amortiguamieto_modal_inherente(self,phi_ord,w,M,zeta):                               # Function to obtain damping matrix using modal damping ratios
        
        m_modal= phi_ord.T@ M @ phi_ord                                                             # Get the modal mass matrix
        dim=len(M)                                                                                  # Dimension of the system
        c_modal=np.zeros((dim,dim))                                                                 # Create a zero matrix for modal damping matrix, not included TMD
                                                                                                    ##----- ASSEMBLE MODAL DAMPING MATRIX -----##
        for i in range(dim):                                                                        # Iterate through each mode
            c_modal[i,i]=2*zeta*m_modal[i,i]*w[i]                                                   # Diagonal entries of modal damping matrix                          
        C=np.linalg.inv(phi_ord.T)@ c_modal @ np.linalg.inv(phi_ord)                                # Transform modal damping matrix to physical coordinates
        return C                                                                                    # Return physical damping matrix without TMD

    def cargar_registro(self):                                                                      # Function to load seismic record in AT2 format using Streamlit
        uploaded_file = st.file_uploader(                                                           # File uploader widget in Streamlit
            "Upload an AT2 file",
            type=["AT2"])
        if uploaded_file is None:                                                                   # If no file is uploaded, show info message and stop execution
            st.info("⬆️ Please upload a seismic record to continue")                                # Message to show in Streamlit
            st.stop()                                                                               # Stop execution until a file is uploaded                                                                                                          

        nombre_archivo = uploaded_file.name                                                         # Get the name of the uploaded file
        content = uploaded_file.read().decode("utf-8", errors="ignore")                             # Read the content of the uploaded file and decode it to a string
        lines = content.splitlines()                                                                # Split the content into lines
        dt = None                                                                                   # Initialize dt variable
        for line in lines:                                                                          # Loop through each line to find the DT value
            if "DT=" in line:                                                                       # Check if the line contains "DT="
                try:                                                                                # Try to extract the DT value
                    dt = float(line.split("DT=")[1].split()[0])                                     # Extract and convert DT value to float
                    break                                                                           # Exit the loop once DT is found
                except:                                                                             # If conversion fails,
                    continue                                                                        

        if dt is None:                                                                              # If DT value was not found, show error message and stop execution
            st.error("❌ The DT value was not found in the file.")                                  # Error message to show in Streamlit
            st.stop()                                                                               # Stop execution

        acc_values = [0.0]                                                                          # Initialize list to store acceleration values with initial zero
        for line in lines:                                                                          # Loop through each line to extract acceleration values                                                                                    
            try:                                                                                    # Try to convert line to float values
                nums = [float(x) for x in line.split()]                                             # Convert each value in the line to float
                acc_values.extend(nums)                                                             # Add the extracted values to the acceleration list
            except ValueError:                                                                      # If conversion fails, skip the line
                continue                                                                            # Skip lines that cannot be converted to float

        n = len(acc_values)                                                                         # Get the number of acceleration data points
        time = np.arange(0, n * dt, dt)                                                             # Create time array based on dt and number of data points
        time = time[:n]                                                                             # Ensure time array length matches acceleration data length

        REG_SISM = np.column_stack((time, acc_values))                                              # Combine time and acceleration values into a 2D array
        df = pd.DataFrame(REG_SISM, columns=["Time (s)", "Acceleration (g)"])                       # Create a DataFrame for better visualization in Streamlit

        st.success("✅ File read successfully.")                                                    # Success message to show in Streamlit
        st.write(f"**File:** {nombre_archivo}")                                                     # Display the name of the uploaded file
        st.write(f"**dt:** {dt:.6f} s")                                                             # Display the dt value  
        st.write(f"**Number of data points:** {n}")                                                 # Display the number of data points

        formato = st.radio(                                                                         # Radio button to select acceleration format                                                                      
            "Acceleration Format:",                                                              
            ["Decimal", "Scientific"],
            horizontal=True )
        df_formatted = df.copy()                                                                    # Create a copy of the DataFrame for formatting
        col = "Acceleration (g)"                                                                    # Column to format
        if formato == "Scientific":                                                                 # Format acceleration values in scientific notation
            df_formatted[col] = df_formatted[col].map(lambda x: f"{x:.6e}")
        else:                                                                                       # Format acceleration values in decimal notation
            df_formatted[col] = df_formatted[col].map(lambda x: f"{x:.9f}")
        st.dataframe(df_formatted, use_container_width=True)                                        # Display the formatted DataFrame in Streamlit
        return df, dt, nombre_archivo                                                               # Return the DataFrame, dt value, and file name

    def mostrar_portico2(self, escala=1.00):                                                        # Function to visualize the 2D frame using Plotly
        fig = go.Figure()                                                                           # Create a new Plotly figure
        for e in self.elementos:                                                                    # Iterate through each element
            ni = self._get_nodo_por_id(e.nodo_i)                                                    # Get the starting node of the element     
            nj = self._get_nodo_por_id(e.nodo_j)                                                    # Get the ending node of the element
            color = "#00BFFF" if e.tipo == "columna" else "#FF4C4C"                             # Determine color based on element type

            fig.add_trace(go.Scatter(                                                               # Add a trace for the element line
                x=[ni.x, nj.x],                                                                     # X-coordinates of the element endpoints    
                y=[ni.y, nj.y],                                                                     # Y-coordinates of the element endpoints
                mode="lines",                                                                       # Type of trace is lines
                line=dict(color=color, width=3*escala),                                             # Line properties
                hovertemplate=(                                                                     # Hover text template
                    f"<b>Elemento E{e.id}</b><br>"                                                  # Bold element ID
                    f"Ni=({ni.x:.2f}, {ni.y:.2f})<br>"                                              # Starting node coordinates
                    f"Nj=({nj.x:.2f}, {nj.y:.2f})<extra></extra>"                                   # Ending node coordinates
                ),
                showlegend=False                                                                    # Do not show legend for this trace
            ))

                                                                                                    # Center text for element ID
            xm = (ni.x + nj.x) / 2                                                                  # Midpoint X-coordinate of the element
            ym = (ni.y + nj.y) / 2+0.5                                                              # Midpoint Y-coordinate of the element

            fig.add_trace(go.Scatter(                                                               # Add a trace for the element ID text
                x=[xm],                                                                             # X-coordinate of the text
                y=[ym],                                                                             # Y-coordinate of the text
                mode="text",                                                                        # Type of trace is text
                text=[f"E{e.id}"],                                                                  # Text to display (element ID)
                textposition="middle center",                                                       # Position of the text
                textfont=dict(size=10*escala, color="blue"),                                        # Text font properties for element ID
                hoverinfo="skip"                                                                    # Skip hover info over text
            ))

                                                                                                    # To draw nodes 
        for n in self.nodos:                                                                        # Iterate through each node
            fig.add_trace(go.Scatter(                                                               # Add a trace for the node
                x=[n.x],                                                                            # X-coordinate of the node
                y=[n.y],                                                                            # Y-coordinate of the node
                mode="markers+text",                                                                # Type of trace is markers and text
                marker=dict(color="white", size=8*escala, line=dict(color="black", width=1)),       # Marker properties for the node
                text=[f"N{n.id}"],                                                                  # Text to display (node ID)
                textposition="top center",                                                          # Position of the text  
                textfont=dict(size=9*escala, color="green"),                                        # Text font properties for nodes
                hovertemplate=f"<b>Nodo N{n.id}</b><br>x={n.x:.2f}<br>y={n.y:.2f}<extra></extra>"   # Hover text template for nodes
            ))

                                                                                                    # Layout configuration
        fig.update_layout(                                                                          # Update layout properties of the figure
            title=dict(                                                                             # Title configuration
                text="2D Frame – Node and Element Numbering",                                       # Title text
                x=0.5,                                                                              # Center the title
                xanchor="center",                                                                   # Anchor title at center    
                font=dict(size=18*escala, family="Arial Black", color="#3A08EF")                  # Title font properties
            ),
            xaxis=dict(                                                                             # X-axis configuration
                title="Distance [m]",                                                              # X-axis title
                showgrid=True,                                                                      # Show grid lines
                gridcolor="#525353",                                                                # Grid line color                         
                zeroline=False,                                                                     # Do not show zero line
                tickcolor="#9CC9FF",                                                              # Tick color
                ticks="outside",                                                                    # Ticks outside the axis        
                ticklen=6,                                                                          # Tick length
                title_standoff=8,                                                                   # Distance between title and axis
                scaleanchor="y",                                                                    # Link x-axis scale to y-axis
                scaleratio=1                                                                        # Equal scaling ratio                                 
            ),
            yaxis=dict(                                                                             # Y-axis configuration
                title="Height [m]",                                                                 # Y-axis title          
                showgrid=True,                                                                      # Show grid lines
                gridcolor="#525353",                                                                # Grid line color
                zeroline=True,                                                                      # Show zero line
                zerolinecolor="#8FB6FF",                                                            # Zero line color
                zerolinewidth=1.1,                                                                   # Zero line width
                tickcolor="#9CC9FF",                                                                 # Tick color
                ticks="outside",                                                                     # Ticks outside the axis
                ticklen=6,                                                                           # Tick length
                title_standoff=8                                                                     # Distance between title and axis                                          
            ),
            paper_bgcolor="#FFFFFF",                                                                # Background color of the figure
            plot_bgcolor="#FCFCFC",                                                                 # Plot area background color
            hovermode="closest",                                                                    # Hover mode configuration
            width=900,                                                                              # Figure width
            height=600,                                                                             # Figure height                                   
            margin=dict(l=60, r=40, t=90, b=70),                                                    # Margins around the figure
            showlegend=False                                                                        # Do not show legend                           
        )

        st.plotly_chart(fig, use_container_width=True)                                              # Display the Plotly figure in Streamlit

    def derivas_maximas(self,U, alturas):                                                           # Function to calculate maximum interstory drifts
        U = np.asarray(U, float)                                                                     # Convert U to a NumPy array of floats
        h = np.asarray(alturas, float).reshape(1, -1)                                               # Convert alturas to a NumPy array and reshape to (1, n_pisos)
                                                                                                    # Calculate interstory drifts
        dU = np.empty_like(U)                                                                       # Create an empty array with the same shape as U
        dU[:, 0]  = U[:, 0]                                                                         # First floor drift is equal to its displacement
        dU[:, 1:] = U[:, 1:] - U[:, :-1]                                                            # Subsequent floor drifts are the difference between consecutive floor displacements
                                                                                                    # Calculate drift ratios
        drift = dU / h                                                                              # Calculate drift ratios by dividing interstory drifts by floor heights
                                                                                                    # Determine maximum drifts
        drift_max = np.max(np.abs(drift), axis=0)                                                   # Calculate maximum absolute drift for each floor
                                                                                                    # Identify critical floor and its drift
        piso_crit = int(np.argmax(drift_max) + 1)                                                   # Find the index of the floor with maximum drift (1-based index)
        drift_global = float(drift_max[piso_crit - 1])                                              # Maximum drift at the critical floor

        return drift, drift_max, piso_crit, drift_global                                            # Return interstory drifts, maximum drifts, critical floor, and global maximum drift

    def Desp_Base(self, REG_SISM_fft):                                                              # Function to calculate base displacements from seismic record
        t = REG_SISM_fft[:, 0]                                                                      # Time array from seismic record
        a = REG_SISM_fft[:, 1].astype(float) * 9.8067                                               # Acceleration array converted to m/s²     
        n = len(a)                                                                                  # Number of data points

        v = np.zeros(n); r = np.zeros(n)                                                            # Initialize velocity and displacement arrays
        v[0] = 0; r[0] = 0                                                                          # Initial conditions: zero velocity and displacement

        for i in range(n-1):                                                                        # Loop through each time step to perform numerical integration
            dt = t[i+1] - t[i]                                                                      # Calculate time step                             
            v[i+1] =  ( a[i+1]-a[i]) * dt/2 + a[i]*dt + v[i]                                        # Update velocity      
            r[i+1] =  (a[i+1]-a[i]) * (dt**2) / 6.0 + a[i]*(dt**2)/2 + v[i]*dt + r[i]               # Update displacement 
        return v, r                                                                                 # Return velocity and displacement arrays




    def plot_filtered_record(self,REG_SISM,dt,nombre_archivo):                                      #function in tab 2 MAIN
        pga_max_idx = np.argmax(np.abs(REG_SISM[:,1]))
        pga_max=REG_SISM[pga_max_idx,1]
                                                                                                                        # Seismic record plot
        fig = px.line(
            x=REG_SISM[:, 0],
            y=REG_SISM[:, 1],
            title=nombre_archivo,
            line_shape="linear"
        )
                                                                                                                        # General style
        fig.update_traces(line=dict(color="blue", width=1.5), name="Seismic Record",showlegend=True )
        fig.update_layout(
            template="simple_white",
            title_font=dict(size=18, family="Arial", color="black"),
            font=dict(size=13),
            xaxis=dict(showgrid=True, zeroline=True, title="Time [s]"),
            yaxis=dict(showgrid=True, zeroline=True, title="Acceleration [g]"),
            width=800, height=400,
            title=dict(x=0.5, xanchor="center")
        )
                                                                                                                        # Max Point of PGA
        fig.add_scatter(
            x=[REG_SISM[pga_max_idx, 0]],
            y=[REG_SISM[pga_max_idx, 1]],
            mode="markers+text",
            name="PGA (g)",
            showlegend=True,
            text=[f"<b>Sa[g]={pga_max:.6f} g</b>"],
            textposition="middle right",
            line=dict(color="red", width=1.5)
        )
                                                                                                                        # FFT analysis
        t = REG_SISM[:, 0]                                                                                              # Time vector
        acc = REG_SISM[:, 1]                                                                                            # Acceleration vector
        n = len(acc)                                                                                                    # Number of samples                                                    
        FFT = np.fft.fft(acc)                                                                                           # FFT calculation                               
        freq = np.fft.fftfreq(n, d=dt)                                                                                  # Frequency vector         
        idx = freq >= 0                                                                                                 # Positive frequencies only
        freq_pos = freq[idx]                                                                                            # Positive frequency vector
        FFT_pos = FFT[idx]                                                                                              # Positive FFT values

        amplitud = np.abs(FFT_pos)                                                                                      # Amplitude spectrum
        amplitud = (2 / n) * np.abs(FFT_pos)                                                                            # Normalize amplitude
        fig_fft = px.line(                                                                                              
            x=freq_pos,
            y=amplitud,
            title="Fourier Spectrum of the Seismic Record",
            labels={
                "x": "Frequency [Hz]",
                "y": "Spectral Amplitude"
            }
        )                                                                                                               # FFT plot
        fig_fft.update_traces(
            line=dict(color="blue", width=1.5),
            name="Acceleration FFT",
            showlegend=True
        )                                                                                                               # General style for FFT plot                    

        fig_fft.update_layout(
            template="simple_white",
            title_font=dict(size=18, family="Arial", color="black"),
            font=dict(size=13),
            xaxis=dict(type="log",showgrid=True, zeroline=True),
            yaxis=dict(showgrid=True, zeroline=True),
            height=400,
            title=dict(x=0.5, xanchor="center"),
            showlegend=True
        )                                                                                                               # Display plots side by side

        col_acc, col_fft = st.columns([2.5, 1.5])                                                                       # Columns for time signal and FFT
        with col_acc:                                                                                                   # Create a column for time signal  
            st.plotly_chart(fig, use_container_width=True)                                                              # Time-domain plot
        with col_fft:                                                                                                   # Create a column for FFT
            st.plotly_chart(fig_fft, use_container_width=True)                                                          # FFT plot

        st.subheader("Band-Pass Filter (Frequencies)")                                                                  # Band-pass filter section
        st.markdown("The application of the following filter is left to " \
        "the informed discretion of the user. It is intended to minimize the " \
        "influence of environmental noise, such as traffic, footsteps, and " \
        "other ambient sounds, which may affect the accuracy of the recorded " \
        "signals. Careful consideration should be given to the frequency " \
        "filtering process, ensuring that the selected frequency ranges are " \
        "consistent with the operational characteristics and sensitivity of the" \
        " measurement instruments. Proper understanding of both the filter design "
        "and the limitations of the recording equipment is essential to avoid " \
        "distortion of structural response data while effectively isolating " \
        "the structural vibrations of interest.")

        f_min = st.number_input(
            "Frecuencia mínima [Hz]",
            min_value=0.0,
            max_value=float(np.max(freq_pos)),
            value=0.0,
            step=0.1
        )                                                                                                               # Minimum frequency input

        f_max = st.number_input(
            "Frecuencia máxima [Hz]",
            min_value=0.0,
            max_value=float(np.max(freq_pos)),
            value=20.0,
            step=0.1
        )                                                                                                              # Maximum frequency input

        if f_min >= f_max:                                                                                              # Validate frequency inputs
            st.error("⚠️ The minimum frequency must be lower than the maximum frequency")                              # Error message
            st.stop()                                                                                                   # Stop execution if invalid
        mask = (freq_pos >= f_min) & (freq_pos <= f_max)                                                                # Band-pass filter mask                                   
        freq_pb = freq_pos[mask]                                                                                        # Filtered frequency vector
        amp_pb = amplitud[mask]                                                                                         # Filtered amplitude spectrum
        fig_fft_pb = px.line(
            x=freq_pb,
            y=amp_pb,
            title=f"Filtered Fourier Spectrum ({f_min:.2f} – {f_max:.2f} Hz)",
            labels={
                "x": "Frequency [Hz]",
                "y": "Spectral Amplitude"
            }
        )                                                                                                               # Filtered FFT plot                                    
        fig_fft_pb.update_traces(line=dict(color="blue", width=1.5),name="Filtered FFT",showlegend=True)                # Filtered FFT style

        fig_fft_pb.update_layout(
            template="simple_white",
            title_font=dict(size=18,family="Arial",color="black"),
            height=400,
            title=dict(x=0.5, xanchor="center"),
            xaxis=dict(type="log",showgrid=True),
            yaxis=dict(showgrid=True)
        )                                                                                                               # Filtered FFT plot
        FFT_full = np.fft.fft(acc)                                                                                      # Full FFT calculation
        freq_full = np.fft.fftfreq(n, d=dt)                                                                             # Full frequency vector
        mask_full = (np.abs(freq_full) >= f_min) & (np.abs(freq_full) <= f_max)                                         # Full frequency mask     
        FFT_filt = FFT_full * mask_full                                                                                 # Apply band-pass filter to full FFT
        acc_filt = np.real(np.fft.ifft(FFT_filt))                                                                       # Filtered time-domain signal
        fig_filt = px.line(
            x=t,
            y=acc_filt,
            title="Filtered Accelerogram (Band-Pass)",
            labels={"x": "Time [s]","y": "Acceleration [g]"},  
        )                                                                                                               # Filtered time-domain plot

        fig_filt.update_traces(
            line=dict(color="blue", width=1.5),
            name="Filtered Record",
            showlegend=True
        )                                                                                                               # Filtered time-domain style

        fig_filt.update_layout(
            template="simple_white",
            title_font=dict(size=18,family="Arial",color="black"),
            height=400,
            title=dict(x=0.5, xanchor="center"),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )                                                                                                               # Filtered time-domain signal plot
            
        col_acc_fil, col_fft2 = st.columns([2.5, 1.5])                                                                  # Columns for filtered time signal and FFT
        with col_acc_fil:
            st.plotly_chart(fig_filt, use_container_width=True)                                                         # Filtered time-domain plot
        with col_fft2:
            st.plotly_chart(fig_fft_pb, use_container_width=True)                                                       # Filtered FFT plot
        return np.array(acc_filt)
    
    def nice_dtick(self,max_abs, n_ticks=5):                                                                              # Function to calculate tick intervals
                
                max_abs = float(max_abs)
                if max_abs <= 0:
                    return 1.0
                raw = (2 * max_abs) / n_ticks  
                exp = 10 ** np.floor(np.log10(raw))
                frac = raw / exp

                if frac <= 1:
                    step = 1
                elif frac <= 2:
                    step = 2
                elif frac <= 5:
                    step = 5
                else:
                    step = 10
                return float(step * exp)
    def nice_ylim(self,max_abs, dtick):                                                                                  # Function to calculate nice y-limits
            if max_abs <= 0:
                return dtick
            return float(np.ceil(max_abs / dtick) * dtick)



    def tmd_analysis(self, REG_SISM,acc_filt,M,K_din3,C,w,T,Mei_pct,Mei_pct_acum,altura_pisos):
        betaBN=0.25                                                                                             # Newmark parameters
        gammaBN=0.5                                                                                             # Newmark parameters
        REG_SISM_fft=np.column_stack([REG_SISM[:, 0],acc_filt])                                                 # Combine time and filtered acceleration                                
        Respuesta_Des,Respuesta_Vel,Respuesta_Acc=Respuesta_tiempo_historiaV2_nb(M,K_din3,C,REG_SISM_fft,gammaBN,betaBN)   # Structural response without TMD 

                                                                                                                #Determine number of modes for 90% mass participation   
        indx90=0                                                                                                # Create a variable to store the index of the mode that reaches 90% mass participation
        for i in range(len(Mei_pct_acum)):                                                                      # Loop through cumulative mass participation
            if Mei_pct_acum[i]>=90:                                                                             # Check if cumulative participation is >= 90%                                         
                indx90=i                                                                                        # Store the index
                break                                                                                           # Exit loop once found
        
                                                                                                                #Location (mass (i), frequency (z), damping ratio (j))
        alturas = np.asarray(altura_pisos, dtype=np.float64)                                                    # Heights of each floor                              
        gdl=len(M)                                                                                              # Degrees of freedom 
        h = np.asarray(alturas, float).reshape(-1, 1)                                                           # Reshape heights into column vector
        deriva_matrix=np.eye(gdl)                                                                               # Identity matrix for drift calculation
        for i in range(gdl-1):                                                                                  # Loop to create drift calculation matrix
            deriva_matrix[i+1,i]=-1                                                                             # Set -1 for lower floor

        deriva=deriva_matrix@Respuesta_Des                                                                      # Calculate interstory drift
        deriva_max=np.max(np.abs(deriva)/h)                                                                     # Maximum interstory drift ratio


        ms2=np.linspace(0.005,0.06,30)*np.sum(M)                                                                # Multiplied by total mass
        ws2=w[0:indx90+1]                                                                                       # Frequencies up to 90% mass participation          
        
        FREQ_OPT_90=[]                                                                                          # Store modal frequencies up to 90% mass participation
        for i in range(indx90+1):                                                                               # Loop through modes up to indx90
            FREQ_OPT_90.append({    
                "w(rad/sec)": w[i],
                "T(sec)": T[i],
                "%Mass Participation": Mei_pct[i],
                "%Cumulative Mass Participation": Mei_pct_acum[i]
            })                                                                                                  # Append modal properties to list                                      
        df_freq_op = pd.DataFrame(FREQ_OPT_90)                                                                  # Create DataFrame from modal properties
        st.markdown("## Modal frequencies of the structure without TMD for modes contributing up to 90 % of mass participation")   # Display heading
        st.dataframe(df_freq_op)                                                                                # Display modal frequencies DataFrame         

        zetas2=np.array([0.05], dtype=np.float64)                                                               # Fixed damping ratio for TMD
        R2,Location_opt2=optimizacion_TMD_nb(M,C,K_din3,zetas2,ws2,ms2,REG_SISM_fft,gammaBN,betaBN,alturas)     # TMD optimization calculation
        i_opt2 = Location_opt2[0]                                                                               # Optimal mass index
        z_opt2=Location_opt2[1]                                                                                 # Optimal frequency index
        j_opt2=Location_opt2[2]                                                                                 # Optimal damping index
        R_i_2 = R2[:, :, 0]                                                                                     # Extract results for fixed damping ratio

        W2, MS2 = np.meshgrid(ws2, ms2/np.sum(M)*100, indexing="xy")                                            # Create meshgrid for plotting
        
        if len(M)>1:                                                                                            # 2D contour plot                                
            fig2, ax = plt.subplots(figsize=(6, 6))                                                             # Create figure and axis
                                                                                                                # Color map for interstory drift ratio reduction
            cont = ax.contourf( W2,MS2,(R_i_2)/deriva_max, levels=20, cmap="viridis")                           # Filled contour plot
            cbar = fig2.colorbar(cont, ax=ax)                                                                   # Color bar
            cbar.set_label("R = max |IDR_TMD| / max |IDR_NOTMD|")                                               # Color bar label
            ax.set_xlabel("ω (rad/s)")                                                                          # X-axis label
            ax.set_ylabel("% mass")                                                                             # Y-axis label
            ax.scatter(ws2[z_opt2],ms2[i_opt2]/np.sum(M)*100, color='red', s=80, label='Punto de interés')      # Highlight optimal point
                                                                                                                # 3D plot
            Z = (R_i_2) / deriva_max                                                                            # Z values for 3D plot
            fig = go.Figure()                                                                                   # Create 3D figure
            fig.add_trace(go.Surface(
                x=W2,
                y=MS2,
                z=Z,
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(
                        text="R = max |IDR_TMD| / max |IDR_NOTMD|",
                        side="right"
                    )
                )
            ))                                                                                                  # Surface plot                             
            
            fig.add_trace(go.Scatter3d(             
                x=[ws2[z_opt2]],
                y=[ms2[i_opt2] / np.sum(M) * 100],
                z=[Z[i_opt2, z_opt2]],
                mode='markers',
                marker=dict(size=6, color='red'),
                name="Optimal point"
            ))                                                                                                  # Highlight optimal point in 3D plot
            fig.update_layout(
                scene=dict(
                    xaxis_title="ω (rad/s)",
                    yaxis_title="% mass",
                    zaxis_title="Interstory drift ratio reduction"
                ),
                width=800,
                height=600,
                margin=dict(l=0, r=0, b=0, t=40)
            )                                                                                                   # Layout settings for 3D plot
            col1, col2 = st.columns([1, 1])                                                                     # Create two columns for plots
            with col1:                                                                                          # First column for 2D contour plot
                st.pyplot(fig2)                                                                                 # Display 2D contour plot
            with col2:                                                                                          # Second column for 3D plot
                st.plotly_chart(fig, use_container_width=True)                                                  # Display 3D plot
        else:                                                                                                   # 1D plot for single degree of freedom
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(ms2 / np.sum(M) * 100,(R_i_2)/deriva_max,'-')                                               # Plot mass vs IDRR
            ax.set_xlabel("% mass")                                                                             # X-axis label
            ax.set_ylabel("R = max |IDR_TMD| / max |IDR_NOTMD|")                                                # Y-axis label
            ax.set_title("Mass vs IDRR (fixed frequency)")                                                      # Plot title
            ax.grid(True)                                                                                       # Enable grid
            st.pyplot(fig)                                                                                      # Display 2D plot
        


        st.markdown("### The interstory drift ratio reduction (IDRR) due to the TMD is evaluated based on the maximum " \
        "interstory drift ratio of the structure with and without the TMD, representing the global reduction" \
        " effect of the device on the structure")                                                               # Set markdown explanation


        valores2 = np.array([ms2[i_opt2]*9.8067,
                                ws2[z_opt2],
                                ms2[i_opt2]*ws2[z_opt2]**2, 
                                R2[i_opt2,z_opt2,j_opt2]/deriva_max]) 
        TMD_OPT2 = pd.DataFrame(valores2,index=["Optimal Weight [ton]",
                                                "Optimal angular frequency [rad/sec]",
                                                "Optimal stiffness[Ton/m]", "Interstory drift ratio reduction due to TMD"], 
                                                columns=["Valor"])
        
        st.dataframe(TMD_OPT2)                                                                                                              # Display optimal TMD parameters DataFrame
        k=ms2[i_opt2]*ws2[z_opt2]**2                                                                                                        # Calculate stiffness
        m=ms2[i_opt2]                                                                                                                       # Mass
        zeta_TMD=zetas2[0]                                                                                                                  # Damping ratio

        C_TMD=Matriz_amortiguamieto_conTMD_nb(C,zeta_TMD,k,m)                                                                               # Damping matrix with TMD
        _,_,_,M_TMD, K_TMD,_ ,_=Propiedades_dinamicas_conTMD_nb(M,K_din3,m,k)                                                               # Dynamic properties with TMD
        Respuesta_Des_TMD,Respuesta_Vel_TMD,Respuesta_Acc_TMD=Respuesta_tiempo_historiaV2_nb(M_TMD,K_TMD,C_TMD,REG_SISM_fft,gammaBN,betaBN)  # Structural response with TMD

        return (
                REG_SISM_fft,
                Respuesta_Des,
                Respuesta_Vel,
                Respuesta_Acc,
                Respuesta_Des_TMD,
                Respuesta_Vel_TMD,
                Respuesta_Acc_TMD,
                deriva_max,
                ms2[i_opt2],
                ws2[z_opt2],
                zetas2[0],
                k,
                R2,
                R_i_2,
                ws2,
                ms2,
                alturas,
                deriva_matrix,
                indx90
            )








    def plot_Comparison_tmd(self,REG_SISM_fft,Respuesta_Des,                                                            #function in tab 4 MAIN
    Respuesta_Vel,Respuesta_Acc,Respuesta_Des_TMD, Respuesta_Vel_TMD,
    Respuesta_Acc_TMD,alturas, modelo, nice_dtick,nice_ylim):

        st.subheader("🎞️ Comparison: WITHOUT TMD vs WITH TMD")                                                          # Subheader for comparison section
        c1, c2, c3, c4 = st.columns([1.2, 1.1, 1.8, 1.2])                                                               # Create columns for input controls
        with c1:                                                                                                        # Units selection
            unit_u = st.selectbox("Units", ["m", "cm", "mm"], index=1)                                                  # Select units for displacement
            fac_u = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[unit_u]                                                       # Conversion factor for displacement
        with c2:                                                                                                        # Frame stride selection
            frame_stride = st.slider("Frame Jump", 1, 20, 1)                                                            # Slider for frame jump
        with c3:                                                                                                        # Response type selection
            resp_kind = st.radio("Analyzed Dynamic Unit", ["u", "v", "a_abs"], horizontal=True, index=0)                # Radio buttons for response type
        with c4:                                                                                                        # Animation speed selection
            fps = st.slider("Velocity (fps)", 1, 60, 20)                                                                # Slider for frames per second

        c5, c6 = st.columns([1.2, 1.0])                                                                                 # Create columns for x-axis scaling options
        with c5:                                                                                                        # Auto scale option
            auto_x = st.checkbox("Auto Scale X (maximum between WITHOUT and WITH TMD)", value=True)                     # Checkbox for auto scaling
        with c6:                                                                                                        # x-axis limit
            xlim_user = st.number_input(f"Common X Range (±) en {unit_u}", min_value=0.0001, value=5.0, step=0.5)       # defined x-axis limit
        n_dof_sin = Respuesta_Des.shape[0]                                                                              # Number of DOF without TMD
        n_dof_con = Respuesta_Des_TMD.shape[0]                                                                          # Number of DOF with TMD
        max_plot_con = max(1, n_dof_con - 1)                                                                            # Max DOF to plot with TMD (excluding TMD DOF)

        c7, c8 = st.columns(2)                                                                                          # Create columns for DOF selection
        with c7:                                                                                                        # DOF selection without TMD
            piso_plot_sin = st.number_input("DOF/Floor to Plot (WITHOUT TMD) 1..N", 1, int(n_dof_sin), int(n_dof_sin), 1)   # Input for DOF to plot without TMD
            dof_sin = int(piso_plot_sin) - 1                                                                            # Convert to zero-based index                           
        with c8:                                                                                                        # DOF selection with TMD
            piso_plot_con = st.number_input("DOF/Floor to Plot (WITH TMD) 1..N", 1, int(max_plot_con), int(max_plot_con), 1)    # Input for DOF to plot with TMD
            dof_con = int(piso_plot_con) - 1                                                                            # Convert to zero-based index
        v, r=modelo.Desp_Base(REG_SISM_fft)                                                                             # Obtain base displacement from seismic record
        
        t_fft = REG_SISM_fft[:, 0].astype(float)                                                                        # Time vector from seismic record
        ag_fft = REG_SISM_fft[:, 1].astype(float)                                                                       # Ground acceleration vector from seismic record
        idx = np.arange(0, len(t_fft), frame_stride, dtype=int)                                                         # Indices for frame skipping
        tA = t_fft[idx]                                                                                                 # Decimated time vector
        agA = ag_fft[idx]                                                                                               # Decimated ground acceleration vector
        ugA = r[idx]*fac_u                                                                                              # Decimated base displacement vector with unit conversion    

        y_floors = np.cumsum(alturas)                                                                                   # Heights of each floor
        y_nodes = np.concatenate(([0.0], y_floors))                                                                     # Node heights including ground level
        n_pisos = len(alturas)                                                                                          # Number of floors

        U_sin = Respuesta_Des.T                                                                                         # Displacement response without TMD (transposed)
        V_sin = Respuesta_Vel.T                                                                                         # Velocity response without TMD (transposed)
        A_sin = Respuesta_Acc.T                                                                                         # Acceleration response without TMD (transposed)

        U_con = Respuesta_Des_TMD.T                                                                                     # Displacement response with TMD (transposed)                                            
        V_con = Respuesta_Vel_TMD.T                                                                                     # Velocity response with TMD (transposed)
        A_con = Respuesta_Acc_TMD.T                                                                                     # Acceleration response with TMD (transposed)

        Aabs_sin_g = (A_sin / 9.8067) + ag_fft.reshape(-1, 1)                                                           # Absolute acceleration without TMD in g
        Aabs_con_g = (A_con / 9.8067) + ag_fft.reshape(-1, 1)                                                           # Absolute acceleration with TMD in g
        Aabs_con_g2 = Aabs_con_g[:,:-1]                                                                                 # Exclude TMD DOF for absolute acceleration with TMD
        max_ag_fft=np.max(np.abs(ag_fft))                                                                               # Maximum ground acceleration for scaling

        uA_sin = U_sin[idx, :] * fac_u                                                                                  # Decimated displacement without TMD with unit conversion
        vA_sin = V_sin[idx, :]                                                                                          # Decimated velocity without TMD
        aA_sin = Aabs_sin_g[idx, :]                                                                                     # Decimated absolute acceleration without TMD in g

        uA_con = U_con[idx, :] * fac_u                                                                                  # Decimated displacement with TMD with unit conversion
        vA_con = V_con[idx, :]                                                                                          # Decimated velocity with TMD
        aA_con = Aabs_con_g2[idx, :]                                                                                     # Decimated absolute acceleration with TMD in g

                                                                                                                        # Determine number of floors to plot
        n_use_sin = min(n_pisos, uA_sin.shape[1])                                                                       # Number of floors to use for plotting without TMD      
        n_use_con = min(n_pisos, uA_con.shape[1])                                                                       # Number of floors to use for plotting with TMD

        if resp_kind == "u":                                                                                            # Select response type to plot
            respA_sin = uA_sin[:, dof_sin]                                                                              # Displacement response without TMD
            respA_con = uA_con[:, dof_con]                                                                              # Displacement response with TMD
            ylab_resp = f"u ({unit_u})"                                                                                 # Y-axis label for displacement
        elif resp_kind == "v":                                                                                          # Velocity response
            respA_sin = vA_sin[:, dof_sin]                                                                              # Velocity response without TMD
            respA_con = vA_con[:, dof_con]                                                                              # Velocity response with TMD
            ylab_resp = "v (m/s)"                                                                                       # Y-axis label for velocity
        else:                                                                                                     
            respA_sin = aA_sin[:, dof_sin]                                                                              # Absolute acceleration response without TMD
            respA_con = aA_con[:, dof_con]                                                                              # Absolute acceleration response with TMD
            ylab_resp = "a_abs (g)"                                                                                     # Y-axis label for absolute acceleration

        xmax_common = float(np.max([
            np.max(np.abs(uA_sin[:, :n_use_sin] + ugA[:, np.newaxis])) if n_use_sin > 0 else 0.0,
            np.max(np.abs(uA_con[:, :n_use_con] + ugA[:, np.newaxis])) if n_use_con > 0 else 0.0]))                     # Maximum displacement for common x-axis limit
        if auto_x:                                                                                                      # Auto scale x-axis
            xlim_common = max(1e-6, 1.2 * xmax_common)                                                                  # Set common x-axis limit with margin
        else:
            xlim_common = float(xlim_user)                                                                              # Use user-defined x-axis limit

        H_total=np.sum(np.asarray(modelo.alturas_pisos, dtype=float))                                                   # Total height of the structure
        if H_total<=15:                                                                                                 # Determine layout height based on total height
            HPx=283                                                                                                     # Base height for layout
        if H_total>15:                                                                                                  # Determine layout height based on total height
            HPx=10*H_total
        
        LayoutHPx=HPx+2*283                                                                                             # Total layout height in pixels

        porcentaje1=HPx/LayoutHPx                                                                                       # Percentage height for first row
        porcentaje2=283/LayoutHPx                                                                                       # Percentage height for second and third rows
        
        fig = make_subplots(
            rows=3, cols=2,
            column_widths=[0.5, 0.5],
            row_heights=[porcentaje1, porcentaje2, porcentaje2],
            vertical_spacing=0.1,
            horizontal_spacing=0.08,
            subplot_titles=(
                f"WITHOUT TMD | {resp_kind} DOF {dof_sin+1}",
                f"WITH TMD | {resp_kind} DOF {dof_con+1}",
            )
        )                                                                                                               # Create subplots for comparison

        x0_sin = np.concatenate(([0.0], uA_sin[0, :n_use_sin]))                                                         # Initial frame displacement without TMD
        x0_con = np.concatenate(([0.0], uA_con[0, :n_use_con]))                                                         # Initial frame displacement with TMD

        fig.add_trace(go.Scatter(
            x=x0_sin, y=y_nodes[:n_use_sin+1],
            mode="lines+markers", marker=dict(size=14),
            name="Structure (WITHOUT TMD)"
        ), row=1, col=1)                                                                                                # Add trace for structure without TMD

        fig.add_trace(go.Scatter(
            x=x0_con, y=y_nodes[:n_use_con+1],
            mode="lines+markers", marker=dict(size=14),
            name="Structure (WITH TMD)"
        ), row=1, col=2)                                                                                                # Add trace for structure with TMD

        
        fig.add_trace(go.Scatter(x=tA, y=agA, mode="lines", name="ag (g)"), row=2, col=1)                               # Add trace for ground acceleration without TMD
        fig.add_trace(go.Scatter(x=[tA[0]], y=[agA[0]], mode="markers", marker=dict(size=10), name="t (ag)"),
                    row=2, col=1)                                                                                       # Add marker for current time without TMD                                   

        
        fig.add_trace(go.Scatter(x=tA, y=agA, mode="lines", name="ag2 (g)"), row=2, col=2)                            # Add trace for ground acceleration with TMD
        fig.add_trace(go.Scatter(x=[tA[0]], y=[agA[0]], mode="markers", marker=dict(size=10), name="t2 (ag)"),
                    row=2, col=2)                                                                                       # Add marker for current time with TMD

        
        fig.add_trace(go.Scatter(x=tA, y=respA_sin, mode="lines", name="resp"), row=3, col=1)
        fig.add_trace(go.Scatter(x=[tA[0]], y=[respA_sin[0]], mode="markers", marker=dict(size=10), name="t_without_TMD (resp)"),
                    row=3, col=1)                                                                                       # Add marker for current response without TMD

        
        fig.add_trace(go.Scatter(x=tA, y=respA_con, mode="lines", name="resp2"), row=3, col=2)
        fig.add_trace(go.Scatter(x=[tA[0]], y=[respA_con[0]], mode="markers", marker=dict(size=10), name="t_with_TMD (resp)"),
                    row=3, col=2)                                                                                       # Add marker for current response with TMD

        frames = []                                                                                                     # Initialize frames list

        # Índices de trazas que SÍ cambian 
        # 0: edificio SIN TMD
        # 1: edificio CON TMD
        # 3: marcador ag SIN TMD
        # 5: marcador ag CON TMD
        # 7: marcador resp SIN TMD
        # 9: marcador resp CON TMD
        dyn_traces = [0, 1, 3, 5, 7, 9]

        for k in range(len(tA)):
            base = ugA[k]                                                                                               # Base displacement at time k                   
            xk_sin = np.concatenate(([base], uA_sin[k, :n_use_sin]+base))                                               # for moving base
            xk_con = np.concatenate(([base], uA_con[k, :n_use_con]+base))                                               # for moving base

            frames.append(go.Frame(
                name=str(k),
                data=[

                    go.Scatter(x=xk_sin, y=y_nodes[:n_use_sin+1]),
                    go.Scatter(x=xk_con, y=y_nodes[:n_use_con+1]),
                    go.Scatter(x=[tA[k]], y=[agA[k]]),
                    go.Scatter(x=[tA[k]], y=[agA[k]]),
                    go.Scatter(x=[tA[k]], y=[respA_sin[k]]),
                    go.Scatter(x=[tA[k]], y=[respA_con[k]]),
                ],
                traces=dyn_traces
            ))                                                                                                          # Append frame to frames list

        fig.frames = frames                                                                                             # Assign frames to figure
        steps = [{
            "method": "animate",
            "args": [[str(k)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
            "label": f"{tA[k]:.2f}s"
        } for k in range(len(tA))]                                                                                      # Create slider steps for each frame                                        

        fig.update_layout(
            height=LayoutHPx,
            title="Compararison : WITHOUT TMD vs WITH TMD",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=70, b=70),
            updatemenus=[{
                "type": "buttons",
                "direction": "left",
                "x": 0.0, "y": -0.20,
                "buttons": [
                    {"label": "▶ Play", "method": "animate",
                    "args": [None, {"frame": {"duration": int(1000/fps), "redraw": True}, "fromcurrent": True}]},
                    {"label": "⏸ Pause", "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
                ]
            }],
            sliders=[{
                "x": 0.05, "y": -0.08,
                "len": 0.90,
                "currentvalue": {"prefix": "t = "},
                "steps": steps
            }]
        )                                                                                                               # Update layout with animation controls         

        max_ag = float(np.max(np.abs(agA)))                                                                             # Maximum ground acceleration for scaling
        dtick_ag = nice_dtick(max_ag, n_ticks=5)                                                                        # Set tick interval for ground acceleration
        ylim_ag = nice_ylim(max_ag, dtick_ag)                                                                           # Y-axis limit for ground acceleration

        max_resp = float(max(np.max(np.abs(respA_sin)), np.max(np.abs(respA_con))))                                     # Maximum response for scaling
        dtick_resp = nice_dtick(max_resp, n_ticks=5)                                                                    # Set tick interval for response
        ylim_resp = nice_ylim(max_resp, dtick_resp)                                                                     # Y-axis limit for response

        fig.update_xaxes(title_text=f"Displacement ({unit_u})",
                        range=[-xlim_common, xlim_common], row=1, col=1)                                                # Update x-axis for structure without TMD
        fig.update_xaxes(title_text=f"Displacement ({unit_u})",
                        range=[-xlim_common, xlim_common], row=1, col=2)                                                # Update x-axis for structure with TMD
        fig.update_yaxes(title_text="Altura (m)", row=1, col=1)                                                         # Update y-axis for structure without TMD
        fig.update_yaxes(title_text="Altura (m)", row=1, col=2)                                                         # Update y-axis for structure with TMD

        fig.update_yaxes(title_text="ag (g)", range=[-ylim_ag, ylim_ag], dtick=dtick_ag, row=2, col=1)                  # Update y-axis for ground acceleration without TMD
        fig.update_yaxes(title_text="ag (g)", range=[-ylim_ag, ylim_ag], dtick=dtick_ag, row=2, col=2)                  # Update y-axis for ground acceleration with TMD

        fig.update_yaxes(title_text=ylab_resp, range=[-ylim_resp, ylim_resp], dtick=dtick_resp, row=3, col=1)            # Update y-axis for response without TMD
        fig.update_yaxes(title_text=ylab_resp, range=[-ylim_resp, ylim_resp], dtick=dtick_resp, row=3, col=2)            # Update y-axis for response with TMD


        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})                # Display the comparison animation chart

        U_sin_use = U_sin[:, :n_pisos]                                                                                  # Use only structural DOFs without TMD                                
        U_con_use = U_con[:, :n_pisos]                                                                                  # Use only structural DOFs with TMD

        drift_sin, driftmax_sin, piso_sin, driftg_sin = modelo.derivas_maximas(U_sin_use, alturas)                      # Calculate maximum drifts without TMD
        drift_con, driftmax_con, piso_con, driftg_con = modelo.derivas_maximas(U_con_use, alturas)                      # Calculate maximum drifts with TMD

        pisos = np.arange(1, n_pisos + 1)                                                                               # Floor numbers
        altura_acum = np.cumsum(alturas)                                                                                # Cumulative heights

        df_drift = pd.DataFrame({
            "Piso": pisos,
            "Drift ratio without TMD (%)": driftmax_sin * 100,
            "Drift ratio with TMD (%)": driftmax_con * 100,
            "Height (m)": altura_acum})                                                                                 # Create DataFrame for drift ratios       

        df_long2 = df_drift.melt(
            id_vars=["Height (m)"],
            value_vars=["Drift ratio without TMD (%)", "Drift ratio with TMD (%)"],
            var_name="Case",
            value_name="Drift ratio (%)")                                                                               # Melt DataFrame for plotting                                
        
        df_zero = pd.DataFrame({
            "Height (m)": [0.0, 0.0],
            "Case": [
                "Drift ratio without TMD (%)",
                "Drift ratio with TMD (%)"
            ],
            "Drift ratio (%)": [0.0, 0.0]
        })                                                                                                              # Base point (0,0) for both cases
        df_long2 = pd.concat([df_zero, df_long2], ignore_index=True)                                                    # Concatenate base point to DataFrame

        st.markdown("""
        **Left:** Plot showing the Interstory drift ratio of the structure under the optimal TMD parameters.  

        **Right:** Plot showing the Maximum absolute floor accelerations under the same TMD parameter configuration.  

        *Note:* The tuned mass damper (TMD) is designed with a damping ratio of **5%**, ensuring effective reduction of structural vibrations while maintaining system stability.
        """)

        fig2 = px.line(df_long2, x="Drift ratio (%)", y="Height (m)", color="Case", markers=True, line_shape="linear")  # Create line plot for drift ratios
        fig2.update_layout(
            height=650,
            width=500,
            yaxis_title="Height (m)",
            yaxis=dict(range=[0, altura_acum[-1] * 1.05]),                                                              # Minimum 0, maximum slightly above the actual value
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top")                                                     # Legend on the right
        )

        accmax_sin = np.max(np.abs(Aabs_sin_g), axis=0)                                                                 # Maximum absolute acceleration without TMD in g
        accmax_con = np.max(np.abs(Aabs_con_g2), axis=0)                                                                # Maximum absolute acceleration with TMD in g 

        pisos = np.arange(1, n_pisos + 1)                                                                               # Floor numbers
        altura_acum = np.cumsum(alturas)                                                                                # Cumulative heights

        df_acc = pd.DataFrame({
            "Piso": pisos,
            "Absolute acceleration without TMD (g)": accmax_sin,
            "Absolute acceleration with TMD (g)": accmax_con,
            "Height (m)": altura_acum
        })                                                                                                              # Create DataFrame for absolute accelerations
        df_long_acc = df_acc.melt(
            id_vars=["Height (m)"],
            value_vars=[
                "Absolute acceleration without TMD (g)",
                "Absolute acceleration with TMD (g)"
            ],
            var_name="Case",
            value_name="Absolute acceleration (g)"
        )                                                                                                              # Melt DataFrame for plotting
        df_zero_acc = pd.DataFrame({
            "Height (m)": [0.0, 0.0],
            "Case": [
                "Absolute acceleration without TMD (g)",
                "Absolute acceleration with TMD (g)"
            ],
            "Absolute acceleration (g)": [max_ag_fft, max_ag_fft]
        })                                                                                                              # Base point (0, max ground acceleration) for both cases

        df_long_acc = pd.concat([df_zero_acc, df_long_acc], ignore_index=True)                                          # Concatenate base point to DataFrame

        fig_acc = px.line(
            df_long_acc,
            x="Absolute acceleration (g)",
            y="Height (m)",
            color="Case",
            markers=True,
            line_shape="linear"
        )                                                                                                               # Create line plot for absolute accelerations

        fig_acc.update_layout(
            height=650,
            width=500,
            xaxis_title="Maximum absolute acceleration (g)",
            yaxis_title="Height (m)",
            xaxis=dict(range=[0, df_long_acc["Absolute acceleration (g)"].max() * 1.05]),  
            yaxis=dict(range=[0, df_long_acc["Height (m)"].max() * 1.05]),               
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top")                       
        )                                                                                                               # Update layout for absolute acceleration plot
        col1, col2 = st.columns([1, 1])                                                                                 # Create two columns for side-by-side plots

        with col1:                                                                                                      # Column where will be displayed drift ratio plot
            st.plotly_chart(fig2, use_container_width=False)                                                            # Display the drift ratio plot       
        with col2:                                                                                                      # Column where will be displayed absolute acceleration plot
            st.plotly_chart(fig_acc, use_container_width=False)                                                         # Display the absolute acceleration plot

        










