import numpy as np                                                                      # Import numpy for numerical operations
from numba import njit                                                                  # Import njit from numba for JIT compilation

@njit                                                                                   # Function to compute dynamic properties with TMD
def Propiedades_dinamicas_conTMD_nb(M, K, m, k):

    dim = len(K)                                                                        # Dimension of the original system                              
    K_TMD = np.zeros((dim+1, dim+1))                                                    # Initialize stiffness matrix with TMD        
    M_TMD = np.zeros((dim+1, dim+1))                                                    # Initialize mass matrix with TMD                                   

    M_TMD[:dim, :dim] = M                                                               # Fill original mass matrix      
    M_TMD[-1, -1] = m                                                                   # Add TMD mass                                     

    K_sub = np.array([[k, -k],                                                          # Stiffness submatrix for TMD
                      [-k, k]])                                                            
    K_TMD[0:dim, 0:dim] = K                                                             # Fill original stiffness matrix                                   
    K_TMD[-2:, -2:] += K_sub                                                            # Add TMD stiffness submatrix                  

    M_TMD_inv = np.linalg.inv(M_TMD)                                                    # Inverse of mass matrix with TMD                            
    A_TMD = M_TMD_inv @ K_TMD                                                           # System matrix with TMD                        
    w2_TMD, phi_TMD = np.linalg.eig(A_TMD)                                              # Eigenvalues and eigenvectors with TMD

    idx_TMD = np.argsort(w2_TMD)                                                        # Indices to sort eigenvalues                        
    w2_ord_TMD = w2_TMD[idx_TMD]                                                        # Sorted eigenvalues                         
    phi_ord_TMD = phi_TMD[:, idx_TMD]                                                   # Sorted eigenvectors             

    w_TMD = np.sqrt(w2_ord_TMD)                                                         # Natural frequencies with TMD
    T_TMD = 2*np.pi / w_TMD                                                             # Periods with TMD                           

    ntmd = M_TMD.shape[1]                                                               # Number of DOF with TMD                         
    rtmd = np.ones((ntmd, 1))                                                           # Vector of ones for participation factors

    Mt = phi_ord_TMD.T @ M_TMD  @ phi_ord_TMD                                           # Modal mass matrix with TMD                        
    den = np.diag(Mt @ phi_ord_TMD)                                                     # Denominator for participation factors                           
    gamma_tmd = (Mt @ rtmd).flatten() / den                                             # Participation factors with TMD

    Mei_tmd = (gamma_tmd**2) * den                                                      # Modal participative masses with TMD   
    Mei_TMD_pct = 100 * Mei_tmd / np.sum(Mei_tmd)                                       # Modal participative masses percentage with TMD

    return phi_ord_TMD, w_TMD, T_TMD, M_TMD, K_TMD, gamma_tmd, Mei_TMD_pct              # Return dynamic properties with TMD

@njit                                                                                   # Function to compute damping matrix with TMD
def Matriz_amortiguamieto_conTMD_nb(C, zeta_TMD, k, m):

    dim = len(C)                                                                        # Dimension of the original system
    dim_tmd = dim + 1                                                                   # Dimension with TMD

    C_TMD = np.zeros((dim_tmd, dim_tmd))                                                # Initialize damping matrix with TMD
    C_TMD[0:dim, 0:dim] = C                                                             # Fill original damping matrix  

    wtmd = np.sqrt(k/m)                                                                 # Natural frequency of TMD
    c_tmd = 2*zeta_TMD*m*wtmd                                                           # Damping coefficient of TMD

    C_TMD[-2:, -2:] += np.array([[ c_tmd, -c_tmd],                                      # Damping submatrix for TMD
                                 [-c_tmd,  c_tmd]])                                     
    return C_TMD                                                                        # Return damping matrix with TMD

@njit                                                                                   # Function to compute time history response
def Respuesta_tiempo_historiaV2_nb(M, K, C, REG_SISM, gamma, beta):

    n = len(M)                                                                          # Number of DOF
    NPTS = REG_SISM.shape[0]                                                            # Number of time points

    Respuesta_Acc = np.zeros((n, NPTS))                                                 # Initialize acceleration response
    Respuesta_Vel = np.zeros((n, NPTS))                                                 # Initialize velocity response            
    Respuesta_Des = np.zeros((n, NPTS))                                                 # Initialize displacement response

    dt = REG_SISM[1, 0] - REG_SISM[0, 0]                                                # Time step size
    V_activacion = np.ones((n, 1))                                                      # Activation vector

    g = 9.8067                                                                          # Acceleration due to gravity (m/s^2)
    MV = M @ V_activacion                                                               # Mass vector                                                                

    a1 = (1.0/(beta*dt**2))*M + (gamma/(beta*dt))*C                                     # Coefficient a1
    a2 = (1.0/(beta*dt))*M + (gamma/beta - 1.0)*C                                       # Coefficient a2
    a3 = (1.0/(2.0*beta) - 1.0)*M + dt*(gamma/(2.0*beta) - 1.0)*C                       # Coefficient a3

    K_TECHO = K + a1                                                                    # Effective stiffness matrix
    for i in range(NPTS - 1):                                                           # Time-stepping loop
        
        des_i  = Respuesta_Des[:,  i:i+1]                                               # Displacement at time step i    
        vel_i  = Respuesta_Vel[:,  i:i+1]                                               # Velocity at time step i
        acc_i  = Respuesta_Acc[:,  i:i+1]                                               # Acceleration at time step i

        p_techo_iter = -g * REG_SISM[i+1, 1] * MV + a1 @ des_i + a2 @ vel_i + a3 @ acc_i    # Effective load vector

        des_ip1 = np.linalg.solve(K_TECHO, p_techo_iter)                                # Displacement at time step i+1
        Respuesta_Des[:, i+1:i+2] = des_ip1                                             # Store displacement
                                                                                        # Update velocity and acceleration
        Respuesta_Vel[:, i+1:i+2] = (gamma/(beta*dt))*(des_ip1 - des_i) \
                                    + (1.0 - gamma/beta)*vel_i \
                                    + dt*(1.0 - gamma/(2.0*beta))*acc_i

        Respuesta_Acc[:, i+1:i+2] = (1.0/(beta*dt**2))*(des_ip1 - des_i) \
                                    - (1.0/(beta*dt))*vel_i \
                                    - (1.0/(2.0*beta) - 1.0)*acc_i
    return Respuesta_Des, Respuesta_Vel, Respuesta_Acc                                  # Return responses

@njit                                                                                   # Function to optimize TMD parameters
def optimizacion_TMD_nb(M, C, K, zetas, ws, ms, REG_SISM_fft, gammaBN, betaBN,alturas):

    nzetas = len(zetas)                                                                 # Number of damping ratios
    nws = len(ws)                                                                       # Number of TMD frequencies
    nms = len(ms)                                                                       # Number of TMD masses
    gdl_tmd = len(M) + 1                                                                # DOF with TMD

    deriva_matrix = np.eye(gdl_tmd)                                                     # Initialize derivative matrix
    for ii in range(gdl_tmd - 1):                                                       # Fill derivative matrix
        deriva_matrix[ii+1, ii] = -1.0                                                  # Derivative relation

    R = np.zeros((nms, nws, nzetas))                                                    # Initialize response matrix                                
                                                                                        # Variables to track the best parameters
    best_val = 1e300                                                                    # Best response value                                  
    best_i = 0                                                                          # Best mass index                               
    best_z = 0                                                                          # Best frequency index
    best_j = 0                                                                          # Best damping ratio index
    #h = np.asarray(alturas, float).reshape(-1, 1)
    for i in range(nms):                                                                # Loop over mass values
        for j in range(nzetas):                                                         # Loop over damping ratios                         
            zeta_TMD = zetas[j]                                                         # Current damping ratio         
            for z in range(nws):                                                        # Loop over frequency values                       
                k = ms[i] * ws[z]**2                                                    # Stiffness of TMD                   
                _, _, _, M_TMD, K_TMD, _, _ = Propiedades_dinamicas_conTMD_nb(M, K, ms[i], k)                                   # Dynamic properties with TMD
                C_TMD = Matriz_amortiguamieto_conTMD_nb(C, zeta_TMD, k, ms[i])                                                  # Damping matrix with TMD

                Respuesta_Des_TMD, _, _ = Respuesta_tiempo_historiaV2_nb(M_TMD, K_TMD, C_TMD, REG_SISM_fft, gammaBN, betaBN)    # Time history response with TMD

                deriva = deriva_matrix @ Respuesta_Des_TMD                               # Compute derivative of response
                deriva=deriva[0:gdl_tmd-1]/alturas.reshape(-1, 1)                        # Normalize by heights
                val = np.max(np.abs(deriva))                                             # Maximum normalized response                        
                R[i, z, j] = val                                                         # Store response value                            

                # actualizar mínimo
                if val < best_val:                                                      # Check for new best value                             
                    best_val = val                                                      # Update best response value                   
                    best_i = i                                                          # Update best mass index                      
                    best_z = z                                                          # Update best frequency index
                    best_j = j                                                          # Update best damping ratio index                         

    Location_opt = np.array([best_i, best_z, best_j], dtype=np.int64)                   # Optimal parameter indices
    return R, Location_opt                                                              # Return response matrix and optimal indices                       