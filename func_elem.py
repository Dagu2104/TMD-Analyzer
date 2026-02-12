import numpy as np
def basic_elem(L,I,A,kx,E,nu,elem_modifier):
    """
    # Prismatic element properties (3 DOF)
    # elem_modifiers: (Am, Sm, Mm)
    # Am = Axial area modifier
    # Sm = Shear area modifier
    # Mm = Moment of inertia modifier
    # kx = Element shape factor
    # L  = Node-to-node element length
    # I  = Element moment of inertia
    # A  = Element cross-sectional area
    # E  = Material Young's modulus
    # nu = Material Poisson's ratio
    """
    Am,Sm,Mm=elem_modifier                                                                                      # Unpack modifiers  
    A_EFE=A*Am                                                                                                  # Effective Area
    I_EFE=I*Mm                                                                                                  # Effective Inertia
    G=E/(2*(1+nu))                                                                                              # Shear Modulus
    As=A*kx                                                                                                     # Shear Area
    phi=(12*E*I_EFE/(G*As*L**2))*Sm                                                                             # Shear Deform. Factor

    k11=(4+phi)*E*I_EFE/(L*(1+phi))                                                                             # Flexural Stiffness
    k12=(2-phi)*E*I_EFE/(L*(1+phi))                                                                             # Coupling Stiffness
    k33=E*A_EFE/L                                                                                               # Axial Stiffness   

    k_basic=np.array([[k11,k12, 0 ],                                                                            # Assembly of Basic Stiffness Matrix
                      [k12,k11, 0 ],
                      [0  , 0 ,k33]])
    
    return k_basic                                                                                              # Return Basic Stiffness Matrix

def local_elem(L,I,A,kx,E,nu,elem_modifier):                                                                    # Function to compute Local Stiffness Matrix
    """
    # Prismatic element properties (3 DOF)
    # elem_modifiers: (Am, Sm, Mm)
    # Am = Axial area modifier
    # Sm = Shear area modifier
    # Mm = Moment of inertia modifier
    # kx = Element shape factor
    # L  = Node-to-node element length
    # I  = Element moment of inertia
    # A  = Element cross-sectional area
    # E  = Material Young's modulus
    # nu = Material Poisson's ratio 
    """
    k_basic=basic_elem(L,I,A,kx,E,nu,elem_modifier)                                                              # Get Basic Stiffness Matrix
    T=np.array([[0, 1/L,  1, 0, -1/L, 0],                                                                        # Transformation Matrix
                [0, 1/L,  0, 0, -1/L, 1],
                [-1,  0,  0, 1,  0,   0]])
    k_local=(T.T)@k_basic@T                                                                                     # Stiffness Matrix in local coordinates
    return k_local                                                                                              # Return Local Stiffness Matrix
                           


def local_rot_elem_normal(L,theta,I,A,kx,E,nu,elem_modifier):                                                   # Function to compute Local Stiffness Matrix with Rotation
    """
    Properties of prismatic element with 6DOF
    elem_modifiers: (Am,Sm,Mm)
        Am = axial area modifier
        Sm = shear area modifier
        Mm = moment of inertia modifier
    tag=0 if Insertion Point is on the axis; 1 if it is on the top face
    C1=distance from rigid node to flexible element (left)
    C2=distance from rigid node to flexible element (right)
    rezf=rigid end zone factor (0 to 1)
    kx= element shape factor
    Lt = length from node to node of the element
    L= free length of the element
    I = moment of inertia of the element
    A = cross-sectional area of the element
    E = modulus of elasticity of the material
    nu= Poisson's ratio of the material  
    """
    k_local_rez=local_elem(L,I,A,kx,E,nu,elem_modifier)                                                         # Get Local Stiffness Matrix
    c=np.cos(theta)                                                                                             # Used to compute rotation matrix
    s=np.sin(theta)                                                                                             # Used to compute rotation matrix

    T_rot=np.array([[ c,s,0, 0,0,0],                                                                            # Rotation Transformation Matrix        
                    [-s,c,0, 0,0,0],
                    [ 0,0,1, 0,0,0],
                    [ 0,0,0, c,s,0],
                    [ 0,0,0,-s,c,0],
                    [ 0,0,0, 0,0,1]
                    ])
    k_rot=(T_rot.T)@k_local_rez@T_rot                                                                           # Stiffness Matrix in rotated coordinates
    return k_rot                                                                                                # Return Local Stiffness Matrix with Rotation