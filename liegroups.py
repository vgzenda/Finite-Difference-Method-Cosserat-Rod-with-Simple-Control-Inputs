import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt


# ---------
def vee_SO3(W):
    """
    This function takes a 3x3 skew-symmetric matrix W and applies the vee operator
    to extract the corresponding vector w.
    
    Parameters:
    - W: 3x3 skew-symmetric numpy matrix
    Returns:
    - w: 3x1 numpy matrix (vector)
    """
    if W.shape == (3,3):
        p = W[2, 1]
        q = W[0, 2]
        r = W[1, 0]
        w = np.array([p, q, r])
        return w
    else: 
        raise "ERROR: wrong size"
    

def hat_SO3(w):
    """
    This function takes a 3x1 vector and generates a 3x3 skew-symmetric matrix.
    
    Parameters:
    - w: 3x1 numpy matrix (vector)
    Returns:
    - W: 3x3 skew-symmetric numpy matrix
    """
    w = w.ravel()
    if w.shape == (3,):
        w1 = w[0]
        w2 = w[1]
        w3 = w[2]

        W = np.array([[0, -w3, w2],
                      [w3, 0, -w1],
                      [-w2, w1, 0]])
        return W
    else:
        raise "ERROR: wrong size"

def hat_SE3(w):
    """
    This function applies the hat operator for SE(3) on the vector w,
    resulting in matrix W, an element of the Lie algebra.
    
    Parameters:
    - w: 6x1 numpy matrix (vector)
    Returns:
    - W: 4x4 se(3) lie-algebra numpy matrix
    """
    phi = w[:3]      # Extract the first 3 elements
    rho = w[3:6]    # Extract the last 3 elements

    Phi = hat_SO3(phi)

    # Construct the matrix W
    W = np.block([[Phi, rho.reshape(3, 1)],    # Add rho1 as a column vector
                  [np.zeros((1, 3)), 0]])       # Append the final row

    return W

def vee_SE3(W):
    """
    This function applies the vee operator for se(3) on the screw W,
    resulting in vector w, an element R6.
    
    Parameters:
    - W: 4x4 se(3) lie-algebra numpy matrix
    Returns:
    - w: 6x1 numpy matrix (vector)
    """
    W_so3 = W[:3,:3]
    w_so3 = vee_SO3(W_so3)
    t_R3 = W[:3,3]
    return np.concatenate([w_so3,t_R3])


# get SE3 inv
def SE3_inv(h):
    """
    This function computes the inverse of an SE3 Rigid transformation
    
    Parameters:
    - g: 4x4 SE(3) numpy matrix
    Returns:
    - g_inv: 4x4 SE(3) numpy matrix
    """
    if h.shape == (4,4):
        R = h[0:3,0:3]
        p = h[0:3,3]
        R = np.transpose(R)
        p = -R@p
        g1 = [R[0,0],R[0,1],R[0,2],p[0]]
        g2 = [R[1,0],R[1,1],R[1,2],p[1]]
        g3 = [R[2,0],R[2,1],R[2,2],p[2]]
        g4 = [0.0,0.0,0.0,1.0]
        return np.array([g1,g2,g3,g4])
    else: 
        raise "ERROR: wrong size"

# get rotation/translation components 
def get_components_SE3(g):
    """
    This function extracts the rotation and translation components of an SE3 matrix g
    
    Parameters:
    - g: 4x4 SE(3) numpy matrix
    Returns:
    - R: 3x3 SO(3) numpy matrix
    - p: 3x1 R3    numpy matrix
    """
    if g.shape == (4,4):
        R = g[0:3,0:3]
        p = g[0:3,3]
        return R, p
    else: 
        raise "ERROR: wrong size"
    

# make rigid motion 
def make_components(R,p):
    """
    This function concatinates the rotation and translation components 
    and returns a SE3 matrix g
    
    Parameters:
    - R: 3x3 SO(3) numpy matrix
    - p: 3x1 R3    numpy matrix
    Returns:
    - g: 4x4 SE(3) numpy matrix
    """
    p = p.ravel()
    if R.shape == (3,3) and p.shape == (3,):
        g1 = [R[0,0],R[0,1],R[0,2],p[0]]
        g2 = [R[1,0],R[1,1],R[1,2],p[1]]
        g3 = [R[2,0],R[2,1],R[2,2],p[2]]
        g4 = [0.0,0.0,0.0,1.0]
        return np.array([g1,g2,g3,g4])
    else:
        raise "ERROR: wrong size"
    
# so(3) exponential map       
def exp_SO3(xi_so):
    """
    Computes the exponential map for rotations SO3
    
    Parameters:
    - xi_so: 3x1 numpy matrix
    Returns:
    - g: 3x3 SO(3) numpy matrix
    """
    xi_so = xi_so.ravel()
    if xi_so.shape == (3,):
        theta = np.linalg.norm(xi_so)
        if theta < 1e-8:
            return np.eye(3)
        else:
            return np.eye(3) + (np.sin(theta)/theta)*hat_SO3(xi_so) + ((1.0-np.cos(theta))/(theta**2))*hat_SO3(xi_so)@hat_SO3(xi_so)
    else: 
        raise "ERROR: wrong size!"
        
    

# se(3) exponential map
def exp_SE3(xi_se):
    """
    Computes the exponential map for rigid-transformations SE3
    
    Parameters:
    - xi_se: 6x1 numpy matrix
    Returns:
    - g: 4x4 SE(3) numpy matrix
    """
    xi_se = xi_se.ravel()
    if xi_se.shape == (6,):
        omega = xi_se[0:3]
        v = xi_se[3:6]
        R = exp_SO3(omega)
        om = np.linalg.norm(omega)
        if om < 1e-10:
            return np.array([[1.0,0.0,0.0,v[0]],[0.0,1.0,0.0,v[1]],[0.0,0.0,1.0,v[2]],[0.0,0.0,0.0,1.0]])
        else:
            Pmat = np.eye(3) + ((1.0-np.cos(om))/om**2)*hat_SO3(omega) + ((om - np.sin(om))/om**3)*hat_SO3(omega)@hat_SO3(omega)
            p = Pmat@v
            g1 = [R[0,0],R[0,1],R[0,2],p[0]]
            g2 = [R[1,0],R[1,1],R[1,2],p[1]]
            g3 = [R[2,0],R[2,1],R[2,2],p[2]]
            g4 = [0.0,0.0,0.0,1.0]
            return np.array([g1,g2,g3,g4])
    else:
        raise "ERROR: wrong size"
    

def get_components_SE3(g):
    """
    Extracts rotation matrix R and translation vector p from 4x4 SE(3) matrix.
    """
    R = g[:3, :3]
    p = g[:3, 3]
    return R, p

def make_components(R, p):
    """
    Builds a 4x4 SE(3) matrix from rotation matrix R and translation vector p.
    """
    p = p.ravel()
    g = np.block([
        [R, p.reshape(3, 1)],
        [np.zeros((1, 3)), np.ones((1, 1))]
    ])
    return g


# Adjoint map
def Adjoint_SE3(g):
    """
    This function computes the Adjoint map for SE3
    
    Parameters:
    - g: 4x4 SE3 numpy matrix
    Returns:
    - Ad_g:  6x6 numpy matrix
    """
    if g.shape == (4,4):
        R = g[0:3,0:3]
        p = g[0:3,3]
        return np.block([[R,np.zeros((3,3))],[hat_SO3(p)@R,R]])
    else:
        raise ValueError("wrong size")
        
# coAdjoint map
def coAdjoint_SE3(g):
    """
    This function computes the coAdjoint map for SE3
    
    Parameters:
    - g: 4x4 SE3 numpy matrix
    Returns:
    - Ad_g_star:  6x6 numpy matrix
    """
    if g.shape == (4,4):
        return np.transpose(Adjoint_SE3(g))
    else:
        raise ValueError("wrong size")

# adjoint map
def adjoint_SE3(x):
    """
    This function computes the adjoint map for se3
    
    Parameters:
    - x: 6x1 SE3 numpy matrix
    Returns:
    - ad_x:  6x6 numpy matrix
    """
    x = x.ravel()
    if x.shape == (6,):
        Omega = x[0:3]
        v = x[3:6]
        return np.block([[hat_SO3(Omega),np.zeros((3,3))],[hat_SO3(v),hat_SO3(Omega)]])
    else: 
        raise ValueError("wrong size")
        
# coadjoint map
def coadjoint_SE3(x):
    """
    This function computes the coadjoint map for se3
    
    Parameters:
    - x: 6x1 SE3 numpy matrix
    Returns:
    - ad_x_star:  6x6 numpy matrix
    """
    x = x.ravel()
    if x.shape == (6,):
        return np.transpose(adjoint_SE3(x))
    else: 
        raise ValueError("wrong size")