import numpy as np

def get_transform(theta):
    '''
    Get the inverse of the stress transformation matrix
    '''
    c = np.cos(theta)
    s = np.sin(theta)

    Tinv = np.zeros((3,3))
    Tinv[0,0] = c**2
    Tinv[0,1] = s**2
    Tinv[0,2] = -2*s*c

    Tinv[1,0] = s**2
    Tinv[1,1] = c**2
    Tinv[1,2] = 2*s*c

    Tinv[2,0] = s*c
    Tinv[2,1] = -s*c
    Tinv[2,2] = c**2 - s**2

    return Tinv

def get_stiffness(E1, E2, nu12, G12):
    '''
    Given the engineernig constants E1, E2, nu12 and G12, compute the
    stiffness in the material reference frame.
    '''

    # Compute the stiffness matrix in the material coordinate frame
    Q = np.zeros((3,3))

    nu21 = nu12*E2/E1
    fact = 1.0/(1.0 - nu12*nu21)

    # Assign the values to Q
    Q[0,0] = fact*E1
    Q[0,1] = fact*E2*nu12

    Q[1,0] = fact*E1*nu21
    Q[1,1] = fact*E2

    Q[2,2] = G12

    return Q

def get_global_stiffness(E1, E2, nu12, G12, thetas):
    '''
    Compute the stiffness matrices for each of the given angles in the
    global coordinate frame.
    '''

    # Get the stiffness in the material frame
    Q = get_stiffness(E1, E2, nu12, G12)

    # Allocate the Cmat array of matrices
    Cmats = np.zeros((len(thetas), 3, 3))

    # Compute the transformed stiffness for each angle
    for i in xrange(len(thetas)):
        Tinv = get_transform(thetas[i])

        # Compute the Qbar matrix
        Cmats[i,:,:] = np.dot(Tinv, np.dot(Q, Tinv.T))

    return Cmats

def get_tsai_wu(Xt, Xc, Yt, Yc, S12):
    '''
    Given the failure properties, compute the Tsai--Wu coefficients
    assuming that there is no F12 interaction term.
    '''

    F1 = (Xc - Xt)/(Xt*Xc)
    F2 = (Yc - Yt)/(Yt*Yc)

    F11 = 1.0/(Xt*Xc)
    F22 = 1.0/(Yt*Yc)
    F12 = 0.0

    F66 = 1.0/(S12*S12)

    return F1, F2, F11, F22, F12, F66

def get_failure_coeffs(E1, E2, nu12, G12,
                       F1, F2, F11, F22, F12, F66, thetas):
    '''
    Given the stiffness matrix coefficients and the Tsai--Wu failure
    criterion, compute the h and G matrices in the global coordinate
    frame.
    '''

    # Get the stiffness in the material frame
    Q = get_stiffness(E1, E2, nu12, G12)

    # Allocate the arrays that will be used
    h = np.zeros((len(thetas), 3))
    G = np.zeros((len(thetas), 3, 3))

    h1 = np.array([F1, F2, 0.0])
    G1 = np.array([[F11, F12, 0.0],
                   [F12, F22, 0.0],
                   [0.0, 0.0, F66]])

    # Compute the
    hb = np.dot(Q, h1)
    Gb = np.dot(Q, np.dot(G1, Q))

    # Compute the transformed stiffness for each angle
    for i in xrange(len(thetas)):
        Tinv = get_transform(thetas[i])

        # Compute the coefficients in the global reference frame
        h[i, :] = np.dot(Tinv, hb)
        G[i,:,:] = np.dot(Tinv, np.dot(Gb, Tinv.T))

    return h, G

def get_isotropic(E, nu):
    '''
    Create a list of isotropic materials
    '''

    Cmats = np.zeros((len(E), 3, 3))

    # Set the constitutive matrix
    for k in xrange(len(E)):
        Cmats[k,0,0] = E[k]/(1.0 - nu[k]**2)
        Cmats[k,0,1] = nu[k]*E[k]/(1.0 - nu[k]**2)
        Cmats[k,1,0] = Cmats[k,0,1]
        Cmats[k,1,1] = Cmats[k,0,0]

        # Compute the shear stiffness
        Cmats[k,2,2] = 0.5*E[k]/(1.0 + nu[k])

    return Cmats

def get_von_mises(E, nu, smax):
    '''
    Get the coefficients for the von Mises failure criterion
    '''

    h = np.zeros((len(E), 3))
    G = np.zeros((len(E), 3, 3))

    # Iterate through all of the materials
    for k in xrange(len(E)):
        scale = (E[k]/(smax[k]*(1.0 - nu[k]**2)))**2
        G[k,0,0] = scale*(1.0 - nu[k] + nu[k]**2)
        G[k,0,1] = -scale*(0.5 - 2*nu[k] + 0.5*nu[k]**2)
        G[k,1,0] = G[k,0,1]
        G[k,1,1] = G[k,0,0]
        G[k,2,2] = scale*(3.0/4.0)*(1.0 - nu[k])**2

    return h, G
