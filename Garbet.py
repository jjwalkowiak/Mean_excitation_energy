import numpy as np
import atomic_data
import scipy.integrate as integrate
import math


def MEE_Garbet_v2(Z, N, A, lambda_Z, unit=27.211):
    """
    Z - atomic number
    N - number of bound electrons
    A - list of A coefficients
    lambda_Z - list of lambda coefficients
    """
    if type(A) is not type(np.array([1])):
        A = np.atleast_1d(A)
    if type(lambda_Z) is not type(np.array([1])):
        lambda_Z = np.atleast_1d(lambda_Z)
    if A.size != lambda_Z.size:
        print('ERROR: A and lambdaZ must have the same dimension')
        print(f'A: {A}\tlambda: {lambda_Z}')
        return 0
    s1 = 0
    for i in range(A.size):
        s1 += A[i]*lambda_Z[i]**-2
    s2 = 0
    for i in range(A.size):
        s2 += A[i]*lambda_Z[i]
    # This is direct implementation of the last term as in the Yves paper, below is shorter version
    # s3 = 0
    # for i in range(A.size):
    #     s3 += 1/2 * A[i]**2*lambda_Z[i]
    # s4 = 0
    # for i in range(A.size):
    #     for j in range(A.size):
    #         if i != j:
    #             s4 += A[i]*A[j] * lambda_Z[i]**2 / (lambda_Z[i] + lambda_Z[j] )
    # I = unit / np.sqrt(6*s1) * np.sqrt((Z-N) * s2 + N * (s3+s4))
    s3 = 0
    for i in range(A.size):
        for j in range(A.size):
            s3 += A[i]*A[j] * lambda_Z[i]**2 / (lambda_Z[i] + lambda_Z[j] )
    I = unit / np.sqrt(6*s1) * np.sqrt((Z-N) * s2 + N * s3)
    return I


def tungsten_MEE_Garbet_v2(log=False, verbose=False):
    # initailize list
    MEE = []
    # prescribe Z for tungsten
    Z = 74
    # Calculate MEE for ions
    for N in range(1, Z+1):
        list_A, list_lambda = atomic_data.get_tungsten_parameters(N)
        MEE.append(MEE_Garbet_v2(Z, N, list_A, list_lambda))
        if verbose: print(N, MEE[-1])

    if log:
        return np.log(MEE)
    else:
        return MEE


def PT_opt_parameters(Z, N):
  # PT Model parameters
    P = np.array([[1.1831,	0.1738,	0.0913,	0.0182,	0.7702],
        [0.8368,	1.0987,	0.9642,	1.2535,	0.2618],
        [0.3841,	0.6170,	1.0000,	1.0000,	1.0000],
        [0.5883,	0.0461,	1.0000,	1.0000,	1.0000]])

  # Calculate lambda as in https://doi.org/10.1063/5.0075859
    lambda_Z =  np.array([ P[0,0]*Z**P[1,0], P[0,1]*Z**P[1,1], P[0,2]*Z**P[1,2], P[0,3]*Z**P[1,3], P[0,4]*Z**P[1,4]])
    n_Z =       np.array([ P[2,0]*Z**P[3,0], P[2,1]*Z**P[3,1], P[2,2]*Z**P[3,2], P[2,3]*Z**P[3,3], P[2,4]*Z**P[3,4]])

    q = (Z-N) / Z
    lambda_i = lambda_Z * ( (1 - q**(n_Z+1)) / (1-q) ) ** (1/2)

    A_i = [min(N, 2), min(max(N - 2, 0), 8), min(max(N - 10, 0), 18), min(max(N - 28, 0), 26), max(N - 54, 0)]
    A_i = [x/N for x in A_i]

    parameters = zip(A_i, lambda_i)
    parameters = [x for x in parameters if x[0] != 0]

    return zip(*parameters)


def MEE_Garbet_PT(Z, log=False, verbose=False):
    """
    Use Yves method with PT_opt
    """
    # initailize list
    MEE = []

    # Calculate MEE for ions
    for N in range(1, Z+1):
        list_A, list_lambda = PT_opt_parameters(Z, N)
        MEE.append(MEE_Garbet_v2(Z, N, list_A, list_lambda))
        if verbose: print(N, MEE[-1])

    if log:
        return np.log(MEE)
    else:
        return MEE


def argon_MEE_Garbet_v2(log=False, verbose=False):
    # initailize list
    MEE = []
    # prescribe Z for tungsten
    Z = 18
    # Calculate MEE for ions
    for N in range(1, Z+1):
        list_A, list_lambda = atomic_data.get_argon_parameters(N)
        MEE.append(MEE_Garbet_v2(Z, N, list_A, list_lambda))
        if verbose: print(N, MEE[-1])

    if log:
        return np.log(MEE)
    else:
        return MEE


def PT_opt_denisty(Z, N):
  # PT Model parameters
    P = np.array([[1.1831,	0.1738,	0.0913,	0.0182,	0.7702],
        [0.8368,	1.0987,	0.9642,	1.2535,	0.2618],
        [0.3841,	0.6170,	1.0000,	1.0000,	1.0000],
        [0.5883,	0.0461,	1.0000,	1.0000,	1.0000]])

  # Create empty array for MEE and scale for electron denisty
    R = np.logspace(-4, 1, 10000)

  # Calculate electron denisty as in https://doi.org/10.1063/5.0075859
    lambda_Z =  np.array([ P[0,0]*Z**P[1,0], P[0,1]*Z**P[1,1], P[0,2]*Z**P[1,2], P[0,3]*Z**P[1,3], P[0,4]*Z**P[1,4]])
    n_Z =       np.array([ P[2,0]*Z**P[3,0], P[2,1]*Z**P[3,1], P[2,2]*Z**P[3,2], P[2,3]*Z**P[3,3], P[2,4]*Z**P[3,4]])

    q = (Z-N) / Z
    lambda_i = lambda_Z * ( (1 - q**(n_Z+1)) / (1-q) ) ** (1/2)

    e_density = (lambda_i[0] ** 2 * min(N, 2) * np.exp(-lambda_i[0] * R))   \
                + (lambda_i[1] ** 2 * min( max(N-2,0) ,8) * np.exp(-lambda_i[1] * R))   \
                + (lambda_i[2] ** 2 * min( max(N-10,0) ,18) * np.exp(-lambda_i[2] * R)) \
                + (lambda_i[3] ** 2 * min( max(N-28,0) ,26) * np.exp(-lambda_i[3] * R)) \
                + (lambda_i[4] ** 2 * max(N-54,0) * np.exp(-lambda_i[4] * R))

    e_density = 1 / (4*math.pi*R) * e_density

    return R, e_density


def TF_kinetic_energy(r, density):
    # https://en.wikipedia.org/wiki/Thomas%E2%80%93Fermi_model
    Ck = 3/10 * (3*np.pi**2)**(2/3) # In atomic units
    K = Ck * 4*np.pi* integrate.trapezoid(density**(5/3) * r**2, r)
    return K


def MEE_Garbet_denisty(Z, rho_dft=False, log=False, unit=27.211 ):
    MEE = []

    for N in range(1, Z+1):
        if rho_dft:
            r, density = dft.load_dft_data(Z, N)
        else:
            r, density = PT_opt_denisty(Z, N)

        Ke = TF_kinetic_energy(r, density)
        K_avg = Ke / N

        density_momentum = 4*np.pi/N * integrate.trapezoid(r**4 * density, r)
        MEE.append( np.sqrt(2*K_avg/density_momentum) * unit)

    if log:
        return np.log(MEE)
    else:
        return MEE


if __name__ == '__main__':
    # tungsten_MEE_Garbet_v2(verbose=True)
    MEE_Garbet_PT(74, verbose=True)

