import numpy as np
import scipy.integrate as integrate
import math


def mod(Z, N):
    return math.exp(1 - (N / Z)) - 0.84


def org(Z, N):
    return np.log(np.sqrt(2))


# Generate MEE from modified LPA based on Walkowiak version of PT e-denisty
def MEE_LPA(Z, func):
    # PT Model parameters
    P = np.array([[1.1831,	0.1738,	0.0913,	0.0182,	0.7702],
        [0.8368,	1.0987,	0.9642,	1.2535,	0.2618],
        [0.3841,	0.6170,	1.0000,	1.0000,	1.0000],
        [0.5883,	0.0461,	1.0000,	1.0000,	1.0000]])

    # Create empty array for MEE and scale for electron denisty
    lnI = np.empty(Z+1) * np.nan
    R = np.logspace(-4, 1, 10000)

    # Calculate electron denisty as in https://doi.org/10.1063/5.0075859
    lambda_Z =  np.array([ P[0,0]*Z**P[1,0], P[0,1]*Z**P[1,1], P[0,2]*Z**P[1,2], P[0,3]*Z**P[1,3], P[0,4]*Z**P[1,4]])
    n_Z =       np.array([ P[2,0]*Z**P[3,0], P[2,1]*Z**P[3,1], P[2,2]*Z**P[3,2], P[2,3]*Z**P[3,3], P[2,4]*Z**P[3,4]])
    for N in range(1, Z+1):
        q = (Z-N) / Z
        lambda_i = lambda_Z * ( (1 - q**(n_Z+1)) / (1-q) ) ** (1/2)

        e_density = (lambda_i[0] ** 2 * min(N, 2) * np.exp(-lambda_i[0] * R))   \
                    + (lambda_i[1] ** 2 * min( max(N-2,0) ,8) * np.exp(-lambda_i[1] * R))   \
                    + (lambda_i[2] ** 2 * min( max(N-10,0) ,18) * np.exp(-lambda_i[2] * R)) \
                    + (lambda_i[3] ** 2 * min( max(N-28,0) ,26) * np.exp(-lambda_i[3] * R)) \
                    + (lambda_i[4] ** 2 * max(N-54,0) * np.exp(-lambda_i[4] * R))

        e_density = 1 / (4*math.pi*R) * e_density

    # Calculate MEE from modified LPA
        local_frequency = np.log(np.sqrt(4*math.pi*e_density), out=np.zeros_like(R), where=e_density!=0)

        lnI_au = 4 * math.pi / N * integrate.trapezoid(R**2 * e_density * local_frequency, R) \
                 + func(Z, N)
        lnI[N] = lnI_au + math.log(27.211)

    MEE = np.exp(lnI)

    return MEE