import numpy as np

def Sauer_Approximation(MEE=None, Z_min=18, Z_max=86):
  # Approximate analytical 2-parameter formula (8) from Sauer, Sabin, Oddershede J Chem Phys 148, 174307 (2018)
  # This method works only for ions with max 14 bound electrons
  MEAN_EXCITATION_ENERGY_FUNCTION_D = (0, 0.00, 0.24, 0.34, 0.41, 0.45, 0.48, 0.50, 0.51, 0.52, 0.55, 0.57, 0.58, 0.59);
  MEAN_EXCITATION_ENERGY_FUNCTION_S_0 = (0, 0.30, 1.51, 2.32, 3.13, 3.90, 4.67, 5.44, 6.21, 6.97, 8.10, 9.08, 10.03, 10.94);
  MAX_NE = len(MEAN_EXCITATION_ENERGY_FUNCTION_D)

  HYDROGEN_MEAN_EXCITATION_ENERGY = 14.99

  if MEE is None:
    MEE2 = np.empty((Z_max+1, MAX_NE+1)) * np.nan
  else:
    MEE2 = np.array(MEE)


  for Z in range(Z_min, Z_max+1):
    for N in range(1, MAX_NE+1):
      D_N = MEAN_EXCITATION_ENERGY_FUNCTION_D[N-1]
      S_N0 = MEAN_EXCITATION_ENERGY_FUNCTION_S_0[N-1]

      A_N = (1-D_N) * (1-D_N)
      B_N = 2*(1-D_N) * (N * D_N - S_N0)
      C_N = (N *D_N - S_N0) * (N * D_N - S_N0)

      MEE2[Z,N] = HYDROGEN_MEAN_EXCITATION_ENERGY * (A_N*Z*Z + B_N*Z + C_N)

  return MEE2

