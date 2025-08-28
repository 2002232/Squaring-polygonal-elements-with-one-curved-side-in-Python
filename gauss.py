# -*- coding: utf-8 -*-
"""gauss.ipynb


import numpy as np

def gauss(N, albet):

#    Gauss quadrature rule.

#    Given a weight function w encoded by the nx2 array albet of the
#    first n recurrence coefficients for the associated orthogonal
#    polynomials, the first column of ab containing the n alpha-
#    coefficients and the second column the n beta-coefficients,
#    the call xw=GAUSS(n,ab) generates the nodes and weights xw of
#    the n-point Gauss quadrature rule for the weight function w.
#    The nodes, in increasing order, are stored in the first
#    column, the n corresponding weights in the second column, of
#    the nx2 array xw.

  ab = np.array(albet)
  if len(ab[:,0]) < N:
      raise ValueError('Input array ab is too short')
  J = np.zeros((N, N))

  for i in range(N):
      J[i][i] = ab[i][0]
  for j in range(N-1):
      J[j+1][j] = np.sqrt(ab[j+1][1])
      J[j][j+1] = J[j+1][j]

  D, V = np.linalg.eigh(J)
  I = np.argsort(D)
  D = np.sort(D)
  V = V[:,I]

  return np.transpose((D , ab[0][1]*V[0,:]**2))
