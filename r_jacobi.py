# -*- coding: utf-8 -*-
"""r_jacobi.ipynb

import numpy as np
from math import gamma

def r_jacobi(N,a=0,b=[]):

#  Recurrence coefficients for monic Jacobi polynomials.
#
#    ab=R_JACOBI(n,a,b) generates the first n recurrence
#    coefficients for monic Jacobi polynomials with parameters
#    a and b. These are orthogonal on [-1,1] relative to the
#    weight function w(t)=(1-t)^a(1+t)^b. The n alpha-coefficients
#    are stored in the first column, the n beta-coefficients in
#    the second column, of the nx2 array ab. The call ab=
#    R_JACOBI(n,a) is the same as ab=R_JACOBI(n,a,a) and
#    ab=R_JACOBI(n) the same as ab=R_JACOBI(n,0,0).
#
#    Supplied by Dirk Laurie, 6-22-1998; edited by Walter
#    Gautschi, 4-4-2002.

  if b == []:
    b = a
  if N<=0 or a<=-1 or b<=-1:
    raise ValueError('parameter(s) out of range')
  nu = (b-a)/(a+b+2)
  mu = (2**(a+b+1))*gamma(a+1)*gamma(b+1)/gamma(a+b+2)

  # Important note: scipy.special.gamma() takes arbitrary np.arrays as input.
  # math.gamma() requires float or single-element numpy arrays, which a
  # crippling limitation in many use cases. –
  # The function scipy.special.gamma allows complex numbers, even though
  # math.gamma doesn't.

  if N == 1:
    return np.array([nu,mu])
  N0 = int(N)
  n = np.array(range(1,N0))
  nab = 2*n+a+b
  A = np.append((b**2-a**2)*np.ones(N0-1)/(nab*(nab+2)),nu)
  n = n[1:]
  nab = nab[1:]
  B1 = 4*(a+1)*(b+1)/((a+b+2)**2*(a+b+3))
  B = 4*(n+a)*(n+b)*n*(n+a+b)/((nab**2)*(nab+1)*(nab-1))
  ab = [mu] + [B1]
  for i in B:
    ab = ab + [i]
  ab = [A] + [ab]


  return   np.transpose(ab)
