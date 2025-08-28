# -*- coding: utf-8 -*-
"""trigauss.ipynb

import numpy as np
from scipy import integrate

def trigauss(n, alpha, beta):

  # by Gaspare Da Fies and Marco Vianello,
  # 8 Nov 2011

  # computes the n+1 angles and weights of a trigonometric gaussian
  # quadrature formula on [alpha,beta], 0<beta-alpha<=pi

  # uses the routines chebyshev.m, gauss.m from
  # W. Gautschi repository (not available online)
  # we suggest to put the following statements
  # ab = zeros(N,2); sig = zeros(N+1,2*N);
  # at the beginning of the body of chebyshev.m to speed-up execution

  # input:
  # n: trigonometric degree of exactness
  # [alpha,beta]: angular interval, 0<beta-alpha<=pi

  # output:
  # tw: (n+1) x 2 array of (angles,weights)

  # the formula integrates the canonical trigonometric basis with accuracy
  # from about 10^(-15) (small omega) to about 10^(-13) (omega-->pi)
  # up to n=300

  def integrand(t, n, omega):
    return np.cos(2*n*np.arccos(np.sin(t/2)/np.sin(omega/2)))

  omega=(beta-alpha)/2       # half-length of the angular interval

  #modified Chebyshev moments by recurrence
  z = np.zeros(n+1)
  z[0] = 2 * omega
  z[n] = integrate.quad(integrand,-omega, \
                         omega, args=(n, omega), limit=5000)[0]

  temp = np.arange(2, 2*n, 2)  # 2,4,...,2n-2
  dl = 1/4 - 1/(4*(temp-1))
  dc = 1/2 - 1/(np.sin(omega/2)**2) - 1/(2*(temp**2-1))
  du = 1/4 + 1/(4*(temp+1))
  d = 4 * np.cos(omega/2)/np.sin(omega/2)/(temp**2 - 1)
  d[n-2] = d[n-2] - du[n-2] * z[n]
  z[1:n] = tridisolve(dl[1:n-1], dc[0:n-1], du[0:n-2], d[0:n-1])
  mom = np.zeros(2*n+2)
  mom[0:2*n+1:2] = z  # fill even indices (0,2,4,...) with z

  # normalization of the moments (monic polynomials)
  k = np.arange(3, len(mom)+1)
  mom[2:] *= np.exp((2-k)*np.log(2))

  # recurrence coeffs of the monic Chebyshev polynomials
  abm = np.zeros((2*n+1, 2))
  abm[:,1] = 0.25
  abm[0,1] = np.pi
  abm[1,1] = 0.5

  # recurrence coeffs for the monic OPS w.r.t. the weight function
  # w(x)=2*sin(omega/2)/sqrt(1-sin^2(omega/2)*x^2) by the
  # modified Chebyshev algorithm
  ab, normsq = chebyshev(n+1,mom,abm)

  # Gaussian formula for the weight function above
  xw = gauss(n+1,ab)

  # angles and weights for the trigonometric gaussian formula
  return  np.column_stack((2*np.arcsin(np.sin(omega/2)*xw[:,0]) + \
   (beta+alpha)/2, xw[:,1]))
  
#-------------------------------------------------------------------------------

def chebyshev(N, Mom, Abm=False):

#  Modified Chebyshev algorithm.

#  Given a weight function w encoded by its first 2n modified
#  moments, stored in the (row) vector mom, relative to monic
#  polynomials defined by the (2n-1)x2 array abm of their
#  recurrence coefficients, [ab,normsq]=CHEBYSHEV(n,mom,abm)
#  generates the array ab of the first n recurrence coefficients
#  of the orthogonal polynomials for the weight function w, and
#  the vector normsq of their squared norms. The n alpha-
#  coefficients are stored in the first column, the n beta-
#  coefficients in the second column, of the nx2 array ab. The
#  call [ab,normsq]=CHEBYSHEV(n,mom) does the same, but using the
#  classical Chebyshev algorithm. If n is larger than the sizes
#  of mom and abm warrant, then n is reduced accordingly.

  mom = np.array(Mom)
  normsq = []
  if N <= 0:
     raise ('N out of range')
  if N > len(mom)/2:
    N = len(mom)/2

  if type(Abm) == bool:
    abm = np.zeros((2,int(2*N-1)))
  else:
    abm = np.array(Abm)
  if len(abm[0]) == 2 and len(abm)!=2:
      abm = abm.T

  if N > (len(abm[0])+1)/2:
    N = (len(abm[0])+1)/2

  if N == 1:
    normsq = [mom[0]]

  N = int(N)
  ab =  np.array([[abm[0][0]+mom[1]/mom[0], mom[0]]]+[[0,0]]*(N-1))
  sig = np.zeros((N+1,2*N))
  sig[1] = mom[0:int(2*N)]

  #NOTE: abm is a matrix with two raws instead of a matrix with two
  # columns, therefore the indexes are reversed.

  for n in range(2,N+1):
    for m in range(n-1,(2*N-n+1)):
      sig[n][m] = sig[n-1][m+1]-(ab[n-2][0]-abm[0][m])*sig[n-1][m]-ab[n-2][1]*sig[n-2][m]+abm[1][m]*sig[n-1][m-1]
    ab[n-1,0] = abm[0][n-1] + sig[n][n]/sig[n][n-1] - sig[n-1][n-1]/sig[n-1][n-2]
    ab[n-1,1] = sig[n][n-1]/sig[n-1][n-2]

  for i in range(N):
    normsq.append(sig[i+1][i])


  return ab, np.array(normsq)

#-------------------------------------------------------------------------------

def tridisolve(a,b,c,d):

  #   TRIDISOLVE  Solve tridiagonal system of equations.
  # From Cleve Moler's Matlab suite
  # http://www.mathworks.it/moler/ncmfilelist.html
  #     x = TRIDISOLVE(a,b,c,d) solves the system of linear equations
  #     b(1)*x(1) + c(1)*x(2) = d(1),
  #     a(j-1)*x(j-1) + b(j)*x(j) + c(j)*x(j+1) = d(j), j = 2:n-1,
  #     a(n-1)*x(n-1) + b(n)*x(n) = d(n).
  #
  #   The algorithm does not use pivoting, so the results might
  #   be inaccurate if abs(b) is much smaller than abs(a)+abs(c).
  #   More robust, but slower, alternatives with pivoting are:
  #     x = T\d where T = diag(a,-1) + diag(b,0) + diag(c,1)
  #     x = S\d where S = spdiags([[a; 0] b [0; c]],[-1 0 1],n,n)

  x = d
  n = len(x)-1
  for j in range(0,n):
    mu = a[j]/b[j]
    b[j+1] = b[j+1] - mu*c[j]
    x[j+1] = x[j+1] - mu*x[j]

  x[n] = x[n]/b[n]
  for i in range(n)[::-1]:
    x[i] = (x[i]-c[i]*x[i+1])/b[i]

  return x

#-------------------------------------------------------------------------------

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
