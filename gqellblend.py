# -*- coding: utf-8 -*-
"""gqellblend.ipynb"""

import numpy as np
from scipy import integrate
from math import gamma

def gqellblend(n, A, B, C, alpha, beta):

  # by Gaspare Da Fies, Alvise Sommariva and Marco Vianello

  # 2 June 2011

  # computes the nodes and weights of a product gaussian formula
  # exact on total-degree bivariate polynomials of degree <=n
  # on the planar region R obtained by linear blending (convex combination)
  # of two trigonometric arcs with parametric equations
  # P(theta)=A1*cos(theta)+B1*sin(theta)+C1
  # Q(theta)=A2*cos(theta)+B2*sin(theta)+C2
  # namely
  # R = {(x,y)=t*P(theta)+(1-t)*Q(theta), t in [0,1], theta in [alpha,beta],
  # 0<beta-alpha<=2*pi}

  # uses the routines:
  #
  # r_jacobi.m, gauss.m from
  # www.cs.purdue.edu/archives/2002/wxg/codes/OPQ.html
  #
  # trigauss.m
  # http://www.math.unipd.it/~marcov/mysoft/trigauss.m
  # this will be soon substituted by an optimized version input

  # input:
  # n: algebraic degree of exactness
  # A,B,C: 2x2 matrices of the parametric arc coefficients:
  # [alpha,beta]: angular interval, 0<beta-alpha<=2*pi

  # output:
  # xyw: 3 columns array of (xnodes,ynodes,weights)

  S1 = abs((A[0][0]-A[1][0])*(B[0][1]-B[1][1])+(A[0][1]-A[1][1])*\
   (B[1][0]-B[0][0]))>10*np.finfo(float).eps
  S2 = abs((C[0][0]-C[1][0])*(B[0][1]-B[1][1])+(C[0][1]-C[1][1])*\
   (B[1][0]-B[0][0]))>10*np.finfo(float).eps
  S3 = abs((A[0][0]-A[1][0])*(C[0][1]-C[1][1])+(A[0][1]-A[1][1])*\
   (C[1][0]-C[0][0]))>10*np.finfo(float).eps

  if S1 or S2 or S3:
    h=1
  else:
    h=0

  S4 = abs(A[0][1]*A[1][0]-A[0][0]*A[1][1]-B[0][1]*B[1][0]+\
           B[0][0]*B[1][1])>10*np.finfo(float).eps
  S5 = abs(A[0][1]*B[1][0]-A[0][0]*B[1][1]+B[0][1]*A[1][0]-\
           B[0][0]*A[1][1])>10*np.finfo(float).eps
  S6 = abs(B[1][0]*(C[0][1]-C[1][1])-B[1][1]*(C[0][0]-C[1][0]))\
  >10*np.finfo(float).eps
  S7 = abs(A[1][0]*(C[0][1]-C[1][1])-A[1][1]*(C[0][0]-C[1][0]))\
  >10*np.finfo(float).eps
  S8 = abs((C[0][0]-C[1][0])*(B[0][1]-B[1][1])+(C[0][1]-C[1][1])*\
   (B[1][0]-B[0][0]))>10*np.finfo(float).eps
  S9 = abs((A[0][0]-A[1][0])*(C[0][1]-C[1][1])+(A[0][1]-A[1][1])*\
   (C[1][0]-C[0][0]))>10*np.finfo(float).eps

  if S4 or S5:
    k = 2
  elif S6 or S7 or S8 or S9:
    k = 1
  else:
    k = 0

  # trigonometric gaussian formula on the arc
  tw=trigauss(n+k,alpha,beta)

  # algebraic gaussian formula on [0,1]
  ab=r_jacobi(int(np.ceil((n+h+1)/2)),0,0)

  xw=gauss(int(np.ceil((n+h+1)/2)),ab)
  xw[:,0]=xw[:,0]/2+1/2
  xw[:,1]=xw[:,1]/2

  #creating the grid
  t, theta = np.meshgrid(xw[:,0], tw[:,0])
  w1, w2 = np.meshgrid(xw[:,1], tw[:,1])

  # nodal cartesian coordinates and weights
  theta = theta.T.flatten()
  t = t.T.flatten()
  w1 = w1.T.flatten()
  w2 = w2.T.flatten()
  s = np.sin(theta)
  c = np.cos(theta)
  p1 = A[0][0]*c+B[0][0]*s+C[0][0]
  p2 = A[0][1]*c+B[0][1]*s+C[0][1]
  q1 = A[1][0]*c+B[1][0]*s+C[1][0]
  q2 = A[1][1]*c+B[1][1]*s+C[1][1]
  dp1 = -A[0][0]*s+B[0][0]*c
  dp2 = -A[0][1]*s+B[0][1]*c
  dq1 = -A[1][0]*s+B[1][0]*c
  dq2 = -A[1][1]*s+B[1][1]*c

  # xyw[:,0] = p1*t+q1*(1-t), p2*t+q2*(1-t)
  # xyw[:,1] = p2*t+q2*(1-t)
  # xyw[:,2] = np.abs((p1-q1)*(dp2*t+dq2*(1-t))-\
  # (p2-q2)*(dp1*t+dq1*(1-t)))*w1*w2)

  return  np.transpose((p1*t+q1*(1-t), p2*t+q2*(1-t), np.abs((p1-q1)*\
   (dp2*t+dq2*(1-t))-(p2-q2)*(dp1*t+dq1*(1-t)))*w1*w2))

#-------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------

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
