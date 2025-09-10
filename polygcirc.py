# -*- coding: utf-8 -*-
"""polygcirc.ipynb"""

# Code author: Giovanni Traversin
# Release date: 05 Sept 2025

import numpy as np
from scipy import integrate
from math import gamma
from scipy.optimize import nnls

# Code author: Giovanni Traversin
# Release date: 05 Sept 2025

def polygcirc(n,V,A1,B1,center,r,conv,pos=1):

  #--------------------------------------------------------------------------
  # OBJECT:

  # Computation of a basic and a compressed positive cubature formula
  # on a polygonal element with a circular side, namely the set
  # union (convex arc) or difference (concave arc) of a convex
  # polygonal element with a circular segment.
  #--------------------------------------------------------------------------
  # INPUT:

  # n: polynomial degree of exactness
  # A1,B1: extrema of the circular arc
  # center,r: circular arc center and radius
  # V: polygon vertices (2-column array of coords)
  #    note that to "v" are added by default the arc extrema, in
  #    couterclockwise order
  # conv: conv=1 for a convex arc, conv=0 for a concave arc
  #
  # WARNING: the figure vertices are a,v(1,:),...,v(end,:),b and MUST BE
  # in COUNTERCLOCKWISE order
  # (the arc ba is clockwise on the circle if concave and counterclockwise
  # if convex)
  #--------------------------------------------------------------------------
  # OUTPUT:

  # xyw: 3-column array of cubature nodes and positive weights
  # xywc: 3-column array of compressed cubature nodes and positive weights
  #--------------------------------------------------------------------------
  # AUTHORS:

  # Authors: E. Artioli, A. Sommariva and M. Vianello
  # Written: April 24, 2018
  # Revised: December 02, 2021
  #--------------------------------------------------------------------------

  # auxiliary function used to stack columns on xyw more easily

  cc = np.array(center)
  v = np.array(V)
  a = np.array(A1)
  b = np.array(B1)
  L = []
  P = []
  subs = []

  def stackxyw(xyw, nw):
    if len(xyw) == 0:
      return nw
    else:
      return np.vstack((xyw, nw))

  # Convex scenario
  if conv==1:
    P = np.vstack([a,v,b,a])
    xyw = np.vstack((polygauss(n,P), circtrap(n,b,a,b,a,cc,r)))
    L += [xyw[:,1].size]
    A = b[:]
    B = a[:]
    C = b[:]
    D = a[:]
    subs += [A,B,C,D]

  #Concave scenario
  else: #conv = 0
    k = 0; l = 0; t = 0; e=[]; z=[]; u=[]; eta=[]; xyw=[]
    aa=a-cc; bb=b-cc
    angleab = np.arccos((aa@bb)/(np.linalg.norm(aa)*np.linalg.norm(bb)))
    anglea = np.angle(aa[0]+1j*aa[1])
    angleb = np.angle(bb[0]+1j*bb[1])

    if angleb<=np.pi:
      clockba = anglea>=angleb and anglea<angleb+np.pi
    else:
      clockba = anglea>angleb-np.pi and anglea<=angleb
    if len(v[:,0])==2:
      v = np.array([v[0], (v[0]+v[1])/2, v[1]])

    vv = v-np.matlib.repmat(cc, len(v[:,0]), 1)

    for i in range(len(v[:,1])):
      angle1 = np.arccos((vv[i]@aa)/(np.linalg.norm(aa)*np.linalg.norm(vv[i])))
      angle2 = np.arccos((vv[i]@bb)/(np.linalg.norm(bb)*np.linalg.norm(vv[i])))

      if angle1<angle2 and angle2>angleab and clockba==0:
        e += [v[i]]
        k+=1
      if clockba==1 or (angle1<=angleab and angle2<=angleab):
        z += [v[i]]
        zz = z[l]-cc
        theta = np.angle(zz[0]+1j*zz[1])
        u += [cc+r*np.array([np.cos(theta), np.sin(theta)])]
        l+=1
      if angle2<angle1 and angle1>angleab and clockba==0:
        eta += [v[i]]
        t+=1

    e = np.array(e); z = np.array(z); u = np.array(u); eta  = np.array(eta)

    if k>1:
      P1 = np.vstack((a, e, a))
      xyw = polygauss(n, P1)
      L += [xyw[:,0].size]

    if k==0 and l>=1:
      if np.linalg.norm(u[0]-a)>10**(-14):
        nw = circtrap(n,a,u[0], z[0], z[0], cc, r)
        L += [nw[:,0].size]
        xyw = nw
        A = a; B = u[0,:]
        C = z[0]; D = z[0]
        subs += [A,B,C,D]

    if k>=1 and l>=1:
      nw = circtrap(n,a,u[0],e[k-1],z[0],cc,r)
      L += [nw[:,0].size]
      xyw = stackxyw(xyw, nw)
      A = a; B = u[0,:]
      C = e[k-1,:]; D = z[0,:]
      subs += [A,B,C,D]

    for j in range(1,l):
      nw=circtrap(n,u[j-1],u[j],z[j-1],z[j],cc,r)
      L += [nw[:,0].size]
      xyw = stackxyw(xyw, nw)
      A = u[j-1]; B = u[j]
      C = z[j-1]; D = z[j]
      subs += [A,B,C,D]

    if t == 0 and l>=1:
      if np.linalg.norm(u[l-1]-b)>10**(-14):
        nw = circtrap(n,u[l-1],b,z[l-1,],z[l-1],cc,r)
        L += [nw[:,0].size]
        xyw = stackxyw(xyw, nw)
        A = u[l-1]; B = b
        C = z[l-1]; D = z[l-1]
        subs += [A,B,C,D]

    if t>=1 and l>=1:
      nw = circtrap(n,u[l-1,:],b,z[l-1,:],eta[0,:],cc,r)
      L += [nw[:,0].size]
      xyw = stackxyw(xyw, nw)
      A = u[l-1]; B = b
      C = z[l-1]; D = eta[0]
      subs += [A,B,C,D]

    if t>1:
      P2=np.vstack((b,eta,b))
      nw=polygauss(n,P2)
      L += [nw[:,1].size]
      xyw = stackxyw(xyw, nw)

    if l==0:
      if k>=1 and t>=1:
        nw = circtrap(n,a,b,e[k-1],eta[0],cc,r)
        L += [nw[:,0].size]
        A = a; B = b
        C = e[k-1]; D = eta[0]
        subs += [A,B,C,D]

      if k==0 and t>=1:
        nw = circtrap(n,a,b,eta[0],eta[0],cc,r)
        L += [nw[:,0].size]
        A = a; B = b
        C = eta[0]; D = eta[0]
        subs += [A,B,C,D]

      if k>=1 and t==0:
        nw = circtrap(n,a,b,e[k-1],e[k-1],cc,r)
        L += [nw[:,0].size]
        A = a; B = b
        C = e[k-1,:]; D = e[k-1,:]
        subs += [A,B,C,D]
      xyw = stackxyw(xyw, nw)


  subs = np.array(subs)
  pts, w, momerr = comprexcub(n,[xyw[:,0], xyw[:,1]],xyw[:,2],pos)
  xywc = np.transpose([pts[:,0], pts[:,1], w])

  return  xyw, xywc, P, L, subs


#-------------------------------------------------------------------------------


def circtrap(n,a,b,c,d,cc,r):

  # by E. Artioli, A. Sommariva, M. Vianello
  # April 2018

  # n: polynomial exactness degree
  # a,b: circular arc extrema coords 1 x 2
  # c,d: base segment extrema coords 1 x 2, ac and bd are the sides
  # cc: circle center coords 1 x 2
  # r: circle radius

  # xyw: 3-column array xyw(:,1:2) nodes, xyw(:,3) weights

  A = np.array(a)
  B = np.array(b)
  C = np.array(c)
  D = np.array(d)
  CC = np.array(cc)

  Z = (A[0]-CC[0])+1j*(A[1]-CC[1])
  W = (B[0]-CC[0])+1j*(B[1]-CC[1])
  az = np.angle(Z)
  aw = np.angle(W)

  if az<=aw:
    if aw-az<=np.pi:
      alpha = az
      beta = aw
      U = C
      V = D
    else:
      alpha = aw
      beta = az+2*np.pi
      U = D
      V = C

  if az>aw:
    if az-aw<=np.pi:
      alpha = aw
      beta = az
      U = D
      V = C
    else:
      alpha = az
      beta = aw+2*np.pi
      U = C
      V = D

  om = (beta-alpha)/2
  g = (beta+alpha)/2
  s=2*np.sin(om)
  A1 = [[r*np.cos(g), r*np.sin(g)], [0, 0]]
  B1 = [[-r*np.sin(g), r*np.cos(g)], [(V[0]-U[0])/s , (V[1]-U[1])/s ]]
  C1 = [[CC[0], CC[1]], [(V[0]+U[0])/2, (V[1]+U[1])/2]]

  return gqellblend(n,A1,B1,C1,-om,om)

#-------------------------------------------------------------------------------

def gqellblend(n, A, B, C, alpha, beta):

  # Original code by Gaspare Da Fies, Alvise Sommariva and Marco Vianello

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


#-------------------------------------------------------------------------------


# Code author: Giovanni Traversin
# Release date: 05 Sept 2025

def comprexcub(deg, x, omega, pos):

  # compression of bivariate cubature formulas by Tchakaloff points
  # or approximate Fekete points
  # useful, for example, in node reduction of algebraic cubature formulas
  # see the web page: http://www.math.unipd.it/~marcov/Tchakaloff.html

  # by Federico Piazzon, Alvise Sommariva and Marco Vianello
  # , May 2016


  # INPUT:
  # deg: polynomial exactness degree
  # X: 2-column array of point coordinates
  # omega: 1-column array of weights
  # pos: NNLS for pos=1, QR with column pivoting for pos=0

  # OUTPUT:
  # pts: 2-column array of extracted points
  # w: 1-column array of corresponding weights (positive for pos=1)
  # momerr: moment reconstruction error

  X = np.array(x)
  if len(X[0])>2:
    X = X.T
  rect = [min(X[:,0]), max(X[:,0]), min(X[:,1]), max(X[:,1])]
  V=chebvand(deg,X,rect)
  Q, R = np.linalg.qr(V)
  Q = np.real(Q)
  orthmom = Q.T @ omega
  match pos:
    case 1:
      weights = LHDM(Q.T,orthmom)
    case _:
      weights = nnls(Q.T,orthmom)[0]
  ind = np.where(abs(weights)>0)
  w = weights[ind]
  return X[ind, :][0], w, np.linalg.norm( Q[ind,:][0].T @ w - orthmom)


#-------------------------------------------------------------------------------


def chebvand(deg, x, rect = None):

# INPUT:
# deg = polynomial degree
# x = 2-column array or 2 arrays of the same length of point coordinates
# rect = 4-component vector such that the rectangle
# [rect(1),rect(2)] x [rect(3),rect(4)] contains X
# If you do not compile the input for rect, the program will use as rect the
# smaller rectangle containing all the points in X.

# OUTPUT:
# V = Chebyshev-Vandermonde matrix at x, graded lexic. order

  X = np.array(x)
  if rect is None:
    rect = [min(X[:,0]), max(X[:,0]), min(X[:,1]), max(X[:,1])]

  # couples with length less or equal to deg
  # graded lexicographical order
  j = np.arange(deg + 1)
  j1, j2 = np.meshgrid(j,j)
  jj = j1 + j2
  dim = (deg+1)*(deg+2)//2
  couples = np.zeros((dim, 2), dtype=int)

  for s in range(deg+1):
    good = np.argwhere(jj == s)
    a = s*(s+1)//2
    for i  in range(len(good)):
      couples[a+i] = [j1[good[i][1],good[i][0]], j2[good[i][1],good[i][0]]]

  # mapping the mesh in the square [-1,1]^2
  a = rect[0]; b = rect[1]; c = rect[2]; d = rect[3]
  # map = [(2*X[:,0]-b-a)/(b-a) , (2*X[:,1]-d-c)/(d-c)]

  # Chebyshev-Vandermonde matrix on the mesh
  T1 = chebpolys(deg, (2*X[:,0]-b-a)/(b-a))
  T2 = chebpolys(deg, (2*X[:,1]-d-c)/(d-c))

  return  T1[:, couples[:, 0]] * T2[:, couples[:, 1]] # = V


#-------------------------------------------------------------------------------


def chebpolys(deg,x1):

# computes the Chebyshev-Vandermonde matrix on the real line by recurrence

# INPUT:
# deg = maximum polynomial degree
# x = list or array of abscissas

# OUTPUT:
# T = Chebyshev-Vandermonde matrix at x,
# T(i,j+1)=T_j(x_i), j=0,...,deg

    x = np.array(x1)
    n = len(x)
    T= np.zeros((n,deg+1))
    t0 = np.ones(n)
    t1 = x[:]
    T[:,0] = t0
    T[:,1] = t1

    for i in range(2,deg+1):
      t2 = 2*x*t1 - t0
      T[:,i] = t2
      t0 = t1[:]
      t1 = t2[:]

    return T


#-------------------------------------------------------------------------------


#Code author: Laura Rinaldi
# https://github.com/laura-rinaldi/Cheap_Splinegauss

def LHDM(C, d, options=None, verbose=0):
# Lawson-Hanson algorithm accelerated by Deviation Maximization (DM).

#     Parameters:
#         C (numpy.ndarray): Least squares matrix.
#         d (numpy.ndarray): Right-hand side vector.
#         options (dict, optional): Optimization parameters:
#             - init (bool): Use ULS initialization if True.
#             - tol (float): Tolerance for projected residual.
#             - k (int): Max number of indices added to Passive set
#               each iteration.
#             - thres (float): Threshold for angle between columns (0 to 1).
#         verbose (int, optional): Verbosity level.

#     Returns:
#         x (numpy.ndarray): Nonnegative solution minimizing ||C*x - d||.
#         resnorm (float): Squared residual norm ||C*x - d||^2.
#         exitflag (int): Exit condition (1: success, 0: exceeded iteration).
#         outeriter (int): Number of outer iterations.
#
    if options is None:
        options = {}

    m, n = C.shape
    nZeros = np.zeros(n)
    wz = np.zeros(n)
    itmax = 2 * m

    # Initialize sets
    P = np.zeros(n, dtype=bool)
    Z = np.ones(n, dtype=bool)
    x = np.zeros(n)

    thres = options.get('thres', 0.2222)
    thres_w = options.get('thres_w', 0.8)
    k = options.get('k', max(1, m // 20))
    tol = options.get('tol',10*np.finfo(float).eps*np.linalg.norm(C, 1)*len(C))
    LHDMflag = k > 1

    if verbose:
        print(f"LHDM({k}){'with ULS initialization'if \
                          options.get('init',False)else ''}")

    if LHDMflag:
        Cnorm = C / np.linalg.norm(C, axis=0)

    # Initialize residual and dual variables
    resid = d - C @ x
    w = C.T @ resid
    outeriter = 0
    totiter = 0

    while np.any(Z) and (np.any(w[Z] > tol) or np.any(x[P] < 0)) and totiter < itmax:
        outeriter += 1
        totiter += 1

        wz[P] = -np.inf
        wz[Z] = w[Z]

        if outeriter == 1 or not LHDMflag:
            t = np.argmax(wz)
        else:
            t = DM(Cnorm, wz, k, thres, thres_w)
            t = t[:min(len(t), m - np.sum(P))]

        addedP = np.shape(t)
        z = np.zeros_like(x)
        P[t] = True
        Z[t] = False
        z[P] = np.linalg.lstsq(C[:, P], d, rcond=None)[0]

        iter = 0
        removedP = 0

        while np.any(z[P] <= 0) and totiter < itmax:
            totiter += 1
            iter += 1
            Q = (z <= 0) & P
            alpha = np.min(x[Q] / (x[Q] - z[Q]))
            x = x + alpha * (z - x)
            t = np.where((np.abs(x) < tol) & P)[0]
            removedP += len(t)
            Z[(np.abs(x) < tol) & P] = True
            P = ~Z
            z[P] = np.linalg.lstsq(C[:, P], d, rcond=None)[0]

        x = z
        resid = d - C @ x
        w = C.T @ resid

    exitflag = 1 if outeriter < itmax else 0
    resnorm = np.dot(resid, resid)

    return x #, resnorm, exitflag, outeriter



def DM(Cnorm, wz, k, thres, thres_w):

#     Deviation Maximization

#     :param Cnorm: 2D NumPy array
#     :param wz: 1D NumPy array
#     :param k: integer
#     :param thres: threshold value
#     :param thres_w: weight threshold multiplier
#     :return: list of indices

    wzI = np.sort(wz)[::-1]  # Sort in descending order
    I = np.argsort(wz)[::-1]  # Get sorted indices
    t = I[0]
    p = [t]

    thres_wloc = thres_w * wzI[0]
    C = np.where(wzI > thres_wloc)[0]  # Get indices of wzI above threshold

    n = C.shape[0]
    add = 1

    for i in range(1, n):
        c = C[i]
        max_dev = np.max(np.abs(Cnorm[:, I[c]].T @ Cnorm[:, p]))

        if max_dev < thres:
            p.insert(0, I[c])
            add += 1

        if add >= k:
            break

    return p
