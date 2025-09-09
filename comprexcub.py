# -*- coding: utf-8 -*-
"""comprexcub.ipynb"""

import numpy as np
from scipy.optimize import nnls
import scipy

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
        print(f"LHDM({k}){'with ULS initialization'if options.get('init',False)else ''}")

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
