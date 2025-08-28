# -*- coding: utf-8 -*-
"""chebvand.ipynb

import numpy as np


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
    good = good[np.argsort(good[:, 1])]
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
