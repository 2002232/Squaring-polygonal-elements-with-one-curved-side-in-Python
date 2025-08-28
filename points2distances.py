# -*- coding: utf-8 -*-
"""points2distances.ipynb"""

import numpy as np

def points2distances(p):

  # Create euclidean distance matrix from point matrix.
  points = np.array(p)

  # All inner products between points.
  distances = points @ np.transpose(points)

  # Vector of squares of norms of points.
  lsq = np.diag(distances)

  a = np.array([[i]*len(points) for i in lsq])

  # distances = np.sqrt(a+np.transpose(a)-2*distances)
  return np.sqrt(a+np.transpose(a)-2*distances)
