# -*- coding: utf-8 -*-
"""auto_rotation.ipynb"""

import numpy as np

# Code author: Giovanni Traversin
# Release date: 05 Sept 2025

def auto_rotation(p, V1, V2):

  # AUTOMATIC ROTATION OF A CONVEX POLYGON SO THAT "GAUSSIAN POINTS",
  # AS IN THE PAPER THEY ARE ALL CONTAINED IN THE CONVEX POLYGON.
  # SEE THE PAPER FOR DETAILS.

  polygon_bd = np.array(p)
  vertex_1 = np.array(V1)
  vertex_2 = np.array(V2)

  # FIND DIRECTION AND ROTATION ANGLE.
  if len(vertex_1) == 0:
    # COMPUTING ALL THE DISTANCES BETWEEN POINTS.
    # A LITTLE TIME CONSUMING AS PROCEDURE.
    distances = points2distances(polygon_bd)
    max_distances = distances.max(1)
    max_distance = distances.max()
    max_row_comp = np.argmax(max_distances)
    max_col_comp = []
    for i in distances:
      max_col_comp.append(np.argmax(i))
    vertex_1 = np.array(polygon_bd[max_col_comp[max_row_comp]])
    vertex_2 = np.array(polygon_bd[max_row_comp])
    direction_axis = (vertex_2-vertex_1)/max_distance
  else:
    direction_axis = (vertex_2-vertex_1)/np.linalg.norm(vertex_2-vertex_1)

  rot_angle_x = np.arccos(direction_axis[0])
  rot_angle_y = np.arccos(direction_axis[1])

  if rot_angle_y <= np.pi/2:
    if rot_angle_x <= np.pi/2:
      rot_angle = -rot_angle_y
    else:
      rot_angle = rot_angle_y
  else:
    if rot_angle_x <= np.pi/2:
      rot_angle = np.pi-rot_angle_y
    else:
      rot_angle = rot_angle_y

  # CLOCKWISE ROTATION.
  rot_matrix = np.array([[np.cos(rot_angle), np.sin(rot_angle)], \
   [-np.sin(rot_angle), np.cos(rot_angle)]])
  number_sides = len(polygon_bd[:,0])-1
  polygon_bd_rot = (rot_matrix @ polygon_bd.T).T
  axis_abscissa = rot_matrix @ vertex_1.T

  return polygon_bd_rot, rot_matrix, rot_angle, axis_abscissa, \
  vertex_1, vertex_2

#-------------------------------------------------------------------------------

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
