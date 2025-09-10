# -*- coding: utf-8 -*-
"""polygauss.ipynb"""

import numpy as np

def polygauss(N, p, P1=[], Q1=[], rotation = 1):

  #--------------------------------------------------------------------------
  # REFERENCE PAPER:
  # [1] A. SOMMARIVA and M. VIANELLO
  # "Gauss-like and triangulation-free cubature over polygons".
  #
  # INPUT:
  #
  # N     : DEGREE OF THE 1 DIMENSIONAL GAUSS-LEGENDRE RULE.
  #
  # polygon_sides: IF THE POLYGON HAS "L" SIDES, "boundary.pts" IS A
  #         VARIABLE CONTAINING ITS VERTICES, ORDERED COUNTERCLOCKWISE.
  #         AS LAST ROW MUST HAVE THE COMPONENTS OF THE FIRST VERTEX.
  #         IN OTHER WORDS, THE FIRST ROW AND LAST ROW ARE EQUAL.
  #         "polygon_sides" IS A "L+1 x 2" MATRIX.
  #
  #            --------- NOT MANDATORY VARIABLES ---------
  #
  # rotation: 0: NO ROTATION.
  #           1: AUTOMATIC.
  #           2: PREFERRED DIRECTION ROTATION BY P, Q.
  #
  # P, Q: DIRECTION THAT FIXES THE ROTATION.
  #
  # OUTPUT:
  #
  # xyw     : THE GAUSS LIKE FORMULA PRODUCES THE NODES (xyw(:,1),xyw(:,2))
  #           AND THE WEIGHTS xyw(:,3) OF A CUBATURE RULE ON THE POLYGON.
  #
  #--------------------------------------------------------------------------
  # EXAMPLE 1 (NO ROTATION.)
  #---------------------------
  #
  # >> xyw=polygauss_2013(2,[0 0; 1 0; 1 1; 0 1; 0 0],0)
  #
  # xyw =
  #
  #     0.2113    0.2113    0.2500
  #     0.2113    0.7887    0.2500
  #     0.7887    0.2113    0.2500
  #     0.7887    0.7887    0.2500
  #
  # >>
  #
  #--------------------------------------------------------------------------
  # EXAMPLE 2 (AUTO ROTATION.)
  #-----------------------------
  #
  # >> xyw=polygauss_2013(2,[0 0; 1 0; 1 1; 0 1; 0 0])
  #
  # xyw =
  #
  #     0.0683    0.0444    0.0078
  #     0.3028    0.1972    0.0556
  #     0.5374    0.3499    0.0616
  #     0.6501    0.4626    0.0616
  #     0.8028    0.6972    0.0556
  #     0.9556    0.9317    0.0078
  #     0.9317    0.9556    0.0078
  #     0.6972    0.8028    0.0556
  #     0.4626    0.6501    0.0616
  #     0.3499    0.5374    0.0616
  #     0.1972    0.3028    0.0556
  #     0.0444    0.0683    0.0078
  #     0.1008    0.0119    0.0078
  #     0.4472    0.0528    0.0556
  #     0.7935    0.0938    0.0616
  #     0.9062    0.2065    0.0616
  #     0.9472    0.5528    0.0556
  #     0.9881    0.8992    0.0078
  #     0.8992    0.9881    0.0078
  #     0.5528    0.9472    0.0556
  #     0.2065    0.9062    0.0616
  #     0.0938    0.7935    0.0616
  #     0.0528    0.4472    0.0556
  #     0.0119    0.1008    0.0078
  #
  # >>

  #--------------------------------------------------------------------------
  # Copyright (C) 2007-2013 Marco Vianello and Alvise Sommariva
  #
  # This program is free software; you can redistribute it and/or modify
  # it under the terms of the GNU General Public License as published by
  # the Free Software Foundation; either version 2 of the License, or
  # (at your option) any later version.
  #
  # This program is distributed in the hope that it will be useful,
  # but WITHOUT ANY WARRANTY; without even the implied warranty of
  # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  # GNU General Public License for more details.
  #
  # You should have received a copy of the GNU General Public License
  # along with this program; if not, write to the Free Software
  # Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
  #
  # Authors:
  # Marco Vianello    <marcov@euler.math.unipd.it>
  # Alvise Sommariva  <alvise@euler.math.unipd.it>
  # Date: April 30, 2013.
  #--------------------------------------------------------------------------

  polygon_sides = np.array(p)
  P = np.array(P1)
  Q = np.array(Q1)
  x_bd = polygon_sides[:,0]
  y_bd = polygon_sides[:,1]

  # "MINIMUM" RECTANGLE CONTAINING POLYGON.
  x_min = min(x_bd)
  x_max=max(x_bd)
  y_min = min(y_bd)
  y_max=max(y_bd);

  cubature_type=4

  # POLYGON ROTATION (IF NECESSARY).
  match rotation:
    case 0:
      #print('\n \t [ROTATION]: NO.')
      rot_matrix = np.eye(2)
      axis_abscissa = np.array([x_min, y_max])-np.array([x_min, y_min])
    case 1:
      #print('\n \t [ROTATION]: AUTOMATIC')
      polygon_sides, rot_matrix, rot_angle, axis_abscissa, P, Q = \
      auto_rotation(polygon_sides,[],[])
    case 2:
      #print('\n \t [ROTATION]: PREFERRED DIRECTION');
      nrm_vect = np.linalg.norm(Q-P)
      if nrm_vect > 0:
        direction_axis = (Q-P)/nrm_vect
        polygon_sides,rot_matrix,rot_angle,axis_abscissa,P,Q = \
        auto_rotation(polygon_sides,P,Q)
      else:
        #print('\n \t [WARNING]: THE DIRECTION VECTOR IS NULL. ')
        #print('USING AUTOMATIC ROTATION.')
        polygon_sides,rot_matrix,rot_angle,axis_abscissa,P,Q = \
        auto_rotation(polygon_sides,P,Q)


  # COMPUTE NODES AND WEIGHTS OF 1D GAUSS-LEGENDRE RULE.
  # TAKEN FROM TREFETHEN PAPER "Is ... Clenshaw-Curtis?".

  # DEGREE "N" (ORDER GAUSS PRIMITIVE)
  s_N, w_N = cubature_rules_1D((N-1),cubature_type)
  N_length = len(s_N)

  # DEGREE "M" (ORDER GAUSS INTEGRATION)
  M = N+1
  s_M, w_M = cubature_rules_1D((M-1),cubature_type)

  # L: NUMBER OF SIDES OF THE POLYGON.
  L = len(polygon_sides[:,1])-1

  #a=0.5;
  a = axis_abscissa[0]

  # COMPUTE 2D NODES (nodes_x,nodes_y) AND WEIGHTS "weights".
  nodes_x=[]
  nodes_y=[]
  weights=[]

  for i in range(L):
     x1=polygon_sides[i][0]
     x2=polygon_sides[i+1][0]
     y1=polygon_sides[i][1]
     y2=polygon_sides[i+1][1]
     if not (x1 == a and x2 == a):
      if y2-y1 != 0:
        if x2-x1 != 0:
          s_M_loc = s_M
          w_M_loc = w_M
        else:
          s_M_loc = s_N
          w_M_loc = w_N

        M_length = len(s_M_loc)
        half_pt_x = (x1+x2)/2
        half_pt_y = (y1+y2)/2
        half_length_x = (x2-x1)/2
        half_length_y = (y2-y1)/2

        #  GAUSSIAN POINTS ON THE SIDE.
        x_gauss_side = half_pt_x+half_length_x*s_M_loc; #SIZE: (M_loc,1)
        y_gauss_side = half_pt_y+half_length_y*s_M_loc; #SIZE: (M_loc,1)

        scaling_fact_plus = (x_gauss_side+a)/2  #SIZE: (M_loc,1)
        scaling_fact_minus = (x_gauss_side-a)/2  #SIZE: (M_loc,1)
        # SIZE:(M_loc,1)
        local_weights = (half_length_y*scaling_fact_minus)*w_M_loc
        # SIZE: (M_loc,N)
        term_1 = np.tile(np.transpose([scaling_fact_plus]),N_length)
        # SIZE: (M_loc,N)
        term_2 = np.tile(np.transpose([scaling_fact_minus]),N_length)
        rep_s_N = np.tile(s_N,(M_length,1))


        # x, y ARE STORED IN MATRICES. A COUPLE WITH THE SAME INDEX
        # IS A POINT, i.e. "P_i=(x(k),y(k))" FOR SOME "k".
        x= term_1+term_2*rep_s_N
        number_rows=len(x)
        number_cols=len(x[0])
        x = x.flatten('F')
        y = np.tile(y_gauss_side,N_length)
        # THE INVERSE OF A ROTATION MATRIX IS ITS TRANSPOSE.
        rot_gauss_pts = rot_matrix.T @ np.array([x,y])


        # GAUSS POINTS IN THE ORIGINAL SYSTEM.
        x_rot = rot_gauss_pts[0,:]
        y_rot = rot_gauss_pts[1,:]

        x_rot = np.reshape(x_rot,(number_cols,number_rows)).T
        y_rot = np.reshape(y_rot,(number_cols,number_rows)).T

        weights = np.append(weights, local_weights)
        for i in x_rot:
          nodes_x.append(i)
        for j in y_rot:
          nodes_y.append(j)

  #nodes_x = np.ravel(np.array(nodes_x),'F')
  #nodes_y = np.ravel(np.array(nodes_y),'F')
  #weights = np.ravel(np.outer(weights, w_N),'F')

  #return nodes_x, nodes_y, weights

  return np.transpose((np.ravel(np.array(nodes_x),'F'), \
                       np.ravel(np.array(nodes_y),'F'), \
                       np.ravel(np.outer(weights, w_N),'F')))


#-------------------------------------------------------------------------------


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

#-------------------------------------------------------------------------------

def cubature_rules_1D(n,cubature_type):

  # SEE WALDVOGEL PAPER. ADDED NODES

  # Weights of the Fejer2, Clenshaw-Curtis and Fejer1 quadrature by DFTs
  # n>1. Nodes: x_k = cos(k*pi/n)

  N = np.array(range(1,n,2))
  l = len(N)
  K = np.array(range(0,n-l))

  match cubature_type:
    case 1:       # FEJER 1
      v0 = np.append([2*np.exp(1j*np.pi*K/n)/(1-4*K**2)],[0]*(l+1))
      v1 = v0[:-1]+np.conjugate(v0[:0:-1])
      weights = np.real(np.fft.ifft(v1))  # Imaginary residual
      k = np.arange(.5 , n+.5)
      nodes = np.cos(k*np.pi/n)

    case 2:       # FEJER 2
      v0 = np.append(2/N/(N-2), [1/N[-1]]+ [0]*(n-l))
      v2 = -v0[:-1]-v0[:0:-1]
      weights = np.append(np.fft.ifft(v2),0)
      k = np.array(range(0,n+1))
      nodes = np.cos(k*np.pi/n)

    case 3:       # CLENSHAW CURTIS
      g0 = -np.ones(n)
      g0[l] = g0[l]+n
      g0[n-l] = g0[n-l]+n
      g=g0/(n**2-1+n%2)
      v0 = np.append(2/N/(N-2), [1/N[-1]]+ [0]*(n-l))
      v2 = -v0[:-1]-v0[:0:-1]
      wcc = np.real(np.fft.ifft(v2+g))  # Imaginary residual
      weights = np.append(wcc, wcc[0])
      k = np.array(range(0,n+1))
      nodes = np.cos(k*np.pi/n)

    case 4:       # GAUSS LEGENDRE
      beta = .5/np.sqrt(1-1/((2*np.array(range(1,n+1)))**2))
      T = np.diag(beta,1)+np.diag(beta,-1)
      nodes,V=np.linalg.eigh(T)
      index = np.argsort(nodes)
      nodes = np.sort(nodes)
      weights = 2*V[0]**2


  return nodes, weights
