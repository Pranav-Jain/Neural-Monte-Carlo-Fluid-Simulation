import numpy as np
import meshio

def boundary_faces(T):
  '''
  Adapted from: https://github.com/alecjacobson/gptoolbox/blob/master/mesh/boundary_faces.m
  Params:
    T: #Tet x 4 tetrahedron elements
  Return:
    F: faces on the boundary of the tetrahedron mesh
  '''
  assert(T.shape[1] == 4)
  allF = np.vstack((T[:,[3,1,2]], T[:,[2,0,3]], T[:,[1,3,0]], T[:,[0,2,1]]))
  # sort rows so that faces are reorder in ascending order of indices
  sortedF = np.sort(allF, axis = 1)
  # determine uniqueness and counts of faces
  u, indices, counts = np.unique(sortedF, return_index=True, return_counts=True, axis = 0)
  # extract faces that only occurred once
  I = indices[counts == 1]
  F = allF[I, :]
  return F


if __name__ == '__main__':
  '''Testing'''
  mesh_path = './data/bunny.mesh'
  mesh = meshio.read(mesh_path)
  faces = mesh.cells_dict['tetra']
  SF = boundary_faces(faces)
  print(mesh.cells_dict)