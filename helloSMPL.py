'''
  Author: Nianhong Jiao
'''

import os
import numpy as np
from time import time

from lib.utils import *
from lib.npModel import *


def npGenerate(beta, pose, trans, gender='n'):
  model = loadModel('./models/basicModel_%s.pkl' % gender)
  verts, faces = model.generate(beta=beta, pose=pose, trans=trans)
  return verts, faces

def main():
  pose_size = 72
  shape_size = 10

  trans = np.zeros(3)
  pose = np.random.rand(pose_size) * .2
  beta = np.random.rand(shape_size) * .03

  '''
    n: neutral
    f: fmale
    m: male
  '''
  gender = 'f'
  t0 = time()
  verts, faces = npGenerate(beta, pose, trans, gender=gender)
  t1 = time()
  print('Run time: %.05f' % (t1-t0))

  # Render model
  w, h = (640, 480)
  render_res = render(verts, faces, w, h)

  # Show
  import cv2
  cv2.imshow('render_SMPL', render_res)
  print ('..Print any key while on the display window')
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # Write obj file
  outmesh_path = './outputMesh/demo.obj'
  save_to_obj(verts, faces, outmesh_path)
  print('Output mesh saved to:', outmesh_path)

if __name__ == '__main__':
  main()