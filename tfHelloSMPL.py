'''
  Author: Nianhong Jiao
'''

import tensorflow as tf
import numpy as np
import pickle
import cv2

from lib.utils import *
from lib.tfModel import *


def tfGenerate(shape, pose, trans, gender):
	pose = tf.constant(pose.reshape([1, -1]), dtype=tf.float32)
	shape = tf.constant(shape.reshape([1, -1]), dtype=tf.float32)
	trans = tf.constant(trans.reshape([1, -1]), dtype=tf.float32)

	model_dir = 'models'
	model = loadSMPL(model_dir, gender=gender)
	v, f = model.tfGenerate(shape, pose, trans)
	sess = tf.Session()
	verts, faces = sess.run([v, f])

	return verts, faces.astype(int)


def main():
	pose_size = 72
	shape_size = 10
	trans = np.zeros(3)
	pose = np.random.rand(pose_size) *.2
	shape = np.random.rand(shape_size) *.02

	gender = 'f'
	verts, faces = tfGenerate(shape, pose, trans, gender=gender)

	w, h = (640, 480)
	render_res = render(verts, faces, w, h)

	cv2.imshow('render_tfSMPL', render_res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	outmeshe_path = 'demo.obj'
	save_to_obj(verts, faces, outmeshe_path)


if __name__ == '__main__':
	main()