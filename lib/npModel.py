'''
  Author: Nianhong Jiao
'''

import numpy as np
import pickle


class loadModel():
	def __init__(self, model_path):
		'''
			Load model.
    		J_regressor: joint locations, sparse matrix, 24 * 6890
    		weights: blend weights, 6890 * 24
    		posedirs: pose, 6890 * 3 * 207(207 = 23 * 9)
    		v_template: T, 6890 * 3
    		shapedirs: shape, 6890 * 3 * 10
    		f: triangulation, 13776 * 3
    		kintree_table: kinematic tree, 2 * 24
    		J: initial 24 joints
    		weights_prior: weights prior, 6890 * 24
		'''

		with open(model_path, 'rb') as f:
			params = pickle.load(f)

			self.J_regressor = params['J_regressor']
			self.weights = np.array(params['weights'])
			self.posedirs = np.array(params['posedirs'])
			self.v_template = np.array(params['v_template'])
			self.shapedirs = np.array(params['shapedirs'])
			self.faces = np.array(params['f'])
			self.kintree_table = params['kintree_table']

		id_joint = {
			self.kintree_table[1, i] : i for i in range(self.kintree_table.shape[1])
		}
		self.parent = {
			i : id_joint[self.kintree_table[0, i]]
			for i in range(1, self.kintree_table.shape[1])
		}

		self.pose_shape = [24, 3]
		self.beta_shape = [10]
		self.trans_shape = [3]

		self.pose = np.zeros(self.pose_shape)
		self.beta = np.zeros(self.beta_shape)
		self.trans = np.zeros(self.trans_shape)

		self.verts = None
		self.J = None
		self.R = None

		self.update()

	def generate(self, beta=None, pose=None, trans=None):
		'''
			Update model.
		'''

		if pose is not None:
			self.pose = pose
		if beta is not None:
			self.beta = beta
		if trans is not None:
			self.trans = trans
		self.update()

		return self.verts, self.faces

	def update(self):
		'''
			Update vertices.
		'''

		# Add shape to model
		v_shaped = self.shapedirs.dot(self.beta) + self.v_template

		# Locate joints
		self.J = self.J_regressor.dot(v_shaped)

		# Rotate joints
		pose_para = self.pose.reshape((-1, 1, 3))
		self.R = self.rodrigues(pose_para)
		I_cube = np.broadcast_to(
			np.expand_dims(np.eye(3), axis=0),
			(self.R.shape[0]-1, 3, 3)
		)

		# Rotation matrix 3 * 3, 207(23 * 9)
		rot_matrix = (self.R[1:] - I_cube).ravel()

		# Add pose to model
		v_posed = v_shaped + self.posedirs.dot(rot_matrix)

		# World transformation of joints
		G = np.empty((self.kintree_table.shape[1], 4, 4))

		# Root joint
		G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape(3, 1))))

		# Relative rotation of child joint with..
		# 	respect to its parent joint in the kinematic tree table.
		for i in range(1, self.kintree_table.shape[1]):
			G[i] = G[self.parent[i]].dot(
				self.with_zeros(
					np.hstack(
						(self.R[i], (self.J[i, :]-self.J[self.parent[i], :]).reshape(3, 1))
						)
					)
				)

		# Remove the transformation of rest pose
		G = G - self.pack(
			np.matmul(
				G, np.hstack(
					(self.J, np.zeros([24, 1]))
					).reshape([24, 4, 1])
				)
			)

		# Transform each vetex
		T = np.tensordot(self.weights, G, axes = [[1], [0]])
		rest_shape = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
		v = np.matmul(T, rest_shape.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
		self.verts = v + self.trans.reshape([1, 3])


	def rodrigues(self, pose_r):
		'''
			Similar to cv2.Rodrigues().
			Input: pose_r [size, 1, 3]
			Return: Rotation matrix of shape, [size, 3, 3]
		'''

		theta = np.linalg.norm(pose_r, axis=(1, 2), keepdims=True)

		# Avoid zero divide
		theta = np.maximum(theta, np.finfo(np.float64).tiny)
		r_hat = pose_r / theta
		cos = np.cos(theta)
		z_stick = np.zeros(theta.shape[0])
		m = np.dstack([
			z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
			r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
			-r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
		).reshape([-1, 3, 3])
		i_cube = np.broadcast_to(
		np.expand_dims(np.eye(3), axis=0),
			[theta.shape[0], 3, 3]
		)
		A = np.transpose(r_hat, axes=[0, 2, 1])
		B = r_hat
		dot = np.matmul(A, B)
		R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m

		return R


	def with_zeros(self, x):
		'''
			Add [0, 0, 0, 1] vector to x([3, 4] matrix).
			Return: [4, 4] matrix
		'''

		return np.vstack((x, np.array([[0., 0., 0., 1.]])))

	def pack(self, x):
		'''
			Add [size, 4, 3] zeros matrix to x([size, 4, 1]).
			Return: [size, 4, 4] matrix
		'''

		return np.dstack((np.zeros((x.shape[0], 4, 3)), x))
