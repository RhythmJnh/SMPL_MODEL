'''
  Author: Nianhong Jiao
'''

import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints

def render(verts, faces, w=640, h=480):
	# Frontal view
	verts[:, 1:3] = -verts[:, 1:3]

	# Create OpenDR renderer
	rn = ColoredRenderer()

	# Assign attributes to renderer
	rn.camera = ProjectPoints(v=verts, rt=np.zeros(3), t=np.array([0., 0., 2.]), f=np.array([w,h])/2., c=np.array([w,h])/2., k=np.zeros(5))
	rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
	rn.set(v=verts, f=faces, bgcolor=np.zeros(3))

	# Construct point light source
	rn.vc = LambertianPointLight(
    	f=rn.f,
    	v=rn.v,
    	num_verts=len(verts),
    	light_pos=np.array([1000, -1000, -2000]),
    	vc=np.ones_like(verts)*.9,
    	light_color=np.array([1., 1., 1.]))

	return rn.r


def save_to_obj(verts, faces, path):
	# Frontal view
	verts[:, 1:3] = -verts[:, 1:3]
	with open(path, 'w') as fp:
		for v in verts:
			fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
		for f in faces + 1:
			fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))