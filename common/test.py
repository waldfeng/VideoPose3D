
from camera import *
import numpy as np
from h36m_dataset import Human36mDataset
from quaternion import *

dataset = Human36mDataset('1.npz')

cam_orientation = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
cam_translation = [1841.1070556640625, 4955.28466796875, 1563.4454345703125]

rot_mat = [[-0.9042074184788829, 0.42657831374650107, 0.020973473936051274], [0.06390493744399675, 0.18368565260974637, -0.9809055713959477], [-0.4222855708380685, -0.8856017859436166, -0.1933503902128034]];

q = rotmat2q( rot_mat )
q_mat = q2rotmat( q )

cam_orientation = np.array(cam_orientation, dtype='float32')
cam_translation = np.array(cam_translation, dtype='float32')

positions = np.random.random((1,1,3)).astype('float32')
print(positions)

pos_3d = world_to_camera(positions, cam_orientation, cam_translation)
print(pos_3d)
pass