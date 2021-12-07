# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from numpy.core.defchararray import equal
from numpy.lib.twodim_base import eye
import torch
import numpy as np

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))
    
    
def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)

def crossMul(in_v): # output is list
    out = [[0,-in_v[2],in_v[1]],[in_v[2],0,-in_v[0]],[-in_v[1],in_v[0],0]]
    return out

def q2rotmat(q): # output is list
    q = vecNormalize(q)
    
    Qw = q[0];
    Qv = q[1:4];
    Qv_cross_mat = crossMul(Qv);
    Qv_neg = (-1.0*np.array(Qv)).tolist()

    leftMat =  Qw*np.eye(4) + np.array([[0]+Qv_neg, [Qv[0]]+Qv_cross_mat[0],    [Qv[1]]+Qv_cross_mat[1],      [Qv[2]]+Qv_cross_mat[2]]);
    rightMat = Qw*np.eye(4) + np.array([[0]+Qv,     [Qv_neg[0]]+Qv_cross_mat[0], [Qv_neg[1]]+Qv_cross_mat[1], [Qv_neg[2]]+Qv_cross_mat[2]]);
    mat = leftMat.dot( rightMat );
    rotMat = mat[1:4,1:4].tolist();
    return rotMat

def rotmat2q( mat ):
    q_1 = ((mat[0][0] + mat[1][1] + mat[2][2] + 1)/4)**0.5;
    q_2 = (mat[2][1] - mat[1][2])/(4*q_1);
    q_3 = (mat[0][2] - mat[2][0])/(4*q_1);
    q_4 = (mat[1][0] - mat[0][1])/(4*q_1);
    q = vecNormalize( [q_1,q_2,q_3,q_4] );
    
    return q;

def vecNormalize( vec ): #vector must be row vector
    if type(vec) != np.ndarray:
        vec = vec / ((np.array(vec).dot(np.array(vec).T))**0.5)
    else:
        vec = vec / ((vec.dot(vec.T))**0.5)
    return vec.tolist()