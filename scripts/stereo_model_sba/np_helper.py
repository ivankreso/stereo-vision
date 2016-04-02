import numpy as np

# compute inverse of Rt transform using orthogonality of R
def inv_Rt(Rt):
    Rt_inv = np.zeros((4,4))
    R_inv = Rt[0:3,0:3].T
    Rt_inv[0:3,0:3] = R_inv
    Rt_inv[0:3,3] = -R_inv.dot(Rt[0:3,3])
    Rt_inv[3,3] = 1.0
    return Rt_inv
