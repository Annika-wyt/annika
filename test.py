import numpy as np


coor2d  = np.array([[ 86.25, 320.  ,   1.  ], 
                   [486.25, 318.75,   1.  ], 
                   [227.25, 312.  ,   1.  ],
                   [351.25, 307.75,   1.  ]])

coor3d = np.array([[1.19704777, 0.40907088, 0.10880016],
                  [ 1.22920676, -0.23438357,  0.10842846],
                  [1.30515043, 0.20414246, 0.10620334],
                  [ 1.33730117, -0.01914946,  0.10970175]])

id = [12, 11, 14, 13]
distance3d = []

def construct_matrix_A(point_3d, point_2d):
    X, Y, Z = point_3d
    u, v, _ = point_2d
    return np.array([
        [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u],
        [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
    ])

def dlt_pnp(world_points, image_points):
    A = []
    for i in range(len(world_points)):
        A.append(construct_matrix_A(world_points[i], image_points[i]))
    print(np.array(A).shape)
    print("=====================")
    A = np.vstack(A)
    print(A.shape)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:]
    H = L.reshape(3, 4)

    # Decompose H to get rotation and translation
    h1, h2, h3 = H[:,0], H[:,1], np.cross(H[:,0], H[:,1])
    norm = np.linalg.norm(np.cross(h1, h2))
    r1, r2, r3 = h1/norm, h2/norm, h3/norm
    t = H[:, 2] / norm
    return np.column_stack((r1, r2, r3, t))

# Example usage

pose = dlt_pnp(coor3d, coor2d)
print("Camera Pose:\n", pose)
