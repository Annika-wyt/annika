import numpy as np
import cv2

# coor2d  = np.array([[86.25, 320. ], 
#                    [486.25, 318.75], 
#                    [227.25, 312.  ],
#                    [351.25, 307.75]])

# coor3d = np.array([[1.19704777, 0.40907088, 0.10880016],
#                   [ 1.22920676, -0.23438357,  0.10842846],
#                   [1.30515043, 0.20414246, 0.10620334],
#                   [ 1.33730117, -0.01914946,  0.10970175]])

coor3d = np.array([[ 1.70152698,  0.3761682,   0.09671503],
                   [ 1.80176858,  0.20927306,  0.10202861],
                   [ 1.85041478,  0.02437692,  0.10989203],
                   [ 1.84000608, -0.18252941,  0.11360352]])
coor2d  = np.array([[194.5, 291.25],
                    [265 , 285.5 ],
                    [332 , 281  ],
                    [405.5, 280.25]])

K = np.array([[514.578298266441, 0.0, 340.0718185830948],
         [0.0, 514.8684665452305, 231.4918039429434],
         [0.0, 0.0, 1.0]])
D = np.array([0.06295602826790396, -0.1840231372229633, -0.004945725015870819, 0.01208470957502327, 0.0])

id = [12, 11, 14, 13]
distance3d = []
print(coor2d)
print(coor3d)

# coor2d = np.array([value['2d'] for value in self.arucoDict.values() if '2d' in value])[:,:2]
# coor3d = np.array([value['3d'] for value in self.arucoDict.values() if '3d' in value])
success, rotation_vector, translation_vector = cv2.solvePnP(coor3d, coor2d, K, D, flags=cv2.SOLVEPNP_EPNP)


def normalize_points(points):
    """ Normalize points to improve the accuracy of the solution. """
    mean = np.mean(points, axis=0)
    std_dev = np.std(points)
    if points.shape[1] == 2:  # 2D points
        scale = np.sqrt(2) / std_dev
        T = np.array([[scale, 0, -scale * mean[0]],
                      [0, scale, -scale * mean[1]],
                      [0, 0, 1]])
    elif points.shape[1] == 3:  # 3D points
        scale = np.sqrt(3) / std_dev
        T = np.array([[scale, 0, 0, -scale * mean[0]],
                      [0, scale, 0, -scale * mean[1]],
                      [0, 0, scale, -scale * mean[2]],
                      [0, 0, 0, 1]])
    else:
        raise ValueError("Points dimension not supported")

    points_homogeneous = np.vstack((points.T, np.ones(points.shape[0])))
    points_normalized = np.dot(T, points_homogeneous)
    return points_normalized[0:-1, :].T, T

#epnp
def make_A(p2d, p3d):
    matrixA = []
    # K_inv = np.linalg.inv(CamInfo['K'].reshape(3,3))
    for p_3d, p_2d in zip(p3d, p2d):
        X, Y, Z = p_3d
        x, y = p_2d
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,-1]
        cy = K[1,-2]
        matrixA.append([-X, -Y, -Z, -1, 0, 0, 0, 0, fx*X, fx*Y, fx*Z, fx])
        matrixA.append([0, 0, 0, 0, -X, -Y, -Z, -1, fy*X, fy*Y, fy*Z, fy])    # A = np.vstack(matrixA)

    # h1, h2, h3 = H[:,0], H[:,1], np.cross(H[:,0], H[:,1])
    # norm = np.linalg.norm(np.cross(h1, h2))
    # r1, r2, r3 = h1/norm, h2/norm, h3/norm
    # t = H[:, 2] / norm
    # result = np.column_stack((r1, r2, r3, t))
    # base = [0,0,0,1]
    # result = np.vstack((result, base))
    # # print("result", result)
    return matrixA

# p2d_normalized, T_2d = normalize_points(coor2d)
# # p3d_homogeneous = np.hstack((coor3d, np.ones((coor3d.shape[0], 1))))
# p3d_normalized, T_3d = normalize_points(coor3d)
# A = make_A(p2d_normalized, p3d_normalized[:,:3])

# U, S, Vh = np.linalg.svd(A)
# L = Vh[-1,:].reshape(3,4)
# L = np.dot((T_2d), np.dot(L, T_3d))
# R = L[:, 0:3]
# T = L[:, 3]
# print(R, T)

# def construct_matrix_A(point_3d, point_2d):
#     X, Y, Z = point_3d
#     u, v, _ = point_2d
#     return np.array([
#         [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u],
#         [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
#     ])

# def dlt_pnp(world_points, image_points):
#     A = []
#     for i in range(len(world_points)):
#         A.append(construct_matrix_A(world_points[i], image_points[i]))
#     print(np.array(A).shape)
#     print("=====================")
#     A = np.vstack(A)
#     print(A.shape)
#     U, S, Vh = np.linalg.svd(A)
#     L = Vh[-1,:]
#     H = L.reshape(3, 4)

#     # Decompose H to get rotation and translation
#     h1, h2, h3 = H[:,0], H[:,1], np.cross(H[:,0], H[:,1])
#     norm = np.linalg.norm(np.cross(h1, h2))
#     r1, r2, r3 = h1/norm, h2/norm, h3/norm
#     t = H[:, 2] / norm
#     return np.column_stack((r1, r2, r3, t))

# # Example usage

# pose = dlt_pnp(coor3d, coor2d)
# print("Camera Pose:\n", pose)

