#! /usr/bin/env python3

import rospy
import tf
import tf2_ros
import tf.transformations as tf_trans
import tf2_geometry_msgs

import numpy as np
import cv2


class p3p:
    def __init__(self):
        pass

    def estimation(self, CamInfo, ArucoInfo, EstimateInfo, method="grunet"):
        distance3d = []
        uniVecs = []
        angles = []

        coor2d = [value['2d'] for value in ArucoInfo.values() if '2d' in value]
        coor3d = [value['3d'] for value in ArucoInfo.values() if '3d' in value]
        id = [value['id'] for value in ArucoInfo.values() if 'id' in value]
        # calculate distance between points
        for ind, coor in enumerate(coor3d):
            distance3d.append(np.linalg.norm(coor - coor3d[(ind+1)% len(coor3d)]))

        for ind, coor in enumerate(coor2d):
            uniVecs.append(coor/np.linalg.norm(coor))

        for ind, uniVec in enumerate(uniVecs):
            angles.append(np.arccos(np.dot(uniVec, uniVecs[(ind+1)%len(uniVecs)])))

        # self.grunert(distance3d, angles)
        matrixA = []
        K_inv = np.linalg.inv(CamInfo['K'].reshape(3,3))
        for point3d, point2d in zip(coor3d, coor2d):
            p2_normalized = K_inv.dot(point2d)  # Convert to normalized coordinates
            u_normalized, v_normalized = p2_normalized[:2] / p2_normalized[2]
            matrixA.append(np.array([[point3d[0], point3d[1], point3d[2], 1, 0, 0, 0, 0, -point2d[0]*point3d[0], -point2d[0]*point3d[1], -point2d[0]*point3d[2], -point2d[0]],
                                     [0, 0, 0, 0, point3d[0], point3d[1], point3d[2], 1,-point2d[1]*point3d[0], -point2d[1]*point3d[1], -point2d[1]*point3d[2], -point2d[1]]]))

        A = np.vstack(matrixA)
        U, S, Vh = np.linalg.svd(A)
        L = Vh[-1,:]
        H = L.reshape(3,4)

        h1, h2, h3 = H[:,0], H[:,1], np.cross(H[:,0], H[:,1])
        norm = np.linalg.norm(np.cross(h1, h2))
        r1, r2, r3 = h1/norm, h2/norm, h3/norm
        t = H[:, 2] / norm
        result = np.column_stack((r1, r2, r3, t))
        base = [0,0,0,1]
        result = np.vstack((result, base))
        # print("result", result)
        return result

    def grunert(self, distance3d, angles): # incorrect update
        for ind, CoorAndAngle in enumerate(zip(distance3d, angles)):
            us = []
            indN = (ind+1)%len(angles)
            indNN = (ind+2)%len(angles)
            print("**********************************************************************************")
            print("ind", ind, "ARUCO ID", id[ind], id[indN], id[indNN])


            a = distance3d[ind]
            b = distance3d[indN]
            c = distance3d[indNN]
            alpha = angles[ind]
            beta = angles[indN]
            gamma = angles[indNN]

            constMinus = (a**2 - c**2)/(b**2)
            constPlus = (a**2 + c**2)/(b**2)
            a4 = (constMinus-1)**2 - (4*c**2/b**2)*(np.cos(alpha)**2)
            a3 = 4*(constMinus*(a-constMinus)*np.cos(beta) - (1- constPlus)*np.cos(alpha)*np.cos(gamma) + 2*(c**2)/(b**2)*(np.cos(alpha)**2)*np.cos(beta))
            a2 = 2*((constMinus**2) - 1 + 2*(constMinus**2)*(np.cos(beta)**2) + 2*(((b**2-c**2)/(b**2))*(np.cos(alpha)**2)) - 4*(constPlus)*np.cos(alpha)*np.cos(beta)*np.cos(gamma) + 2*(((b**2-a**2)/(b**2))*(np.cos(gamma)**2)))
            a1 = 4*(-(constMinus)*(1 + constMinus)*np.cos(beta) + 2*((a**2)/(b**2))*(np.cos(gamma)**2)*np.cos(beta) - (1 - constPlus)*np.cos(alpha)*np.cos(gamma))
            a0 = (1 + constMinus)**2 - 4*(a**2)/(b**2)*(np.cos(gamma)**2)

            # constMinus = ((distance3d[ind]**2 - distance3d[indNN]**2)/distance3d[indN]**2) #(a^2 - c^2)/b^2
            # constPlus = ((distance3d[ind]**2 + distance3d[indNN]**2)/distance3d[indN]**2) #(a^2 + c^2)/b^2

            # a4 = (constMinus - 1)**2 - 4*(distance3d[indNN]**2/distance3d[indN]**2)*(np.cos(angles[ind])**2)
            # a3 = 4*(constMinus*(1-constMinus)*np.cos(angles[indN]) - (1-constPlus)*np.cos(angles[ind])*np.cos(angles[indNN]) + 2*((distance3d[indNN])**2/(distance3d[indN])**2)*((np.cos(angles[ind]))**2)*np.cos(angles[indN]))
            # a2 = 2*((constMinus)**2-1+2*(constMinus**2)*np.cos(angles[indN])**2 + 2*((distance3d[indN]**2 - distance3d[indNN]**2)/distance3d[indN]**2)*np.cos(angles[ind]**2)-4*(constPlus)*np.cos(angles[ind])*np.cos(angles[indN])*np.cos(angles[indNN])+2*((distance3d[indN]**2-distance3d[ind]**2)/distance3d[indN]**2)*np.cos(angles[indNN])**2)
            # a1 = 4*(-constMinus*(1+constMinus)*np.cos(angles[indN]) + 2*distance3d[ind]**2/distance3d[indN]**2*np.cos(angles[indNN])**2*np.cos(angles[indN]) - (1-constPlus)*np.cos(angles[ind])*angles[indNN])
            # a0 = (1 + constMinus)**2 - 4*(distance3d[ind]**2/distance3d[indN]**2)*(np.cos(angles[indNN])**2)

            roots = np.roots([a4, a3, a2, a1, a0])
            print("ROOTS", roots)
            vs = roots[np.isreal(roots)].real
            print("REAL ROOTS", vs)
            vs = vs[vs>0]
            for v in vs:
                num = (-1 + constMinus)*(v**2) - 2*(constMinus)*np.cos(beta*v) + 1 + constMinus
                den = 2*(np.cos(gamma) - v*np.cos(alpha))
                # num = (-1+constMinus)*root**2 - 2*(constMinus)*np.cos(angles[indN])*root + 1 + constMinus
                # den = 2*(np.cos(angles[indNN]) - root*angles[ind])
                us.append(num/den)

            for u, v in zip(us, vs):
                s1 = (c**2)/(1+(u**2)-2*u*np.cos(gamma))
                s2 = (b**2)/(1 + (v**2) - 2*v*np.cos(beta))
                s3 = (a**2)/((u**2) + (v**2) - 2*u*v*np.cos(alpha))
                # s1 = (distance3d[indNN]**2)/(1 + uv[0]**2 - 2* uv[0] * np.cos(angles[indNN]))
                # s2 = (distance3d[indN]**2)/(1 + uv[1]**2 - 2* uv[1]* np.cos(angles[indN]))
                # s3 = (distance3d[ind]**2)/(uv[0]**2 + uv[1]**2 - 2*uv[0]*uv[1]*np.cos(angles[ind]))

                print("=========================================")
                print(s1)
                print("++++++++++++++++++++++++++++++")
                print(s2)
                print("++++++++++++++++++++++++++++++")
                print(s3)
                print("=========================================")
            print("**********************************************************************************")

