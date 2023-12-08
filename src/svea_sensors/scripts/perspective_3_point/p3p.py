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

    def estimation(CamInfo, ArucoInfo, EstimateInfo, method="grunet"):
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
            angles.append(np.cos(np.dot(uniVec, uniVecs[(ind+1)%len(uniVecs)])))

        print("================================")
        for ind, CoorAndAngle in enumerate(zip(distance3d, angles)):
            u = []
            indN = (ind+1)%len(angles)
            indNN = (ind+2)%len(angles)
            print("ARUCO ID", id[ind], id[indN], id[indNN])

            constMinus = ((distance3d[ind]**2 - distance3d[indNN]**2)/distance3d[indN]**2) #(a^2 - c^2)/b^2
            constPlus = ((distance3d[ind]**2 + distance3d[indNN]**2)/distance3d[indN]**2) #(a^2 + c^2)/b^2

            a4 = (constMinus - 1)**2 - 4*(distance3d[indNN]**2/distance3d[indN]**2)*(np.cos(angles[ind])**2)
            a3 = 4*(constMinus*(1-constMinus)*np.cos(angles[indN]) - (1-constPlus)*np.cos(angles[ind])*np.cos(angles[indNN]) + 2*((distance3d[indNN])**2/(distance3d[indN])**2)*((np.cos(angles[ind]))**2)*np.cos(angles[indN]))
            a2 = 2*((constMinus)**2-1+2*(constMinus**2)*np.cos(angles[indN])**2 + 2*((distance3d[indN]**2 - distance3d[indNN]**2)/distance3d[indN]**2)*np.cos(angles[ind]**2)-4*(constPlus)*np.cos(angles[ind])*np.cos(angles[indN])*np.cos(angles[indNN])+2*((distance3d[indN]**2-distance3d[ind]**2)/distance3d[indN]**2)*np.cos(angles[indNN])**2)
            a1 = 4*(-constMinus*(1+constMinus)*np.cos(angles[indN]) + 2*distance3d[ind]**2/distance3d[indN]**2*np.cos(angles[indNN])**2*np.cos(angles[indN]) - (1-constPlus)*np.cos(angles[ind])*angles[indNN])
            a0 = (1 + constMinus)**2 - 4*(distance3d[ind]**2/distance3d[indN]**2)*(np.cos(angles[indNN])**2)
            # v
            v = np.roots([a4, a3, a2, a1, a0])
        
            for root in v:
                num = (-1+constMinus)*root**2 - 2*(constMinus)*np.cos(angles[indN])*root + 1 + constMinus
                den = 2*(np.cos(angles[indNN]) - root*angles[ind])
                u.append(num/den)
            # print("roots", roots)
            # print("u", u)
            for uv in zip(u, v):
                s1 = (distance3d[indNN]**2)/(1 + uv[0]**2 - 2* uv[0] * np.cos(angles[indNN]))
                s2 = (distance3d[indN]**2)/(1 + uv[1]**2 - 2* uv[1]* np.cos(angles[indN]))
                s3 = (distance3d[ind]**2)/(uv[0]**2 + uv[1]**2 - 2*uv[0]*uv[1]*np.cos(angles[ind]))
                print(s1)
                print("+++++++++++++++++++++++++++++")
                print(s2)
                print("+++++++++++++++++++++++++++++")
                print(s3)
                break
            print("==================================")
            break