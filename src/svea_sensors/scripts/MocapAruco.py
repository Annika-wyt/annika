#!/usr/bin/env python3

import numpy as np
import cv2
from cv2 import aruco

import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped, Point, Quaternion, Vector3, PoseStamped
from svea_msgs.msg import Aruco, ArucoArray
import tf2_geometry_msgs
from message_filters import Subscriber, ApproximateTimeSynchronizer

ARUCOLIST = [10,11,12,13,14,15]

class MocapAruco:
    def __init__(self):
        # Initialization
        rospy.init_node("MocapAruco")

        # Parameters
        self.debugMode = rospy.get_param('~debugMode', True)

        # Subscriber
        Aruco2Ddetection = Subscriber('/aruco/2Ddetection', ArucoArray)
        # MocapAruco0 = Subscriber('/qualisys/arucoAnchor/pose', PoseStamped)
        MocapAruco10 = Subscriber('/qualisys/aruco10/pose', PoseStamped)
        MocapAruco11 = Subscriber('/qualisys/aruco11/pose', PoseStamped)
        MocapAruco12 = Subscriber('/qualisys/aruco12/pose', PoseStamped)
        MocapAruco13 = Subscriber('/qualisys/aruco13/pose', PoseStamped)
        MocapAruco14 = Subscriber('/qualisys/aruco14/pose', PoseStamped)
        MocapAruco15 = Subscriber('/qualisys/aruco15/pose', PoseStamped)

        sync = ApproximateTimeSynchronizer([Aruco2Ddetection, MocapAruco10, MocapAruco11, MocapAruco12, MocapAruco13, MocapAruco14, MocapAruco15], queue_size=10, slop=5)
        sync.registerCallback(self.ArucoCombine2D3D)

        # Publisher
        self.MapArucoPub = rospy.Publisher('/aruco/detection', ArucoArray, queue_size=5)    

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()
        
    def run(self):
        rospy.spin()

    def ArucoCombine2D3D(self, A2Dmsg, MA10msg, MA11msg, MA12msg, MA13msg, MA14msg, MA15msg):
        for aruco in A2Dmsg.arucos:
            if aruco.marker.id in ARUCOLIST: 
                currentAruco = locals()["MA{}msg".format(aruco.marker.id)]
                try:
                    MapMocapTransform = self.buffer.lookup_transform('map', 'mocap', aruco.marker.header.stamp, rospy.Duration(0.5))
                    MapAruco = tf2_geometry_msgs.do_transform_pose(currentAruco, MapMocapTransform)
                    aruco.marker.pose.pose.position = MapAruco.pose.position
                    aruco.marker.pose.pose.orientation = MapAruco.pose.orientation
                    aruco.marker.pose.covariance = self.FillCovaraince()
                    if self.debugMode:
                        self.debug(aruco.marker)
                        # print(aruco.marker.id)
                        # print(MapAruco)
                except Exception as e:
                    rospy.logerr(f'{e}')
        self.MapArucoPub.publish(A2Dmsg)
            
    def FillCovaraince(self):
        TransCov = np.eye(3,6, dtype=float)*1e-3
        RotCov = np.eye(3,6, k=3, dtype=float)*1e-3*5
        return np.vstack((TransCov, RotCov)).flatten()

    def debug(self, marker):
        #################################################################
        ##################### FOR SHOWING MAP_ARUCO #####################
        #################################################################
        msg2 = TransformStamped()
        msg2.header.stamp = marker.header.stamp
        msg2.header.frame_id = "map"
        msg2.child_frame_id = "MapAruco" + str(marker.id)
        msg2.transform.translation = marker.pose.pose.position
        msg2.transform.rotation = marker.pose.pose.orientation
        self.br.sendTransform(msg2)

if __name__ == '__main__':
    MocapAruco().run()