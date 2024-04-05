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

ARUCOLIST = [10,11,12,13,15]

class MocapAruco:
    def __init__(self):
        # Initialization
        rospy.init_node("MocapAruco")

        # Parameters
        self.debugMode = rospy.get_param('~debugMode', True)

        # Subscriber
        MocapAruco10 = Subscriber('/qualisys/aruco10/pose', PoseStamped)
        MocapAruco11 = Subscriber('/qualisys/aruco11/pose', PoseStamped)
        MocapAruco12 = Subscriber('/qualisys/aruco12/pose', PoseStamped)
        MocapAruco13 = Subscriber('/qualisys/aruco13/pose', PoseStamped)
        MocapAruco14 = Subscriber('/qualisys/aruco14/pose', PoseStamped)
        MocapAruco15 = Subscriber('/qualisys/aruco15/pose', PoseStamped)
        sync = ApproximateTimeSynchronizer([MocapAruco10, MocapAruco11, MocapAruco12, MocapAruco13, MocapAruco15], queue_size=1, slop=2)
        sync.registerCallback(self.GroundtruthCallback)

        # Publisher
        self.MapArucoPub = rospy.Publisher('/aruco/detection/Groundtruth', ArucoArray, queue_size=1)    

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()
        
    def run(self):
        rospy.spin()

    def GroundtruthCallback(self, MA10msg, MA11msg, MA12msg, MA13msg, MA15msg):
        arucoarray = ArucoArray()
        for marker in ARUCOLIST: 
            ArucoGroundtruth = Aruco()
            currentAruco = locals()["MA{}msg".format(marker)]
            try:
                MapMocapTransform = self.buffer.lookup_transform('map', 'mocap', currentAruco.header.stamp, rospy.Duration(0.5))
                MapAruco = tf2_geometry_msgs.do_transform_pose(currentAruco, MapMocapTransform)
                ArucoGroundtruth.marker.header = MapAruco.header
                ArucoGroundtruth.marker.header.seq = currentAruco.header.seq
                ArucoGroundtruth.marker.id = marker
                ArucoGroundtruth.marker.pose.pose.position = MapAruco.pose.position
                ArucoGroundtruth.marker.pose.pose.orientation = MapAruco.pose.orientation
                ArucoGroundtruth.marker.pose.covariance = self.FillCovaraince()
                if self.debugMode:
                    self.debug(ArucoGroundtruth.marker)
                    # print(aruco.marker.id)
                    # print(MapAruco)
            except Exception as e:
                rospy.logerr(f'{e}')
            arucoarray.arucos.append(ArucoGroundtruth)
        arucoarray.header = currentAruco.header
        arucoarray.header.frame_id = "map"
        self.MapArucoPub.publish(arucoarray)
            
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