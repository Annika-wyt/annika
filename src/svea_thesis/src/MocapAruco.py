#!/usr/bin/env python3

import numpy as np
import cv2
from cv2 import aruco

import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped, Point, Quaternion, Vector3, PoseStamped
from svea_thesis.msg import Aruco, ArucoArray
import tf2_geometry_msgs
from message_filters import Subscriber, ApproximateTimeSynchronizer

ARUCOLIST = [10,11,12,13,14,15]

class MocapAruco:
    def __init__(self):
        # Initialization
        rospy.init_node("MocapAruco")

        # Parameters
        self.debugMode = rospy.get_param('~debugMode', True)

        self.header = None
        self.seq = 0
        self.arucoMsg = {
            "10" : None,
            "11" : None,
            "12" : None,
            "13" : None,
            "14" : None,
            "15" : None
        }

        self.publishing = False

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

        # Subscriber
        MocapAruco10 = rospy.Subscriber('/qualisys/aruco10/pose', PoseStamped, self.arucoCallback, callback_args="10")
        MocapAruco11 = rospy.Subscriber('/qualisys/aruco11/pose', PoseStamped, self.arucoCallback, callback_args="11")
        MocapAruco12 = rospy.Subscriber('/qualisys/aruco12/pose', PoseStamped, self.arucoCallback, callback_args="12")
        MocapAruco13 = rospy.Subscriber('/qualisys/aruco13/pose', PoseStamped, self.arucoCallback, callback_args="13")
        MocapAruco14 = rospy.Subscriber('/qualisys/aruco14/pose', PoseStamped, self.arucoCallback, callback_args="14")
        MocapAruco15 = rospy.Subscriber('/qualisys/aruco15/pose', PoseStamped, self.arucoCallback, callback_args="15")
        # sync = ApproximateTimeSynchronizer([MocapAruco10, MocapAruco11, MocapAruco12, MocapAruco13, MocapAruco14, MocapAruco15], queue_size=1, slop=2)
        # sync.registerCallback(self.GroundtruthCallback)

        # Publisher
        self.MapArucoPub = rospy.Publisher('/aruco/detection/Groundtruth', ArucoArray, queue_size=1)    
        
    def run(self):
        while not rospy.is_shutdown():
            self.publishAruco()
            rospy.sleep(0.05)

    def publishAruco(self):
        self.publishing = True
        arucoarray = ArucoArray()
        for topic_name, position in self.arucoMsg.items():
            if position != None:
                ArucoGroundtruth = Aruco()
                ArucoGroundtruth.marker.header = self.header
                ArucoGroundtruth.marker.id = int(topic_name)
                ArucoGroundtruth.marker.pose.pose.position = Point(*position[:3])
                ArucoGroundtruth.marker.pose.pose.orientation = Quaternion(*position[3:])
                ArucoGroundtruth.marker.pose.covariance = self.FillCovaraince()
                if self.debugMode:
                    self.debug(ArucoGroundtruth.marker)
                    # print(aruco.marker.id)
                    # print(MapAruco)
                arucoarray.arucos.append(ArucoGroundtruth)
                self.arucoMsg[topic_name] = None
            if self.header != None:
                arucoarray.header = self.header
            else:
                arucoarray.header.stamp = rospy.Time.now()
            arucoarray.header.seq = self.seq
            self.seq += 1
            arucoarray.header.frame_id = "map"
            self.MapArucoPub.publish(arucoarray)
        self.publishing = False

    def arucoCallback(self, msg, topic_name):
        if not self.publishing:
            try:
                MapMocapTransform = self.buffer.lookup_transform('map', 'mocap', rospy.Time(), rospy.Duration(0.5))
                MapAruco = tf2_geometry_msgs.do_transform_pose(msg, MapMocapTransform)
                self.header = msg.header
                self.arucoMsg[topic_name] = [MapAruco.pose.position.x, MapAruco.pose.position.y, MapAruco.pose.position.z, MapAruco.pose.orientation.x, MapAruco.pose.orientation.y, MapAruco.pose.orientation.z, MapAruco.pose.orientation.w]
            except Exception as e:
                rospy.logerr(f'{e}')

    def GroundtruthCallback(self, MA10msg, MA11msg, MA12msg, MA13msg, MA14msg, MA15msg):
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