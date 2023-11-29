#! /usr/bin/env python3

# import numpy as np

import rospy

import tf
import tf2_ros
import tf2_geometry_msgs
# from tf import transformations 
from tf.transformations import quaternion_from_euler #euler_from_quaternion, 

# from tf2_msgs.msg import TFMessage
# from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Quaternion, PoseStamped #Point, 
from aruco_msgs.msg import Marker, MarkerArray
from svea_msgs.msg import Aruco, ArucoArray


class perspective_3_point:

    def __init__(self):
        # Initalize the node 
        rospy.init_node('perspective_3_point')

        # Get parameters from launch file

        # Subscriber
        # self.SUB_ARUCO_POSE = rospy.get_param('~pub_aruco_pose', '/aruco/pose')
        rospy.Subscriber('/aruco/detection', ArucoArray, self.ArucoDetectionCallback, queue_size=1)
        
        # Publisher
        
        # for plotting
        # rospy.Subscriber('/qualisys/aruco11/pose', PoseStamped, self.aruco_callback, queue_size=1)
        # rospy.Subscriber('/qualisys/aruco12/pose', PoseStamped, self.aruco_callback, queue_size=1)
        # rospy.Subscriber('/qualisys/aruco13/pose', PoseStamped, self.aruco_callback, queue_size=1)

        # Variable
        self.aruco_2D = []
        self.aruco_3D = []
        self.aruco_id = []

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

    def run(self):
        rospy.spin()

    def ArucoDetectionCallback(self, msg):
        for aruco in msg.arucos:
            if aruco.marker.id in self.aruco_id:
                ind = self.aruco_id.index(aruco.marker.id)
                self.aruco_2D[ind] = [aruco.image_x, aruco.image_y]
                self.aruco_3D[ind] = [aruco.marker.pose.pose.position.x, aruco.marker.pose.pose.position.y, aruco.marker.pose.pose.position.z]
            else:
                self.aruco_id.append(aruco.marker.id)
                self.aruco_2D.append([aruco.image_x, aruco.image_y])
                self.aruco_3D.append([aruco.marker.pose.pose.position.x, aruco.marker.pose.pose.position.y, aruco.marker.pose.pose.position.z])
        

if __name__ == '__main__':
    perspective_3_point().run()