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

###################################################################
###Estimate the pose of SVEA bsaed on the ArUco marker detection###
###################################################################

class aruco_pose:

    def __init__(self):
        # Initalize the node 
        rospy.init_node('aruco_pose')

        # Get parameters from launch file
        self.aruco_pose_topic = rospy.get_param("~aruco_pose_topic", "/aruco/detection")
        self.aruco_id = rospy.get_param("~aruco_id", 0)

        # Subscriber
        rospy.Subscriber(self.aruco_pose_topic, ArucoArray, self.aruco_callback, queue_size=1)
        
        # Publisher
        # self.pose_pub = rospy.Publisher("static/pose", PoseWithCovarianceStamped, queue_size=1) #publish to pose0 in ekf
        self.setEKFPose = rospy.Publisher("/set_pose", PoseWithCovarianceStamped, queue_size=1) 
        
        # Variable
        self.gps_msg = None
        self.location = None
        self.frame = 'aruco' + str(self.aruco_id)

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

        # Convaraince for ArUco marker detection
        self.lin_cov = 1e-6
        self.ang_cov = 1e-6

    def run(self):
        rospy.spin()

    def aruco_callback(self, msg):
        for aruco in msg.arucos:
            if aruco.marker.id == self.aruco_id:
                if (aruco.marker.pose.pose.position.x**2 + aruco.marker.pose.pose.position.y**2 + aruco.marker.pose.pose.position.z**2) <= 1.5:
                    self.transform_aruco(aruco.marker)
                
    ## Estimate the pose of SVEA based on the ArUco marker detection
    ## Assume the ArUco marker and the Map frame coincide
    def transform_aruco(self, marker):
        try:
            # frame_id: map, child_frame_id: aruco
            transform_aruco_map = self.buffer.lookup_transform("map", 'aruco0_actual', rospy.Time.now(), rospy.Duration(0.5)) 
            # frame_id: aruco , child_frame_id: base_link
            transform_baselink_aruco = self.buffer.lookup_transform(self.frame, "base_link", marker.header.stamp, rospy.Duration(0.5)) 

            pose_aruco_baselink = PoseStamped()
            pose_aruco_baselink.header = transform_baselink_aruco.header
            pose_aruco_baselink.pose.position = transform_baselink_aruco.transform.translation
            pose_aruco_baselink.pose.orientation = transform_baselink_aruco.transform.rotation
            
            adjust_orientation = TransformStamped()
            adjust_orientation.header = marker.header
            adjust_orientation.header.frame_id = "map"
            adjust_orientation.child_frame_id = self.frame
            adjust_orientation.transform.rotation = Quaternion(*quaternion_from_euler(0,0,0))

            # Adjust the orientation of the ArUco marker since it is not different from the map frame
            position = tf2_geometry_msgs.do_transform_pose(pose_aruco_baselink, adjust_orientation) 

            #frame_id = map child_frame: baselink
            position_final = tf2_geometry_msgs.do_transform_pose(position, transform_aruco_map) 
            position_final.pose.position.z = 0.0

            self.publish_pose(position_final.pose.position, position_final.pose.orientation, marker.header.stamp)

            # Publish the transformation
            self.broadcast_pose(position_final.pose.position, position_final.pose.orientation, marker.header.stamp)

            # rospy.loginfo("Received ARUCO")
        except Exception as e:
            rospy.logerr(e)

    def broadcast_pose(self, translation, quaternion, time):
        msg = TransformStamped()
        msg.header.stamp = time
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.transform.translation = translation
        msg.transform.rotation = quaternion
        self.br.sendTransform(msg)

    def publish_pose(self, translation, quaternion, time):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = time
        msg.header.frame_id = "map"
        msg.pose.pose.position = translation
        msg.pose.pose.orientation = quaternion
        self.cov_matrix = [self.lin_cov, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, self.lin_cov, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, self.lin_cov, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, self.ang_cov, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, self.ang_cov, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, self.ang_cov]
        msg.pose.covariance = self.cov_matrix
        self.setEKFPose.publish(msg)

if __name__ == '__main__':
    aruco_pose().run()