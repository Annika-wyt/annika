#! /usr/bin/env python3

import rospy
import tf
import tf2_ros
import tf.transformations as tf_trans
from aruco_msgs.msg import Marker, MarkerArray
from svea_msgs.msg import Aruco, ArucoArray
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage

import numpy as np
import cv2

class perspective_3_point:

    def __init__(self):
        # Initalize the node 
        rospy.init_node('perspective_3_point')

        # Get parameters from launch file

        # Subscriber
        # self.SUB_ARUCO_POSE = rospy.get_param('~pub_aruco_pose', '/aruco/pose')
        rospy.Subscriber('/aruco/detection', ArucoArray, self.ArucoDetectionCallback, queue_size=1)
        rospy.Subscriber('/camera/camera_info', CameraInfo, self.CameraInfoCallback, queue_size=1)
        # rospy.Subscriber('/tf', TFMessage, self.InitialGuessCameraPose, queue_size=1)
        
        # Publisher
        
        # for plotting
        # rospy.Subscriber('/qualisys/aruco11/pose', PoseStamped, self.aruco_callback, queue_size=1)
        # rospy.Subscriber('/qualisys/aruco12/pose', PoseStamped, self.aruco_callback, queue_size=1)
        # rospy.Subscriber('/qualisys/aruco13/pose', PoseStamped, self.aruco_callback, queue_size=1)

        # Variable
        self.aruco_2D = []
        self.aruco_3D = []
        self.aruco_id = []

        self.cameraD, self.cameraK = None, None

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

    def run(self):
        rospy.spin()

    def InitialGuessCameraPose(self, timestamp):
        # use wheel encoder, imu, controller input to guess the initial pose of camera at every timestamp
        try:
            transformMapCamera = self.buffer.lookup_transform("map", "camera", timestamp, rospy.Duration(4))
            estimatedRotation = tf_trans.quaternion_matrix([transformMapCamera.transform.rotation.x,
                                                            transformMapCamera.transform.rotation.y,
                                                            transformMapCamera.transform.rotation.z,
                                                            transformMapCamera.transform.rotation.w])[:3, :3]
            estimatedRotation, _ = cv2.Rodrigues(estimatedRotation)
            estimatedTranslation = tf_trans.translation_matrix([transformMapCamera.transform.translation.x, 
                                                                transformMapCamera.transform.translation.y,
                                                                transformMapCamera.transform.translation.z])[:3, -1]
            return estimatedTranslation, estimatedRotation
        except Exception as e:
            rospy.logerr(f"No transform from map to camera, \n {e}")
            return None, None
    
    def ArucoDetectionCallback(self, msg):
        self.aruco_2D = []
        self.aruco_3D = []
        self.aruco_id = []
        for aruco in msg.arucos:
            self.aruco_id.append(aruco.marker.id)
            self.aruco_2D.append([aruco.image_x, aruco.image_y])
            self.aruco_3D.append([aruco.marker.pose.pose.position.x, aruco.marker.pose.pose.position.y, aruco.marker.pose.pose.position.z])
        if len(self.aruco_id) >= 3:
            self.estimation(np.array(self.aruco_2D), np.array(self.aruco_3D), aruco.marker.header.stamp)

    def CameraInfoCallback(self, msg):
        self.cameraD = msg.D
        self.cameraK = msg.K
    
    def estimation(self, aruco2D, aruco3D, timestamp):
        estimatedTranslation, estimatedRotation = self.InitialGuessCameraPose(timestamp)
        success, rotation, translation = None, None, None
        if estimatedTranslation == None or estimatedRotation == None:
            if len(self.aruco_id >= 4):
                success, rotation, translation = cv2.solvePnP(aruco3D, aruco2D, self.cameraK, self.cameraD)
            else:
                rospy.logerr(f"cannot find estimated Translation or Rotation and less than 4 landmarks")
        else:
            options = {'flags': cv2.SOLVEPNP_ITERATIVE, 'useExtrinsicGuess': True}
            success, rotation, translation = cv2.solvePnP(aruco3D, aruco2D, self.cameraK, self.cameraD, rvec=estimatedRotation, tvec=estimatedTranslation, options=options)
        print("====================================")
        print(success)
        print("====================================")
        print(rotation)
        print("====================================")
        print(translation)
if __name__ == '__main__':
    perspective_3_point().run()