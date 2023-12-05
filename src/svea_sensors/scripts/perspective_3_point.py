#! /usr/bin/env python3

import rospy
import tf
import tf2_ros
import tf.transformations as tf_trans
from aruco_msgs.msg import Marker, MarkerArray
from svea_msgs.msg import Aruco, ArucoArray
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, TransformStamped

import numpy as np
import cv2

class perspective_3_point:

    def __init__(self):
        # Initalize the node 
        rospy.init_node('perspective_3_point')

        # Get parameters from launch file

        # Subscriber
        rospy.Subscriber('/aruco/detection', ArucoArray, self.ArucoDetectionCallback, queue_size=1)
        rospy.Subscriber('/camera/camera_info', CameraInfo, self.CameraInfoCallback, queue_size=1)
        rospy.Subscriber('/ekf_estimation/odometry/filtered', Odometry, self.InitialGuessCameraPoseCallback, queue_size=1)
        
        # Publisher
        self.PosePub = rospy.Publisher('/p3p/odometry', Odometry, queue_size=1)

        # for plotting
        # rospy.Subscriber('/qualisys/aruco11/pose', PoseStamped, self.aruco_callback, queue_size=1)
        # rospy.Subscriber('/qualisys/aruco12/pose', PoseStamped, self.aruco_callback, queue_size=1)
        # rospy.Subscriber('/qualisys/aruco13/pose', PoseStamped, self.aruco_callback, queue_size=1)

        # Variable
        self.aruco2D = []
        self.aruco3D = []
        self.arucoId = []
        self.header, self.estimatedRotation, self.estimatedTranslation = None, [], []
        self.RUNNING = False
        # self.DECODEARUCO = False
        self.cameraD, self.cameraK = None, None

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

    def run(self):
        rospy.spin()

    def InitialGuessCameraPoseCallback(self, msg):
        # if self.DECODEARUCO:
        if len(self.estimatedTranslation) == 0:
            self.estimatedTranslation = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=np.float32)
        if len(self.estimatedRotation) == 0:
            self.estimatedRotation, _ = cv2.Rodrigues(tf_trans.quaternion_matrix([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]))
            self.estimatedRotation = self.estimatedRotation.T[0]
            
    def ArucoDetectionCallback(self, msg):
        # self.DECODEARUCO = True
        self.aruco2D = []
        self.aruco3D = []
        self.arucoId = []
        for aruco in msg.arucos:
            self.header = aruco.marker.header
            self.arucoId.append(aruco.marker.id)
            self.aruco2D.append(tuple([aruco.image_x, aruco.image_y]))
            self.aruco3D.append(tuple([aruco.marker.pose.pose.position.x, aruco.marker.pose.pose.position.y, aruco.marker.pose.pose.position.z]))
        if len(self.arucoId) >= 3:
            self.estimation(np.array(self.aruco2D, dtype=np.float32), np.array(self.aruco3D, dtype=np.float32), self.header, self.estimatedRotation, self.estimatedTranslation)
        # self.header, self.estimatedTranslation, self.estimatedRotation = None, [], []

    def CameraInfoCallback(self, msg):
        self.cameraD = np.array(msg.D, dtype=np.float64)
        self.cameraK = np.array(msg.K, dtype=np.float64).reshape((3,3))
    
    def estimation(self, aruco2D, aruco3D, header, estimatedRotation, estimatedTranslation):
        success, rotation, translation = None, None, None
        if len(estimatedTranslation) == 0 or len(estimatedRotation) == 0:
            # self.DECODEARUCO = False
            if len(self.arucoId) >= 4:
                success, rotation, translation = cv2.solvePnP(aruco3D, aruco2D, self.cameraK, self.cameraD, flags=cv2.SOLVEPNP_P3P)
            else:
                rospy.logerr(f"cannot find estimated Translation or Rotation and less than 4 landmarks")
        else:
            success, rotation, translation = cv2.solvePnP(aruco3D, aruco2D, self.cameraK, self.cameraD, rvec=estimatedRotation, tvec=estimatedTranslation, flags=cv2.SOLVEPNP_P3P)
        # rospy.loginfo(f"\n ==================================== \n {success}, \n {rotation}, \n {translation} \n ====================================")
        if success:
            self.publish_pose(rotation, translation, header)
    
    def publish_pose(self, rotationVec, translationVec, header):
        translation = tf_trans.translation_matrix(translationVec.reshape((3,)))
        translation[:3,:3] = cv2.Rodrigues(rotationVec)[0]
        rotation = tf_trans.quaternion_from_matrix(translation)
        msg = Odometry()
        msg.header = header
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position = Point(*translation[:3,-1])
        msg.pose.pose.orientation = Quaternion(*rotation)
        # msg.pose.covariance = self.FillCovaraince()
        msg.pose.covariance = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        self.PosePub.publish(msg)
        msg2 = TransformStamped()
        msg2.header.stamp = header.stamp
        msg2.header.frame_id = "map"
        msg2.child_frame_id = "base_link"
        msg2.transform.translation = Point(*translation[:3,-1])
        msg2.transform.rotation = Quaternion(*rotation)
        self.br.sendTransform(msg2)
    
    def FillCovaraince(self):
        pass

if __name__ == '__main__':
    perspective_3_point().run()