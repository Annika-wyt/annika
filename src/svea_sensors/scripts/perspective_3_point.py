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
from geometry_msgs.msg import Point, Quaternion

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
        self.header, self.estimatedRotation, self.estimatedTranslation = None, None, None
        self.RUNNING = False
        self.DECODEARUCO = False
        self.cameraD, self.cameraK = None, None

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

    def run(self):
        rospy.spin()

    def InitialGuessCameraPoseCallback(self, msg):
        if self.DECODEARUCO and self.estimatedRotation == None and self.estimatedTranslation == None:
            # get initial guess
            rospy.loginfo(f"qua_mat, \n {tf_trans.quaternion_matrix(msg.pose.pose.orientation)}")
            self.estimatedRotation = cv2.Rodrigues(tf_trans.quaternion_matrix(msg.pose.pose.orientation))
            self.estimatedTranslation = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
    
    def ArucoDetectionCallback(self, msg):
        self.DECODEARUCO = True
        self.aruco2D = []
        self.aruco3D = []
        self.arucoId = []
        for aruco in msg.arucos:
            self.arucoId.append(aruco.marker.id)
            self.aruco2D.append(([aruco.image_x, aruco.image_y]))
            self.aruco3D.append([aruco.marker.pose.pose.position.x, aruco.marker.pose.pose.position.y, aruco.marker.pose.pose.position.z])
        if len(self.arucoId) >= 3:
            self.estimation(np.array(self.aruco2D, dtype=np.float64).T, np.array(self.aruco3D, dtype=np.float64).T, self.header, self.estimatedRotation, self.estimatedTranslation)
        self.header, self.estimatedTranslation, self.estimatedRotation = None, None, None
        self.DECODEARUCO = False

    def CameraInfoCallback(self, msg):
        self.cameraD = np.array(msg.D, dtype=np.float64)
        self.cameraK = np.array(msg.K, dtype=np.float64)
    
    def estimation(self, aruco2D, aruco3D, header, estimatedRotation, estimatedTranslation):
        success, rotation, translation = None, None, None
        if estimatedTranslation == None or estimatedRotation == None:
            if len(self.arucoId) >= 4:
                rospy.loginfo("no guess")
                rospy.loginfo(f"aruco3D, aruco2D, {aruco3D}, \n {aruco2D}")
                temp = cv2.solvePnP(aruco3D, aruco2D, self.cameraK, self.cameraD)
                print(temp)
                # success, rotation, translation = cv2.solvePnP(aruco3D, aruco2D, self.cameraK, self.cameraD)
            else:
                rospy.logerr(f"cannot find estimated Translation or Rotation and less than 4 landmarks")
        else:
            rospy.loginfo("Guess")
            options = {'flags': cv2.SOLVEPNP_ITERATIVE, 'useExtrinsicGuess': True}
            success, rotation, translation = cv2.solvePnP(aruco3D.T, aruco2D.T, self.cameraK, self.cameraD, rvec=estimatedRotation, tvec=estimatedTranslation, options=options)
        # print(f"==================================== \n, {success}, \n {rotation}, \n {translation}, \n ====================================")
        # if success:
            # self.publish_pose(rotation, translation, header)

    def publish_pose(self, rotationVec, translationVec, header):
        rotation = tf_trans.quaternion_from_matrix(cv2.Rodrigues(rotationVec))
        translation = tf_trans.translation_matrix(translationVec)
        msg = Odometry()
        msg.header = header
        msg.child_frame_id = "base_link"
        msg.pose.pose.position = Point(*translation)
        msg.pose.pose.orientation = Quaternion(*rotation)
        # msg.pose.covariance = ##something meaningful???
        self.PosePub(msg)

if __name__ == '__main__':
    perspective_3_point().run()