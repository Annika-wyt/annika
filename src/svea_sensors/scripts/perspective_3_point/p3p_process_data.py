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
from geometry_msgs.msg import Point, Quaternion, TransformStamped, PoseStamped
import tf2_geometry_msgs

from message_filters import Subscriber, ApproximateTimeSynchronizer

from p3p import p3p

import numpy as np
import cv2

class p3p_process_data:

    def __init__(self):
        # Initalize the node 
        rospy.init_node('p3p_process_data')

        # Get parameters from launch file

        # Subscriber
        ArucoDetection = Subscriber('/aruco/detection', ArucoArray)
        CameraInformaton = Subscriber('/camera/camera_info', CameraInfo)
        InitialGuessCameraPose = Subscriber('/ekf_estimation/odometry/filtered', Odometry)

        # sync = ApproximateTimeSynchronizer([ArucoDetection, CameraInformaton, InitialGuessCameraPose], queue_size=10, slop=0.1)
        sync = ApproximateTimeSynchronizer([ArucoDetection, CameraInformaton], queue_size=10, slop=0.1)
        sync.registerCallback(self.ArucoDectAndCameraAndInitialGuessCallback)
        
        # Publisher
        self.PosePub = rospy.Publisher('/p3p/odometry', Odometry, queue_size=1)

        self.solvePNP = p3p()

        # Variable
        self.arucoDict = {}
        self.cameraDict = {'D': None, 'K': None}
        self.estimateDict = {'Header': None, 'Translation': None, 'Rotation': None}

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

    def run(self):
        rospy.spin()

    # def ArucoDectAndCameraAndInitialGuessCallback(self, ArucoMsg, CameraMsg, PoseMsg):
    def ArucoDectAndCameraAndInitialGuessCallback(self, ArucoMsg, CameraMsg):
        # Get Aruco
        self.arucoDict = {}
        for aruco in ArucoMsg.arucos:
            self.header = aruco.marker.header
            self.arucoDict[aruco.marker.id] = {
                'id': aruco.marker.id,
                '2d': np.array([aruco.image_x, aruco.image_y]),
                '3d': np.array([aruco.marker.pose.pose.position.x, aruco.marker.pose.pose.position.y, aruco.marker.pose.pose.position.z])}

        # Get Camera Info
        self.cameraDict['D'] = np.array(CameraMsg.D)
        self.cameraDict['K'] = np.array(CameraMsg.K)
        # Get Initial Guess
        # try:
        #     OdomMapTrans = self.buffer.lookup_transform("map", 'odom', PoseMsg.header.stamp, rospy.Duration(0.5))
        #     poseEKF = PoseStamped()
        #     poseEKF.header = PoseMsg.header
        #     poseEKF.pose.position = PoseMsg.pose.pose.position
        #     poseEKF.pose.orientation = PoseMsg.pose.pose.orientation
        #     position = tf2_geometry_msgs.do_transform_pose(poseEKF, OdomMapTrans) 
            
        #     self.estimateDict['Header'] = PoseMsg.header
        #     self.estimateDict['Translation'] = np.array([position.pose.position.x, position.pose.position.y, position.pose.position.z], dtype=np.float32)
        #     self.estimateDict['Rotation'] = cv2.Rodrigues(tf_trans.quaternion_matrix([position.pose.orientation.x, position.pose.orientation.y, position.pose.orientation.z, position.pose.orientation.w]))[0].T[0]
        
        # except Exception as e:
        #     rospy.logerr(f"{e}")
        
        # Run P3P
        # result, success = 
        # print("going to p3p estimation")
        
        coor2d = np.array([value['2d'] for value in self.arucoDict.values() if '2d' in value], dtype=np.float32)
        coor3d = np.array([value['3d'] for value in self.arucoDict.values() if '3d' in value], dtype=np.float32)

        cam_matrix = np.array(self.cameraDict['K'], dtype=np.float32).reshape(3,3)
        dis_matrix = np.array(self.cameraDict['D'], dtype=np.float32)
        if len(coor3d) >= 4:
            success, rotation_vector, translation_vector = cv2.solvePnP(coor3d, coor2d, cam_matrix, dis_matrix, flags=cv2.SOLVEPNP_EPNP)
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            # result = self.solvePNP.estimation(self.cameraDict, self.arucoDict, self.estimateDict)
            result = np.hstack((rotation_matrix, translation_vector))
            result = np.vstack((result, [0,0,0,1]))
            self.publish_pose(result, self.header)
            # if success:
                # self.publish_pose(success)

    def publish_pose(self, result, header):
        translation = result[:-1,-1]
        rotation = tf_trans.quaternion_from_matrix(result)
        rotation = rotation/np.linalg.norm(rotation)

        temppose = PoseStamped()
        temppose.header = header
        temppose.pose.position = Point(*translation)
        temppose.pose.orientation = Quaternion(*rotation)

        transform_aruco_map = self.buffer.lookup_transform("base_link", 'camera', header.stamp, rospy.Duration(0.5))  #rospy.Time.now()
        position = tf2_geometry_msgs.do_transform_pose(temppose, transform_aruco_map) 


        msg = Odometry()
        msg.header = header
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link" #should be the est of cam
        # msg.child_frame_id = "camera"
        msg.pose.pose.position = position.pose.position #Point(*translation)
        msg.pose.pose.orientation = position.pose.orientation #Quaternion(*rotation)
        # msg.pose.covariance = self.FillCovaraince()
        msg.pose.covariance = [1e-3, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 1e-3, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 1e-3, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 1e-3, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 1e-3, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 1e-3]
        self.PosePub.publish(msg)


        msg2 = TransformStamped()
        msg2.header.stamp = header.stamp
        msg2.header.frame_id = "map"
        msg2.child_frame_id = "base_link_est" #should be the est of cam
        # msg2.transform.translation = Point(*[0,0,0])
        msg2.transform.translation = position.pose.position #Point(*translation)
        msg2.transform.rotation = position.pose.orientation #Quaternion(*rotation)
        # rospy.loginfo("SENDING NEW POSE ODOM BASELINK FROM P3P")
        print("translation", translation)
        print("rotation", rotation)
        # self.br.sendTransform(msg2)
    
    def FillCovaraince(self):
        pass

if __name__ == '__main__':
    p3p_process_data().run()