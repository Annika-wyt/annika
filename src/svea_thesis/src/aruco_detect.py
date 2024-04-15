#!/usr/bin/env python3

import numpy as np
import cv2
from cv2 import aruco

import rospy
import tf2_ros
import tf_conversions
import message_filters as mf
from sensor_msgs.msg import Image, CameraInfo
from aruco_msgs.msg import Marker, MarkerArray
from visualization_msgs.msg import Marker as VM
from geometry_msgs.msg import TransformStamped, Point, Quaternion, Vector3, PoseStamped
from cv_bridge import CvBridge
from svea_msgs.msg import Aruco, ArucoArray
from std_msgs.msg import ColorRGBA
import tf2_geometry_msgs

arucoAnchorList = [0,1,2,3,4,5]
arucoList = [10,11,12,13,14,15]

class aruco_detect:

    def __init__(self):

        ## Initialize node
        rospy.init_node('aruco_detect')

        ## Parameters
        self.SUB_IMAGE = rospy.get_param('~sub_image', '/camera/image_raw')
        self.SUB_CAMERA_INFO = rospy.get_param('~camera_info', '/camera/camera_info')
        
        self.ARUCO_DICT_NAME = rospy.get_param('~aruco_dict', 'DICT_4X4_250')
        self.ARUCO_SIZE = rospy.get_param('~aruco_size', '0.365')
        self.ARUCO_TF_NAME = rospy.get_param('~aruco_tf_name', 'arucoCamera')
        self.PUB_ARUCO_POSE = rospy.get_param('~pub_aruco_pose', '/aruco/pose')

        self.base_link = "svea2"

        ## Aruco
        self.aruco_size = float(self.ARUCO_SIZE)

        dict_name = getattr(aruco, self.ARUCO_DICT_NAME)
        self.aruco_dict = aruco.Dictionary_get(dict_name)

        ## TF2
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

        ## Publishers
        # self.pub_aruco_pose = rospy.Publisher(self.PUB_ARUCO_POSE, MarkerArray, queue_size=5)
        self.pub_aruco_marker = rospy.Publisher('/aruco/marker', VM, queue_size=5)
        self.pub_aruco_marker_coordinate = rospy.Publisher('/aruco/detection', ArucoArray, queue_size=5)
        self.image_pub = rospy.Publisher('/camera/image_detect', Image, queue_size=10)

        ## Subscribers
        ts = mf.TimeSynchronizer([
            mf.Subscriber(self.SUB_IMAGE, Image),
            mf.Subscriber(self.SUB_CAMERA_INFO, CameraInfo),
        ], queue_size=1)
        ts.registerCallback(self.callback)
        rospy.loginfo(self.SUB_IMAGE)

    def run(self):
        rospy.spin()

    def callback(self, image, camera_info):
        # convert to grayscale
        gray = bridge.imgmsg_to_cv2(image, 'mono8')
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict)

        if ids is None:
            CoorArray = ArucoArray()
            CoorArray.header = image.header
            CoorArray.header.frame_id = self.base_link
            self.pub_aruco_marker_coordinate.publish(CoorArray)
            return
            
        rvecs, tvecs = aruco.estimatePoseSingleMarkers(
            corners,
            self.aruco_size,
            np.array(camera_info.K).reshape((3, 3)),          # camera matrix
            np.array(camera_info.D).reshape((1, 5)),          # camera distortion
        )[:2] # [:2] due to python2/python3 compatibility

        markerArray = MarkerArray()
        CoorArray = ArucoArray()

        for aruco_id, aruco_corner, rvec, tvec in zip(ids, corners, rvecs, tvecs):
            mtx = np.zeros((4, 4))
            mtx[:3, :3] = cv2.Rodrigues(rvec)[0]
            mtx[:3, 3] = tvec
            mtx[3, 3] = 1
            translation = tf_conversions.transformations.translation_from_matrix(mtx)
            rotation = tf_conversions.transformations.quaternion_from_matrix(mtx)

            t = TransformStamped()
            t.child_frame_id = self.ARUCO_TF_NAME + str(aruco_id[0])

            if aruco_id in arucoList:
                try: 
                    ## Broadcast
                    arucoPose = PoseStamped()
                    arucoPose.header = image.header
                    arucoPose.pose.position = Point(*translation)
                    arucoPose.pose.orientation = Quaternion(*rotation)
                    transform_aruco_map = self.buffer.lookup_transform(self.base_link, 'camera', image.header.stamp, rospy.Duration(0.5))  #rospy.Time.now()
                    position = tf2_geometry_msgs.do_transform_pose(arucoPose, transform_aruco_map) 
                    # print("arucoPose", arucoPose)

                    t.header = position.header
                    t.transform.translation = position.pose.position
                    t.transform.rotation = position.pose.orientation
                except Exception as e:
                    rospy.logerr(f"{e}")

            elif aruco_id in arucoAnchorList:
                t.header = image.header
                t.transform.translation = Point(*translation)
                t.transform.rotation = Quaternion(*rotation)

            self.br.sendTransform(t)

            #get the 2D image coordinate of the aruco marker
            aruco_msg = Aruco()
            markerCorners = aruco_corner[0]

            # aruco_msg.header = image.header
            aruco_msg.image_x, aruco_msg.image_y = np.mean(markerCorners, axis=0)

            cv2.circle(gray, (int(aruco_msg.image_x), int(aruco_msg.image_y)), 50, (0,255,0), 2)
            self.image_pub.publish(CvBridge().cv2_to_imgmsg(gray, "passthrough"))

            ## Publish  
            marker = Marker()
            marker.header = t.header
            marker.header.seq = image.header.seq
            marker.id = int(aruco_id)
            marker.pose.pose.position = t.transform.translation
            marker.pose.pose.orientation = t.transform.rotation
            marker.confidence = 1 # NOTE: Set this to something more relevant?
            aruco_msg.marker = marker
            
            Vmarker = VM()
            Vmarker.header = t.header
            Vmarker.id = int(aruco_id)
            Vmarker.pose.position = t.transform.translation
            Vmarker.pose.orientation = t.transform.rotation
            Vmarker.scale = Vector3(*[0.1, 0.1, 0.1])
            Vmarker.color = ColorRGBA(*[0, 1, 0, 1])
            markerArray.markers.append(marker)
            self.pub_aruco_marker.publish(Vmarker)
            if aruco_id in arucoList:
                CoorArray.header = t.header
                CoorArray.arucos.append(aruco_msg)
        self.pub_aruco_marker_coordinate.publish(CoorArray)

if __name__ == '__main__':
    ##  Global resources  ##
    bridge = CvBridge()

    ##  Start node  ##
    aruco_detect().run()