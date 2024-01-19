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
        self.pub_aruco_marker_coordinate = rospy.Publisher('/aruco/2Ddetection', ArucoArray, queue_size=5)
        self.image_pub = rospy.Publisher('/your_modified_image_topic', Image, queue_size=10)
        # rospy.loginfo(self.PUB_ARUCO_POSE)

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
            return
            
        rvecs, tvecs = aruco.estimatePoseSingleMarkers(
            corners,
            self.aruco_size,
            np.array(camera_info.K).reshape((3, 3)),          # camera matrix
            np.array(camera_info.D).reshape((1, 5)),          # camera distortion
        )[:2] # [:2] due to python2/python3 compatibility

        markerArray = MarkerArray()
        CoorArray = ArucoArray()
        CoorArray.header = image.header
        
        ####################################################
        ################ FAKE ARUCO INPUT ##################
        ####################################################

        # tempidlist = [10,11,12,13,14]
        # temp2dlist = [[110.75, 312.75], [321.5, 183.25],[226.75, 300.25],[334.5,295.25],[256.75, 185.25]]
        # temp3dtranslist = [[1.3375, 0.0656, 0.11208],[1.9656, -0.45626, 0.40923],[1.5389166883418355, -0.14251363529411656, 0.10319558148637761],[1.52224, -0.3869, 0.10880],[1.96482, -0.251415, 0.39841]]
        # temp3drotlist = [[0.5830346806744786, -0.46681744013238335, -0.41429800114788345, 0.520104994173221],[0.5268924827774656, -0.4530495126004113, -0.41710572284442976, 0.5857928530594703],[0.3401925386391215, -0.4785159393007294, -0.6300308911726228, 0.5082839842623817],[0.483782002498004, -0.488529528618286, -0.512462885282696, 0.5144663885374339],[0.4838306475100693, -0.40487469023736683, -0.5031715243574452, 0.5905952986698999]]

        # for aruco_id, imagexy, translation, rotation in zip(tempidlist, temp2dlist, temp3dtranslist, temp3drotlist):
        #     aruco_msg = Aruco()
        #     aruco_msg.image_x = imagexy[0]
        #     aruco_msg.image_y = imagexy[1]

        #     ## Publish
        #     marker = Marker()
        #     marker.header = image.header
        #     marker.id = int(aruco_id)
        #     marker.pose.pose.position = Point(*translation)
        #     marker.pose.pose.orientation = Quaternion(*rotation)
        #     marker.confidence = 1 # NOTE: Set this to something more relevant?
        #     aruco_msg.marker = marker

        #     Vmarker = VM()
        #     Vmarker.header = image.header
        #     Vmarker.id = int(aruco_id)
        #     Vmarker.pose.position = Point(*translation)
        #     Vmarker.pose.orientation = Quaternion(*rotation)
        #     Vmarker.scale = Vector3(*[0.1, 0.1, 0.1])
        #     Vmarker.color = ColorRGBA(*[0, 1, 0, 1])
        #     markerArray.markers.append(marker)
        #     CoorArray.arucos.append(aruco_msg)
        #     self.pub_aruco_marker.publish(Vmarker)
        # self.pub_aruco_marker_coordinate.publish(CoorArray)

        for aruco_id, aruco_corner, rvec, tvec in zip(ids, corners, rvecs, tvecs):
            mtx = np.zeros((4, 4))
            mtx[:3, :3] = cv2.Rodrigues(rvec)[0]
            mtx[:3, 3] = tvec
            mtx[3, 3] = 1
            translation = tf_conversions.transformations.translation_from_matrix(mtx)
            rotation = tf_conversions.transformations.quaternion_from_matrix(mtx)

            # if False:
            arucoAnchorList = [0,1,2,3,4,5]
            arucoList = [10,11,12,13,14]
            if aruco_id in arucoList:
                try: 
                    ## Broadcast
                    arucoPose = PoseStamped()
                    arucoPose.header = image.header
                    arucoPose.pose.position = Point(*translation)
                    arucoPose.pose.orientation = Quaternion(*rotation)
                    transform_aruco_map = self.buffer.lookup_transform("map", 'camera', image.header.stamp, rospy.Duration(0.5))  #rospy.Time.now()
                    position = tf2_geometry_msgs.do_transform_pose(arucoPose, transform_aruco_map) 
                    # print("arucoPose", arucoPose)

                    t = TransformStamped()
                    t.header = position.header
                    t.child_frame_id = self.ARUCO_TF_NAME + str(aruco_id[0])
                    t.transform.translation = position.pose.position
                    t.transform.rotation = position.pose.orientation
                    self.br.sendTransform(t)
                    # print("t", t)

                    #get the 2D image coordinate of the aruco marker
                    aruco_msg = Aruco()
                    markerCorners = aruco_corner[0]

                    # aruco_msg.header = image.header
                    aruco_msg.image_x, aruco_msg.image_y = np.mean(markerCorners, axis=0)

                    cv2.circle(gray, (int(aruco_msg.image_x), int(aruco_msg.image_y)), 50, (0,255,0), 2)
                    self.image_pub.publish(CvBridge().cv2_to_imgmsg(gray, "passthrough"))

                    ## Publish  
                    marker = Marker()
                    marker.header = position.header
                    marker.id = int(aruco_id)
                    marker.pose.pose.position = position.pose.position
                    marker.pose.pose.orientation = position.pose.orientation
                    marker.confidence = 1 # NOTE: Set this to something more relevant?
                    aruco_msg.marker = marker
                    aruco_msg.marker.pose.pose.position = Vector3(*[0, 0, 0])
                    aruco_msg.marker.pose.pose.orientation = Quaternion(*[0, 0, 0, 1])
                    
                    Vmarker = VM()
                    Vmarker.header = position.header
                    Vmarker.id = int(aruco_id)
                    Vmarker.pose.position = position.pose.position
                    Vmarker.pose.orientation = position.pose.orientation
                    Vmarker.scale = Vector3(*[0.1, 0.1, 0.1])
                    Vmarker.color = ColorRGBA(*[0, 1, 0, 1])
                    markerArray.markers.append(marker)
                    CoorArray.arucos.append(aruco_msg)
                    self.pub_aruco_marker.publish(Vmarker)
                except Exception as e:
                    # pass
                    rospy.logerr(f"{e}")
                
            elif aruco_id in arucoAnchorList:
                ## Broadcast
                t = TransformStamped()
                t.header = image.header
                t.child_frame_id = self.ARUCO_TF_NAME + str(aruco_id[0])
                t.transform.translation = Point(*translation)
                t.transform.rotation = Quaternion(*rotation)

                self.br.sendTransform(t)

                ##get the 2D image coordinate of the aruco marker
                aruco_msg = Aruco()
                markerCorners = aruco_corner[0]

                # aruco_msg.header = image.header
                aruco_msg.image_x, aruco_msg.image_y = np.mean(markerCorners, axis=0)
                
                ## Publish
                marker = Marker()
                marker.header = image.header
                marker.id = int(aruco_id)
                marker.pose.pose.position = Point(*translation)
                marker.pose.pose.orientation = Quaternion(*rotation)
                marker.confidence = 1 # NOTE: Set this to something more relevant?
                aruco_msg.marker = marker
                
                Vmarker = VM()
                Vmarker.header = image.header
                Vmarker.id = int(aruco_id)
                Vmarker.pose.position = Point(*translation)
                Vmarker.pose.orientation = Quaternion(*rotation)
                Vmarker.scale = Vector3(*[0.1, 0.1, 0.1])
                Vmarker.color = ColorRGBA(*[0, 1, 0, 1])
                markerArray.markers.append(marker)
                CoorArray.arucos.append(aruco_msg)
                self.pub_aruco_marker.publish(Vmarker)

        self.pub_aruco_marker_coordinate.publish(CoorArray)

        # see if need the header from the markerArray
        # markerArray.header = 
        # self.pub_aruco_pose.publish(markerArray)


if __name__ == '__main__':
    ##  Global resources  ##
    bridge = CvBridge()

    ##  Start node  ##
    aruco_detect().run()