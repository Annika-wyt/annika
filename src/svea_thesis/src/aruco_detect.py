#!/usr/bin/env python3
import psutil
import os

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
from geometry_msgs.msg import TransformStamped, Point, Quaternion, Vector3, PoseStamped, Vector3Stamped
from cv_bridge import CvBridge
from svea_msgs.msg import Aruco, ArucoArray
from std_msgs.msg import ColorRGBA
import tf2_geometry_msgs

arucoAnchorList = [0,1,2,3,4,5]
arucoList = [10,11,12,13,14,15,16,17,18,19,20,21]

class aruco_detect:

    def __init__(self):

        ## Initialize node
        rospy.init_node('aruco_detect')

        ## Parameters
        self.SUB_IMAGE = rospy.get_param('~sub_image', '/camera/image_raw')
        self.SUB_CAMERA_INFO = rospy.get_param('~camera_info', '/camera/camera_info')
        
        self.ARUCO_DICT_NAME = rospy.get_param('~aruco_dict', 'DICT_4X4_250')
        self.ARUCO_SIZE = rospy.get_param('~aruco_size', '0.1')
        self.ARUCO_TF_NAME = rospy.get_param('~aruco_tf_name', 'arucoCamera')
        self.PUB_ARUCO_POSE = rospy.get_param('~pub_aruco_pose', '/aruco/pose')
        
        self.debug = True
        
        self.base_link = "svea5"

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

        ## Static Transform
        self.transform_aruco_base = None
        while self.transform_aruco_base == None and not rospy.is_shutdown():
            self.GetStaticTransform()

        ## Subscribers
        ts = mf.ApproximateTimeSynchronizer([
            mf.Subscriber(self.SUB_IMAGE, Image),
            mf.Subscriber(self.SUB_CAMERA_INFO, CameraInfo),
        ], queue_size=1, slop=0.5)
        ts.registerCallback(self.callback)
        rospy.loginfo(self.SUB_IMAGE)


    def GetStaticTransform(self):
        try: 
            self.transform_aruco_base = self.buffer.lookup_transform(self.base_link, 'camera', rospy.Time(), rospy.Duration(0.5))
        except Exception as e:
            print(f"/aruco_detect/GetStaticTransform: {e}")

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
                    position = tf2_geometry_msgs.do_transform_pose(arucoPose, self.transform_aruco_base) 
                    
                    # undistorted_points = cv2.undistortPoints(np.array([[camera_x, camera_y]]), np.array(camera_info.K).reshape((3, 3)), np.array(camera_info.D).reshape((1, 5)), R=np.array(camera_info.R).reshape((3, 3)), P=np.array(camera_info.P).reshape((3, 4)))
                    

                    t.header = image.header
                    t.header.frame_id = "camera" #self.base_link
                    t.transform.translation = Point(*translation) #position.pose.position
                    t.transform.rotation =    Quaternion(*rotation) #position.pose.orientation
                except Exception as e:
                    rospy.logerr(f"{e}")

            elif aruco_id in arucoAnchorList:
                t.header = image.header
                t.transform.translation = Point(*translation)
                t.transform.rotation =    Quaternion(*rotation)
            
            else:
                continue

            self.br.sendTransform(t)
            markerCorners = aruco_corner[0]
            center_x, center_y = np.mean(markerCorners, axis=0)
            fx, fy = camera_info.K[0], camera_info.K[4]
            cx, cy = camera_info.K[2], camera_info.K[5]

            # fx, fy = camera_info.P[0], camera_info.P[5]
            # cx, cy = camera_info.P[2], camera_info.P[6]s
            camera_x = (center_x - cx) / fx
            camera_y = (center_y - cy) / fy


            #get the 2D image coordinate of the aruco marker
            aruco_msg = Aruco()
            # temp = np.array([position2D.pose.position.x, position2D.pose.position.y, position2D.pose.position.z])
            # temp /= np.linalg.norm(temp)
            aruco_msg.image_x = camera_x
            aruco_msg.image_y = camera_y

            #TODO: either add image_z or share the raw pose of image_x and image_y

            # debug
            if self.debug:
                cv2.circle(gray, (int(aruco_msg.image_x), int(aruco_msg.image_y)), 50, (0,255,0), 2)
                
                cv2.circle(gray, (int(markerCorners[0][0]), int(markerCorners[0][1])), 20, (0,255,0), 4)
                cv2.circle(gray, (int(markerCorners[1][0]), int(markerCorners[1][1])), 20, (0,255,0), 4)
                cv2.circle(gray, (int(markerCorners[2][0]), int(markerCorners[2][1])), 20, (0,255,0), 4)
                cv2.circle(gray, (int(markerCorners[3][0]), int(markerCorners[3][1])), 20, (0,255,0), 4)
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
            Vmarker.header.frame_id = self.base_link
            Vmarker.id = int(aruco_id)
            Vmarker.type = VM.SPHERE
            Vmarker.action = VM.ADD
            Vmarker.pose.position = t.transform.translation
            Vmarker.pose.orientation = t.transform.rotation
            Vmarker.scale = Vector3(*[0.1, 0.1, 0.1])
            Vmarker.color = ColorRGBA(*[0, 1, 0, 1])
            markerArray.markers.append(marker)
            self.pub_aruco_marker.publish(Vmarker)
            # if aruco_id in arucoList:
            CoorArray.header = t.header
            CoorArray.arucos.append(aruco_msg)
        self.pub_aruco_marker_coordinate.publish(CoorArray)

if __name__ == '__main__':
    ##  Global resources  ##
    bridge = CvBridge()

    current_pid = os.getpid()
    process = psutil.Process(current_pid)
    process.cpu_affinity([1])
    ##  Start node  ##
    aruco_detect().run()