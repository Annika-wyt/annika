#! /usr/bin/env python3

import rospy
import numpy as np
import tf2_ros

from message_filters import Subscriber, ApproximateTimeSynchronizer

from ricatti_observer import riccati_observer

from geometry_msgs.msg import TwistWithCovarianceStamped
from svea_msgs.msg import Aruco, ArucoArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped, TransformStamped, Point, Quaternion, Vector3, PoseStamped
import tf2_geometry_msgs


class ricatti_estimation():
    def __init__(self):
        rospy.init_node('ricatti_estimation')

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.StaticTransformBroadcaster()

        self.stop = 0
        ##############################################
        ################# Subscriber #################

        # rospy.Subscriber('/actuation_to_twist', TwistWithCovarianceStamped, self.TwistCallback)
        # rospy.Subscriber('/aruco/detection', ArucoArray, self.LandmarkCallback)
        #TODO: change the aruco_detect.py so that it will publish ArucoArray even with no data
        Twist = Subscriber('/odometry/filtered/global', Odometry)
        Landmark = Subscriber('/aruco/detection', ArucoArray)
        LandmarkGroudtruth = Subscriber('/aruco/detection/groundtruth', ArucoArray)
        sync = ApproximateTimeSynchronizer([Twist, Landmark, LandmarkGroudtruth], queue_size=1, slop=0.1)
        sync.registerCallback(self.TwistAndLandmarkCallback)
        self.seq = 0
        self.t = 0
        ## Sub to state
        ################# Subscriber #################
        ##############################################

        ##############################################
        ################# Variables ##################

        # self.linear_velocity = None
        # self.angular_velocity = None
        # self.lanmark = []

        ################# Variables ##################
        ##############################################

        ##############################################
        ################# Parameters #################
        k = rospy.get_param("~k", 1)
        q = rospy.get_param("~q", 10)
        v1 = rospy.get_param("~v1", 0.1)
        v2 = rospy.get_param("~v2", 1)
        # estpose = rospy.get_param("~estpose", np.array([0, 0, 0]))
        # estori = rospy.get_param("~estori", np.array([1, 0, 0, 0]))
        self.estpose = np.array([-2, 4, 3])
        self.estori = np.array([1, 0, 0, 0]) #w, x, y, z
        # estori = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2]) #x, y, z, w
        ################# Parameters #################
        ##############################################
        self.riccati_obj = riccati_observer(
        stepsize                = 0.1,
        tol                     = 1e-2 * 6,
        which_eq                = 0,
        p_hat                   = self.estpose, # sth from state or just input from lanuch file,
        Lambda_bar_0            = self.estori, # sth from state or just input from lanuch file,  # quaternion: w, x, y, z
        z_appear                = np.array([]),
        k                       = k,
        q                       = q, 
        v                       = np.array([v1, v2]),
        p_ricatti               = np.array([1, 100])
        )
        
        self.pub_EstPose()
    def pub_EstPose(self):
        msg = TransformStamped()
        msg.header.seq = self.seq
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link_est'
        msg.transform.translation = Vector3(*self.estpose)
        msg.transform.rotation = Quaternion(*self.estori) #x, y, z, w
        self.br.sendTransform(msg)
        self.seq += 1

    def pub_landmark(self, landmark,id):
        msg = TransformStamped()
        msg.header = landmark.header
        msg.child_frame_id = 'lm' + str(id)
        msg.transform.translation = landmark.pose.position
        msg.transform.rotation = landmark.pose.orientation #x, y, z, w
        self.br.sendTransform(msg)

    def TwistCallback(self, msg):
        self.linear_velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.angular_velocity = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])

    def TwistAndLandmarkCallback(self, TwistMsg, LandmarkMsg, LandmarkGroudtruthMsg):
        # if self.stop < :
        if self.t <= 5:
            linear_velocity = np.array([TwistMsg.twist.twist.linear.x, TwistMsg.twist.twist.linear.y, TwistMsg.twist.twist.linear.z])
            angular_velocity = np.array([TwistMsg.twist.twist.angular.x, TwistMsg.twist.twist.angular.y, TwistMsg.twist.twist.angular.z])
            landmark = []
            landmarkGroundtruth = []
            for aruco in LandmarkMsg.arucos:
                landmark.append([aruco.marker.pose.pose.position.x, aruco.marker.pose.pose.position.y, aruco.marker.pose.pose.position.z])
            for aruco in LandmarkGroudtruthMsg.arucos:
                transform_map_baselink = self.buffer.lookup_transform('base_link_est',"map", LandmarkGroudtruthMsg.header.stamp, rospy.Duration(0.5))  #rospy.Time.now()
                position = tf2_geometry_msgs.do_transform_pose(aruco.marker.pose, transform_map_baselink) 
                self.pub_landmark(position, aruco.marker.id)
                landmarkGroundtruth.append([position.pose.position.x, position.pose.position.y, position.pose.position.z])
                # print(position)
            if len(landmark) != 0:
                self.riccati_obj.update_z(landmark)
            if len(landmarkGroundtruth) != 0:
                self.riccati_obj.update_z_groundtruth(landmarkGroundtruth)
            self.riccati_obj.update_linear_velocity(linear_velocity)
            self.riccati_obj.update_angular_velocity(angular_velocity)
            solt, dt, soly = self.riccati_obj.step_simulation()
            self.t = solt
            self.estpose = soly[4:7]
            self.estori = np.hstack((soly[1:4], soly[0]))
            self.pub_EstPose()
            print("===================================================")
            self.stop += 1

    def run(self):
        rospy.spin()            

if __name__ == '__main__':
    ricatti_estimation().run()