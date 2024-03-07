#! /usr/bin/env python3

import rospy
import numpy as np

from message_filters import Subscriber, ApproximateTimeSynchronizer

from ricatti_observer import riccati_observer

from geometry_msgs.msg import TwistWithCovarianceStamped
from svea_msgs.msg import Aruco, ArucoArray


class ricatti_estimation():
    def __init__(self):
        rospy.init_node('ricatti_estimation')


        ##############################################
        ################# Subscriber #################

        # rospy.Subscriber('/actuation_to_twist', TwistWithCovarianceStamped, self.TwistCallback)
        # rospy.Subscriber('/aruco/detection', ArucoArray, self.LandmarkCallback)
        #TODO: change the aruco_detect.py so that it will publish ArucoArray even with no data
        Twist = Subscriber('/actuation_to_twist', TwistWithCovarianceStamped)
        Landmark = Subscriber('/aruco/2Ddetection', ArucoArray)
        sync = ApproximateTimeSynchronizer([Twist, Landmark], queue_size=1, slop=0.1)
        sync.registerCallback(self.TwistAndLandmarkCallback)


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


        ################# Parameters #################
        ##############################################

        self.riccati_obj = riccati_observer(
        use_adaptive            = True,
        quaternion              = True,
        # time                    = (0, 100),
        stepsize                = 0.1,
        tol                     = 1e-2 * 3,
        # noise                   = False,
        which_eq                = 2,
        which_omega             = "z",
        # with_image_hz_sim       = False,
        # image_hz                = 60, 
        # randomize_image_input   = False,
        p_hat                   = # sth from state or just input from lanuch file,
        Lambda_bar_0            = # sth from state or just input from lanuch file,  # quaternion: w, x, y, z
        z_appear                = np.array([]),
        k                       = k,
        q                       = [q], 
        v                       = np.array([v1. v2]),
        p_ricatti               = np.array([1, 100])
        )
        
    def TwistCallback(self, msg):
        self.linear_velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.angular_velocity = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])

    def TwistAndCameraCallback(self, TwistMsg, LandmarkMsg):
        linear_velocity = np.array([TwistMsg.twist.twist.linear.x, TwistMsg.twist.twist.linear.y, TwistMsg.twist.twist.linear.z])
        angular_velocity = np.array([TwistMsg.twist.twist.angular.x, TwistMsg.twist.twist.angular.y, TwistMsg.twist.twist.angular.z])
        landmark = []
        for aruco in LandmarkMsg:
            landmark.append([aruco.marker.pose.pose.position.x, aruco.marker.pose.pose.position.y, aruco.marker.pose.pose.position.z])
        if len(landmark) != 0:
            self.riccati_obj.update_z(landmark)
        self.riccati_obj.update_linear_velocity(linear_velocity)
        self.riccati_obj.update_angular_velocity(angular_velocity)
        self.riccati_obj.step_simulation()

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    ricatti_estimation().run()