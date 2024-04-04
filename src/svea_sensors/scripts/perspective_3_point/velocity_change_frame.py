#! /usr/bin/env python3

import rospy
import numpy as np
import tf2_ros

from geometry_msgs.msg import TwistStamped, Vector3Stamped, Vector3
from nav_msgs.msg import Odometry
import tf2_geometry_msgs
from message_filters import Subscriber, ApproximateTimeSynchronizer

class velocity_change_frame():
    def __init__(self):
        rospy.init_node("velocity_change_frame")
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()
        self.svea_frame_name = "base_link"
        Vel = Subscriber("/qualisys/svea2/velocity", TwistStamped)
        Odom = Subscriber("/qualisys/svea2/odom", Odometry)
        sync = ApproximateTimeSynchronizer([Vel, Odom], queue_size=1, slop=3)
        sync.registerCallback(self.TransformVelcity)

        self.TwistPub = rospy.Publisher("/odometry/filtered", Odometry, queue_size=1)
        

    def run(self):
        rospy.spin()

    def TransformVelcity(self, VelMsg, OdomMsg):
        # odommsg = Odometry()
        # odommsg.header = TwistMsg.header
        # odommsg.twist.twist.linear = TwistMsg.twist.linear
        # odommsg.twist.twist.angular = TwistMsg.twist.angular
        # self.TwistPub.publish(odommsg)

        try:
            transform_base_map = self.buffer.lookup_transform(self.svea_frame_name, "mocap", rospy.Time.now(), rospy.Duration(0.5))
            transMsg = Vector3Stamped()
            transMsg.header = VelMsg.header
            transMsg.vector = VelMsg.twist.linear
            trans_linear = tf2_geometry_msgs.do_transform_vector3(transMsg, transform_base_map).vector
            transMsg.vector = VelMsg.twist.angular
            trans_angular = tf2_geometry_msgs.do_transform_vector3(transMsg, transform_base_map).vector
            odommsg = Odometry()
            odommsg.header = VelMsg.header
            odommsg.header.frame_id = "map"
            odommsg.child_frame_id = "base_link"
            odommsg.twist.twist.linear = trans_linear
            odommsg.twist.twist.angular = trans_angular
            self.TwistPub.publish(odommsg)
        except:
            try:
                transform_base_map = self.buffer.lookup_transform(self.svea_frame_name, "mocap", rospy.Time.now(), rospy.Duration(0.5))
                transMsg = Vector3Stamped()
                transMsg.header = VelMsg.header
                transMsg.vector = VelMsg.twist.linear
                trans_linear = tf2_geometry_msgs.do_transform_vector3(transMsg, transform_base_map).vector
                transMsg.vector = VelMsg.twist.angular
                trans_angular = tf2_geometry_msgs.do_transform_vector3(transMsg, transform_base_map).vector
                odommsg = Odometry()
                odommsg.header = VelMsg.header
                odommsg.header.frame_id = "map"
                odommsg.twist.twist.linear = trans_linear
                odommsg.twist.twist.angular = trans_angular
                self.TwistPub.publish(odommsg)
            except Exception as e:
                rospy.logerr(f"Velocity change frame node: {e}")


if __name__ == "__main__":
    velocity_change_frame().run()