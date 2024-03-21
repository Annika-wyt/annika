#! /usr/bin/env python3

import rospy
import tf2_ros

from geometry_msgs.msg import PoseStamped, TransformStamped
class diff_est_act():
    def __init__(self):
        rospy.init_node("diff_est_act")

        self.pub = rospy.Publisher("/diff_est_act", TransformStamped, queue_size=10)
        self.svea_frame_name = "svea2"
        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

    def run(self):
        while not rospy.is_shutdown():
            self.pubFun()

    def pubFun(self):
        try:
            transform = self.buffer.lookup_transform(self.svea_frame_name, "base_link_est", rospy.Time.now(), rospy.Duration(2))  
            self.pub.publish(transform)

        except Exception as e:
            print(e)

if __name__ == "__main__":
    diff_est_act().run()