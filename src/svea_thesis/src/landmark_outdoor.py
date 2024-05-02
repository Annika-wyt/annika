#!/usr/bin/env python3

import rospy
import tf2_ros

from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
class landmark_outdoor:
    def __init__(self):
        rospy.init_node('landmark_outdoor')
        self.landmark = {
            11 : [1.0  , 0.7 , 0.08],
            12 : [-1.0 , 0.9 , 0.08],
            14 : [0.6  , 1.1 , 0.08],
            15 : [-0.6 , 1.3 , 0.08]
        }

        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.static_br = tf2_ros.StaticTransformBroadcaster()
        self.seq = 0

    def publishLandmark(self):
        msg = TransformStamped()
        for key, items in self.landmark.items():
            msg.header.stamp = rospy.Time.now()
            msg.header.seq = self.seq
            msg.header.frame_id = "map_ref"
            msg.child_frame_id = "aruco" + str(key)
            msg.transform.translation = Vector3(*items)
            msg.transform.rotation = Quaternion(*[0, 0, 0, 1])
            self.static_br.sendTransform(msg)
        self.seq += 1
    def run(self):
        while not rospy.is_shutdown():
            self.publishLandmark()

if __name__ == '__main__':
    landmark_outdoor().run()