#!/usr/bin/env python3

import rospy
import tf2_ros
import numpy as np

from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, Point
from svea_msgs.msg import Aruco, ArucoArray

class landmark_outdoor:
    def __init__(self):
        rospy.init_node('landmark_outdoor')
        self.landmark = {
            11 : [1.0  , 0.7 , 0.08],
            12 : [-1.0 , 0.9 , 0.08],
            14 : [0.6  , 1.1 , 0.08],
            15 : [-0.6 , 1.3 , 0.08]
        }

        self.MapArucoPub = rospy.Publisher('/aruco/detection/Groundtruth', ArucoArray, queue_size=1)    

        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.static_br = tf2_ros.StaticTransformBroadcaster()
        self.seq = 0
    
    def FillCovaraince(self):
        TransCov = np.eye(3,6, dtype=float)*1e-3
        RotCov = np.eye(3,6, k=3, dtype=float)*1e-3*5
        return np.vstack((TransCov, RotCov)).flatten()
    
    def publishLandmark(self):
        msg = TransformStamped()
        arucoarray = ArucoArray()

        for key, items in self.landmark.items():
            msg.header.stamp = rospy.Time.now()
            msg.header.seq = self.seq
            msg.header.frame_id = "map_ref"
            msg.child_frame_id = "aruco" + str(key)
            msg.transform.translation = Vector3(*items)
            msg.transform.rotation = Quaternion(*[0, 0, 0, 1])
            self.static_br.sendTransform(msg)

            arucoitem = Aruco()
            arucoitem.marker.header = msg.header
            arucoitem.marker.id = key
            arucoitem.marker.pose.pose.position = Point(*items)
            arucoitem.marker.pose.pose.orientation = Quaternion(*[0, 0, 0, 1])
            arucoitem.marker.pose.covariance = self.FillCovaraince()
            arucoarray.arucos.append(arucoitem)
        self.seq += 1
        self.MapArucoPub.publish(arucoarray)
        
    def run(self):
        while not rospy.is_shutdown():
            self.publishLandmark()

if __name__ == '__main__':
    landmark_outdoor().run()