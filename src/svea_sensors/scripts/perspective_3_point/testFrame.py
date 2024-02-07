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
from geometry_msgs.msg import Point, Quaternion, TransformStamped, PoseStamped, Vector3
import tf2_geometry_msgs
import tf_conversions
from message_filters import Subscriber, ApproximateTimeSynchronizer

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot


class testFrame():
    def __init__(self):
        # Initalize the node 
        rospy.init_node('testFrame')

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.sbr = tf2_ros.StaticTransformBroadcaster()
        self.br = tf2_ros.TransformBroadcaster()

        self.pose = None
        self.landmarks = [[0, 0, 0], [5, 0, 0], [2.5, 2.5, 0]]
        self.rot = np.eye(3)
        self.seq = 0
        self.t = 0
        self.R = np.eye(3)

    def run(self):
        while not rospy.is_shutdown():
            self.pub_map_odom()
            self.pub_map_base()
            self.pub_landmarks()
            self.pub_dir()
            self.seq += 1
            self.t += 0.1
            rospy.Rate.sleep(rospy.Rate(10))

    def pub_map_odom(self):
        msg = TransformStamped()
        msg.header.seq = self.seq
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'odom'
        msg.transform.translation = Vector3(*[2.5+2.5*np.cos(0.4*0), 2.5*np.sin(0.4*0), 10])
        msg.transform.rotation = Quaternion(*[0, 0, 0, 1])
        self.sbr.sendTransform(msg)

    def pub_map_base(self):
        msg = TransformStamped()
        msg.header.seq = self.seq
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'
        self.pose = [2.5+2.5*np.cos(0.4*self.t), 2.5*np.sin(0.4*self.t), 10]
        msg.transform.translation = Vector3(*self.pose)
        omega = np.array([[0.1*np.sin(self.t), 0.4*np.cos(2*self.t), 0.6*self.t]])
        rot = ScipyRot.from_matrix(np.cross(self.rot, omega))
        self.rot = ScipyRot.as_matrix(rot)
        qua = rot.as_quat()
        msg.transform.rotation = Quaternion(*qua)
        self.br.sendTransform(msg)

    def pub_landmarks(self):
        for idx, lm in enumerate(self.landmarks):
            msg = TransformStamped()
            msg.header.seq = self.seq
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'map'
            msg.child_frame_id = 'landmark' + str(idx)
            msg.transform.translation = Vector3(*lm)
            msg.transform.rotation = Quaternion(*[0, 0, 0, 1])
            self.br.sendTransform(msg)

    def pub_dir(self):
        for idx, lm in enumerate(self.landmarks):
            msg = TransformStamped()
            msg.header.seq = self.seq
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'base_link'
            msg.child_frame_id = 'dir' + str(idx)
            pose = np.array(self.pose)
            lm = np.array(lm)
            d = np.matmul(np.transpose(self.rot), (pose-lm)/np.linalg.norm(pose-lm))
            msg.transform.translation = Vector3(*d)
            msg.transform.rotation = Quaternion(*[0, 0, 0, 1])
            self.br.sendTransform(msg)

if __name__ == '__main__':
    testFrame().run()
        