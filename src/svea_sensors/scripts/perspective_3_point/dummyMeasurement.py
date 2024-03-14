#! /usr/bin/env python3

import rospy
import numpy as np
import tf2_ros

from svea_msgs.msg import Aruco, ArucoArray
from geometry_msgs.msg import TwistWithCovarianceStamped, TransformStamped, Point, Quaternion, Vector3, PoseStamped

from nav_msgs.msg import Odometry
import tf2_geometry_msgs

class dummyMeasurement():
    def __init__(self):
        rospy.init_node("dummyMeasurement")

        rospy.Subscriber("/qualisys/svea5/odom", Odometry, self.Odomcallback)
        
        self.landmarkPub = rospy.Publisher("/aruco/detection/groundtruth", ArucoArray, queue_size=1)
        self.landmark2Pub = rospy.Publisher("/aruco/detection", ArucoArray, queue_size=1)
        self.twistPub = rospy.Publisher("/odometry/filtered/global", Odometry, queue_size=1)
        self.seq = 0
        self.frame = "map" #TODO: need to add a transformation such that the landmark is in base_link frame
        self.time = rospy.Time.now()

        self.startTime = rospy.Time.now()
        self.current_time = (self.time - self.startTime).to_sec()
        
        self.motion = "linear" #"static", "linear", "angular", "both"

        self.landmarkId = np.array([12, 13, 14, 15])
        lm1 = PoseStamped()
        lm1.header.seq = self.seq
        lm1.header.stamp = self.time
        lm1.header.frame_id = self.frame
        lm1.pose.position = Point(*[5, 5, 1])
        lm1.pose.orientation = Quaternion(*[0, 0, 0, 1])

        lm2 = PoseStamped()
        lm2.header.seq = self.seq
        lm2.header.stamp = self.time
        lm2.header.frame_id = self.frame
        lm2.pose.position = Point(*[5, 0, 1])
        lm2.pose.orientation = Quaternion(*[0, 0, 0, 1])

        lm3 = PoseStamped()
        lm3.header.seq = self.seq
        lm3.header.stamp = self.time
        lm3.header.frame_id = self.frame
        lm3.pose.position = Point(*[0, 0, 1])
        lm3.pose.orientation = Quaternion(*[0, 0, 0, 1])
                   
        lm4 = PoseStamped()
        lm4.header.seq = self.seq
        lm4.header.stamp = self.time
        lm4.header.frame_id = self.frame
        lm4.pose.position = Point(*[10, 0, 5])
        lm4.pose.orientation = Quaternion(*[0, 0, 0, 1])

        self.landmarkPose = [lm1, lm2, lm3, lm4]
        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()
        self.sbr = tf2_ros.StaticTransformBroadcaster()

    def Odomcallback(self, msg):
        self.publishBaselink(msg)
        self.publishTwist(msg)   

    def run(self):
        while not rospy.is_shutdown():
            self.time = rospy.Time.now()
            self.current_time = (self.time - self.startTime).to_sec()
            if self.motion != "bag":
                self.publishBaselink(None)
                self.publishTwist(None)
            self.publishAruco()
            self.seq += 1
            rospy.sleep(0.001)

    def publishTwist(self, odomMsg):
        odometryMsg = Odometry()
        odometryMsg.header.seq = self.seq
        odometryMsg.header.stamp = self.time
        odometryMsg.header.frame_id = self.frame
        odometryMsg.child_frame_id = "base_link"
        if self.motion == "static":
            linear = [0, 0, 0]
            angular = [0, 0, 0]
        elif self.motion == "linear":
            linear = [0.5, 0, 0]
            angular = [0, 0, 0]
        elif self.motion == "angular":
            linear = [0, 0, 0]
            angular = [0, 0, 0.3]
        elif self.motion == "both":
            linear = [0.3, 0, 0]
            angular = [0, 0, 0.3]
        elif self.motion == "bag":
            linear = [odomMsg.twist.twist.linear.x, odomMsg.twist.twist.linear.y, odomMsg.twist.twist.linear.z]
            angular = [odomMsg.twist.twist.angular.x, odomMsg.twist.twist.angular.y, odomMsg.twist.twist.angular.z]

        odometryMsg.twist.twist.linear = Vector3(*linear)
        odometryMsg.twist.twist.angular = Vector3(*angular)
        self.twistPub.publish(odometryMsg)

    def publishBaselink(self, odomMsg):
        msg = TransformStamped()
        msg.header.seq = self.seq
        msg.header.stamp = self.time
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'
        if self.motion == "static": 
            msg.transform.translation = Vector3(*[5, 0, 10])
            msg.transform.rotation = Quaternion(*[0, 0, 0, 1]) #x, y, z, w
        elif self.motion == "linear":
            msg.transform.translation = Vector3(*[0.5*self.current_time, 0, 10])
            msg.transform.rotation = Quaternion(*[0, 0, 0, 1]) #x, y, z, w
        elif self.motion == "angular":
            msg.transform.translation = Vector3(*[5, 0, 10])
            # msg.transform.rotation = Quaternion(*[0, 0, 0, 1]) #x, y, z, w
            msg.transform.rotation = Quaternion(*[0, 0, np.sin(0.3*self.current_time/2), np.cos(0.3*self.current_time/2)]) #x, y, z, w
        elif self.motion == "both":
            msg.transform.translation = Vector3(*[0.3*self.current_time, 0, 10])
            msg.transform.rotation = Quaternion(*[0, 0, 0, 1]) #x, y, z, w
            # msg.transform.rotation = Quaternion(*[0, 0, np.sin(0.3*self.current_time/2), np.cos(0.3*self.current_time/2)]) #x, y, z, w

        elif self.motion == "bag":
            msg.transform.translation = odomMsg.pose.pose.position
            msg.transform.rotation = odomMsg.pose.pose.orientation #x, y, z, w


        self.br.sendTransform(msg)

    def publishAruco(self):

        for id, landmark in zip(self.landmarkId, self.landmarkPose):
            msg = TransformStamped()
            msg.header.seq = self.seq
            msg.header.stamp = self.time
            msg.header.frame_id = 'map'
            msg.child_frame_id = 'lmGroundTruth' + str(id)
            msg.transform.translation = landmark.pose.position
            msg.transform.rotation = landmark.pose.orientation #x, y, z, w
            self.br.sendTransform(msg)

        arucoArrayMsg = ArucoArray()
        arucoArrayMsg.header.seq = self.seq
        arucoArrayMsg.header.stamp = self.time
        arucoArrayMsg.header.frame_id = self.frame

        arucoArrayMsg2 = ArucoArray()
        arucoArrayMsg2.header.seq = self.seq
        arucoArrayMsg2.header.stamp = self.time
        arucoArrayMsg2.header.frame_id = self.frame

        try:
            transform_map_baselink = self.buffer.lookup_transform('base_link',"map", self.time, rospy.Duration(0.5))  #rospy.Time.now()

            for id, landmark in zip(self.landmarkId, self.landmarkPose):
                position = tf2_geometry_msgs.do_transform_pose(landmark, transform_map_baselink) 
                arucoMsg = Aruco()
                arucoMsg2 = Aruco()
                arucoMsg.marker.header.seq = self.seq
                arucoMsg.marker.header.stamp = self.time
                arucoMsg.marker.header.frame_id = self.frame
                arucoMsg.marker.id = id
                arucoMsg.marker.pose.pose.position = landmark.pose.position
                arucoMsg.marker.pose.pose.orientation = landmark.pose.orientation
                arucoArrayMsg.arucos.append(arucoMsg)

                arucoMsg2.marker.header = position.header
                arucoMsg2.marker.id = id
                arucoMsg2.marker.pose.pose.position = position.pose.position
                arucoMsg2.marker.pose.pose.orientation = position.pose.orientation
                arucoArrayMsg2.arucos.append(arucoMsg2)

                msg = TransformStamped()
                msg.header.seq = self.seq
                msg.header.stamp = self.time
                msg.header.frame_id = 'base_link'
                msg.child_frame_id = 'lmBaseLink' + str(id)
                msg.transform.translation = position.pose.position
                msg.transform.rotation = position.pose.orientation #x, y, z, w
                self.br.sendTransform(msg)

            self.landmarkPub.publish(arucoArrayMsg)
            self.landmark2Pub.publish(arucoArrayMsg2)
        except:
            print("waiting for transform between base link and map")

if __name__ == "__main__":
    dummyMeasurement().run()
