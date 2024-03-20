#! /usr/bin/env python3

import rospy
import numpy as np
import tf2_ros

from svea_msgs.msg import Aruco, ArucoArray
from geometry_msgs.msg import TwistWithCovarianceStamped, TransformStamped, Point, Quaternion, Vector3, PoseStamped

from nav_msgs.msg import Odometry
import tf2_geometry_msgs

import pandas as pd

AveAll = True
class dummyMeasurement():
    def __init__(self):
        rospy.init_node("dummyMeasurement")
        self.svea_frame_name = "svea2"

        rospy.Subscriber("/qualisys/" + self.svea_frame_name + "/odom", Odometry, self.Odomcallback)
        if AveAll:
            self.linearXRunningAvg = np.array([0, 0, 0])
            self.angularZRunningAvg = np.array([0, 0, 0])
        else:
            self.linearXRunningAvg = 0
            self.angularZRunningAvg = 0
        self.alpha = 2/(20+1)
        self.alphaAng = 2/(30+1)
        self.landmarkPub = rospy.Publisher("/aruco/detection/groundtruth", ArucoArray, queue_size=10)
        self.landmark2Pub = rospy.Publisher("/aruco/detection", ArucoArray, queue_size=10)
        self.twistPub = rospy.Publisher("/odometry/filtered/global", Odometry, queue_size=10)
        self.seq = 0
        self.frame = "map" #TODO: need to add a transformation such that the landmark is in base_link frame
        self.time = rospy.Time.now()

        self.startTime = rospy.Time.now()
        self.current_time = (self.time - self.startTime).to_sec()
        
        self.motion = "bag" #"static", "linear", "angular", "both"
        self.stepCounter = 0 


        if self.motion == "test":
            sim_solution = pd.read_csv("/home/annika/ITRL/kth_thesis/simulated_result/angular.txt", header=None)
            sim_solution = sim_solution.to_numpy().reshape((-1, 8))
            # print("sim_sol shape", np.shape(sim_solution))
            self.ori = sim_solution[:,0:4]
            self.pose = sim_solution[:,4:7]
            self.sim_time = sim_solution[:,-1]

        self.landmarkId = np.array([12, 13, 14, 15, 16])
        landmark_pose = np.array([[5, 5, 1], [-5, 0, 1], [-3, -3, 1], [-6, 2, 1], [4, 0, 1]])
        self.landmarkPose = []
        for lId, lpose in zip(self.landmarkId, landmark_pose):
            lm = PoseStamped()
            lm.header.seq = self.seq
            lm.header.stamp = self.time
            lm.header.frame_id = self.frame
            lm.pose.position = Point(*lpose)
            lm.pose.orientation = Quaternion(*[0, 0, 0, 1])
            self.landmarkPose.append(lm)

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
            rospy.sleep(0.005)

    def publishTwist(self, odomMsg):
        odometryMsg = Odometry()
        odometryMsg.header.seq = self.seq
        odometryMsg.header.stamp = self.time
        odometryMsg.header.frame_id = self.frame
        odometryMsg.child_frame_id = self.svea_frame_name
        if self.motion == "static":
            linear = [0, 0, 0]
            angular = [0, 0, 0]
        elif self.motion == "linear":
            linear = [-np.sin(0.4*self.current_time), 0, 0]
            angular = [0, 0, 0]
        elif self.motion == "angular":
            linear = [0, 0, 0]
            angular = [0, 0, 0.3]
        elif self.motion == "both":
            linear = [0.3, 0, 0]
            angular = [0, 0, 0.3]
        elif self.motion == "bag":
            if AveAll:
                self.linearXRunningAvg = self.alpha * np.array([odomMsg.twist.twist.linear.x, odomMsg.twist.twist.linear.y, odomMsg.twist.twist.linear.z]) + (1 - self.alpha) * self.linearXRunningAvg
                self.angularZRunningAvg = self.alphaAng * np.array([odomMsg.twist.twist.angular.x, odomMsg.twist.twist.angular.y, odomMsg.twist.twist.angular.z]) + (1 - self.alphaAng) * self.angularZRunningAvg
                linear = self.linearXRunningAvg 
                angular = self.angularZRunningAvg
            else:
                self.linearXRunningAvg = self.alpha * odomMsg.twist.twist.linear.x + (1 - self.alpha) * self.linearXRunningAvg
                self.angularZRunningAvg = self.alphaAng * odomMsg.twist.twist.angular.z + (1 - self.alphaAng) * self.angularZRunningAvg
                linear = [self.linearXRunningAvg, odomMsg.twist.twist.linear.y, odomMsg.twist.twist.linear.z]
                angular = [odomMsg.twist.twist.angular.x, odomMsg.twist.twist.angular.y, self.angularZRunningAvg]
        elif self.motion == "test":
            linear = [0, 0, 0]
            angular = [0, 0, 0.3]

        odometryMsg.twist.twist.linear = Vector3(*linear)
        odometryMsg.twist.twist.angular = Vector3(*angular)
        self.twistPub.publish(odometryMsg)

    def publishBaselink(self, odomMsg):
        msg = TransformStamped()
        msg.header.seq = self.seq
        msg.header.stamp = self.time
        msg.header.frame_id = "map"
        msg.child_frame_id = self.svea_frame_name
        if self.motion == "static": 
            msg.transform.translation = Vector3(*[5, 0, 10])
            msg.transform.rotation = Quaternion(*[0, 0, 0, 1]) #x, y, z, w
        elif self.motion == "linear":
            msg.transform.translation = Vector3(*[2.5+2.5*np.cos(0.4*self.current_time), 0, 0])
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
            return 
            # msg.transform.translation = odomMsg.pose.pose.position
            # msg.transform.rotation = odomMsg.pose.pose.orientation #x, y, z, w
        elif self.motion == "test":
            print(self.pose[self.stepCounter])
            print(self.ori[self.stepCounter])
            
            msg.transform.translation = Vector3(*self.pose[self.stepCounter])
            msg.transform.rotation = Quaternion(*self.ori[self.stepCounter]) #x, y, z, w
            self.stepCounter += 1

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
            transform_map_baselink = self.buffer.lookup_transform(self.svea_frame_name,"map", self.time, rospy.Duration(0.5))  #rospy.Time.now()

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
                msg.header.frame_id = self.svea_frame_name
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
