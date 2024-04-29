#! /usr/bin/env python3

import rospy
import numpy as np
import tf2_ros

from svea_msgs.msg import Aruco, ArucoArray
from geometry_msgs.msg import TwistWithCovarianceStamped, TransformStamped, Point, Quaternion, Vector3, PoseStamped, TwistStamped, Vector3Stamped

from nav_msgs.msg import Odometry
import tf2_geometry_msgs
import tf
# import pandas as pd

from message_filters import Subscriber, ApproximateTimeSynchronizer
from visualization_msgs.msg import Marker as VM
from visualization_msgs.msg import MarkerArray as VMA
from std_msgs.msg import Float32, ColorRGBA

from sensor_msgs.msg import CameraInfo

import cv2

AveAll = True
camera_matrix = np.array([[514.578298266441, 0, 340.0718185830948], [0, 514.8684665452305, 231.4918039429434], [0, 0, 1]])
distortion_coeffs = np.array([0.06295602826790396, -0.1840231372229633, -0.004945725015870819, 0.01208470957502327, 0])
class dummyMeasurement():
    def __init__(self):
        rospy.init_node("dummyMeasurement")
        self.svea_frame_name = "svea5"

        OdomTopic = Subscriber("/qualisys/" + self.svea_frame_name + "/odom", Odometry)
        VelTopic = Subscriber("/qualisys/" + self.svea_frame_name + "/velocity", TwistStamped)
        sync = ApproximateTimeSynchronizer([OdomTopic, VelTopic], queue_size=1, slop=0.1)
        # sync.registerCallback(self.OdomVelCallback)

        rospy.Subscriber("/qualisys/svea5/odom", Odometry, self.odomCallback)
        rospy.Subscriber("/actuation_twist", TwistWithCovarianceStamped, self.TwistCallback)

        self.visualArucoPub = rospy.Publisher("/visual/cameraaruco", VMA, queue_size=10)
        self.campub = rospy.Publisher("/camera/camera_info", CameraInfo, queue_size = 10)

        if AveAll:
            self.linearXRunningAvg = np.array([0, 0, 0])
            self.angularZRunningAvg = np.array([0, 0, 0])
        else:
            self.linearXRunningAvg = 0
            self.angularZRunningAvg = 0
        self.alpha = 2/(30+1)
        self.alphaAng = 2/(30+1)
        self.landmarkPub = rospy.Publisher("/aruco/detection/Groundtruth", ArucoArray, queue_size=1) #map frame
        self.landmark2Pub = rospy.Publisher("/aruco/detection", ArucoArray, queue_size=1) #svea2 frame; actual direction
        self.twistPub = rospy.Publisher("/odometry/filtered", Odometry, queue_size=1)
        self.seq = 0
        self.frame = "map" #TODO: need to add a transformation such that the landmark is in base_link frame
        self.time = None

        self.startTime = None
        self.current_time = 0
        
        self.motion = "static" #"static", "linear", "angular", "both"
        self.stepCounter = 0 


        if self.motion == "test":
            # sim_solution = pd.read_csv("/home/annika/ITRL/kth_thesis/simulated_result/angular.txt", header=None)
            sim_solution = sim_solution.to_numpy().reshape((-1, 8))
            # print("sim_sol shape", np.shape(sim_solution))
            self.ori = sim_solution[:,0:4]
            self.pose = sim_solution[:,4:7]
            self.sim_time = sim_solution[:,-1]
        z=1
        self.landmarkId = np.array([11,12,14,15,16,17,18,19,20,21])
        # landmark_pose = np.array([[-1.74, -2.34, 0.046], [-1.80, -1.85, 0.06], [-2, -2.2, 0.16], [1.5, 0.5, 1], [-1, 1, 0.5], [1, -1, 0]])
        # landmark_pose = np.array([[1.2,  0.65,  0.08],
        #                           [1.2,  -0.65, 0.08],
        #                           [-1.2, 0.65,  0.08],
        #                           [-1.2, -0.65, 0.08],
        #                           [0.65, 1.2,   0.08],
        #                           [-0.65,1.2 ,  0.08],
        #                           [0.65, -1.2 , 0.08],
        #                           [-0.65,-1.2 , 0.08],
        #                           [0.0,  -1.0 , 0.08],
        #                           [0.0,  1.0 ,  0.08],
        #                           [-1.0, 0.0 ,  0.08],
        #                           [1.0,  0.0 ,  0.08]])
        landmark_pose = np.array([[-0.3982733459472656, -0.3341578674316406, 0.10122600555419922], 
                                  [-1.021384765625, -1.3663646240234375, 0.08791256713867188],
                                  [-1.590159423828125, -0.7866044311523438, 0.0821039810180664],
                                  [-0.7896022338867188, 0.3022557373046875, 0.07362690734863281]])
        
        
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

    def TwistCallback(self, msg):
        odometryMsg = Odometry()
        odometryMsg.header.seq = self.seq
        odometryMsg.header.stamp = self.time
        odometryMsg.header.frame_id = self.frame
        odometryMsg.child_frame_id = self.svea_frame_name
        if AveAll:
            self.linearXRunningAvg = self.alpha * np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]) + (1 - self.alpha) * self.linearXRunningAvg
            self.angularZRunningAvg = self.alphaAng * np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]) + (1 - self.alphaAng) * self.angularZRunningAvg
            linear = self.linearXRunningAvg 
            angular = self.angularZRunningAvg
        odometryMsg.twist.twist.linear = Vector3(*linear)
        odometryMsg.twist.twist.angular = Vector3(*angular)
        self.twistPub.publish(odometryMsg)

    def odomCallback(self, msg):
        t = TransformStamped()
        t.header = msg.header
        t.header.frame_id = "map"
        t.child_frame_id = "svea5"
        t.transform.translation = msg.pose.pose.position
        t.transform.rotation = msg.pose.pose.orientation

        self.br.sendTransform(t)

    def OdomVelCallback(self, Odommsg, Velmsg):
        try:
            self.publishBaselink(Odommsg)
            transform_base_map = self.buffer.lookup_transform("self.svea_frame_name", "map", rospy.Time(), rospy.Duration(0.5))
            transMsg = Vector3Stamped()
            print("transform_base_map", transform_base_map)
            transMsg.header = Velmsg.header
            transMsg.vector = Velmsg.twist.linear
            print("tf2", tf2_geometry_msgs.do_transform_vector3(transMsg, transform_base_map))
            trans_linear = tf2_geometry_msgs.do_transform_vector3(transMsg, transform_base_map).vector
            self.publishTwist(Velmsg, trans_linear)   
        except Exception as e:
            print(e)
    def run(self):
        while not rospy.is_shutdown():
            # self.pubCameraInfo()
            for i in range(1):
                if self.startTime == None:
                    self.startTime = rospy.Time.now()
                    self.time = self.startTime
                    self.current_time = (self.time - self.startTime).to_sec()
                else:
                    self.time = rospy.Time.now()
                    self.current_time = (self.time - self.startTime).to_sec()
                # if self.motion != "bag":
                    # self.publishBaselink(None)
                    # self.publishTwist(None, None)
                    # rospy.sleep(0.001)
            self.publishAruco()
            self.seq += 1

    def pubCameraInfo(self):
        cam = CameraInfo()
        cam.height = 720
        cam.width = 1280
        cam.distortion_model = "plumb_bob"
        cam.D = [0.1047628804269683, -0.2294792634248884, -0.00246960020570932, -0.002432772303304122, 0]
        cam.K = [1046.017603382867, 0, 637.7297160039146, 0, 1045.709085897633, 344.7307046566843, 0, 0, 1]
        cam.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        cam.P = [1053.760165048383, 0, 634.6699965276354, 0, 0, 1059.559386601993, 343.5753848730548, 0, 0, 0, 1, 0]
        self.campub.publish(cam)

    def publishTwist(self, Msg, transform):
        odometryMsg = Odometry()
        odometryMsg.header.seq = self.seq
        odometryMsg.header.stamp = self.time
        odometryMsg.header.frame_id = self.frame
        odometryMsg.child_frame_id = self.svea_frame_name
        if self.motion == "static":
            linear = [0, 0, 0]
            angular = [0, 0, 0]
        elif self.motion == "linear":
            linear = [0.2, 0, 0]
            # linear = [np.sin(0.2*self.current_time), 0, 0]
            angular = [0, 0, 0]
            odometryMsg.pose.pose.position = Vector3(*[0.2*self.current_time-5, 2.5, 4.5])
            rotation = np.array([0, 0, 0, 0.5])
            rotation /= np.linalg.norm(rotation)
            odometryMsg.pose.pose.orientation = Quaternion(*rotation) #x, y, z, w
        elif self.motion == "angular":
            linear = [-np.sin(0.4*self.current_time), np.cos(0.4*self.current_time), 0]
            angular = [0, 0, 0.3]
        elif self.motion == "both":
            linear = [0.3, 0, 0]
            angular = [0, 0, 0.3]
        elif self.motion == "bag":
            if AveAll:
                self.linearXRunningAvg = self.alpha * np.array([transform.x, transform.y, transform.z]) + (1 - self.alpha) * self.linearXRunningAvg
                self.angularZRunningAvg = self.alphaAng * np.array([Msg.twist.angular.x, Msg.twist.angular.y, Msg.twist.angular.z]) + (1 - self.alphaAng) * self.angularZRunningAvg
                linear = self.linearXRunningAvg 
                angular = self.angularZRunningAvg
            else:
                self.linearXRunningAvg = self.alpha *transform.x + (1 - self.alpha) * self.linearXRunningAvg
                self.angularZRunningAvg = self.alphaAng * Msg.twist.angular.z + (1 - self.alphaAng) * self.angularZRunningAvg
                linear = [self.linearXRunningAvg, transform.y, transform.z]
                angular = [Msg.twist.angular.x, Msg.twist.angular.y, self.angularZRunningAvg]
        elif self.motion == "test":
            linear = [0, 0, 0]
            angular = [0, 0, 0.3]

        odometryMsg.twist.twist.linear = Vector3(*linear)
        odometryMsg.twist.twist.angular = Vector3(*angular)
        self.twistPub.publish(odometryMsg)
        # print(odometryMsg)

    def publishBaselink(self, odomMsg):
        msg = TransformStamped()
        msg.header.seq = self.seq
        msg.header.stamp = self.time
        msg.header.frame_id = "map"
        msg.child_frame_id = self.svea_frame_name
        if self.motion == "static": 
            msg.transform.translation = Vector3(*[-1.5, 0.5 , 0])
            rotation = [0, 0, 0, 1]
            rotation = rotation/np.linalg.norm(rotation)
            msg.transform.rotation = Quaternion(*rotation) #x, y, z, w
        elif self.motion == "linear":
            # msg.transform.translation = Vector3(*[-1/0.2*np.cos(0.2*self.current_time) + 1/0.2 - 2, -2, 0])
            msg.transform.translation = Vector3(*[0.2*self.current_time-2, -2, 0])
            rotation = np.array([0, 0, 0, 0.5])

            # msg.transform.translation = Vector3(*[0, -1/0.4*np.cos(0.4*self.current_time) + 1/0.4,0])
            # rotation = np.array([0, 0, 0.5, 0.5])
            rotation /= np.linalg.norm(rotation)
            msg.transform.rotation = Quaternion(*rotation) #x, y, z, w
        elif self.motion == "angular":
            msg.transform.translation = Vector3(*[2*np.cos(0.4*self.current_time), 2*np.sin(0.4*self.current_time), 0])
            # msg.transform.rotation = Quaternion(*[0, 0, 0, 1]) #x, y, z, w
            theta = 0.4 * self.current_time + np.pi/2
            quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
            msg.transform.rotation = Quaternion(*quaternion) #x, y, z, w
        elif self.motion == "both":
            msg.transform.translation = Vector3(*[0.3*self.current_time, 0, 10])
            msg.transform.rotation = Quaternion(*[0, 0, 0, 1]) #x, y, z, w
            # msg.transform.rotation = Quaternion(*[0, 0, np.sin(0.3*self.current_time/2), np.cos(0.3*self.current_time/2)]) #x, y, z, w
        elif self.motion == "bag":
            msg.transform.translation = odomMsg.pose.pose.position
            msg.transform.rotation = odomMsg.pose.pose.orientation #x, y, z, w
        elif self.motion == "test":
            print(self.pose[self.stepCounter])
            print(self.ori[self.stepCounter])
            
            msg.transform.translation = Vector3(*self.pose[self.stepCounter])
            msg.transform.rotation = Quaternion(*self.ori[self.stepCounter]) #x, y, z, w
            self.stepCounter += 1
        self.br.sendTransform(msg)
        visualarr = VMA()
        visualmarker = VM()
        visualmarker.header.stamp = self.time
        visualmarker.header.frame_id = "map"
        visualmarker.ns = "sveapose"
        visualmarker.id = 1
        visualmarker.type = VM.CUBE
        visualmarker.action = VM.ADD
        visualmarker.pose.position.x = msg.transform.translation.x
        visualmarker.pose.position.y = msg.transform.translation.y
        visualmarker.pose.position.z = msg.transform.translation.z
        visualmarker.pose.orientation.w = msg.transform.rotation.w
        visualmarker.pose.orientation.x = msg.transform.rotation.x
        visualmarker.pose.orientation.y = msg.transform.rotation.y
        visualmarker.pose.orientation.z = msg.transform.rotation.z
        visualmarker.scale = Vector3(*[0.2, 0.2, 0.2])
        visualmarker.color = ColorRGBA(*[0, 1, 0, 1.0])
        visualarr.markers.append(visualmarker)
        self.visualArucoPub.publish(visualarr)

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
        visualarr = VMA()
        list2d = np.array([[0.13244211551237228,0.03685241983429774], [-0.47923498405224174, 0.014646746223398321], [-0.24344701507965696, 0.0006049232047413362], [0.2617662646995518, 0.014973300247088019]])

        try:
            transform_map_baselink = self.buffer.lookup_transform("camera","map", rospy.Time(), rospy.Duration(0.5))  #rospy.Time.now()
            count = 0
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
                points_3d = np.array([position.pose.position.x, position.pose.position.y, position.pose.position.z])

                arucoMsg2.image_x = list2d[count][0]
                arucoMsg2.image_y = list2d[count][1]
                count += 1
                arucoArrayMsg2.arucos.append(arucoMsg2)

                msg = TransformStamped()
                msg.header.seq = self.seq
                msg.header.stamp = self.time
                msg.header.frame_id = "camera"
                msg.child_frame_id = 'lmBaseLink' + str(id)
                msg.transform.translation = position.pose.position
                msg.transform.rotation = position.pose.orientation #x, y, z, w
                self.br.sendTransform(msg)

                visualmarker = VM()
                visualmarker.header = arucoMsg.marker.header
                visualmarker.header.frame_id = "map"
                visualmarker.ns = "Aruco"
                visualmarker.id = id
                visualmarker.type = VM.SPHERE
                visualmarker.action = VM.ADD
                visualmarker.pose = arucoMsg.marker.pose.pose
                visualmarker.scale = Vector3(*[0.2, 0.2, 0.2])
                visualmarker.color = ColorRGBA(*[0.0, 0.0, 1.0, 1.0])
                visualarr.markers.append(visualmarker)


            self.visualArucoPub.publish(visualarr)


            self.landmarkPub.publish(arucoArrayMsg)
            self.landmark2Pub.publish(arucoArrayMsg2)
        except Exception as e:
            print(f"{e}")
            # print("waiting for transform between " + self.svea_frame_name + " and map")

if __name__ == "__main__":
    dummyMeasurement().run()
