#! /usr/bin/env python3
import psutil
import os

import rospy
import numpy as np
import tf2_ros

from message_filters import Subscriber, ApproximateTimeSynchronizer

from riccati_observer import riccati_observer

from std_msgs.msg import Float32, ColorRGBA
from svea_msgs.msg import Aruco, ArucoArray, riccati_setup
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped, TransformStamped, Point, Quaternion, Vector3, PoseStamped, Vector3Stamped, TwistStamped, PoseArray, Pose
from sensor_msgs.msg import CameraInfo
import tf2_geometry_msgs
import tf.transformations

from visualization_msgs.msg import Marker as VM
from visualization_msgs.msg import MarkerArray as VMA

WITH_LANDMARK = True

ERROR_MSG = {-1:"No error",
             0: "Not enough source points",
             1: "Algined source points",
             2: "Three non-aligned source points: moving along one of the straight lines of the danger cylinder and passing through a source point",
             3: "Three non-aligned source points: motionless C in danger cylinder",
             4: "Three non-aligned source points: moving on danger cylinder but not along any lines (weak)",
             5: "Four + non-aligned source points: on horopter curve",
            }

class riccati_estimation():
    def __init__(self):
        rospy.init_node('riccati_estimation')

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()
        self.sbr = tf2_ros.StaticTransformBroadcaster()

        ##############################################
        ################# Variables ##################
        self.timeStamp = None
        self.startTime = None
        self.svea_frame_name = "base_link"
        self.seq = 0
        self.t = 0
        
        self.camera_to_base_transform = None
        while self.camera_to_base_transform == None:
            self.GetStaticTransform()
        self.CameraInfo = None

        # for debug
        self.stop = 0
        ################# Variables ##################
        ##############################################

        ##############################################
        ################# Publisher ##################
        self.dtPublisher = rospy.Publisher("/riccati/dt", Float32, queue_size=10)
        self.RiccatiSetupPublisher = rospy.Publisher("/riccati/setup", riccati_setup, queue_size=10)
        self.RiccatiDirPublisher = rospy.Publisher("/riccati/diretions", PoseArray, queue_size=10)

        self.usedLandmark = rospy.Publisher('/aruco/used', ArucoArray, queue_size=10)
        self.debugTopic = rospy.Publisher('/riccati/debug', TransformStamped, queue_size=1)
        self.visualArucoPub = rospy.Publisher("/visual/cameraaruco", VMA, queue_size=10)
        ################# Publisher ##################
        ##############################################

        ##############################################
        ################# Parameters #################
        k = rospy.get_param("~k", 150)
        q = rospy.get_param("~q", 10)
        v1 = rospy.get_param("~v1", 1)
        v2 = rospy.get_param("~v2", 10)

        self.estpose = np.array([1.7, -0.5, 0], dtype=np.float64)
        self.estori = np.array([0, 0, 0, -1], dtype=np.float64) #w, x, y, z #base_link
        self.initpose = self.estpose
        self.estori /= np.linalg.norm(self.estori)
        self.initori = self.estori #w, x, y, z
        ################# Parameters #################
        ##############################################
        self.riccati_obj = riccati_observer(
        stepsize                = 0.1,
        tol                     = 1e-2 * 1, #1e-2 * 3,
        which_eq                = 0,
        p_hat                   = self.estpose, # sth from state or just input from lanuch file,
        Lambda_bar_0            = self.estori, #np.hstack((self.estori[-1], self.estori[0:-1])), # sth from state or just input from lanuch file,  # quaternion: w, x, y, z
        z_appear                = np.array([]),
        k                       = k,
        q                       = q, 
        v                       = np.array([v1, v2]),
        p_riccati               = np.array([1, 100])
        )
        ##############################################
        ################# Subscriber #################

        Twist = Subscriber('/actuation_twist', TwistWithCovarianceStamped)
        # Twist = Subscriber('/odometry/filtered', Odometry)
        
        Landmark = Subscriber('/aruco/detection/more', ArucoArray)
        LandmarkGroudtruth = Subscriber('/aruco/detection/Groundtruth', ArucoArray)

        if WITH_LANDMARK:
            sync = ApproximateTimeSynchronizer([Landmark, LandmarkGroudtruth], queue_size=1, slop=1) #maybe????, but should only apply to cases with changing velocity
            sync.registerCallback(self.TwistAndLandmarkCallback)
        else:
            sync = ApproximateTimeSynchronizer([Twist], queue_size=1, slop=0.2)
            sync.registerCallback(self.TwistSyncCallback)

        self.cameraSub = rospy.Subscriber("/camera/camera_info", CameraInfo, self.CameraInfoCallback)
        ################# Subscriber #################
        ##############################################
        
        self.pubinit = False
        # self.pub_EstPose(rospy.Time.now(), 0)
        self.changeFrame(rospy.Time.now())
    def GetStaticTransform(self):
        try:
            # self.camera_to_base_transform = self.buffer.lookup_transform("camera", self.svea_frame_name, rospy.Time(), rospy.Duration(2)) #frame id = camera, child = svea5
            transCamBase = self.buffer.lookup_transform(self.svea_frame_name, "camera", rospy.Time(), rospy.Duration(2)) #frame id = svea5, child = camera 
            self.camera_to_base_transform = PoseStamped()
            self.camera_to_base_transform.pose.position = transCamBase.transform.translation
            self.camera_to_base_transform.pose.orientation = transCamBase.transform.rotation
            print(self.camera_to_base_transform)
        except Exception as e:
            print(f"/ricatti_estimation/GetStaticTransform: {e}")

    def CameraInfoCallback(self, msg):
        self.CameraInfo = msg
        self.cameraSub.unregister()

    def changeFrame(self, timeStamp):
        # print("CheckFrame")
        pose = np.matmul(self.riccati_obj.rodrigues_formula(self.estori), self.estpose) #baselink in map frame
        msg2 = TransformStamped()
        msg2.header.seq = self.seq
        msg2.header.stamp = timeStamp #+ rospy.Duration(dt)
        msg2.header.frame_id = 'map'
        msg2.child_frame_id = 'base_link_est'

        msg2.transform.translation.x = pose[0]
        msg2.transform.translation.y = pose[1]
        msg2.transform.translation.z = pose[2]
        msg2.transform.rotation.w = self.estori[0]
        msg2.transform.rotation.x = self.estori[1]
        msg2.transform.rotation.y = self.estori[2]
        msg2.transform.rotation.z = self.estori[3]
        self.sbr.sendTransform(msg2)
        # print(msg2)
        transformed_direction = tf2_geometry_msgs.do_transform_pose(self.camera_to_base_transform, msg2) 
        # print("transformed_direction", transformed_direction)
        msg = TransformStamped()
        msg.header.seq = self.seq
        msg.header.stamp = timeStamp #+ rospy.Duration(dt)
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'camera_est'

        msg.transform.translation = transformed_direction.pose.position
        msg.transform.rotation = transformed_direction.pose.orientation
        self.estori = np.array([transformed_direction.pose.orientation.w, transformed_direction.pose.orientation.x, transformed_direction.pose.orientation.y, transformed_direction.pose.orientation.z])
        self.estpose = np.array([transformed_direction.pose.position.x, transformed_direction.pose.position.y, transformed_direction.pose.position.z])
        pose = np.matmul(np.transpose(self.riccati_obj.rodrigues_formula(self.estori)), self.estpose)
        self.estpose = pose
        self.riccati_obj.set_init(self.estori, self.estpose)
        self.sbr.sendTransform(msg)
        # print(msg)
        self.seq += 1
        self.pubinit = True

    def pub_EstPose(self, timeStamp, dt):
            # print("pub_EstPose")
            pose = np.matmul(self.riccati_obj.rodrigues_formula(self.estori), self.estpose) #baselink in map frame

            msg2 = TransformStamped()
            msg2.header.seq = self.seq
            msg2.header.stamp = timeStamp #+ rospy.Duration(dt)
            msg2.header.frame_id = 'map'
            msg2.child_frame_id = 'camera_est'

            msg2.transform.translation.x = pose[0]
            msg2.transform.translation.y = pose[1]
            msg2.transform.translation.z = pose[2]
            msg2.transform.rotation.w = self.estori[0]
            msg2.transform.rotation.x = self.estori[1]
            msg2.transform.rotation.y = self.estori[2]
            msg2.transform.rotation.z = self.estori[3]

            self.sbr.sendTransform(msg2)
            # print(msg2)
            self.debugTopic.publish(msg2)
            transBaseCam = self.buffer.lookup_transform("camera", self.svea_frame_name, rospy.Time(), rospy.Duration(2)) #frame id = svea5, child = camera 
            transBaseCamPose = PoseStamped()
            transBaseCamPose.pose.position = transBaseCam.transform.translation
            transBaseCamPose.pose.orientation = transBaseCam.transform.rotation
            transformed_direction = tf2_geometry_msgs.do_transform_pose(transBaseCamPose, msg2) 

            msg = TransformStamped()
            msg.header.seq = self.seq
            msg.header.stamp = timeStamp #+ rospy.Duration(dt)
            msg.header.frame_id = 'map'
            msg.child_frame_id = 'base_link_est'

            msg.transform.translation = transformed_direction.pose.position
            msg.transform.rotation = transformed_direction.pose.orientation

            # print(msg)
            self.sbr.sendTransform(msg)

            self.seq += 1

            visualarr = VMA()
            visualmarker = VM()
            visualmarker.header.stamp = timeStamp
            visualmarker.header.frame_id = "map"
            visualmarker.ns = "sveapose"
            visualmarker.id = 0
            visualmarker.type = VM.CUBE
            visualmarker.action = VM.ADD
            visualmarker.pose.position.x = msg2.transform.translation.x
            visualmarker.pose.position.y = msg2.transform.translation.y
            visualmarker.pose.position.z = msg2.transform.translation.z
            visualmarker.pose.orientation.w = self.estori[0]
            visualmarker.pose.orientation.x = self.estori[1]
            visualmarker.pose.orientation.y = self.estori[2]
            visualmarker.pose.orientation.z = self.estori[3]
            visualmarker.scale = Vector3(*[0.2, 0.2, 0.2])
            visualmarker.color = ColorRGBA(*[255.0/450.0, 92.0/450.0, 103.0/450.0, 1.0])
            visualarr.markers.append(visualmarker)
            self.visualArucoPub.publish(visualarr)

    def pubRiccatiMsg(self):
        riccatiMsg = riccati_setup()
        riccatiMsg.stepsize = self.riccati_obj.stepsize
        riccatiMsg.tol = self.riccati_obj.tol
        riccatiMsg.which_eq = self.riccati_obj.which_eq
        riccatiMsg.k = self.riccati_obj.k
        riccatiMsg.q = self.riccati_obj.q
        riccatiMsg.v = self.riccati_obj.v
        riccatiMsg.pose.position = Point(*self.initpose)
        riccatiMsg.pose.orientation.w = self.initori[0]
        riccatiMsg.pose.orientation.x = self.initori[1]
        riccatiMsg.pose.orientation.y = self.initori[2]
        riccatiMsg.pose.orientation.z = self.initori[3]

        self.RiccatiSetupPublisher.publish(riccatiMsg)

    def visualizeAruco(self, ArucoList, landmark):
        visualarr = VMA()
        for idx, aruco in enumerate(ArucoList.arucos):
            visualmarker = VM()
            visualmarker.header = aruco.marker.header
            visualmarker.header.frame_id = "svea5"
            visualmarker.ns = "cameraAruco"
            visualmarker.id = aruco.marker.id
            visualmarker.type = VM.SPHERE
            visualmarker.action = VM.ADD
            visualmarker.pose.position = Point(*landmark[idx])
            visualmarker.pose.orientation = Quaternion(*[0, 0, 0, 1])
            visualmarker.scale = Vector3(*[0.2, 0.2, 0.2])
            visualmarker.color = ColorRGBA(*[1.0, 0.0, 0.0, 1.0])
            visualarr.markers.append(visualmarker)
        self.visualArucoPub.publish(visualarr)
    
    def visualDirection(self, z, msg, arucoId):
        dirs = self.riccati_obj.calculate_direction(z)
        Pmsg = PoseArray()
        Pmsg.header = msg.header

        for landmark_idx in range(len(z)):
            d = dirs[landmark_idx]
            Tmsg = TransformStamped()
            Tmsg.header = msg.header
            Tmsg.header.frame_id = "camera_est"
            Tmsg.child_frame_id = 'Direction' + str(arucoId[landmark_idx])
            Tmsg.transform.translation = Vector3(*d)
            Tmsg.transform.rotation = Quaternion(*[0, 0, 0, 1]) #x, y, z, w
            self.sbr.sendTransform(Tmsg)
            posemsg = Pose()
            posemsg.position = Point(*d)
            posemsg.orientation = Quaternion(*[0, 0, 0, 1])
            Pmsg.poses.append(posemsg)
        self.RiccatiDirPublisher.publish(Pmsg)

    def TwistAndLandmarkCallback(self, LandmarkMsg, LandmarkGroudtruthMsg):
        # if self.stop <1:
        if self.camera_to_base_transform != None and self.CameraInfo != None and self.pubinit:
            if self.startTime == None:
                self.startTime = LandmarkMsg.header.stamp
            self.pubRiccatiMsg()
            self.timeStamp = LandmarkMsg.header.stamp
            
            linear_velocity = np.array([0, 0, 0])
            angular_velocity = np.array([0, 0, 0])

            landmark = []
            landmarkGroundTruth = []
            arucosUsed = ArucoArray()
            arucosUsed.header = LandmarkMsg.header
            arucoId = []

            for aruco in LandmarkMsg.arucos:
                temp = np.array([aruco.image_x, aruco.image_y, 1])
                temp /= np.linalg.norm(temp)
                landmark.append(temp)
                arucosUsed.arucos.append(aruco)
                arucoId.append(aruco.marker.id)
                # print(aruco.marker.id)
                # print("2d", temp)
                # b = np.array([aruco.marker.pose.pose.position.x, aruco.marker.pose.pose.position.y, aruco.marker.pose.pose.position.z])
                # b /= np.linalg.norm(b)
                # print("3d", b)
                for ArucoGroundtruth in LandmarkGroudtruthMsg.arucos:
                    if ArucoGroundtruth.marker.id == aruco.marker.id:
                        # in map frame
                        landmarkGroundTruth.append([ArucoGroundtruth.marker.pose.pose.position.x, ArucoGroundtruth.marker.pose.pose.position.y, ArucoGroundtruth.marker.pose.pose.position.z])
                        break

            self.riccati_obj.update_measurement(angular_velocity, linear_velocity, landmark, landmarkGroundTruth, (self.timeStamp - self.startTime).to_sec())
            self.usedLandmark.publish(arucosUsed)

            self.visualizeAruco(arucosUsed, landmark)
            self.visualDirection(landmark, LandmarkMsg, arucoId)
            
            solt, dt, soly, errMsg = self.riccati_obj.step_simulation()
            dtMsg = Float32()
            dtMsg.data = dt
            self.dtPublisher.publish(dtMsg)
            self.t = solt

            ############################# Quaternion
            ############################# 
            self.estpose = soly[4:7]
            # print("self.estpose", self.estpose)
            self.estori = soly[0:4]
            self.estori /= np.linalg.norm(self.estori)
            ############################# Quaternion
            #############################

            ############################# 
            ############################# Rot Mat        
            # self.estpose = soly[9:12]
            # Rotmat = soly[0:9].reshape((3,3))
            # Translmat = soly[9:12].reshape((3,1))
            # Transmat = np.hstack((Rotmat, Translmat))
            # Transmat = np.vstack((Transmat, np.array([0,0,0,1])))
            # self.estori = tf.transformations.quaternion_from_matrix(Transmat) 
            # self.estori = np.concatenate(([self.estori[-1]], self.estori[0:-1]))
            # self.estori /= np.linalg.norm(self.estori)
            ############################# Rot Mat        
            #############################        
            self.pub_EstPose(self.timeStamp, dt)
            print("===================================================")
            self.stop += 1


    def TwistSyncCallback(self, TwistMsg):
        if self.startTime == None:
            self.startTime = TwistMsg.header.stamp
        self.pubRiccatiMsg()
        self.timeStamp = TwistMsg.header.stamp

        linear_velocity = np.array([TwistMsg.twist.twist.linear.x, TwistMsg.twist.twist.linear.y, TwistMsg.twist.twist.linear.z])
        angular_velocity = np.array([TwistMsg.twist.twist.angular.x, TwistMsg.twist.twist.angular.y, TwistMsg.twist.twist.angular.z])
        # angular_velocity = np.array([0, 0, 0])
        print("linear_velocity ", linear_velocity)
        print("angular_velocity ", angular_velocity)

        self.riccati_obj.update_measurement(angular_velocity, linear_velocity, [], [], (self.timeStamp - self.startTime).to_sec())
        solt, dt, soly, errMsg = self.riccati_obj.step_simulation()
        rospy.loginfo(f'errMsg: {ERROR_MSG[errMsg]}')
        dtMsg = Float32()
        dtMsg.data = dt
        self.dtPublisher.publish(dtMsg)
        self.t = solt

        ############################# Quaternion
        ############################# 
        self.estpose = soly[4:7]
        print("self.estpose", self.estpose)
        self.estori = soly[0:4]
        self.estori /= np.linalg.norm(self.estori)
        ############################# Quaternion
        #############################

        ############################# 
        ############################# Rot Mat        
        # self.estpose = soly[9:12]
        # Rotmat = soly[0:9].reshape((3,3))
        # Translmat = soly[9:12].reshape((3,1))
        # Transmat = np.hstack((Rotmat, Translmat))
        # Transmat = np.vstack((Transmat, np.array([0,0,0,1])))
        # self.estori = tf.transformations.quaternion_from_matrix(Transmat) 
        # self.estori = np.concatenate(([self.estori[-1]], self.estori[0:-1]))
        # self.estori /= np.linalg.norm(self.estori)
        ############################# Rot Mat        
        #############################        
        self.pub_EstPose(self.timeStamp, dt)
        print("===================================================")
        self.stop += 1

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    current_pid = os.getpid()
    process = psutil.Process(current_pid)
    process.cpu_affinity([2])
    riccati_estimation().run()