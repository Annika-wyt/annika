#! /usr/bin/env python3

import rospy
import numpy as np
import tf2_ros

from message_filters import Subscriber, ApproximateTimeSynchronizer

from ricatti_observer import riccati_observer

from std_msgs.msg import Float32, ColorRGBA
from svea_msgs.msg import Aruco, ArucoArray, riccati_setup
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped, TransformStamped, Point, Quaternion, Vector3, PoseStamped, Vector3Stamped, TwistStamped
import tf2_geometry_msgs
import tf.transformations

from visualization_msgs.msg import Marker as VM
from visualization_msgs.msg import MarkerArray as VMA

WITH_LANDMARK = True
class ricatti_estimation():
    def __init__(self):
        rospy.init_node('ricatti_estimation')

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()
        self.sbr = tf2_ros.StaticTransformBroadcaster()

        self.stop = 0
        # self.startTime = rospy.Time.now()
        self.startTime = None
        self.svea_frame_name = "svea2"
        ##############################################
        ################# Subscriber #################

        # rospy.Subscriber('/actuation_to_twist', TwistWithCovarianceStamped, self.TwistCallback)
        # rospy.Subscriber('/aruco/detection', ArucoArray, self.LandmarkCallback)
        #TODO: change the aruco_detect.py so that it will publish ArucoArray even with no data
        
        self.seq = 0
        
        self.t = 0
        ## Sub to state
        ################# Subscriber #################
        ##############################################

        self.dtPublisher = rospy.Publisher("/ricatti/dt", Float32, queue_size=10)
        self.RiccatiSetupPublisher = rospy.Publisher("/ricatti/setup", riccati_setup, queue_size=10)
        ##############################################
        ################# Variables ##################

        # self.linear_velocity = None
        # self.angular_velocity = None
        # self.lanmark = []

        ################# Variables ##################
        ##############################################
        self.timeStamp = None
        self.syncCallback = False
        self.INtwistcallback = False
        ##############################################
        ################# Parameters #################
        k = rospy.get_param("~k", 8)
        q = rospy.get_param("~q", 10)
        v1 = rospy.get_param("~v1", 0.1)
        v2 = rospy.get_param("~v2", 0.8)
        self.arucoIdToBeUsed = np.array([10,11,12,13,14,15,16,17,18]) #12-18
        # estpose = rospy.get_param("~estpose", np.array([0, 0, 0]))
        # estori = rospy.get_param("~estori", np.array([1, 0, 0, 0]))
        # self.estori = np.array([0, 0, -0.5, 0.5], dtype=np.float64) #x, y, z, w
        self.estpose = np.array([0, 0, 0], dtype=np.float64)
        # self.estori = np.array([0.5, 0, 0, 0], dtype=np.float64) #w, x, y, z
        self.estori = np.array([0, 0, 0, -0.5], dtype=np.float64) #w, x, y, z
        self.initpose = self.estpose
        self.estori /= np.linalg.norm(self.estori)
        self.initori = self.estori #w, x, y, z
        # estori = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2]) #x, y, z, w
        ################# Parameters #################
        ##############################################
        self.riccati_obj = riccati_observer(
        stepsize                = 0.01,
        tol                     = 1e-2 * 5,
        which_eq                = 0,
        p_hat                   = self.estpose, # sth from state or just input from lanuch file,
        Lambda_bar_0            = self.estori, #np.hstack((self.estori[-1], self.estori[0:-1])), # sth from state or just input from lanuch file,  # quaternion: w, x, y, z
        z_appear                = np.array([]),
        k                       = k,
        q                       = q, 
        v                       = np.array([v1, v2]),
        p_ricatti               = np.array([1, 100])
        )
        Twist = Subscriber('/actuation_twist', TwistWithCovarianceStamped)
        # Twist = Subscriber('/odometry/filtered', Odometry)
        
        Landmark = Subscriber('/aruco/detection', ArucoArray)
        LandmarkGroudtruth = Subscriber('/aruco/detection/Groundtruth', ArucoArray)

        if WITH_LANDMARK:
            sync = ApproximateTimeSynchronizer([Twist, Landmark, LandmarkGroudtruth], queue_size=1, slop=3)
            sync.registerCallback(self.TwistAndLandmarkCallback)
            rospy.Subscriber('/actuation_twist', TwistWithCovarianceStamped, self.twistCallback)
        else:
            sync = ApproximateTimeSynchronizer([Twist], queue_size=10, slop=2)
            sync.registerCallback(self.TwistSyncCallback)

        self.usedLandmark = rospy.Publisher('/aruco/used', ArucoArray, queue_size=10)
        self.debugTopic = rospy.Publisher('/riccati/debug', TransformStamped, queue_size=1)
        self.visualArucoPub = rospy.Publisher("/visual/cameraaruco", VMA, queue_size=10)
        
        self.time = rospy.Time.now()
        self.pubOdom2(self.time)
        self.pub_EstPose(self.time, 0)



    def pubOdom2(self, timeStamp):
        msg = TransformStamped()
        msg.header.seq = self.seq
        msg.header.stamp = timeStamp # + rospy.Duration(dt)
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'odom2'
        msg.transform.translation.x = 0 #self.initpose[0]
        msg.transform.translation.y = 0 #self.initpose[1]
        msg.transform.translation.z = 0 #self.initpose[2]
        # msg.transform.rotation = Quaternion(*self.estori) #x, y, z, w
        msg.transform.rotation.w = self.estori[0]
        msg.transform.rotation.x = self.estori[1]
        msg.transform.rotation.y = self.estori[2]
        msg.transform.rotation.z = self.estori[3]
        self.sbr.sendTransform(msg)
        self.seq += 1

    def pub_EstPose(self, timeStamp, dt):
        # self.pubOdom2(timeStamp)
        msg = TransformStamped()
        msg.header.seq = self.seq
        msg.header.stamp = timeStamp # + rospy.Duration(dt)
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link_est'
        pose = np.matmul(self.riccati_obj.rodrigues_formula(self.estori), self.estpose) #map frame
        msg.transform.translation.x = pose[0] #self.estpose -> in B frame
        msg.transform.translation.y = pose[1]
        msg.transform.translation.z = pose[2]
        # msg.transform.rotation = Quaternion(*self.estori) #x, y, z, w
        msg.transform.rotation.w = self.estori[0]
        msg.transform.rotation.x = self.estori[1]
        msg.transform.rotation.y = self.estori[2]
        msg.transform.rotation.z = self.estori[3]
        self.sbr.sendTransform(msg)
        self.seq += 1
        self.debugTopic.publish(msg)

    def pub_landmark(self, landmark,id):
        msg = TransformStamped()
        msg.header = landmark.header
        msg.header.frame_id = self.svea_frame_name
        msg.child_frame_id = 'lm' + str(id)
        msg.transform.translation = landmark.pose.position
        msg.transform.rotation = landmark.pose.orientation #x, y, z, w
        self.sbr.sendTransform(msg)

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

    def visualizeAruco(self, ArucoList):
        visualarr = VMA()
        for aruco in ArucoList.arucos:
            visualmarker = VM()
            visualmarker.header = aruco.marker.header
            visualmarker.header.frame_id = "base_link_est"
            visualmarker.ns = "cameraAruco"
            visualmarker.id = aruco.marker.id
            visualmarker.type = VM.SPHERE
            visualmarker.action = VM.ADD
            visualmarker.pose = aruco.marker.pose.pose
            visualmarker.scale = Vector3(*[0.2, 0.2, 0.2])
            visualmarker.color = ColorRGBA(*[1.0, 0.0, 0.0, 1.0])
            visualarr.markers.append(visualmarker)
        self.visualArucoPub.publish(visualarr)

    def visualDirection(self, z, msg, arucoId):
        for landmark_idx in range(len(z)):
            d = np.array(z[landmark_idx]/ np.linalg.norm(z[landmark_idx]))
            msg = TransformStamped()
            msg.header = msg.header
            msg.header.frame_id = "base_link_est"
            msg.child_frame_id = 'Direction' + str(arucoId[landmark_idx])
            msg.transform.translation = Vector3(*d)
            msg.transform.rotation = Quaternion(*[0, 0, 0, 1]) #x, y, z, w
            self.sbr.sendTransform(msg)

    def twistCallback(self, TwistMsg):
        if not self.syncCallback:
            if self.startTime == None:
                self.startTime = TwistMsg.header.stamp
            self.INtwistcallback = True

            linear_velocity = np.array([TwistMsg.twist.twist.linear.x, TwistMsg.twist.twist.linear.y, TwistMsg.twist.twist.linear.z])
            angular_velocity = np.array([TwistMsg.twist.twist.angular.x, TwistMsg.twist.twist.angular.y, TwistMsg.twist.twist.angular.z])
            self.timeStamp = TwistMsg.header.stamp
            self.riccati_obj.update_measurement(angular_velocity, linear_velocity, [], [], (self.timeStamp - self.startTime).to_sec())
            arucosUsed = ArucoArray()
            self.usedLandmark.publish(arucosUsed)
            self.visualizeAruco(arucosUsed)
            solt, dt, soly = self.riccati_obj.step_simulation()
            dtMsg = Float32()
            dtMsg.data = dt
            self.dtPublisher.publish(dtMsg)
            self.t = solt
            self.estpose = soly[4:7]
            print("self.estpose", self.estpose)
            self.estori = soly[0:4]
            self.estori /= np.linalg.norm(self.estori)
            self.pub_EstPose(self.timeStamp, dt)
            print("===================================================")
            self.INtwistcallback = False


    def TwistAndLandmarkCallback(self, TwistMsg, LandmarkMsg, LandmarkGroudtruthMsg):
        if not self.INtwistcallback:
            self.syncCallback = True
            if self.startTime == None:
                self.startTime = TwistMsg.header.stamp
            self.pubRiccatiMsg()
            self.timeStamp = TwistMsg.header.stamp

            linear_velocity = np.array([TwistMsg.twist.twist.linear.x, TwistMsg.twist.twist.linear.y, TwistMsg.twist.twist.linear.z])
            angular_velocity = np.array([TwistMsg.twist.twist.angular.x, TwistMsg.twist.twist.angular.y, TwistMsg.twist.twist.angular.z])

            landmark = []
            landmarkGroundTruth = []
            arucosUsed = ArucoArray()
            arucosUsed.header = LandmarkMsg.header
            arucoId = []
            for aruco in LandmarkMsg.arucos:
                landmark.append([aruco.marker.pose.pose.position.x, aruco.marker.pose.pose.position.y, aruco.marker.pose.pose.position.z])
                arucosUsed.arucos.append(aruco)
                arucoId.append(aruco.marker.id)
                for ArucoGroundtruth in LandmarkGroudtruthMsg.arucos:
                    if ArucoGroundtruth.marker.id == aruco.marker.id:
                        # in map frame
                        landmarkGroundTruth.append([ArucoGroundtruth.marker.pose.pose.position.x, ArucoGroundtruth.marker.pose.pose.position.y, ArucoGroundtruth.marker.pose.pose.position.z])
                        break
            print("linear_velocity ", linear_velocity)
            if len(landmark) < 3:
                self.riccati_obj.update_measurement(angular_velocity, linear_velocity, [], [], (self.timeStamp - self.startTime).to_sec())
                arucosUsed.arucos = []
                self.usedLandmark.publish(arucosUsed)
            else:
                self.riccati_obj.update_measurement(angular_velocity, linear_velocity, landmark, landmarkGroundTruth, (self.timeStamp - self.startTime).to_sec())
                self.usedLandmark.publish(arucosUsed)
            self.visualizeAruco(arucosUsed)
            self.visualDirection(landmark, LandmarkMsg, arucoId)
            solt, dt, soly = self.riccati_obj.step_simulation()
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
            self.syncCallback = False


    def TwistSyncCallback(self, TwistMsg):
        print("hi")
        if self.startTime == None:
            self.startTime = TwistMsg.header.stamp
        self.pubRiccatiMsg()
        self.timeStamp = TwistMsg.header.stamp

        linear_velocity = np.array([TwistMsg.twist.twist.linear.x, TwistMsg.twist.twist.linear.y, TwistMsg.twist.twist.linear.z])
        angular_velocity = np.array([TwistMsg.twist.twist.angular.x, TwistMsg.twist.twist.angular.y, TwistMsg.twist.twist.angular.z])
        print("linear_velocity ", linear_velocity)

        self.riccati_obj.update_measurement(angular_velocity, linear_velocity, [], [], (self.timeStamp - self.startTime).to_sec())
        solt, dt, soly = self.riccati_obj.step_simulation()
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
    ricatti_estimation().run()