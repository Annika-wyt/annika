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
import pandas as pd

class testFrame():
    def __init__(self):
        # Initalize the node 
        rospy.init_node('testFrame')

        # Read CSV
        path = '/home/annika/ITRL/kth_thesis/ricatti_noiseless_solution.txt'
        df = pd.read_csv(path, sep='\t', header=None)

        self.px = df.iloc[:,0].tolist() #Actual p x
        self.py = df.iloc[:,1].tolist() #Actual p x
        self.pz = df.iloc[:,2].tolist() #Actual p x

        self.estpx = df.iloc[:,3].tolist() # plot_est_p
        self.estpy = df.iloc[:,4].tolist() # plot_est_p
        self.estpz = df.iloc[:,5].tolist() # plot_est_p

        self.estpbarx = df.iloc[:,6].tolist() # sol_est_p_bar
        self.estpbary = df.iloc[:,7].tolist() # sol_est_p_bar
        self.estpbarz = df.iloc[:,8].tolist() # sol_est_p_bar

        self.errlambdabarx = df.iloc[:,9].tolist() # plot_err_lambda_bar 
        self.errlambdabary = df.iloc[:,10].tolist() # plot_err_lambda_bar 
        self.errlambdabarz = df.iloc[:,11].tolist() # plot_err_lambda_bar 
        self.errlambdabarw = df.iloc[:,12].tolist() # plot_err_lambda_bar 

        self.estlambdabarx = df.iloc[:,13].tolist() # plot_est_lambda_bar 
        self.estlambdabary = df.iloc[:,14].tolist() # plot_est_lambda_bar 
        self.estlambdabarz = df.iloc[:,15].tolist() # plot_est_lambda_bar 
        self.estlambdabarw = df.iloc[:,16].tolist() # plot_est_lambda_bar 

        self.actlambdabarx = df.iloc[:,17].tolist() # plot_act_lambda_bar
        self.actlambdabary = df.iloc[:,18].tolist() # plot_act_lambda_bar
        self.actlambdabarz = df.iloc[:,19].tolist() # plot_act_lambda_bar
        self.actlambdabarw = df.iloc[:,20].tolist() # plot_act_lambda_bar

        self.time = df.iloc[:,21].tolist() # time


        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.sbr = tf2_ros.StaticTransformBroadcaster()
        self.br = tf2_ros.TransformBroadcaster()

        self.pose = None
        self.estpose = None
        self.landmarks = [[0, 0, 0], [5, 0, 0], [2.5, 2.5, 0]]
        self.rot = None
        self.estrot = None
        self.seq = 0
        self.seqt = 0
        self.t = 0
        self.R = np.eye(3)

    def run(self):
        while not rospy.is_shutdown():
            self.pub_map_odom()
            self.pub_map_base()
            self.pub_landmarks()
            self.pub_dir()
            self.seq = (self.seq+1)%100
            self.t = self.time[self.seq]
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
        self.pose = [self.px[self.seq], self.py[self.seq], self.pz[self.seq]]
        msg.transform.translation = Vector3(*self.pose)
        self.rot = [self.actlambdabarx[self.seq], self.actlambdabary[self.seq], self.actlambdabarz[self.seq], self.actlambdabarw[self.seq]]
        msg.transform.rotation = Quaternion(*self.rot)
        self.br.sendTransform(msg)

        msg.header.frame_id = 'map'
        msg.child_frame_id = 'est_base_link'
        self.estpose = [self.estpx[self.seq], self.estpy[self.seq], self.estpz[self.seq]]
        msg.transform.translation = Vector3(*self.estpose)
        self.estrot = [self.estlambdabarx[self.seq], self.estlambdabary[self.seq], self.estlambdabarz[self.seq], self.estlambdabarw[self.seq]]
        msg.transform.rotation = Quaternion(*self.estrot)
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

            msg.header.frame_id = 'base_link'
            msg.child_frame_id = 'est_landmark' + str(idx)
            rot = ScipyRot.from_quat(self.estrot)
            rot = ScipyRot.as_matrix(rot)
            msg.transform.translation = Vector3(*np.matmul(np.transpose(rot), lm))
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
            rot = ScipyRot.from_quat(self.rot)
            rot = ScipyRot.as_matrix(rot)
            d = -np.matmul(np.transpose(rot), (pose-lm)/np.linalg.norm(pose-lm)) ## ADDED A -VE HERE
            msg.transform.translation = Vector3(*d)
            msg.transform.rotation = Quaternion(*[0, 0, 0, 1])
            self.br.sendTransform(msg)

if __name__ == '__main__':
    testFrame().run()
        