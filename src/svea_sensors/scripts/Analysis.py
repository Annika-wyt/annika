#!/usr/bin/env python3

import rospy
import tf2_ros
import threading
import numpy as np
import geometry_msgs.msg
from tf.transformations import euler_from_quaternion
import subprocess
import matplotlib.pyplot as plt
import time

BAGNUMBER = 1
BagPaths = {
    1: '/home/annika/ITRL/kth_thesis/rosbag/2024-01-22/working_epnp_with_EKF.bag',
    2: ' '
    }

Bag = BagPaths[BAGNUMBER]
ARUCOLIST = [10, 11, 12, 13, 14]
class Analysis:
    def __init__(self):
        rospy.init_node("Analysis")
        rospy.set_param('/use_sim_time', True)
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()
        self.Aruco = {
            10 : [],
            11 : [],
            12 : [],
            13 : [],
            14 : []
        }
        self.n = len(self.Aruco)
        self.END = False
        self.start_time = None
        self.rosbag_thread = threading.Thread(target=self.play_rosbag, args=(BagPaths[BAGNUMBER],))
        self.rosbag_thread.start()
        self.getTransform()
        self.rosbag_thread.join()
        self.save_to_file("ARUCO.txt")

    def play_rosbag(self, bag_file):
        try:
            # subprocess.run(['rosbag', 'play', bag_file], check=True)
            subprocess.run(['rosbag', 'play', bag_file, '-r 60'], check=True)
        except subprocess.CalledProcessError as e:
            rospy.logerr(f"Failed to play rosbag: {e}")
        finally:
            self.END = True

    def getTransform(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not self.END:
            for id in ARUCOLIST: 
                try:
                    trans1 = self.buffer.lookup_transform("MapAruco" + str(id), "arucoCamera" + str(id), rospy.Time.now())
                    if self.start_time == None:
                        self.start_time = trans1.header.stamp
                    self.Aruco[id].append([trans1.transform.translation.x, trans1.transform.translation.y, trans1.transform.translation.z, (trans1.header.stamp - self.start_time).to_sec()])
                except Exception as e:
                    rospy.logerr(f'{e}')

    def save_to_file(self, filename):
        """Save the ArUco data to a text file."""
        for aruco_id, data in self.Aruco.items():
            plt.figure(figsize=(10, 3))
              # Convert data list to a string of columns
                # data_str = ', '.join(map(str, data))
                # file.write(f'{data_str}\n')
            dataArray = np.array(data)
            # print(self.Aruco)
            plt.plot(dataArray[:,3], dataArray[:,0], color='r', label=f'Aruco {aruco_id} - x')
            plt.plot(dataArray[:,3], dataArray[:,1], color='g', label=f'Aruco {aruco_id} - y')
            plt.plot(dataArray[:,3], dataArray[:,2], color='b', label=f'Aruco {aruco_id} - z')
            # plt.axhline(y=np.median(dataArray[:, 0]), color='r', linestyle='--', label=f'Median x - {round(np.median(dataArray[:, 0]), 3)}')
            # plt.axhline(y=np.median(dataArray[:, 1]), color='g', linestyle='--', label=f'Median y - {round(np.median(dataArray[:, 1]), 3)}')
            # plt.axhline(y=np.median(dataArray[:, 2]), color='b', linestyle='--', label=f'Median z - {round(np.median(dataArray[:, 2]), 3)}')
            plt.legend(loc='upper right')
            plt.title("Error between MOCAP and Camera Detection")
            plt.xlabel('Time (in second)')
            plt.ylabel('Error (in m)')
            # plt.xlim(0, len(dataArray))
            plt.ylim(-3, 3)
        plt.show()
        print("show plt")


if __name__ == '__main__':
    analysis = Analysis()
