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
import os 

BAGNUMBER = 1

BagPaths = {
    1: '/home/annika/ITRL/kth_thesis/rosbag/2024-01-22/working_epnp_with_EKF.bag',
    2: '/home/annika/ITRL/kth_thesis/rosbag/2024-01-22/Working_alone_epnp.bag',
    3: '/home/annika/ITRL/kth_thesis/rosbag/2024-01-22/EKF_epnp_4landmarks.bag',
    4: '/home/annika/ITRL/kth_thesis/rosbag/2024-01-25/epnp_alone_4_landmarks.bag'
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
        self.stateDiff = []
        self.n = len(self.Aruco)
        self.END = False
        self.start_time_trans1 = None
        self.start_time_state = None
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
            try:
                stateDiff = self.buffer.lookup_transform("svea5", "base_link", rospy.Time.now())
                if self.start_time_state == None:
                    self.start_time_state = stateDiff.header.stamp
                self.stateDiff.append([stateDiff.transform.translation.x, stateDiff.transform.translation.y, stateDiff.transform.translation.z, (stateDiff.header.stamp - self.start_time_state).to_sec()])
            except Exception as e:
                rospy.logerr(f'At State: {e}')
                pass
            for id in ARUCOLIST: 
                try:
                    trans1 = self.buffer.lookup_transform("MapAruco" + str(id), "arucoCamera" + str(id), rospy.Time.now())
                    if self.start_time_trans1 == None:
                        self.start_time_trans1 = trans1.header.stamp
                    self.Aruco[id].append([trans1.transform.translation.x, trans1.transform.translation.y, trans1.transform.translation.z, (trans1.header.stamp - self.start_time_trans1).to_sec()])
                except Exception as e:
                    rospy.logerr(f'At Aruco: {e}')
                    # pass

    def save_to_file(self, filename):
        """Save the ArUco data to a text file."""
        plt.figure(figsize=(10, 3))
        self.stateDiff = np.array(self.stateDiff)
        plt.plot(self.stateDiff[:,3], self.stateDiff[:,0], color='r', linestyle='-', label=f'State - x')
        plt.plot(self.stateDiff[:,3], self.stateDiff[:,1], color='g', linestyle='-', label=f'State - y')
        plt.plot(self.stateDiff[:,3], self.stateDiff[:,2], color='b', linestyle='-', label=f'State - z')
        # plt.axhline(y=np.median(self.stateDiff[:, 0]), color='r', linestyle='--', label=f'Median x - {round(np.median(self.stateDiff[:, 0]), 3)}')
        # plt.axhline(y=np.median(self.stateDiff[:, 1]), color='g', linestyle='--', label=f'Median y - {round(np.median(self.stateDiff[:, 1]), 3)}')
        # plt.axhline(y=np.median(self.stateDiff[:, 2]), color='b', linestyle='--', label=f'Median z - {round(np.median(self.stateDiff[:, 2]), 3)}')
        plt.legend(loc='upper right')
        plt.title("Error between MOCAP and Estimation")
        plt.xlabel('Time (in second)')
        plt.ylabel('Error (in m)')
        # plt.xlim(0, len(dataArray))
        plt.ylim(-3, 3)
        plt.savefig('/home/annika/ITRL/kth_thesis/plots/' + os.path.splitext(os.path.basename(Bag))[0] + 'stateDiff.png', format='png', dpi=300, bbox_inches='tight')
        
        for aruco_id, data in self.Aruco.items():
            plt.figure(figsize=(10, 3))
              # Convert data list to a string of columns
                # data_str = ', '.join(map(str, data))
                # file.write(f'{data_str}\n')
            dataArray = np.array(data)
            plt.plot(dataArray[:,3], dataArray[:,0], color='r', label=f'Aruco {aruco_id} - x')
            plt.plot(dataArray[:,3], dataArray[:,1], color='g', label=f'Aruco {aruco_id} - y')
            plt.plot(dataArray[:,3], dataArray[:,2], color='b', label=f'Aruco {aruco_id} - z')
            plt.axhline(y=np.median(dataArray[:, 0]), color='r', linestyle='--', label=f'Median x - {round(np.median(dataArray[:, 0]), 3)}')
            plt.axhline(y=np.median(dataArray[:, 1]), color='g', linestyle='--', label=f'Median y - {round(np.median(dataArray[:, 1]), 3)}')
            plt.axhline(y=np.median(dataArray[:, 2]), color='b', linestyle='--', label=f'Median z - {round(np.median(dataArray[:, 2]), 3)}')
            plt.legend(loc='upper right')
            plt.title("Error between MOCAP and Camera Detection")
            plt.xlabel('Time (in second)')
            plt.ylabel('Error (in m)')
            # plt.xlim(0, len(dataArray))
            plt.ylim(-3, 3)
            plt.savefig('/home/annika/ITRL/kth_thesis/plots/' + os.path.splitext(os.path.basename(Bag))[0] + '_Aruco' + str(aruco_id) + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    analysis = Analysis()
