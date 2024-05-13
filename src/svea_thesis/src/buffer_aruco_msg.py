#!/usr/bin/env python3

import rospy
from svea_thesis.msg import Aruco, ArucoArray

class Republisher:
    def __init__(self):
        rospy.init_node('republisher_node', anonymous=True)
        self.original_topic = '/aruco/detection'  # Change this to your original topic
        self.republish_topic = '/aruco/detection/more'  # Change this to your republish topic
        self.rate = rospy.Rate(40)  # Change the rate as desired (5 Hz in this example)
        self.buffered_message = None
        rospy.Subscriber(self.original_topic, ArucoArray, self.callback)
        self.publisher = rospy.Publisher(self.republish_topic, ArucoArray, queue_size=10)

    def callback(self, msg):
        # Buffer the incoming message
        self.buffered_message = msg

    def republish(self):
        while not rospy.is_shutdown():
            if self.buffered_message is not None:
                # Republish the buffered message
                self.publisher.publish(self.buffered_message)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        republisher = Republisher()
        republisher.republish()
    except rospy.ROSInterruptException:
        pass
