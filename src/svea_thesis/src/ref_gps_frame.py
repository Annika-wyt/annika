#!/usr/bin/env python3
import rospy
import tf2_ros

from pyproj import Proj

from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from sensor_msgs.msg import NavSatFix

####################################################
### Publish the reference gps frame in utm frame ### 
####################################################

class reference_gps_frame:
    def __init__(self):
        # Initialize node
        rospy.init_node('reference_gps_frame')

        # Reference GPS latitude, Longitude
        self.reference_aruco = rospy.get_param("~reference_aruco", [59.350775, 18.068076]) #ITRL
        self.reference_map = rospy.get_param("~reference_map", [59.350775,18.068094]) #


        # Publisher
        self.reference_gps_publisher = rospy.Publisher('/reference_points', NavSatFix, queue_size=10)

        # Transformation
        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.static_br = tf2_ros.StaticTransformBroadcaster()

        # Variables
        self.rate = rospy.Rate(10)
        self.seq = 0

    def PublishReferenceGpsFrame(self, point, name):
        projection = Proj(proj='utm', zone=34, ellps='WGS84')
        utm = projection(point[1], point[0])
        FixedGpsFrame = TransformStamped()
        FixedGpsFrame.header.stamp = rospy.Time.now()
        FixedGpsFrame.header.frame_id = "utm"
        FixedGpsFrame.child_frame_id = name
        FixedGpsFrame.transform.translation = Vector3(*[utm[0], utm[1], 0.0])
        FixedGpsFrame.transform.rotation = Quaternion(*[0.0, 0.0, 0.0, 1.0])
        self.static_br.sendTransform(FixedGpsFrame)

        reference_msg = NavSatFix()
        reference_msg.header.stamp = rospy.Time.now()
        reference_msg.header.frame_id = "utm"
        reference_msg.header.seq = self.seq

        reference_msg.latitude = point[0]
        reference_msg.longitude = point[1]
        self.reference_gps_publisher.publish(reference_msg)
        self.seq += 1

    def run(self):
        while not rospy.is_shutdown():
            self.PublishReferenceGpsFrame(self.reference_aruco, "aruco1")
            self.PublishReferenceGpsFrame(self.reference_map, "map_ref")
            self.rate.sleep()

if __name__ == '__main__':
    reference_gps_frame().run()