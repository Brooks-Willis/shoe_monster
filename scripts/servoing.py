#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from shoe_monster.msg import Target
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

class Servoing(object):
    """docstring for Servoing"""
    def __init__(self):
        self.target = rospy.Subscriber('target', Target, self.target_received)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.scan = rospy.Subscriber('scan', LaserScan, self.scan_received)
        self.object_dist_pub = rospy.Publisher('object_distance', Float32)
        self.velocity = Twist(Vector3(0.0, 0.0, 0.0),Vector3(0.0, 0.0, 0.0))
        self.turn_percent = 0
        self.velocity_percent = 0
        self.camera_FOV = 52 #In degrees
        self.valid_ranges = {} #Dict of valid data points and angles in degrees 

    def idle(self):
        """This should be an idle scanning for shoes behavior"""
        if self.turn_percent >= 0:
            self.velocity = Twist(Vector3(0.0, 0.0, 0.0),Vector3(0.0, 0.0, 0.4))
        else:
            self.velocity = Twist(Vector3(0.0, 0.0, 0.0),Vector3(0.0, 0.0, -0.4))
        print "Idling"

    def object_angle(self):
        return int(self.camera_FOV * (self.x_target/self.x_img_size) - self.camera_FOV/2.0) # angle reletive to 0 (in center)

    def track(self):
        """Determines the heading to the object and creates the Twist message"""
        x_center = int(self.x_img_size/2.0)
        
        # Simple Proportional controller
        self.turn_percent = -(self.x_target-x_center)/x_center
        velocity = Vector3(0.8*self.velocity_percent, 0.0, 0.0)
        turn = Vector3(0.0, 0.0, 0.5*self.turn_percent)
        self.velocity = Twist(velocity, turn)

    def scan_received(self, msg):
        """This should look in the general area where we are seeing a shoe to 
        determine the distance to said shoe"""
        min_angle = 0 #Angle to nearest object
        min_distance = 5 #Distance to closest object
        max_distance = 5 #Clipping off readings above this distance
        k = -int(self.camera_FOV/2) - 2   #for plus to on eaither side...used later for averaging with data outside view
        distances_roi = {}
        lists = []

        # Looks at angles within the bound of the image (plus 2 on either side)
        for i in range(int(self.camera_FOV/2.0) + 2)+range(360 - (int(self.camera_FOV/2.0) + 2),360):
            if msg.ranges[i] > 0 and msg.ranges[i] <= max_distance:
                if msg.ranges[i] < min_distance:
                    min_distance = msg.ranges[i]
                distances_roi[k] = msg.ranges[i]
            else:
                distances_roi[k] = 5
            k += 1

        print distances_roi
        obj_angle = self.object_angle()
        obj_dists = []

        for i in range(-2,2):
            print obj_angle + i
            obj_dists.append(distances_roi[obj_angle + i])

        self.object_distance = sum(obj_dists)/(len(obj_dists)*1.0)
        self.velocity_percent = min_distance/max_distance

        self.object_dist_pub.publish(self.object_distance)

        print "obj_angle", obj_angle
        print 'dist of object', self.object_distance

    def target_received(self, msg):
        """Determines behavior based off if neato can currently see a target"""
        if msg.x == -1:
            self.idle()
        else:
            #Recast msg to local vars
            self.x_target = float(msg.x)
            self.y_target = float(msg.y)
            self.x_img_size = float(msg.x_img_size)
            self.x_img_size = float(msg.y_img_size)

            self.track()

    def excecute(self):
        rospy.init_node('cmd_vel', anonymous=True)
        r = rospy.Rate(10)
        while not(rospy.is_shutdown()):
            self.cmd_vel.publish(self.velocity)
            r.sleep()


if __name__ == '__main__':
    try:
        servoing = Servoing()
        servoing.excecute()

    except rospy.ROSInterruptException:
        pass
