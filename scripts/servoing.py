#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from shoe_monster.msg import Target
from sensor_msgs.msg import LaserScan

class Servoing(object):
    """docstring for Servoing"""
    def __init__(self):
        self.target = rospy.Subscriber('target', Target, self.target_received)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.scan = rospy.Subscriber('scan', LaserScan, self.scan_received)
        self.velocity = Twist(Vector3(0.0, 0.0, 0.0),Vector3(0.0, 0.0, 0.0))
        self.turn_percent = 0
        self.velocity_percent = 0

    def idle(self):
        """This should be an idle scanning for shoes behavior"""
        if self.turn_percent >= 0:
            self.velocity = Twist(Vector3(0.0, 0.0, 0.0),Vector3(0.0, 0.0, 0.4))
        else:
            self.velocity = Twist(Vector3(0.0, 0.0, 0.0),Vector3(0.0, 0.0, -0.4))
        print "Idling"

    def track(self):
        """Determines the heading to the object, and creates the Twist message"""
        x_center = self.x_img_size/2.0
            
        self.turn_percent = -(self.x_target-x_center)/x_center
        velocity = Vector3(0.8*self.velocity_percent, 0.0, 0.0)
        turn = Vector3(0.0, 0.0, 0.5*self.turn_percent)
        print "Tracking"
        self.velocity = Twist(velocity, turn)

    def scan_received(self, msg):
        """This should look in the general area where we are seeing a shoe to 
        determine the distance to said shoe"""
        l_bound = 45 #Degrees off of zero (measure from camera)
        r_bound = -45 #Degrees off of zero (measure from camera)

        valid_ranges = {} #Dict of valid data points and angles in degrees
        min_angle = 0 #Angle to nearest object
        min_distance = 0 #Distance to closest object
        max_distance = 5 #Clipping off readings above this distance
        
        for i in range(46)+range(315,360): 
            if msg.ranges[i] > 0 and msg.ranges[i] <= max_distance:
                valid_ranges[i]=(msg.ranges[i])
        #print len(valid_ranges)
        if len(valid_ranges) > 0: #Keeps last values if no new objects detected
            min_angle = min(valid_ranges, key=valid_ranges.get)
            min_distance = valid_ranges[min_angle]
        #print'angle', min_angle, 'distance', min_distance
        self.velocity_percent = min_distance/max_distance

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
