#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from shoe_monster.msg import Target

class Servoing(object):
    """docstring for Servoing"""
    def __init__(self):
        self.target = rospy.Subscriber('target', Target, self.target_received)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.scan = rospy.Subscriber('scan', LaserScan, self.scan_received)
        self.velocity = Twist(Vector3(0.0, 0.0, 0.0),Vector3(0.0, 0.0, 0.0))

    def idle(self):
        """This should be an idle scanning for shoes behavior"""
        print "Idling"

    def track(self):
        """Determines the heading to the object, and creates the Twist message"""
        x_center = self.x_img_size/2.0
            
        self.turn_percent = -(self.x_target-x_center)/x_center
        velocity = Vector3(0.5, 0.0, 0.0) #Add velocity based on distance
        print x_center, self.x_target, self.turn_percent
        turn = Vector3(0.0, 0.0, 0.4*self.turn_percent)

        self.velocity = Twist(velocity, turn)

    def scan_received(self, msg):
        """This should look in the general area where we are seeing a shoe to 
        determine the distance to said shoe"""
        l_bound = #Degrees off of zero (measure from camera)
        r_bound = #Degrees off of zero (measure from camera)


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


if __name__ == '__main__':
    try:
        servoing = Servoing()
        servoing.excecute()

    except rospy.ROSInterruptException:
        pass
