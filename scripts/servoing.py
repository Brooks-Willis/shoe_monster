#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from shoe_monster.msg import target

class Servoing(object):
    """docstring for Servoing"""
    def __init__(self):
        self.target = rospy.Subscriber('target', target, self.target_received)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    def idle(self):
        print "Idling"

    def track(self):
        x_center = self.x_img_size/2
            
        self.turn_percent = (self.x_target-x_center)/x_center
        velocity = Vector3(0.5, 0.0, 0.0) #Add velocity based on distance
        turn = Vector3(0.0, 0.0, 0.4*self.turn_percent)

        velocity_msg = Twist(velocity, turn)
        self.cmd_vel.publish(velocity_msg)


    def target_received(self, msg):
        if msg.x == -1:
            self.idle()
        else:
            self.x_target = msg.x
            self.y_target = msg.y
            self.x_img_size = msg.x_img_size
            self.x_img_size = msg.y_img_size

            self.track()

if __name__ == '__main__':
    test = Servoing()
    while True:
        try:
            print test.turn_percent
        except:
            print "No Data"