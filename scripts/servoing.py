#!/usr/bin/env python

import rospy
import numpy as np

class Servoing(object):
    """docstring for Servoing"""
    def __init__(self, target=(0,0)):
        self.sub = rospy.Subscriber('target', Target, self.target_received)
        self.xtarget = target[0]
        self.ytarget = target[1]


