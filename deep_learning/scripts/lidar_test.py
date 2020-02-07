#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pylab as plt
import rospy
import sys
import os
import json
import time
import math
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan, Image, CompressedImage, Joy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped

rospy.init_node('predict')

class BaseClass(object):
    def __init__(self):
        #self.bridge = CvBridge()
        # Global option
        self.debug = False
        self.rate = rospy.Rate(5)
        self.lidar = "sweep"  # "sweep"/"rplidar"
        self.angle = 0  # initial angle
        self.speed = 0.4  # initial speed

        # Lidar
        self.ranges = []

        # PID
        # Kp : oransal katsayı, hatanın çarpıldığı katsayı
        # Kd : hatanın değişiminin çarpıldığı katsayı
        # Ki : steady state erroru önlemek toplam hatanın çarpılacağı katsayı
        self.kp = 0.1  # 1
        self.kd = 0.01  # 0.5
        self.ki = 0.001  # 0.005

        # Duvara olan uzaklık
        self.desired_distance = 0.3

        # PID errors
        self.error = 0  # şimdiki hata
        self.error_prev = 0  # bir önceki durum hatası
        self.error_total = 0  # toplam hata

        # ROS Subs and Pubs
        #rospy.Subscriber('/zed/right/image_rect_color/compressed',
        #                 CompressedImage, self.zed_callback, queue_size=1)
        #rospy.Subscriber('/scan', LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback, queue_size=1)

#self.pub = rospy.Publisher(
        #    '/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)


    def lidar_callback(self, data):
        #rospy.loginfo('#--- LIDAR Callback ---#')
        if self.lidar == "rplidar":
            ranges = data.ranges[270:] + data.ranges[:90]
        elif self.lidar == "sweep":
            ranges = list(reversed(data.ranges[90:270])) # reverse list to get clockwise degrees
            
        # subsample values to only 17: [-80 -70 ... 0 ... 70 80]
        self.ranges = []
        for i in range(10, 180, 10):
            self.ranges.append(min(ranges[i-7:i+7]))  # 7 for some overlapping
        print(len(self.ranges))
        print(self.ranges)

    def pid_drive(self):
        rospy.loginfo('#--- Obstacle detected ---#')
        # Hata = İstenen uzaklık - anlık uzaklık
        error = self.desired_distance - max(self.ranges)
        # Hatanın ne kadar değiştiğinin değeri
        e_delta = error - self.error_prev
        self.error_total += error
        # -1 ile çarpılmasının sebebi direksiyon açısını doğru verebilmek.
        self.angle = -1 * (self.kp * error + e_delta *
                           self.kd + self.error_total * self.ki)
        self.error_prev = error


    def pipeline(self):
        try:
            self.pid_drive()

        except Exception as e:
            print(e)
        self.rate.sleep()

    def nn_model(self):
        jstr = json.loads(open(self.model_name + '.json').read())
        model = model_from_json(jstr)
        model.load_weights(self.model_name + '.h5')
        return model


drive = BaseClass()

if __name__ == '__main__':
    while not rospy.is_shutdown():
        drive.pipeline()
    rospy.spin()
