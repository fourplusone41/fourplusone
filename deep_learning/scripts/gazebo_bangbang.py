#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os, math
import time
import rospkg
import sys
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan


class bangbang_pid(object):
	def __init__(self):
		self.pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
		self.lidar = rospy.Subscriber('/scan',LaserScan,self.lidar_call)
		
		self.steering_angle = 0
		self.error_prev = 0

		self.rate = rospy.Rate(500)
		
		# Duvara olan uzaklık
		self.desired_distance = 2


	def lidar_call(self,data):

		# Her açı değeri için bir değer dönmektedir
		rangesCount = len(data.ranges)
		middle = rangesCount / 2
 		# Sol ve biraz ön tarafa doğru olan alan
		ranges = data.ranges[middle + 20:rangesCount - 80]

		# inf olarak dönen değerler olabiliyor, onları silmek için
		y = [s for s in ranges if not math.isinf(s)]
		
		# Hata = İstenen uzaklık - anlık uzaklık
		error = self.desired_distance - max(y)

		if error > 0:
		    # sola dön
			self.steering_angle = -0.34
		else:
		# sağa dön
			self.steering_angle = 0.34
		
		msg = AckermannDriveStamped()
		print("angle: ",self.steering_angle)
		print("distance", max(y))
		# output karşılatığımız hata, 0 olmalı
	
		msg.drive.speed = 3
		msg.drive.steering_angle = self.steering_angle
		# -1 sağ, 1 sol
		self.pub.publish(msg)
		self.rate.sleep()

	
if __name__ == '__main__':
	rospy.init_node('lidar_pid')
	control = bangbang_pid()
	rospy.spin()
