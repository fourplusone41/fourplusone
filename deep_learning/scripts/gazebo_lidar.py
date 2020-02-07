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
		self.error_prev = 0 # toplam hata
		self.total_error = 0 # bir önceki durum hatası
		# PID parametreleri
		# Kp : oransal katsayı, hatanın çarpıldığı katsayı
		# Kd : hatanın değişiminin çarpıldığı katsayı
		# Ki : steady state erroru önlemek toplam hatanın çarpılacağı katsayı
		
		kp_Value=1
		kd_Value=0.5
		ki_Value=0.005

		self.kp = kp_Value  #1
		self.kd = kd_Value #0.5
		self.ki = ki_Value #0.005
		
		self.file = open(str(kp_Value)+"-"+str(kd_Value)+"-"+str(ki_Value)+".txt","a")

		self.rate = rospy.Rate(500)
		
		# Duvara olan uzaklık
		self.desired_distance = 2



	def lidar_call(self,data):

		
                # Her açı değeri için bir değer dönmektedir
		rangesCount = len(data.ranges)
                print(rangesCount)
                print(data.ranges)
		middle = rangesCount / 2
 		# Sol ve biraz ön tarafa doğru olan alan
		ranges = data.ranges[middle + 20:rangesCount - 80]

		# inf olarak dönen değerler olabiliyor, onları silmek için
		y = [s for s in ranges if not math.isinf(s)]
                #print(y)
		
		# Hata = İstenen uzaklık - anlık uzaklık
		error = self.desired_distance - max(y)

		# Hatanın ne kadar değiştiğinin değeri
		e_prev = error - self.error_prev
		
		self.total_error += error

		# -1 ile çarpılmasının sebebi direksiyon açısını doğru verebilmek. 
		self.steering_angle =  -1 * (self.kp * error + e_prev * self.kd + self.total_error * self.ki)
		self.error_prev = error
		msg = AckermannDriveStamped()
		print("angle: ",self.steering_angle)
		print("distance", max(y))
		print("\n")

		self.file.write(str(self.steering_angle)+"\n") 

		msg.drive.speed = 3
		msg.drive.steering_angle = self.steering_angle
		
		self.pub.publish(msg)
		self.rate.sleep()

	
if __name__ == '__main__':
	rospy.init_node('lidar_pid')
	control = bangbang_pid()
	rospy.spin()
