#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
import std_msgs.msg
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge


rospy.init_node('opencv', anonymous=True)
bridge = CvBridge()

class follow_color:
	def __init__(self):
		rospy.Subscriber('/zed/right/image_rect_color', Image, self.zed_callback)
		self.pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
		self.cmd = AckermannDriveStamped()


	def get_object(self, hsv):
		hsv_lower_filter = (0, 147, 0)
		hsv_upper_filter = (22, 255, 255)
		mask = cv2.inRange(hsv, hsv_lower_filter, hsv_upper_filter)
		_, contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		
		X = 0
		Radius = 0
		if len(contours) > 0 :
			center_list = []
			radius_list = []
			for cnt in contours:
				(x,y),radius = cv2.minEnclosingCircle(cnt)
				center_list.append((int(x),int(y)))
				radius_list.append(int(radius))
			ix = np.argmax(radius_list)
			#print 'Nesne Konumu : ',center_list[ix]
			#print 'Nesne Buyuklugu : ',radius_list[ix]
			X = center_list[ix][0]	
			Radius = radius_list[ix]
		
		return	X,Radius


	def zed_callback(self, data):
		cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
		#cv2.imshow('asd', cv2_img)
		#if cv2.waitKey(25) & 0xFF == ord('q'):
		#	pass
		hsv = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2HSV)
		X,R = self.get_object(hsv)
		print 'Obejct : ' ,X, R
		angle = -0.34*(X - 640) / 640
		print 'Angle:' + str(angle)	
		if R >= 200 :
			speed = 0
		else:
			speed = 0.5
		print angle		
		self.cmd.drive.speed = speed				
		self.cmd.drive.steering_angle = angle
		self.pub.publish(self.cmd)


f = follow_color()

if __name__ == '__main__':
    rospy.spin()
