#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import signal
import threading

# ROS Libraries
import rospkg
import sys
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy

# OpenCV & Numpy Libraries
import cv2
import numpy as np

class seyir_logger:
        def __init__(self):

            self.debug = False
            self.resize = False
            self.rate = rospy.Rate(15)

            path = '/home/nvidia/data/'
            if not os.path.exists(path):
                os.makedirs(path)
            i = 1
            while True:
                dname = path+'%03d'%i
                if os.path.exists(dname):
                    i += 1
                else:
                    rospy.loginfo("directory: " + dname)
                    os.makedirs(dname)
                    break

            self.path = dname+'/'
            self.file_name_npy = self.path + '/training_data.npy'
            self.file_name_csv = self.path + '/training_data.csv'
            self.file_csv = file(self.file_name_csv, "w+") 
            self.file_csv.write('FileName,Speed,Angle\n')
            self.index = 0
            self.training_data = np.array([0,0,0])
            self.speed = None
            self.angle = None
            self.record = None
            self.cv2_img = None
            self.zed_camera = rospy.Subscriber('/zed/right/image_rect_color/compressed', CompressedImage, self.zed_callback)
            self.sub = rospy.Subscriber('/ackermann_cmd', AckermannDriveStamped, self.drive_call, queue_size=1)
            self.joy = rospy.Subscriber('/joy', Joy, self.joy_callback)
            self.lock = threading.Lock()

        def joy_callback(self, data):
            if len(data.buttons)!=0 and data.buttons[5]!=0:
	        self.record = True
            else:
                self.record = False
        
        def drive_call(self, data):
            if self.debug:
                rospy.loginfo(data)
            self.angle = data.drive.steering_angle
            self.speed = data.drive.speed
        
        def zed_callback(self,data):
            try:
                np_arr = np.fromstring(data.data, np.uint8)
                with self.lock:
                    cv2_tmp = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if self.resize == True:
                        self.cv2_img = cv2.resize(cv2_tmp,(640,360),interpolation=cv2.INTER_AREA)
                    else:
                        self.cv2_img = cv2_tmp
            except Exception as e:
                rospy.loginfo(e)

        def write(self):
            try:
                
                if self.cv2_img is None:
                    rospy.loginfo('Camera could not be detected!')
                if self.speed is None:
                    rospy.loginfo('Speed could not be detected!')
                if self.angle is None:
                    rospy.loginfo('Angle could not be detected!')

                #if not self.cv2_img is None and not self.speed is None and not self.angle is None and self.record:
                if not self.cv2_img is None and not self.speed is None and not self.angle is None:
                    fname = self.path+'%05d.jpg'%self.index
                    
                    line = '%05d.jpg'%self.index +','+ str(self.speed)+ ','+str(self.angle)+'\n'
                    self.file_csv.write(line)
                    with self.lock:
                        cv2.imwrite(fname,self.cv2_img)
                    rospy.loginfo('N: {} Speed: {} Angle: {}'.format(self.index, self.speed, self.angle))

                    #generated_data = np.array(['%05d.jpg'%self.index,self.speed,self.angle])
                    #self.training_data = np.vstack((self.training_data, generated_data))
                    #np.save(self.file_name_npy, self.training_data)

                    if self.debug:
                        cv2.imshow('Image', self.cv2_img)
                        k = cv2.waitKey(10)

                    self.index += 1
                
            except Exception,e:
                rospy.loginfo(e)
            self.rate.sleep()

def exit_gracefully(signal,frame):
    rospy.loginfo('Exiting, wait for it...')
    sys.exit(0)

if __name__ == '__main__':
    rospy.init_node('collect_data')
    logger = seyir_logger()
    while not rospy.is_shutdown():
        signal.signal(signal.SIGINT, exit_gracefully)
        #logger.write()
        #if logger.speed != 0.0:
        if logger.record:
            logger.write()
        #else:
        #    print("Not moving")
    rospy.spin()
