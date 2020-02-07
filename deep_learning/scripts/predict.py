#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import json
import signal
import statistics
from pyzbar import pyzbar
from threading import Thread

import cv2
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String, Int32, Float64, Float32MultiArray
from sensor_msgs.msg import LaserScan, Image, CompressedImage, Joy

#import keras
import tensorflow as tf
from keras import backend as K
from keras.models import load_model, model_from_json
# from cv_bridge import CvBridge, CvBridgeError

# Destroy the current TF graph and create a new one
K.clear_session()

# set session config
config = tf.ConfigProto(allow_soft_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.set_session(tf.Session(config=config))

# enable 16-bit
K.set_floatx('float16')


class BaseClass(object):
    def __init__(self):
        #self.bridge = CvBridge()
        # Global option
        self.debug = False
        self.rate = rospy.Rate(10)
        self.lidar = "rplidar"  # "sweep"/"rplidar"
        self.angle = 0  # initial angle
        self.speed = 0.4  # initial speed

        # Lidar
        self.ranges = []

        # Nvidia End-to-End model
        self.model_name_drive = '/home/nvidia/models/2019_04_11_DRIVE_float16_Adam_model'
        self.model_name_park = '/home/nvidia/models/2019_04_13_PARK_float16_Adam_model'
        self.model_drive = self.nn_model()
        self.model_park = self.nn_model2()
        self.out = None  # model prediction
        self.graph_behavioral = tf.get_default_graph()

        # ZED output
        self.image = None
        self.crop_top = 160 * 2
        self.crop_bottom = 340 * 2

        self.flags = {"PID": False, "PID2DL": False, "DL2PID": False, "STOP": False, "WAIT4G": False, "WAIT4P": False, "LEFT": False, "PEDESTRIAN": False, "LOOSE": False, "PARK": False, "AREA": False, "OFF": False, "KEEPL": False, "VISIBLEP": False, "EXIT": False, "SR": False}
        
        self.ped_counter = 0
        self.left_counter = 0
        self.park_counter = 0
        self.prepark_counter = 0
        self.loose_counter = 0
        self.engine_off_counter = 0
        self.keep_counter = 0
        self.s_counter = 0

        # QR

        self.qr_desired = None
        self.qr_read = None

        # PID
        # Kp : oransal katsayı, hatanın çarpıldığı katsayı
        # Kd : hatanın değişiminin çarpıldığı katsayı
        # Ki : steady state erroru önlemek toplam hatanın çarpılacağı katsayı
        self.kp = 0.1  # 1
        self.kd = 0.01  # 0.5
        self.ki = 0.001  # 0.005

        # Duvara olan uzaklık
        self.desired_distance = 0.5

        # PID errors
        self.error = 0  # şimdiki hata
        self.error_prev = 0  # bir önceki durum hatası
        self.error_total = 0  # toplam hata

        self.traffic_sign = None

        # Localization
        self.travelled_x        = [] # passed points
        self.travelled_y        = []

        self.x_offset = 0
        self.y_offset = 0

        self.destination = None

        # ROS Subs and Pubs
        rospy.Subscriber('/zed/right/image_rect_color/compressed',
                         CompressedImage, self.zed_callback, queue_size=1)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber('/zed_yolo', String, self.zedyolo_callback, queue_size=1)
        rospy.Subscriber('/exit_number',Int32, self.exit_callback, queue_size=1)
        rospy.Subscriber('/target_coordinates',Float32MultiArray, self.coord_callback, queue_size=1)
        self.pub = rospy.Publisher(
            '/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)

    def exit_callback(self, data):
        self.qr_desired = data.data
    
    def coord_callback(self, data):
        self.destination = data.data
        
    def zed_callback(self, data):
        np_arr = np.fromstring(data.data, np.uint8)
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        #self.qr_read = self.decode_image()

        if self.debug:
            cv2.imshow('Image', self.image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

    def lidar_callback(self, data):
        if self.lidar == "rplidar":
            ranges = data.ranges[270:] + data.ranges[:90]
        elif self.lidar == "sweep":
            ranges = data.ranges[90:270]

        # subsample values to only 17: [-80 -70 ... 0 ... 70 80]
        self.ranges = []
        for i in range(10, 180, 10):
            tmp = min(ranges[i-7:i+7]) # 7 for some overlapping
            if not math.isinf(tmp):
                self.ranges.append(tmp)  
        self.pid_check()
        #rospy.loginfo(len(self.ranges))
        #rospy.loginfo(self.ranges)

    def odom_callback(self, data):
        y = data.pose.pose.position
        self.current_position = (y.x*100 + self.x_offset, y.y*100 + self.y_offset) # Converted to cm 
        self.travelled_x.append(self.current_position[0])
        self.travelled_y.append(self.current_position[1])

    def zedyolo_callback(self, data):
        tmp_str = data.data
        full_list = tmp_str.split(" ")
        del full_list[-1]

        signs_list = []
        
        for i in range(0,len(full_list),4):
            tmp = []
            tmp.extend([full_list[i],float(full_list[i+1]),float(full_list[i+2]),int(full_list[i+3])])
            signs_list.append(tmp)

        #rospy.loginfo(signs_list)

        # only take action when distance is less than 1 m
        if not signs_list:
            self.traffic_sign = None

        self.flags["PEDESTRIAN"] = False
        self.flags["VISIBLEP"] = False
        self.qr_read = None
        for indx, sign in enumerate(signs_list):
            if sign[0] == "Park_Area":
                self.flags["VISIBLEP"] = True
                break
            self.flags["VISIBLEP"] = False

        
        for indx, sign in enumerate(signs_list):
            if sign[0] == "Pedestrian":
                if sign[2] <= 1.2 or (sign[2] == 1.73 and sign[3] >= 100):
                    self.flags["PEDESTRIAN"] = True
                elif sign[3] >= 120:
                    self.flags["PEDESTRIAN"] = True
                del signs_list[indx]
                break
            self.flags["PEDESTRIAN"] = False

        for indx, sign in enumerate(signs_list):
            if sign[2] <= 1:
                self.traffic_sign = (sign[0], sign[2])
                break
            elif sign[2] == 1.73 and sign[3] >= 80:
                self.traffic_sign = (sign[0], 80)
                break
            elif sign[3] >= 90:
                self.traffic_sign = (sign[0], 80)
                break
            self.traffic_sign = None

        rospy.loginfo("[SIGN] Detected Sign at <=1m: {}".format(self.traffic_sign))  

    def decode_image(self):
        if not self.image is None:
            barcodes = pyzbar.decode(self.image)
            for barcode in barcodes:

                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type
                rospy.loginfo("[QR] Found {} barcode: {}".format(barcodeType, barcodeData))
                return barcodeData
                #try:
                    #d = Int32()
                    #d.data = int(barcodeData)
                    #self.pub.publish(d)
                    # print the barcode type and data to the terminal
                    #print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
                #except:
                    #pass

    def pid_check(self):
        try: 
            #if not self.flags["PID"] and min(self.ranges[6:10]) < 0.6:  # angles in [-20(-7),20(+7)] degrees
            if min(self.ranges[6:10]) < 1:  # angles in [-20(-7),20(+7)] degrees
                self.flags["PID"] = True

            elif min(self.ranges[6:10]) > 1:

                if self.flags["PID"] and min(self.ranges[13:15]) > 1:
                    self.flags["PID"] = False
                    self.flags["PID2DL"] = True

                elif self.flags["PID2DL"] and min(self.ranges[15:]) > 1:
                    self.flags["PID"] = False
                    self.flags["PID2DL"] = False
                    # reset PID errors
                    self.error = 0
                    self.error_prev = 0
                    self.error_total = 0

        except Exception as e:
            rospy.loginfo(e)
        #self.rate.sleep()
    
    def pid_drive(self):
        rospy.loginfo('[DRIVE] Driving with PID')
        # average distance on each side
        left_avg = statistics.mean(self.ranges[0:(len(self.ranges)/2)+2])
        right_avg = statistics.mean(self.ranges[(len(self.ranges)/2)+2:len(self.ranges)])

        ind = self.ranges.index(max(self.ranges))
        # Hata = İstenen uzaklık - anlık uzaklık
        error = self.desired_distance - max(self.ranges)
        # Hatanın ne kadar değiştiğinin değeri
        e_delta = error - self.error_prev
        self.error_total += error
        # -1 ile çarpılmasının sebebi direksiyon açısını doğru verebilmek.
        self.angle = -1 * (self.kp * error + e_delta *
                           self.kd + self.error_total * self.ki)

        if left_avg >= right_avg:
            self.angle = -1*abs(self.angle)
        else:
            self.angle = abs(self.angle)
        #if self.angle is inf:
        self.error_prev = error

    def dl_drive(self):
        rospy.loginfo('[DRIVE] Driving with DL')
        with self.graph_behavioral.as_default():
            cv2_img = cv2.resize(
                self.image[self.crop_top:self.crop_bottom, :, :], (320, 90))
            cv2_img = cv2_img.reshape(1, 90, 320, 3)
            self.out = self.model_drive.predict(cv2_img, batch_size=1)
            self.angle = self.out[0][0] / 3.0

    def dl_park(self):
        rospy.loginfo('[DRIVE] Parking with DL')
        with self.graph_behavioral.as_default():
            cv2_img = cv2.resize(
                self.image[self.crop_top:self.crop_bottom, :, :], (320, 90))
            cv2_img = cv2_img.reshape(1, 90, 320, 3)
            self.out = self.model_park.predict(cv2_img, batch_size=1)
            self.angle = self.out[0][0] / 3.0

    def driver_logic(self):
        if self.traffic_sign:
            "tabela kontrolu burda eklenecek, 1 isiklar, 2 yaya, 3 calisma"

            sign = self.traffic_sign[0] # sign name
            dist = self.traffic_sign[1] # sign distance
            
            if sign == "Lights_Red":
                self.flags["STOP"] = True
                self.flags["WAIT4G"] = True
            
            elif sign == "Lights_Green":
                self.flags["STOP"] = False
                self.flags["WAIT4G"] = False
                #self.dl_drive()

            elif sign == "Turn_Left":
                self.flags["LEFT"] = True

            #elif sign == "Crosswalk":
            elif self.flags["PEDESTRIAN"]:
                rospy.loginfo('#--- Pedestrian Detected')
                self.flags["STOP"] = True
                self.flags["WAIT4P"] = True
                #elif not self.flags["PEDESTRIAN"]:
                #        rospy.loginfo('Pedestrian Left')
                #        self.flags["STOP"] = False
                #        self.flags["WAIT4P"] = False

            elif sign == "Loose_Chippings":
                self.flags["LOOSE"] = True

            elif sign == "Park":
                self.flags["PARK"] = True
            
            elif sign == "Park_Area":
                self.flags["AREA"] = True
            
            elif sign == "Roadworks":
                self.flags["KEEPL"] = True

            elif sign == "Straight_Right":
                self.flags["SR"] = True
                self.qr_read = self.decode_image()
 
        else:
            self.flags["STOP"] = False
        
        if self.flags["WAIT4P"] and not self.flags["PEDESTRIAN"]:
            self.ped_counter += 1
            if self.ped_counter == 20:
                rospy.loginfo('#--- Pedestrian Passed')
                self.flags["WAIT4P"] = False
                self.flags["STOP"] = False
                self.ped_counter = 0

        if self.flags["STOP"] or self.flags["WAIT4G"] or self.flags["WAIT4P"] or self.flags["OFF"]:        
            rospy.loginfo("STOP: {}, WAIT4G: {}, WAIT4P: {}, OFF: {}".format(self.flags["STOP"],self.flags["WAIT4G"],self.flags["WAIT4P"],self.flags["OFF"]))

        else:
            if self.flags["PID"]:
                self.pid_drive()
            elif self.flags["PID2DL"]:
                rospy.loginfo("#--- Returning to Lane {}")
                self.angle = -0.3

            elif self.flags["LEFT"]:
                rospy.loginfo("#--- Turning left {}".format(self.left_counter))
                self.angle = 0.3
                self.left_counter += 1
                if self.left_counter == 20:
                    self.left_counter = 0
                    self.flags["LEFT"] = False
                    #rospy.loginfo("#--- Turning sequence finished {} {}".format(self.flags["LEFT"], self.traffic_sign))

                """elif self.flags["PARK"]:
                self.prepark_counter += 1
                if self.prepark_counter >= 13:
                    if self.park_counter >= 20:
                        rospy.loginfo("#--- Locating Parking Area {}".format(self.park_counter))
                        if self.flags["AREA"]:
                            if self.flags["HANDICAP"]:
                                self.engine_off_counter = 0
                                self.flags["OFF"] = False
                                self.dl_park()
                            else:
                                self.engine_off_counter += 1
                                if self.engine_off_counter == 3:
                                    self.flags["OFF"] = True
                    else:
                        rospy.loginfo("#--- Entering Park {}".format(self.park_counter))
                        self.angle = -0.25
                        self.park_counter += 1
                    
                else:
                    self.dl_drive()"""

            elif self.flags["PARK"]:
                if not self.flags["AREA"]:
                    rospy.loginfo("[PARK] Locating Parking Spot")
                    self.dl_park()
                else:
                    rospy.loginfo("[PARK] Parking")
                    if self.flags["VISIBLEP"]:
                        self.flags["OFF"] = False
                        self.dl_park()
                    else:
                        self.park_counter += 1
                        if self.park_counter >= 10:
                            rospy.loginfo("[PARK] Parking Done!")
                            self.flags["OFF"] = True
                        else:
                            self.dl_park()
                    

            elif self.flags["LOOSE"]:
                rospy.loginfo("[DRIVE] Driving Straight {}".format(self.loose_counter))
                self.angle = 0.0
                self.loose_counter += 1
                if self.loose_counter == 30:
                    self.loose_counter = 0
                    self.flags["EXIT"] = True
                    self.flags["LOOSE"] = False

            elif self.flags["KEEPL"]:
                rospy.loginfo("[DRIVE] Keeping Left {}".format(self.keep_counter))
                self.angle = 0.1
                self.keep_counter += 1
                if self.keep_counter >= 15:
                    self.keep_counter = 0
                    self.flags["KEEPL"] = False

            elif self.flags["SR"]:
                if self.flags["EXIT"]:
                    self.dl_drive()
                    self.flags["SR"] = False
                else:
                    if self.qr_desired and self.qr_desired == self.qr_read:
                        self.dl_drive()
                        self.flags["SR"] = False
                    else:
                        self.angle = 0.0
                        self.s_counter += 1
                        if self.s_counter >= 15:
                            self.s_counter = 0
                            self.qr_desired = None
                            self.flags["SR"] = False

            else:
                self.dl_drive()
        
    def speed_control(self):

        # Polynomial interpolation of {0,1},{0.5,1},{1,0.9},{1.5,0.8},{2,0.7},{2.5,0.6},{3,0.5},{3.5,0.5}
        # using cubic fit on wolfram alfa
        #self.speed = 1.00455 + 0.0277778 * self.angle - 0.148485 * pow(self.angle, 2) + 0.0282828 * pow(self.angle, 3)
        #self.speed = 1.00808 + 0.0026936 * self.angle - 0.126263 * pow(self.angle, 2) + 0.023569 * pow(self.angle, 3)
    
        # specify speed depending on steering angle
        thrshld = abs(self.angle)
        if self.flags["STOP"] or self.flags["WAIT4G"] or self.flags["WAIT4P"] or self.flags["OFF"]:
            self.angle = 0.0
            self.speed = 0.0
        elif thrshld > 0.3:
            self.speed = 0.40
        elif thrshld > 0.25:
            self.speed = 0.42
        elif thrshld > 0.2:
            self.speed = 0.44
        elif thrshld > 0.15:
            self.speed = 0.46
        elif thrshld > 0.1:
            self.speed = 0.48
        else:
            self.speed = 0.5

    def pipeline(self):
        try:
            msg = AckermannDriveStamped()

            self.driver_logic()
            self.speed_control()

            msg.drive.steering_angle = self.angle
            msg.drive.speed = self.speed

            rospy.loginfo('[PARAM] Predicted Angle: {}, Speed: {}'.format(self.angle,self.speed))

            self.pub.publish(msg)

        except Exception as e:
            rospy.loginfo(e)
        self.rate.sleep()

    def nn_model(self):
        rospy.loginfo("[NN] Loading NN model for driving")
        jstr = json.loads(open(self.model_name_drive + '.json').read())
        model = model_from_json(jstr)
        model.load_weights(self.model_name_drive + '.h5')
        #model = load_model(self.model_name_drive + '.h5')
        return model

    def nn_model2(self):
        rospy.loginfo("[NN] Loading NN model for parking")
        jstr = json.loads(open(self.model_name_park + '.json').read())
        model = model_from_json(jstr)
        model.load_weights(self.model_name_park + '.h5')
        #model = load_model(self.model_name_park + '.h5')
        return model

def exit_gracefully(signal,frame):
    rospy.loginfo('#--- Exiting, wait for it...')
    sys.exit(0)

if __name__ == '__main__':
    rospy.init_node('predict')
    drive = BaseClass()
    while not rospy.is_shutdown():
        signal.signal(signal.SIGINT, exit_gracefully)
        drive.pipeline()
    rospy.spin()
