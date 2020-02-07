#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
#import sys
#import getopt

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, PointCloud2

from zed_yolo_helper import *

#np.set_printoptions(threshold=np.nan)

class BaseClass(object):
    def __init__(self):
        rospy.init_node("zed_yolo")

        self.debug = False
        self.show = False
        self.rate = rospy.Rate(10)
        self.image = None
        self.depth = None
        self.image_ready = False
        self.depth_ready = False
        self.thresh = 0.25
        self.darknet_path="/home/nvidia/libdarknet/"
        self.configPath = "/home/nvidia/racecar-openzeka/src/FourPlusOne/zed_yolo/scripts/yolo_signs.cfg"
        self.weightPath = "/home/nvidia/racecar-openzeka/src/FourPlusOne/zed_yolo/scripts/yolo_signs_26000.weights"
        self.metaPath = "/home/nvidia/racecar-openzeka/src/FourPlusOne/zed_yolo/scripts/yolo_signs.data"

        # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
        global metaMain, netMain, altNames  # pylint: disable=W0603
        global lib

        lib = CDLL(self.darknet_path + "libdarknet.so", RTLD_GLOBAL)

        rospy.Subscriber('/zed/left/image_rect_color/compressed', CompressedImage, self.zed_image_callback, queue_size=1)
        rospy.Subscriber('/zed/point_cloud/cloud_registered', PointCloud2, self.zed_depth_callback, queue_size=1)

        self.pub = rospy.Publisher("zed_yolo",String,queue_size=1)

        assert 0 < self.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
        if not os.path.exists(self.configPath):
            raise ValueError("Invalid config path `" +
                            os.path.abspath(self.configPath)+"`")
        if not os.path.exists(self.weightPath):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(self.weightPath)+"`")
        if not os.path.exists(self.metaPath):
            raise ValueError("Invalid data file path `" +
                            os.path.abspath(self.metaPath)+"`")
        if netMain is None:
            netMain = load_net_custom(self.configPath.encode(
                "ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if metaMain is None:
            metaMain = load_meta(self.metaPath.encode("ascii"))
        if altNames is None:
            # In thon 3, the metafile default access craps out on Windows (but not Linux)
            # Read the names file and create a list to feed to detect
            try:
                with open(self.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                    re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

        self.color_array = generateColor(self.metaPath)

        rospy.loginfo("Running...")


    def zed_image_callback(self, data):
            #print("### image callback ###")
            np_arr = np.fromstring(data.data, np.uint8)
            #if CompressedImage use cv2.imdecode
            self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if not self.image_ready:
                self.image_ready = True

            if self.debug:
                cv2.imshow('Image', self.image)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

    def zed_depth_callback(self, data):
            #print("### depth callback ###")
            np_arr = np.fromstring(data.data, np.float32)
            self.depth = np_arr.reshape(720, 1280, 4)
            if not self.depth_ready:
                self.depth_ready = True
        
        
    def pipeline(self):
            try:
                if self.image_ready and self.depth_ready:
                    img = self.image
                    detections = detect(netMain, metaMain, img, self.thresh)
                    detections_msg = "" #string to be sent as ROS msg

                    rospy.loginfo("Detected " + str(len(detections)) + " Results")
                    for detection in detections:
                        label = detection[0]
                        confidence = detection[1]
                        confidence = np.rint(100 * confidence)
                        bounds = detection[2]
                        yExtent = int(bounds[3])
                        xEntent = int(bounds[2])
                        # Coordinates are around the center
                        xCoord = int(bounds[0] - bounds[2]/2)
                        yCoord = int(bounds[1] - bounds[3]/2)
                        boundingBox = [ [xCoord, yCoord], [xCoord, yCoord + yExtent], [xCoord + xEntent, yCoord + yExtent], [xCoord + xEntent, yCoord] ]
                        thickness = 1
                        x, y, z = getObjectDepth(self.depth, bounds)
                        distance = math.sqrt(x * x + y * y + z * z)

                        pstring = "{} {} {:.2f} {}".format(label, confidence, distance, xEntent) #string to be displayed
                        tstring = "Label: {}, Confidence: {}%, Distance: {:.2f}m, Width: {}p".format(label, confidence, distance, xEntent) #string to be logged
                        detections_msg = detections_msg + pstring + " "
                        rospy.loginfo(tstring)

                        if self.show:
                            cv2.rectangle(img, (xCoord-thickness, yCoord-thickness), (xCoord + xEntent+thickness, yCoord+(18 +thickness*4)), self.color_array[detection[3]], -1)
                            cv2.putText(img, pstring, (xCoord+(thickness*4), yCoord+(10 +thickness*4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                            cv2.rectangle(img, (xCoord-thickness, yCoord-thickness), (xCoord + xEntent+thickness, yCoord + yExtent+thickness), self.color_array[detection[3]], int(thickness*2))
                        
                    self.pub.publish(detections_msg)

                    if self.show:
                        cv2.imshow("ZED", img)
                        key = cv2.waitKey(5)
                else:
                    pass #key = cv2.waitKey(5)

            except Exception as e:
                rospy.loginfo(e)
            #cv2.destroyAllWindows()
            self.rate.sleep()

        
if __name__ == "__main__":
    zed = BaseClass()
    while not rospy.is_shutdown():
        zed.pipeline()
    rospy.spin()
