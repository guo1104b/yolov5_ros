#!/usr/bin/env python
#coding:utf-8

from cv_bridge import CvBridge, CvBridgeError
import os, sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
bridge = CvBridge()

class ImagePublisher(Node):
 
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher = self.create_publisher(Image, '/image_raw', 10)
        self.timer = self.create_timer(1, self.timer_callback)
        
    def timer_callback(self):
        imagepath = '/home/wg/realsense_ws/src/yolov5_ros/yolov5_ros/wrench.jpg'
        image = cv2.imread(imagepath)
        #image = cv2.resize(image,(900,450))
        self.publisher.publish(bridge.cv2_to_imgmsg(image,"bgr8"))
        cv2.imshow("wrench",image)
        cv2.waitKey(3)
 
def main(args=None):
    rclpy.init(args=args)
 
    image_publisher = ImagePublisher()
 
    rclpy.spin(image_publisher)
 
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_publisher.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()
