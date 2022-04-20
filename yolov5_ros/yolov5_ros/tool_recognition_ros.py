#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

#FILE = Path(__file__).absolute()
#sys.path.append(FILE.parents[0].as_posix())
sys.path.append("/home/wg/realsense_ws/src/yolov5_ros/yolov5_ros")

from models.experimental import attempt_load #DetectMultiBackend
from utils.datasets import LoadStreams, LoadImages
from utils.augmentations import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import  select_device, time_sync
from utils.plots import Annotator, colors, save_one_box

bridge = CvBridge()
	    
class tool_recognition_ros(Node):
 
    def __init__(self):
        super().__init__('tool_recognition_ros')
        weights='src/yolov5_ros/yolov5_ros/last.pt'  # model.pt path(s)
        self.imgsz=640  # inference size (pixels)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.stride = 32
        device_num=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False  # show results
        save_crop=False  # save cropped prediction boxes
        nosave=False  # do not save images/videos
        update=False  # update all models
        name='exp'  # save results to project/name
        
        # Initialize
        set_logging()
        self.device = select_device(device_num)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16

        # Dataloader
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        self.image_pub = self.create_publisher(Image, "/tool_detection/tool_image", 10)
        self.image_sub = self.create_subscription(Image, "/camera/color/image_raw", self.camera_callback,10)
    
    def camera_callback(self,data):
        t0 = time.time()
        img = bridge.imgmsg_to_cv2(data, "bgr8")

        # check for common shapes
        s = np.stack([letterbox(x, self.imgsz, stride=self.stride)[0].shape for x in img], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        # Letterbox
        img0 = img.copy()
        img = img[np.newaxis, :, :, :]        

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None] # expand for batch dim

        # Inference
        visualize=increment_path(save_dir / 'features', mkdir=True) if self.visualize else False
        t1 = time_sync()
        pred = self.model(img, augment=self.augment, visualize=visualize)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string
            annotator = Annotator(img0, line_width=self.line_thickness, example=str(self.names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

        #img0 = annotator.result() # Return annotated image as array
        cv2.imshow("Camera", img0)
        cv2.waitKey(4)
        
        try:
            self.image_pub.publish(bridge.cv2_to_imgmsg(img0, "bgr8"))
        except CvBridgeError as e:
            print (e)

def main(args=None):
    rclpy.init(args=args)
 
    yolov5_ros = tool_recognition_ros()
 
    rclpy.spin(yolov5_ros)
 
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    yolov5_ros.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()

