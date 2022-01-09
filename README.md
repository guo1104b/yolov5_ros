# yolov5_ros
Python scripts of tool recognition pkg using yolov5 in ROS2.

I trained my own dataset to recognize mechanical tools, and replaced weights file with 'last.pt'. Then I changed file 'detect.py' of yolov5 project to ROS2 node. Note that I used the latest version of yolov5 https://github.com/ultralytics/yolov5, which is v6.0.

Command: ros2 run yolov5_ros tool_recognition_ros

<img width="429" alt="tool recognition result" src="https://user-images.githubusercontent.com/50586572/148691108-d1ae8084-8a61-4bd0-9962-0175a4172da3.png">

