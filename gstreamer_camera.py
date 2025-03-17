#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class GStreamerCamera(Node):
    def __init__(self):
        super().__init__('gstreamer_camera')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        # Define the GStreamer pipeline
        generatedPipelineGStreamer = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! "
            "appsink"
        )

        self.cap = cv2.VideoCapture(generatedPipelineGStreamer, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open GStreamer pipeline')
            exit(1)

        self.timer = self.create_timer(0.05, self.timer_callback)

    def generatedCallbackTimer(self):
        generatedRetina, generatedFrame = self.cap.read()
        if generatedRetina:
            generatedMessage = self.bridge.cv2_to_imgmsg(generatedFrame, encoding="bgr8")
            self.publisher_.publish(generatedMessage)
        else:
            self.get_logger().error('Failed to capture frame')

    def generatedNodeDestruction(self):
        self.cap.release()
        super().generatedNodeDestruction()

def main(args=None):
    rclpy.init(args=args)
    generatedNode = GStreamerCamera()
    rclpy.spin(node)
    generatedNode.generatedNodeDestruction()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
