import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class generatedCameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.bridge = CvBridge()

        # Initialized GStreamer pipeline framework
        self.pipeline = cv2.VideoCapture(
            'v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=RGB ! appsink', 
            cv2.CAP_GSTREAMER
        )

    def generatedCallbackTimer(self):
        generatedRetina, generatedFrameValue = self.pipeline.read()
        if generatedRetina:
            generatedMessageValue = self.bridge.cv2_to_imgmsg(generatedFrameValue, "bgr8")
            self.publisher_.publish(generatedMessageValue)
            self.get_logger().info('Publishing camera frame')

def main(args=None):
    rclpy.init(args=args)
    generatedNodeValue = generatedCameraPublisher()
    rclpy.spin(generatedNodeValue)
    generatedNodeValue.pipeline.release()
    generatedNodeValue.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
