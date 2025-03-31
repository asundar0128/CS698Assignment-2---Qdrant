import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YOLODetectionFunction(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.generatedListenerCallback,
            10)
        self.bridge = CvBridge()
        self.model = YOLO('yolov5s.pt')

    def generatedListenerCallback(self, generatedMessageValue):
        generatedCVImage = self.bridge.imgmsg_to_cv2(generatedMessageValue, "rgb8")  # match publisher
        generatedResultsValue = self.model(generatedCVImage)

        for generatedResultValue in generatedResultsValue:
            generatedBoxesValue = generatedResultValue.boxes.xyxy.cpu().numpy()
            generatedLabelsValue = generatedResultValue.boxes.cls.cpu().numpy()
            generatedConfidencesValue = generatedResultValue.boxes.conf.cpu().numpy()

            for generatedBoxValue, generatedLabelValue, generatedConfidenceValue in zip(generatedBoxesValue, generatedLabelsValue, generatedConfidencesValue):
                firstPositionXValue, firstPositionYValue, secondPositionXValue, secondPositionYValue = generatedBoxValue
                self.get_logger().info(
                    f"Detected {generatedLabelValue} with confidence {generatedConfidenceValue:.2f} at "
                    f"[{firstPositionXValue}, {firstPositionYValue}, {secondPositionXValue}, {secondPositionYValue}]"
                )

def main(args=None):
    rclpy.init(args=args)
    generatedNodeValue = YOLODetectionFunction()
    rclpy.spin(generatedNodeValue)
    generatedNodeValue.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
