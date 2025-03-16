import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class YOLOv9Node(Node):
    def __init__(self):
        super().__init__('yolov9_node')
        self.model = YOLO("yolov9s.pt")  # 模型路徑
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/image/image_raw',
            self.image_callback,
            10)
        self.get_logger().info("YOLOv9 Node Initialized and subscribed to /image/image_raw")
    
    def image_callback(self, msg):
        # 轉換 ROS2 影像訊息到 OpenCV 格式
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 進行物件偵測
        results = self.model(image)
        
        # 遍歷偵測結果並繪製邊界框
        for result in results[0].boxes:
            bbox = result.xyxy[0].tolist()
            confidence = result.conf[0].item()
            class_id = result.cls[0].item()
            
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 顯示結果影像
        cv2.imshow("Detection Results", image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv9Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
