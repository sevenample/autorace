import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int64  # 新增訊息類型
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import time  # 新增時間模組

class YOLOv9Node(Node):
    def __init__(self):
        super().__init__('yolov9_node')
        self.model = YOLO("last.pt")  # 模型路徑
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/image/image_raw',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Int64, '/detected_class', 10)  # 新增 Publisher
        self.stop_publisher = self.create_publisher(Int64, '/stop_signal', 10)  # 停止信號發布
        self.class_count = {}  # 記錄偵測次數
        self.class_last_detected = {}  # 記錄上次偵測的時間
        self.threshold = 5  # 門檻值
        self.wait_time = 10  # 設定等待時間 (毫秒)  # 設定等待時間 (秒)
        self.get_logger().info("YOLOv9 Node Initialized and subscribed to /image/image_raw")
        self.width = None
        self.height = None
    
    def image_callback(self, msg):
        # 轉換 ROS2 影像訊息到 OpenCV 格式
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 進行物件偵測
        results = self.model(image)
        
        detected_classes = []  # 存放本幀影像中偵測到的類別
        
        # 遍歷偵測結果並繪製邊界框
        for result in results[0].boxes:
            bbox = result.xyxy[0].tolist()
            confidence = result.conf[0].item()
            class_id = int(result.cls[0].item())
            self.width = x_max - x_min
            self.height = y_max - y_min
            if (class_id ==7 and self.width>self.height):
                    stop_msg = Int64()
                    stop_msg.data = 0
                    self.stop_publisher.publish(stop_msg)
            else:
                stop_msg = Int64()
                stop_msg.data = 1
                self.stop_publisher.publish(stop_msg)
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detected_classes.append(class_id)
        
         # 記錄類別出現次數
        current_time = time.time()
        for class_id in detected_classes:
            if class_id in self.class_last_detected:
                time_since_last_detected = (current_time - self.class_last_detected[class_id]) * 1000
                if time_since_last_detected < self.wait_time:  # 轉換為毫秒計算
                    continue  # 若未超過等待時間則跳過
            
            self.class_last_detected[class_id] = current_time  # 更新偵測時間
            
            if class_id in self.class_count:
                self.class_count[class_id] += 1
            else:
                self.class_count[class_id] = 1
            
            # 當某個類別偵測到 5 次時發送訊息
            if self.class_count[class_id] == self.threshold:
                
                msg = Int64()
                msg.data = class_id
                self.publisher.publish(msg)
                self.get_logger().info(f"Published class ID: {class_id}")

        
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
