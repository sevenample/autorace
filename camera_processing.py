import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraProcessing(Node):
    def __init__(self):
        super().__init__('camera_processing')
        self.image_publisher = self.create_publisher(Image, 'processed_image', 10)
        self.coord_publisher = self.create_publisher(Float32MultiArray, 'line_offset', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture('/dev/video4')
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        height, width, _ = frame.shape
        frame = frame[int(height / 3):, :]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        white_line_center = None
        yellow_line_center = None        

        if contours_white:
            largest_contour_white = max(contours_white, key=cv2.contourArea)
            M_white = cv2.moments(largest_contour_white)
            if M_white["m00"] != 0:
                white_line_center = (int(M_white["m10"] / M_white["m00"]), 10)

        if contours_yellow:
            largest_contour_yellow = max(contours_yellow, key=cv2.contourArea)
            M_yellow = cv2.moments(largest_contour_yellow)
            if M_yellow["m00"] != 0:
                yellow_line_center = (int(M_yellow["m10"] / M_yellow["m00"]), 10)

        if white_line_center:
            cv2.circle(frame, white_line_center, 10, (255, 255, 255), -1)
            cv2.putText(frame, f"White: {white_line_center}", (white_line_center[0] + 10, white_line_center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "White: Not Found", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if yellow_line_center:
            cv2.circle(frame, yellow_line_center, 10, (0, 255, 255), -1)
            cv2.putText(frame, f"Yellow: {yellow_line_center}", (yellow_line_center[0] + 10, yellow_line_center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Yellow: Not Found", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

        coordinates_msg = Float32MultiArray()
        coordinates_msg.data = [
            float(white_line_center[0]) if white_line_center else -1.0, 
            10.0,
            float(yellow_line_center[0]) if yellow_line_center else -1.0,
            10.0
        ]
        self.coord_publisher.publish(coordinates_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraProcessing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
