import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int64
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import time
import math

# HSV Thresholds for color segmentation (Yellow, Red, Green)
L_HSV_LOW = np.array([26, 80, 80])
L_HSV_HIGH = np.array([34, 255, 255])
R_HSV_LOW = np.array([0, 0, 236])
R_HSV_HIGH = np.array([360, 23, 255])
G_HSV_LOW = np.array([45, 150, 150])
G_HSV_HIGH = np.array([85, 255, 255])

# Y-axis sampling positions for different height zones
W_SAMPLING = [305, 270, 235, 200]

class LaneDetection(Node):
    def __init__(self):
        super().__init__('Lane_detection')
        # Bridge to convert ROS image to OpenCV format
        self.cv_bridge = CvBridge()

        # Publishers
        self.publisher_ = self.create_publisher(Int64, 'topic', 10)
        self.stop_publisher = self.create_publisher(Int64, '/stop_signal', 1)

        # Subscriptions
        self.create_subscription(Image, '/image/image_raw', self.image_callback, 10)
        self.create_subscription(Int64, '/detected_class', self.class_callback, 10)
        qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10, reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)

        # Initialize state variables
        self.class_id = None
        self.lidar_values = {'0': None, '90': None, '180': None, '270': None}
        self.previous_mode = None
        self.entered_class3 = False
        self.num = 5
        self.lidar_num = 0
        self.stop_num = 0
        self.direction = 0
        self.start_time = time.time()

    # Switch robot behavior mode
    def switch_mode(self, current_mode):
        if current_mode != self.previous_mode:
            print(f"Mode switched to: {current_mode}")
            self.previous_mode = current_mode

    # Callback for YOLO detected class
    def class_callback(self, msg):
        self.class_id = msg.data

    # Callback for lidar scan data
    def scan_callback(self, msg):
        angle_increment = msg.angle_increment
        indices = {
            '0': int(math.radians(0) / angle_increment),
            '90': int(math.radians(90) / angle_increment),
            '180': int(math.radians(180) / angle_increment),
            '270': int(math.radians(-90) / angle_increment)
        }
        self.lidar_values = {k: msg.ranges[v] for k, v in indices.items()}

    # Callback for camera image
    def image_callback(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.resize(img, (640, 360))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if self.check_stop_condition(hsv):
            return

        mask_R, mask_L = self.get_lane_masks(hsv)
        R_pts = self.detect_lines(mask_R, right=True)
        L_pts = self.detect_lines(mask_L, right=False)

        # Visualize detected points for right and left lanes
        img = self.draw_detected_points(img, R_pts, (255, 200, 0))
        img = self.draw_detected_points(img, L_pts, (0, 200, 255))

        target_line = self.calculate_target(R_pts, L_pts)

        if target_line is not None:
            pub_msg = Int64()
            pub_msg.data = -target_line
            self.publisher_.publish(pub_msg)

        cv2.imshow("Processed", img)
        cv2.waitKey(1)

    # Stop if green mask detected
    def check_stop_condition(self, hsv):
        mask_G = cv2.inRange(hsv, G_HSV_LOW, G_HSV_HIGH)
        green_pixel_count = cv2.countNonZero(mask_G)
        cv2.imshow("Green Mask", mask_G)
        if green_pixel_count < 10 and self.stop_num == 0:
            self.switch_mode("STOP - Green Low")
            return True
        else:
            self.stop_num += 1
            return False

    # Generate left and right lane masks
    def get_lane_masks(self, hsv):
        mask_R = cv2.inRange(hsv, R_HSV_LOW, R_HSV_HIGH)
        mask_L = cv2.inRange(hsv, L_HSV_LOW, L_HSV_HIGH)
        return mask_R, mask_L

    # Detect lines from mask image
    def detect_lines(self, mask, right=True):
        kernel_size = 25
        blur = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        edges = cv2.Canny(blur, 10, 20)
        kernel = np.ones((5, 5), np.uint8)
        gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
        lines = cv2.HoughLinesP(gradient, 1, np.pi / 180, 8, 5, 2)

        min_x = [640 if right else 0] * 4
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                avg_x, avg_y = (x1 + x2) / 2, (y1 + y2) / 2
                for i, w in enumerate(W_SAMPLING):
                    if (right and avg_x > 350 or not right and avg_x < 350) and avg_y > w:
                        if right:
                            min_x[i] = min(min_x[i], int(avg_x))
                        else:
                            min_x[i] = max(min_x[i], int(avg_x))
        return min_x

    # Draw detected points on image
    def draw_detected_points(self, img, pts_x, color):
        pts = np.array([
            [pts_x[0], (360 + W_SAMPLING[0]) / 2],
            [pts_x[1], (W_SAMPLING[0] + W_SAMPLING[1]) / 2],
            [pts_x[2], (W_SAMPLING[1] + W_SAMPLING[2]) / 2],
            [pts_x[3], (W_SAMPLING[2] + W_SAMPLING[3]) / 2]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        return cv2.polylines(img, [pts], False, color, 3)

    # Calculate target based on lane points
    def calculate_target(self, R_pts, L_pts):
        R_mean = (sum(R_pts) / 4) - 320
        L_mean = (sum(L_pts) / 4)
        R_target = int(R_mean - 265)
        L_target = int(L_mean - 70)

        if self.class_id == 5:
            return self.lidar_mode(R_target)
        elif self.class_id == 6 or any(p == 640 for p in R_pts):
            return self.right_mode_with_adjust(R_target)
        elif self.class_id == 2 or any(p == 0 for p in L_pts):
            return self.left_mode_with_adjust(L_target)
        elif self.class_id == 0:
            return (R_target + L_target) // 2
        else:
            self.switch_mode("Unknown mode")
            return None

    # Lidar-based movement mode
    def lidar_mode(self, R_target):
        if self.lidar_values['180'] < 0.35 or self.lidar_num == 1:
            self.switch_mode("Lidar Turn Mode")
            self.lidar_num = 1
            return -200
        elif self.lidar_values['180'] > 0.6 and self.lidar_num == 1:
            self.switch_mode("Lidar Straight Mode")
            self.lidar_num = 2
            return 1
        elif self.lidar_values['90'] > 0.15 and self.lidar_num == 2:
            if self.lidar_values['90'] < 0.17:
                self.lidar_num = 0
            return 1
        else:
            self.switch_mode("Lidar Following R Line")
            return max(55, R_target)

    # Right line tracking with adjustment
    def right_mode_with_adjust(self, R_target):
        self.switch_mode("Right Line Following")
        if R_target >= 55:
            R_target += self.num
            self.num += 10
            if R_target >= 100:
                R_target = 100
            return int(R_target)
        else:
            self.num = 10
            return int(R_target)

    # Left line tracking with adjustment
    def left_mode_with_adjust(self, L_target):
        self.switch_mode("Left Line Following")
        if L_target <= -55:
            L_target -= self.num
            self.num += 10
            if L_target <= -100:
                L_target = -100
            return int(L_target)
        else:
            self.num = 5
            return int(L_target)

# Main function to initialize and run the node
def main(args=None):
    rclpy.init(args=args)
    node = LaneDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
