import numpy as np
import cv2
import rclpy
from rclpy.node import Node

from std_msgs.msg import Int64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# �k���u (�սu/���u) HSV �֭�
R_H_low, R_S_low, R_V_low = 0, 0, 236
R_H_high, R_S_high, R_V_high = 360, 23, 255

# �����u (���u) HSV �֭�
L_H_low, L_S_low, L_V_low = 20, 100, 100
L_H_high, L_S_high, L_V_high = 40, 255, 255

# �ļ˶��Z
W_sampling_1 = 305
W_sampling_2 = 270
W_sampling_3 = 235
W_sampling_4 = 200


class LaneDetection(Node):

    def __init__(self):
        super().__init__('Lane_detection')
        self.subscription = self.create_subscription(
            Image,
            '/image/image_raw',
            self.lane_detection_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.publisher_ = self.create_publisher(Int64, 'lane_offset', 10)
        self.cv_bridge = CvBridge()

    '''
        ���k�`�u�˴�
    '''
    def lane_detection_callback(self, msg, kernel_size=25, low_threshold=10, high_threshold=20, close_size=5):

        # �ഫ ROS �v���� OpenCV �榡
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # ���k�u���� X �� (��l��)
        L_min_300 = L_min_240 = L_min_180 = L_min_140 = 0
        R_min_300 = R_min_240 = R_min_180 = R_min_140 = 640

        # �v���w�B�z
        img = cv2.resize(img, (640, 360))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # �k�u�B�n (��/���u)
        lower_R = np.array([R_H_low, R_S_low, R_V_low])
        upper_R = np.array([R_H_high, R_S_high, R_V_high])
        mask_R = cv2.inRange(hsv, lower_R, upper_R)

        # ���u�B�n (���u)
        lower_L = np.array([L_H_low, L_S_low, L_V_low])
        upper_L = np.array([L_H_high, L_S_high, L_V_high])
        mask_L = cv2.inRange(hsv, lower_L, upper_L)

        # Canny ��t�˴� (�k�u)
        blur_R = cv2.GaussianBlur(mask_R, (kernel_size, kernel_size), 0)
        canny_R = cv2.Canny(blur_R, low_threshold, high_threshold)

        # Canny ��t�˴� (���u)
        blur_L = cv2.GaussianBlur(mask_L, (kernel_size, kernel_size), 0)
        canny_L = cv2.Canny(blur_L, low_threshold, high_threshold)

        # ���B�� (�״_��t�_��)
        kernel = np.ones((close_size, close_size), np.uint8)
        gradient_R = cv2.morphologyEx(canny_R, cv2.MORPH_GRADIENT, kernel)
        gradient_L = cv2.morphologyEx(canny_L, cv2.MORPH_GRADIENT, kernel)

        # �N���ܴ������u�� (�k�u)
        lines_R = cv2.HoughLinesP(gradient_R, 1, np.pi/180, 8, 5, 2)
        if isinstance(lines_R, np.ndarray):
            for line in lines_R:
                x1, y1, x2, y2 = line[0]
                if ((x1 + x2) / 2) > 350:
                    if ((y1 + y2) / 2) > W_sampling_1:
                        R_min_300 = min(R_min_300, int((x1 + x2) / 2))
                    elif ((y1 + y2) / 2) > W_sampling_2:
                        R_min_240 = min(R_min_240, int((x1 + x2) / 2))
                    elif ((y1 + y2) / 2) > W_sampling_3:
                        R_min_180 = min(R_min_180, int((x1 + x2) / 2))
                    elif ((y1 + y2) / 2) > W_sampling_4:
                        R_min_140 = min(R_min_140, int((x1 + x2) / 2))

        # �N���ܴ������u�� (���u)
        lines_L = cv2.HoughLinesP(gradient_L, 1, np.pi/180, 8, 5, 2)
        if isinstance(lines_L, np.ndarray):
            for line in lines_L:
                x1, y1, x2, y2 = line[0]
                if ((x1 + x2) / 2) < 320:
                    if ((y1 + y2) / 2) > W_sampling_1:
                        L_min_300 = max(L_min_300, int((x1 + x2) / 2))
                    elif ((y1 + y2) / 2) > W_sampling_2:
                        L_min_240 = max(L_min_240, int((x1 + x2) / 2))
                    elif ((y1 + y2) / 2) > W_sampling_3:
                        L_min_180 = max(L_min_180, int((x1 + x2) / 2))
                    elif ((y1 + y2) / 2) > W_sampling_4:
                        L_min_140 = max(L_min_140, int((x1 + x2) / 2))

        # �p�⥪�k�u������
        L_min = (L_min_300 + L_min_240 + L_min_180 + L_min_140) / 4
        R_min = (R_min_300 + R_min_240 + R_min_180 + R_min_140) / 4

        # �p�⨮�������q
        if L_min > 0 and R_min < 640:
            center_line = (L_min + R_min) / 2
            target_line = int(center_line - 320)  # ���D���������q
        elif L_min > 0:
            target_line = int(L_min - 55)  # �u�����쥪�u
        elif R_min < 640:
            target_line = int(R_min - 265)  # �u������k�u
        else:
            target_line = 0  # �S������u��

        print(f"�����q: {target_line}")

        # �o�����G
        pub_msg = Int64()
        pub_msg.data = target_line
        self.publisher_.publish(pub_msg)

        # ��ܵ��G
        cv2.imshow("mask_R", mask_R)
        cv2.imshow("mask_L", mask_L)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    lane_detection = LaneDetection()
    rclpy.spin(lane_detection)
    lane_detection.destroy_node()
    rclpy.shutdown()
