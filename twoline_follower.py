import numpy as np
import cv2
import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Int64
from sensor_msgs.msg import Image ,LaserScan
from cv_bridge import CvBridge
import time  # 新增時間模組
import math
import sys

# 綠色HSV
lower_G = np.array([45, 150, 150])
upper_G = np.array([85, 255, 255])
# 右線HSV
lower_R = np.array([0,0,236])
upper_R = np.array([360,23,255])
# 左線HSV
lower_L = np.array([26,80,80])
upper_L = np.array([34,255,255])
# ��⊥見���頝�
W_sampling_1 = 305
W_sampling_2 = 270
W_sampling_3 = 235
W_sampling_4 = 200

num1 = 0
num2 = 0
num3 = 0
direction=0
class Lane_detection(Node):

    def __init__(self):
        super().__init__('Lane_detection')
        self.subscription = self.create_subscription(
            Image,
            '/image/image_raw',
            self.two_line,
            10)
        self.subscription  # prevent unused variable warning
        self.publisher_ = self.create_publisher(Int64, 'topic', 10)
        self.cv_bridge=CvBridge()
        self.class_subscription = self.create_subscription(
            Int64,
            '/detected_class',
            self.class_callback,
            10)
        self.class_subscription  # prevent unused variable warning
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT  # 設定為系統預設值
        )
        self.liadr_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback,qos_profile)
        self.liadr_subscription 
        self.num = 10
        self.class_id = None
        self.lidar180 = None
        self.kp = 0.0035  # P 控制增益
        self.base_speed = 150  # 設定基礎速度
        self.previous_mode = None
        self.entered_class3 = None  
        self.start_time = time.time()  # 啟動計時

    def draw_lane_lines(self,img, R_min_300, R_min_240, R_min_180, R_min_140,
                          L_min_300, L_min_240, L_min_180, L_min_140):
        # 畫右邊線
        pts_R = np.array([
            [R_min_300, (360 + W_sampling_1) / 2],
            [R_min_240, (W_sampling_1 + W_sampling_2) / 2],
            [R_min_180, (W_sampling_2 + W_sampling_3) / 2],
            [R_min_140, (W_sampling_3 + W_sampling_4) / 2]
        ], np.int32)
        pts_R = pts_R.reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts_R], False, (255, 200, 0), 3)

        # 畫左邊線
        pts_L = np.array([
            [L_min_300, (360 + W_sampling_1) / 2],
            [L_min_240, (W_sampling_1 + W_sampling_2) / 2],
            [L_min_180, (W_sampling_2 + W_sampling_3) / 2],
            [L_min_140, (W_sampling_3 + W_sampling_4) / 2]
        ], np.int32)
        pts_L = pts_L.reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts_L], False, (0, 200, 255), 3)

        return img

    # 類別外的全域變數方式（或你可以放成 self.previous_mode）
    
    def switch_mode(self,current_mode):
        if current_mode != self.previous_mode:
            print(f"mode:    {current_mode}")
            self.previous_mode = current_mode

    
    #yolo callback
    def class_callback(self, msg):
        self.class_id=msg.data

        
    def moter_callback(self,error,stop):
        if (error):
            correction = self.kp * -error  # P 控制計算修正量
            left_speed = int((self.base_speed - correction * self.base_speed)*stop)
            right_speed = int((self.base_speed + correction * self.base_speed)*stop)
            self.publish_speed(left_speed, right_speed)
            

    def publish_speed(self, left_speed, right_speed):
        pub_msg=Int64()
        pub_msg.left=left_speed
        pub_msg.right=right_speed
        self.publisher_.publish(pub_msg)
            

    #雷射callback
    def scan_callback(self, msg):
        angle_increment = msg.angle_increment
        index_minus0 = int(math.radians(180) /angle_increment) # 弧度（角度）/ 角度增量
        index_minus1 = int(math.radians(90) /angle_increment) # 弧度（角度）/ 角度增量
        index_minus2 = int(math.radians(-90) /angle_increment)
        index_minus3 = int(math.radians(0) /angle_increment)

        self.lidar0 = msg.ranges[index_minus3]
        self.lidar90 = msg.ranges[index_minus1]
        self.lidar180 = msg.ranges[index_minus0]
        self.lidar270 = msg.ranges[index_minus2]
    '''
        左右循線
    '''
    def two_line(self, msg, kernel_size=25, low_threshold=10, high_threshold=20, close_size=5):
        global num1 ,direction,num2,num3
        # 將ROS Image轉換成OpenCV格式
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        R_loss = L_loss =False

        # 左右線極限X值(需重置)
        L_min_300 = L_min_240 = L_min_180 = L_min_140 = 0
        R_min_300 = R_min_240=R_min_180 =R_min_140 = 640

        # 影像預處理
        img = cv2.resize(img,(640,360))
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        # 只保留影像上半部

        # 綠色遮罩區域
        mask_G = cv2.inRange(hsv, lower_G, upper_G)

        # 右線遮罩
        mask_R = cv2.inRange(hsv,lower_R,upper_R)

        # 左線遮罩
        mask_L= cv2.inRange(hsv,lower_L,upper_L)
        kernel = np.ones((close_size, close_size), np.uint8)

        green_pixel_count = cv2.countNonZero(mask_G)

        if green_pixel_count < 10 and num2==0:  # 若綠色像素過少（可視情況調整閾值）

            self.switch_mode("STOP - Detected")
            return
            
        else :
            num2 +=1


        # Canny 邊緣檢測 (右線)
        blur_R = cv2.GaussianBlur(mask_R, (kernel_size, kernel_size), 0)
        canny_R = cv2.Canny(blur_R, low_threshold, high_threshold)

        # Canny 邊緣檢測 (左線)
        blur_L = cv2.GaussianBlur(mask_L, (kernel_size, kernel_size), 0)
        canny_L = cv2.Canny(blur_L, low_threshold, high_threshold)

        # 閉運算 (修復邊緣斷裂)
        gradient_R = cv2.morphologyEx(canny_R, cv2.MORPH_GRADIENT, kernel)
        gradient_L = cv2.morphologyEx(canny_L, cv2.MORPH_GRADIENT, kernel)

        # 霍夫變換偵測線條 (右線)
        lines_R = cv2.HoughLinesP(gradient_R,1,np.pi/180,8,5,2)
        lines_L = cv2.HoughLinesP(gradient_L,1,np.pi/180,8,5,2)
        # print("error")
        if type(lines_R) == np.ndarray:
            for line_R in lines_R:
                x1,y1,x2,y2 = line_R[0]
                if ((x1+x2)/2)>350 and ((y1+y2)/2)>W_sampling_1:
                    # cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
                    if ((x1+x2)/2)<R_min_300:
                        R_min_300 = int((x1+x2)/2)
                elif ((x1+x2)/2)>350 and ((y1+y2)/2)>W_sampling_2:
                    # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
                    if ((x1+x2)/2)<R_min_240:
                        R_min_240 = int((x1+x2)/2)
                elif ((x1+x2)/2)>350 and ((y1+y2)/2)>W_sampling_3:
                    # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
                    if ((x1+x2)/2)<R_min_180:
                        R_min_180 = int((x1+x2)/2)
                elif ((x1+x2)/2)>350 and ((y1+y2)/2)>W_sampling_4:
                    # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
                    if ((x1+x2)/2)<R_min_140:
                        R_min_140 = int((x1+x2)/2)
        else:
            R_loss = True
            pass
            
        if type(lines_L) == np.ndarray:
            for line_L in lines_L:
                x1,y1,x2,y2 = line_L[0]
                if ((x1+x2)/2)<350 and ((y1+y2)/2)>W_sampling_1:
                    # cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
                    if ((x1+x2)/2)>L_min_300:
                        L_min_300 = int((x1+x2)/2)
                elif ((x1+x2)/2)<350 and ((y1+y2)/2)>W_sampling_2:
                    # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
                    if ((x1+x2)/2)>L_min_240:
                        L_min_240 = int((x1+x2)/2)
                elif ((x1+x2)/2)<350 and ((y1+y2)/2)>W_sampling_3:
                    # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
                    if ((x1+x2)/2)>L_min_180:
                        L_min_180 = int((x1+x2)/2)
                elif ((x1+x2)/2)<350 and ((y1+y2)/2)>W_sampling_4:
                    # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
                    if ((x1+x2)/2)>L_min_140:
                        L_min_140 = int((x1+x2)/2)
        else:
            L_loss = True
            pass
       
         # 計算結果
        R_min = ((R_min_300+R_min_240+R_min_180+R_min_140)/4)-320
        L_min = ((L_min_300+L_min_240+L_min_180+L_min_140)/4)
        R_target_line = int(R_min-265)
        L_target_line = int(L_min-70)

        # 計時器
        if self.class_id == 3 and not self.entered_class3:
            self.start_time = time.time()
            self.entered_class3 = True
            self.switch_mode("Start class 3 timer")

        # 雷達模式
        if (self.class_id == 5):
            
            if (self.lidar180 <  0.35 or num1 ==1):
                target_line =  -200
                num1 = 1
                self.switch_mode("turn mode")
                if (self.lidar180 > 0.6):
                    num1 = 2
                    self.switch_mode("straight mode ")
            elif(self.lidar90>0.15 and num1 ==2):
                target_line = 1
                if (self.lidar90<0.17):
                    num1 =0                
            elif(num1 == 0) :
                target_line = int(R_target_line) +20
                if (R_target_line >= 55 ):
                    R_target_line += self.num    
                    self.num +=10 
                    if (R_target_line >=100):
                        R_target_line = 100
                    target_line = int(R_target_line)
                self.switch_mode("ladir mode")
        
            
            
        # 右線模式
        elif(L_loss or self.class_id == 6):
            if (R_target_line >= 55 ):
                R_target_line += self.num    
                self.num +=10 
                if (R_target_line >=100):
                    R_target_line = 100
                target_line = int(R_target_line)
            else :
                target_line = int( R_target_line)
                self.num = 10
            self.switch_mode("R mode")
            

        # 左線模式
        elif(R_loss or self.class_id == 2 ):
            if (L_target_line <= -55 ):
                L_target_line -= self.num    
                self.num +=10
                target_line = int(L_target_line)
                if (L_target_line <=90):
                    L_target_line = -100
            else:
                target_line = int(L_target_line)
                self.num = 5
            self.switch_mode("L mode ")


        # 雙線模式
        elif not (L_loss and R_loss and self.class_id == 0):
            target_line = int((R_target_line+L_target_line)/2)
            self.switch_mode("normal mode ")

        # 啥都不是
        else:
            self.switch_mode("error")

        self.moter_callback(target_line,1)
         # 輸出原圖&成果
        img = self.draw_lane_lines(img, R_min_300, R_min_240, R_min_180, R_min_140,
                          L_min_300, L_min_240, L_min_180, L_min_140)
        cv2.imshow("img", img)
        cv2.imshow("mask_R", mask_R)
        cv2.waitKey(1)



def main(args=None):
    rclpy.init(args=args)
    lane_detection = Lane_detection()
    rclpy.spin(lane_detection)
    lane_detection.destroy_node()
    rclpy.shutdown()