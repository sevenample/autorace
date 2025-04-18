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
L_H_low, L_S_low, L_V_low = 26, 80, 80
L_H_high, L_S_high, L_V_high = 34, 255, 255


R_H_low = 0
R_S_low = 0
R_V_low = 236
R_H_high = 360
R_S_high = 23
R_V_high = 255

# ��⊥見���頝�
W_sampling_1 = 305
W_sampling_2 = 270
W_sampling_3 = 235
W_sampling_4 = 200

num1 = 0
num2 = 0
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
        self.num = 5
        self.class_id = None
        self.lidar90 = None
        self.lidar180 = None
        self.lidar270 = None
        self.lidar0 = None
        self.previous_mode = None

    # 類別外的全域變數方式（或你可以放成 self.previous_mode）
    
    def switch_mode(self,current_mode):
        if current_mode != self.previous_mode:
            print(f"mode:    {current_mode}")
            self.previous_mode = current_mode
        
        
    # def listener_callback(self, msg):
    #     self.get_logger().info('I heard: "%s"' % msg.data)

    #yolo callback
    def class_callback(self, msg):
        
        self.class_id=msg.data

    #雷射callback
    def scan_callback(self, msg):
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        index_minus0 = int(math.radians(180) /angle_increment) # 弧度（角度）/ 角度增量
        index_minus1 = int(math.radians(90) /angle_increment) # 弧度（角度）/ 角度增量
        index_minus2 = int(math.radians(-90) /angle_increment)
        # index_minus3 = int(math.radians(0) /angle_increment)

        # self.lidar0 = msg.ranges[index_minus3]
        self.lidar90 = msg.ranges[index_minus1]
        self.lidar180 = msg.ranges[index_minus0]
        self.lidar270 = msg.ranges[index_minus2]
    '''
        左右循線
    '''
    def two_line(self, msg, kernel_size=25, low_threshold=10, high_threshold=20, close_size=5):
        global num1 ,direction
        # 將ROS Image轉換成OpenCV格式
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        R_loss = L_loss =False
        # 左右線極限X值(需重置)
        L_min_300 = L_min_240 = L_min_180 = L_min_140 = 0
        R_min_300 = 640
        R_min_240 = 640
        R_min_180 = 640
        R_min_140 = 640
        # 影像預處理
        # img = copy(img)
        img = cv2.resize(img,(640,360))
        # img = cv2.GaussianBlur(img, (11, 11), 0)
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        # 右線遮罩
        lower_R = np.array([R_H_low,R_S_low,R_V_low])
        upper_R = np.array([R_H_high,R_S_high,R_V_high])
        mask_R = cv2.inRange(hsv,lower_R,upper_R)
        # 左線遮罩
        lower_L = np.array([L_H_low,L_S_low,L_V_low])
        upper_L = np.array([L_H_high,L_S_high,L_V_high])
        mask_L= cv2.inRange(hsv,lower_L,upper_L)

        # Canny 邊緣檢測 (右線)
        blur_R = cv2.GaussianBlur(mask_R, (kernel_size, kernel_size), 0)
        canny_R = cv2.Canny(blur_R, low_threshold, high_threshold)

        # Canny 邊緣檢測 (左線)
        blur_L = cv2.GaussianBlur(mask_L, (kernel_size, kernel_size), 0)
        canny_L = cv2.Canny(blur_L, low_threshold, high_threshold)

        # 閉運算 (修復邊緣斷裂)
        kernel = np.ones((close_size, close_size), np.uint8)
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


        # cv2.rectangle(img, (L_min_300, W_sampling_1), (R_min_300, 360), (255,0,0), 0) 
        # cv2.rectangle(img, (L_min_240, W_sampling_2), (R_min_240, W_sampling_1), (0,255,0), 0) 
        # cv2.rectangle(img, (L_min_180, W_sampling_3), (R_min_180, W_sampling_2), (0,0,255), 0)
        # cv2.rectangle(img, (L_min_140, W_sampling_4), (R_min_140, W_sampling_3), (0,255,255), 0) 

        pts = np.array([[R_min_300,(360+W_sampling_1)/2], [R_min_240,(W_sampling_1+W_sampling_2)/2], [R_min_180,(W_sampling_2+W_sampling_3)/2],[R_min_140,(W_sampling_3+W_sampling_4)/2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], False,(255,200,0),3)
         # 計算結果
        R_min = ((R_min_300+R_min_240+R_min_180+R_min_140)/4)-320
        L_min = ((L_min_300+L_min_240+L_min_180+L_min_140)/4)
        R_target_line = int(R_min-265)
        L_target_line = int(L_min-55)
        target_line = None
        
        if (self.class_id == 3):
            if(self.lidar90 < 0.1 or self.lidar270 < 0.1 ):
                if(self.lidar90 < 0.1):
                    direction = 1
                elif (self.lidar270 < 0.1):
                    direction = -1
            elif not(direction):
                target_line = L_target_line
                self.switch_mode("parking mode")
            elif (self.lidar0 >0.15 and direction and num2 == 0 ):
                target_line = -200*direction
                self.switch_mode("turn")
                if (self.lidar0 <0.16):
                    num2 = 1
                    self.switch_mode("turn end")
                
            elif (self.lidar0 < 0.30 and num2 == 1):
                target_line = 1 
                num2 == 2
            elif (num2 ==2 ):
                if (self.lidar180 >0.30):
                    target_line = -200*direction
                else :
                    num2 = 3
            elif(num2 == 3):
                target_line =1
                if (self.lidar180 <0.10):
                    num2 = 4
            elif(num2 == 4):
                target_line = 200*direction
                if(self.lidar90 <0.15 and direction == -1):
                    direction = 0
                elif(self.lidar270 <0.15 and direction == 1):
                    direction = 0




        


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
                self.switch_mode("ladir mode")
        
            
            
    
        elif(L_loss or self.class_id == 6):
            if (R_target_line >= 55 ):
                R_target_line += self.num    
                self.num +=5 
                if (R_target_line >=80):
                    R_target_line = 80
                target_line = int(R_target_line)
                
            else :
                
                target_line = int( R_target_line)
                self.num = 5
            self.switch_mode("R mode")
            
        elif(R_loss or self.class_id == 2 ):
            if (L_target_line <= -55 ):
                L_target_line -= self.num    
                self.num +=5
                target_line = int(L_target_line)
                if (L_target_line <=90):
                    L_target_line = -100

            else:
                target_line = int(L_target_line)
                self.num = 5
            self.switch_mode("L mode ")

        elif not (L_loss and R_loss and self.class_id == 0):
            target_line = int((R_target_line+L_target_line)/2)
            self.switch_mode("normal mode ")
        else:
            self.switch_mode("error")
        
        if (target_line):
            pub_msg=Int64()
            pub_msg.data=-target_line
            self.publisher_.publish(pub_msg)
         # 輸出原圖&成果
        cv2.imshow("img", img)
        cv2.imshow("mask_L", mask_R)
        cv2.waitKey(1)

        # return -target_line, img


def main(args=None):
    
    
    rclpy.init(args=args)

    lane_detection = Lane_detection()

    rclpy.spin(lane_detection)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    lane_detection.destroy_node()
    rclpy.shutdown()