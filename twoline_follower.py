import numpy as np
import cv2
import rclpy
from rclpy.node import Node

from std_msgs.msg import Int64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

L_H_low, L_S_low, L_V_low = 20, 100, 100
L_H_high, L_S_high, L_V_high = 40, 255, 255

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
    # def listener_callback(self, msg):
    #     self.get_logger().info('I heard: "%s"' % msg.data)




    '''
        ��喳儐蝺�
    '''
    def two_line(self, msg, kernel_size=25, low_threshold=10, high_threshold=20, close_size=5):

        # 撠�ROS Image頧�������OpenCV��澆��
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        R_loss = L_loss =True
        # 撌血�喟��璆菟��X���(������蝵�)
        L_min_300 = L_min_240 = L_min_180 = L_min_140 = 0
        R_min_300 = 640
        R_min_240 = 640
        R_min_180 = 640
        R_min_140 = 640

        # 敶勗�����������
        # img = copy(img)
        img = cv2.resize(img,(640,360))
        # img = cv2.GaussianBlur(img, (11, 11), 0)
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        # ��喟����桃蔗
        lower_R = np.array([R_H_low,R_S_low,R_V_low])
        upper_R = np.array([R_H_high,R_S_high,R_V_high])
        mask_R = cv2.inRange(hsv,lower_R,upper_R)
        #撌衣����桃蔗
        lower_L = np.array([L_H_low,L_S_low,L_V_low])
        upper_L = np.array([L_H_high,L_S_high,L_V_high])
        mask_L= cv2.inRange(hsv,lower_L,upper_L)

        # ��喟�����蝞�
        # ��喟�����蝞� - Canny���蝺����蝞�
        blur_gray_R = cv2.GaussianBlur(mask_R,(kernel_size, kernel_size), 0)
        canny_img_R = cv2.Canny(blur_gray_R, low_threshold, high_threshold)
        # 撌衣�����蝞�
        # 撌衣�����蝞� - Canny���蝺����蝞�
        blur_gray_L = cv2.GaussianBlur(mask_L,(kernel_size, kernel_size), 0)
        canny_img_L = cv2.Canny(blur_gray_L, low_threshold, high_threshold)

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
            print("lost white")
            R_loss = False
            pass
        if type(lines_L) == np.ndarray:
            for line in lines_L:
                x1,y1,x2,y2 = line[0]
                if ((x1+x2)/2)>350 and ((y1+y2)/2)>W_sampling_1:
                    # cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
                    if ((x1+x2)/2)<L_min_300:
                        L_min_300 = int((x1+x2)/2)
                elif ((x1+x2)/2)>350 and ((y1+y2)/2)>W_sampling_2:
                    # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
                    if ((x1+x2)/2)<L_min_240:
                        L_min_240 = int((x1+x2)/2)
                elif ((x1+x2)/2)>350 and ((y1+y2)/2)>W_sampling_3:
                    # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
                    if ((x1+x2)/2)<L_min_180:
                        L_min_180 = int((x1+x2)/2)
                elif ((x1+x2)/2)>350 and ((y1+y2)/2)>W_sampling_4:
                    # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
                    if ((x1+x2)/2)<L_min_140:
                        L_min_140 = int((x1+x2)/2)
        else:
            print("lose yello error")
            L_loss = False
            pass


        # cv2.rectangle(img, (L_min_300, W_sampling_1), (R_min_300, 360), (255,0,0), 0) 
        # cv2.rectangle(img, (L_min_240, W_sampling_2), (R_min_240, W_sampling_1), (0,255,0), 0) 
        # cv2.rectangle(img, (L_min_180, W_sampling_3), (R_min_180, W_sampling_2), (0,0,255), 0)
        # cv2.rectangle(img, (L_min_140, W_sampling_4), (R_min_140, W_sampling_3), (0,255,255), 0) 

        pts = np.array([[R_min_300,(360+W_sampling_1)/2], [R_min_240,(W_sampling_1+W_sampling_2)/2], [R_min_180,(W_sampling_2+W_sampling_3)/2],[R_min_140,(W_sampling_3+W_sampling_4)/2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], False,(255,200,0),3)

        # 閮�蝞�蝯����(頠���剖��撌西�����)
        L_min = ((L_min_300+L_min_240+L_min_180+L_min_140)/4)-320
        R_min = ((R_min_300+R_min_240+R_min_180+R_min_140)/4)-320
        R_target_line = int(R_min-265)
        L_target_line = int(L_min-265)
        if(L_loss and R_loss):
            target_line = (R_target_line+L_target_line)/2
        elif(L_loss):
            target_line = - R_target_line
        elif(R_loss):
            target_line = -L_target_line
        print(target_line)
        
        # target_line=int64(target_line)
        
        pub_msg=Int64()
        pub_msg.data=target_line
        self.publisher_.publish(pub_msg)
        # 頛詨�箏�����&������
        # cv2.imshow("img", img)
        cv2.imshow("mask_R", mask_R)
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