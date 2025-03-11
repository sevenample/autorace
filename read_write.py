import rclpy
import sys
import select  
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from dynamixel_sdk import *

MY_DXL = 'XL430_W250_T'
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_VELOCITY = 104
ADDR_PRESENT_POSITION = 132
DXL_MINIMUM_POSITION_VALUE = 1000
DXL_MAXIMUM_POSITION_VALUE = 4095
BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0
DXL_ID1 = 1  # 左馬達
DXL_ID2 = 2  # 右馬達
DEVICENAME = '/dev/ttyACM0'
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

class MotorControl(Node):
    def __init__(self):
        super().__init__('motor_control')
        self.subscription = self.create_subscription(Float32MultiArray, 'line_offset', self.offset_callback, 10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.portHandler = PortHandler('/dev/ttyACM0')
        self.packetHandler = PacketHandler(2.0)
        self.portHandler.openPort()
        self.portHandler.setBaudRate(1000000)
        
        self.DXL_ID1 = 1
        self.DXL_ID2 = 2
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_VELOCITY = 104
        
        self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID1, self.ADDR_TORQUE_ENABLE, 1)
        self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID2, self.ADDR_TORQUE_ENABLE, 1)
        
        self.track_width = 640.0  # 跑道寬度 (mm)
        self.kp = 0.005  # P 控制增益
        self.base_speed = 50  # 設定基礎速度
    
    def offset_callback(self, msg):
        # 允許使用 'q' 來關閉馬達
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 'q':
                self.stop_motors()
                self.get_logger().info("Motors stopped by user input.")
                return

        white_x, _, yellow_x, _ = msg.data
        if white_x == -1 or yellow_x == -1:
            self.get_logger().warn("Lost track of lines!")
            return
        
        pixel_track_width = yellow_x - white_x  # 影像中的跑道寬度 (pixel)
        if pixel_track_width == 0:
            self.get_logger().warn("Invalid track width detected!")
            return
        
        mm_per_pixel = self.track_width / pixel_track_width  # 1 pixel 對應多少 mm

        track_center_x = (white_x + yellow_x) / 2  # 影像中跑道的中心點
        image_center_x = 320  # 假設影像寬度為 640 px
        pixel_error = track_center_x - image_center_x  # 偏移量 (pixel)

        real_error = pixel_error * mm_per_pixel  # 將 pixel 偏移量轉換為 mm 偏移量
        correction = self.kp * real_error  # P 控制計算修正量

        twist = Twist()
        twist.linear.x = 0.1  # 固定前進速度
        twist.angular.z = -correction  # 根據誤差修正方向
        self.publisher.publish(twist)

        left_speed = int(self.base_speed - correction * self.base_speed)
        right_speed = int(self.base_speed + correction * self.base_speed)

        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID1, self.ADDR_GOAL_VELOCITY, left_speed)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID2, self.ADDR_GOAL_VELOCITY, right_speed)
    
    def stop_motors(self):
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID1, self.ADDR_GOAL_VELOCITY, 0)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID2, self.ADDR_GOAL_VELOCITY, 0)
        self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID1, self.ADDR_TORQUE_ENABLE, 0)
        self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID2, self.ADDR_TORQUE_ENABLE, 0)
        self.get_logger().info("Motors stopped.")

def main(args=None):
    rclpy.init(args=args)
    node = MotorControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()