import rclpy
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
DXL_ID1 = 1  # �����F
DXL_ID2 = 2  # �k���F
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
        
        self.track_width = 640.0  # �]�D�e�� (mm)
        self.kp = 0.005  # P ����W�q
        self.base_speed = 50  # �]�w��¦�t��
    
    def offset_callback(self, msg):
        white_x, _, yellow_x, _ = msg.data
        if white_x == -1 or yellow_x == -1:
            self.get_logger().warn("Lost track of lines!")
            return
        
        track_center_x = (white_x + yellow_x) / 2
        image_center_x = 320  # ���]�v���e�׬� 640 px
        error = track_center_x - image_center_x
        self.get_logger().warn(error)
        correction = self.kp * error  # P ���
        
        twist = Twist()
        twist.linear.x = 0.1  # �T�w�e�i�t��
        twist.angular.z = -correction  # �ھڻ~�t�ץ���V
        self.publisher.publish(twist)
        
        left_speed = int(self.base_speed - correction * self.base_speed)
        right_speed = int(self.base_speed + correction * self.base_speed)
        
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID1, self.ADDR_GOAL_VELOCITY, left_speed)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID2, self.ADDR_GOAL_VELOCITY, right_speed)

def main(args=None):
    rclpy.init(args=args)
    node = MotorControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
