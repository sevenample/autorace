import rclpy
from rclpy.node import Node
from std_msgs.msg import Int64
from geometry_msgs.msg import Twist
from dynamixel_sdk import *

class MotorControlRight(Node):
    def __init__(self):
        super().__init__('motor_control_right')
        self.subscription = self.create_subscription(Int64, 'topic', self.offset_callback, 10)
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
        self.previous_mode = None

    def switch_mode(self,current_mode):
        if current_mode != self.previous_mode:
            print(f"mode:    {current_mode}")
            self.previous_mode = current_mode
    def offset_callback(self, msg):
        left_speed = msg.left
        right_speed =msg.right
        self.switch_mode("start")
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID1, self.ADDR_GOAL_VELOCITY, left_speed)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID2, self.ADDR_GOAL_VELOCITY, right_speed)

def main(args=None):
    rclpy.init(args=args)
    node = MotorControlRight()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
