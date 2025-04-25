import rclpy
from rclpy.node import Node
from std_msgs.msg import Int64
from geometry_msgs.msg import Twist
from dynamixel_sdk import *

class MotorControlRight(Node):
    def __init__(self):
        super().__init__('motor_control_right')
        self.subscription = self.create_subscription(Int64, 'topic', self.offset_callback, 10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.stop_subscription = self.create_subscription(Int64, '/stop_signal', self.stop_callback, 1)

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
        
        self.kp = 0.0035  # P 控制增益
        self.base_speed = 150  # 設定基礎速度
        self.stop_signal=1
        self.previous_mode = None

    def switch_mode(self,current_mode):
        if current_mode != self.previous_mode:
            print(f"mode:    {current_mode}")
            self.previous_mode = current_mode
    def stop_callback(self, msg):
        self.stop_signal = msg.data
        self.get_logger().info(f'Received stop signal: {self.stop_signal}')

    
    def offset_callback(self, msg):
        error = msg.data  # 來自 `two_line.py` 的偏移量
        correction = self.kp * error  # P 控制計算修正量
        twist = Twist()
        twist.linear.x = 0.1  # 固定前進速度
        twist.angular.z = -correction  # 根據誤差修正方向
        self.publisher.publish(twist)

        left_speed = int((self.base_speed - correction * self.base_speed)*self.stop_signal)
        right_speed = int((self.base_speed + correction * self.base_speed)*self.stop_signal)
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
