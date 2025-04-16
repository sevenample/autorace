import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, '/image/image_raw', 10)
        self.timer = self.create_timer(0.001, self.publish_image)
        self.cap = cv2.VideoCapture('/dev/video0') 
        # 嘗試關閉自動曝光（不同裝置可能有不同數值）
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 有些相機使用 0.75 表示開啟，0.25 表示手動
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 210)  # 數值可依需求調整（單位與範圍視相機而定）

        self.bridge = CvBridge()

        if not self.cap.isOpened():
            self.get_logger().error('Unable to open the camera.')
            raise RuntimeError('Unable to open the camera.')

    def publish_image(self):
        ret, frame = self.cap.read()
        # frame_flipped = cv2.flip(frame, 1)
        if ret:
            try:
                ros_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.publisher_.publish(ros_image_msg)
                self.get_logger().info('Image published to camera_image topic')
            except Exception as e:
                self.get_logger().error(f'Error converting and publishing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    try:
        camera_publisher = CameraPublisher()
        rclpy.spin(camera_publisher)
    except Exception as e:
        print(f'Error during execution: {str(e)}')
    finally:
        if camera_publisher:
            camera_publisher.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
