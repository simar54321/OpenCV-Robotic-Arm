import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class IntelligentController(Node):
    def __init__(self):
        super().__init__('intelligent_controller')
        self.subscription = self.create_subscription(Image, '/wrist_camera/wrist_camera/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

        # Add the Gripper Publisher alongside the Arm Publisher
        self.arm_publisher = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_publisher = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        self.state = "SCANNING"
        self.scan_direction = 1
        self.current_base_angle = 0.0
        self.current_shoulder_angle = 1.0
        self.last_command_time = time.time()

        self.get_logger().info("Intelligent Controller Active. Ready for final pick!")

    def sweep_arm(self):
        current_time = time.time()
        if current_time - self.last_command_time < 0.5:
            return
        self.last_command_time = current_time

        msg = JointTrajectory()
        msg.joint_names = ['joint_base', 'shoulder', 'wristmotion']

        self.current_base_angle += (0.15 * self.scan_direction)
        if self.current_base_angle > 1.0 or self.current_base_angle < -1.0:
            self.scan_direction *= -1

        self.current_shoulder_angle = 1.0

        point = JointTrajectoryPoint()
        point.positions = [self.current_base_angle, self.current_shoulder_angle, 0.0]
        point.time_from_start = Duration(sec=0, nanosec=500000000)
        msg.points = [point]
        self.arm_publisher.publish(msg)

    def center_on_target(self, cx, cy, img_width, img_height):
        error_x = cx - (img_width / 2)
        error_y = cy - (img_height / 2)

        # THE TRIGGER: If we are within 15 pixels of dead-center, grab it!
        if abs(error_x) < 15 and abs(error_y) < 15:
            self.state = "GRASPING"
            self.get_logger().info("Target Locked! Executing Grasp Sequence...")
            self.execute_pick()
            return

        current_time = time.time()
        if current_time - self.last_command_time < 0.5:
            return
        self.last_command_time = current_time

        self.get_logger().info(f"Tracking! Error X: {error_x}, Y: {error_y}")

        p_gain_base = 0.002
        p_gain_shoulder = -0.002

        self.current_base_angle += (error_x * p_gain_base)
        self.current_shoulder_angle += (error_y * p_gain_shoulder)

        self.current_base_angle = max(min(self.current_base_angle, 1.57), -1.57)
        self.current_shoulder_angle = max(min(self.current_shoulder_angle, 1.0), -0.5)

        msg = JointTrajectory()
        msg.joint_names = ['joint_base', 'shoulder', 'wristmotion']

        point = JointTrajectoryPoint()
        point.positions = [self.current_base_angle, self.current_shoulder_angle, 0.0] 
        point.time_from_start = Duration(sec=0, nanosec=500000000)

        msg.points = [point]
        self.arm_publisher.publish(msg)

    def execute_pick(self):
        # 1. Close the Grippers
        gripper_msg = JointTrajectory()
        gripper_msg.joint_names = ['joint_gripper1', 'joint_gripper2']
        g_point = JointTrajectoryPoint()
        g_point.positions = [0.0, 0.0] # 0.0 closes the parallel fingers
        g_point.time_from_start = Duration(sec=1, nanosec=0)
        gripper_msg.points = [g_point]
        self.gripper_publisher.publish(gripper_msg)

        self.get_logger().info("Gripper closing... waiting 2 seconds for physics.")
        time.sleep(2.0) # Let Gazebo physics secure the block

        # 2. Lift the Arm
        self.get_logger().info("Lifting the block!")
        arm_msg = JointTrajectory()
        arm_msg.joint_names = ['joint_base', 'shoulder', 'wristmotion']
        arm_point = JointTrajectoryPoint()
        arm_point.positions = [self.current_base_angle, 0.5, 0.0] # Lift shoulder back up
        arm_point.time_from_start = Duration(sec=2, nanosec=0)
        arm_msg.points = [arm_point]
        self.arm_publisher.publish(arm_msg)

        # Mark as done so it stops tracking
        self.state = "DONE"

    def image_callback(self, msg):
        # Stop processing vision if we are currently grabbing or holding the box
        if self.state in ["GRASPING", "DONE"]:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        height, width, _ = cv_image.shape

        mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = mask1 + mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.drawContours(cv_image, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(cv_image, (cx, cy), 5, (255, 0, 0), -1)

                self.center_on_target(cx, cy, width, height)
        else:
            self.state = "SCANNING"
            self.sweep_arm()

        cv2.imshow("Intelligent Controller", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = IntelligentController()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
