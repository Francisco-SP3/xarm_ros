#!/usr/bin/env python3
# Xarm6 + depth camera demo for Roborregos Candidates 2023
# by Emiliano Flores
# Reads the current published frame from the camera, allows for the user to click on a point and the xarm will go to that point
import rospy
import sys
import cv2
import moveit_commander
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
# cmamera info
from sensor_msgs.msg import CameraInfo, PointCloud2
# point stamped
from geometry_msgs.msg import PointStamped, PoseStamped
# marker
from visualization_msgs.msg import Marker
import threading
import tf

class demoReto:
    def __init__(self) -> None:
        rospy.init_node('demoReto2023Client')
        rospy.loginfo("Demo Reto 2023 Client is now running")
        self.ARM_GROUP = "xarm6"
        self.GRIPPER = "xarm_gripper"
        # Safe position in degrees: [0, -45, 0, 0, 45, 0]
        # in radians
        self.SAFE_POSITION = [0, -0.785, 0, 0, 0.785, 0]
        # Observation position in degrees: [0, -90, -60, 0, 115, 0]
        self.OBSERVATION_POSITION = [0, -1.57, -1.047, 0, 2.007, 0]
        # offset in z axis, how much space will the arm leave between the selected point and its goal position
        self.Z_OFFSET = -0.025 # 2.5cm below poit, to pick cube

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._img_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self._img_callback)
        self._depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self._depth_callback)
        self._camera_info_sub = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self._camera_info_callback)
        self._point_cloud_sub = rospy.Subscriber('/camera/depth/points', PointCloud2, self._point_cloud_callback)
        self.listener = tf.TransformListener()
        self._marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self._frame = None
        
        # arm
        self._moveit = moveit_commander.MoveGroupCommander(self.ARM_GROUP, wait_for_servers=0.0)
        # gripper
        self._gripper = moveit_commander.MoveGroupCommander(self.GRIPPER)
        # set arm speed. Max is 1.0
        self._moveit.set_max_velocity_scaling_factor(1.0)
        self._gripper.set_max_velocity_scaling_factor(1.0)
        # allow target unaccuracy
        self._moveit.set_goal_position_tolerance(0.01)

        self._initialPositionThread = threading.Thread(target=self._initialPosition)
        self._initialPositionThread.start()

        self.moving_gripper = False

        self.main()


    def _img_callback(self, data):
        with self._lock:
            self._frame = self._bridge.imgmsg_to_cv2(data)
            cv2.imshow('frame', self._frame)

    
    def _depth_callback(self, data):
        with self._lock:
            self._depth_frame = self._bridge.imgmsg_to_cv2(data)
    
    def _point_cloud_callback(self, data):
        with self._lock:
            self._point_cloud = data
    
    def _camera_info_callback(self, data):
        with self._lock:
            self._camera_info = data
    
    def _click_callback(self, event, x, y, flags, param):

        # Left click to go to a cube
        if event == cv2.EVENT_LBUTTONDOWN:
            # GO TO A CUBE, which means go to a point slightly below clicked point
            self.Z_OFFSET = -0.025 # 2.5cm below poit, to pick cube
            # get the depth value
            depth = self._depth_frame[y, x]
            rospy.loginfo(f"Clicked at ({x}, {y}) with depth {depth}")
            # convert to 3D point for the arm to go
            point_world = self.pixel_to_3d(x, y, depth)
            # Move the arm to the 3D point and pointing down
            if not self.moving_arm:
                self._moveArmThread = threading.Thread(target=self._moveArmtoPoint, args=(point_world,))
                self._moveArmThread.start()
            else:
                rospy.logwarn("Arm is already moving, please wait")

        # right click to go to a point
        if event == cv2.EVENT_RBUTTONDOWN:
            # GO TO A POINT, which means go a point slightly above clicked point
            self.Z_OFFSET = 0.03 # 3cm above point, to avoid hitting cube or table
            # get the depth value
            depth = self._depth_frame[y, x]
            rospy.loginfo(f"Clicked at ({x}, {y}) with depth {depth}")
            # convert to 3D point for the arm to go
            point_world = self.pixel_to_3d(x, y, depth)
            # Move the arm to the 3D point and pointing down
            if not self.moving_arm:
                self._moveArmThread = threading.Thread(target=self._moveArmtoPoint, args=(point_world,))
                self._moveArmThread.start()
            else:
                rospy.logwarn("Arm is already moving, please wait")


    
    def pixel_to_3d(self, x, y, depth):
        # Use the camera info to get the camera parameters
        fx = self._camera_info.K[0]
        fy = self._camera_info.K[4]
        cx = self._camera_info.K[2]
        cy = self._camera_info.K[5]

        # Convert the 2D point to a 3D point in the camera frame
        z = depth
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy

        # Create a PointStamped message for the 3D point
        point = PointStamped()
        point.header.frame_id = self._camera_info.header.frame_id
        point.header.stamp = rospy.Time(0)
        point.point.x = x
        point.point.y = y
        point.point.z = z

        # rospy.loginfo(f"3D point in the camera frame: {point}")

        # Transform the point to the world frame
        self.listener.waitForTransform("world", point.header.frame_id, rospy.Time(0), rospy.Duration(4.0))
        point_world = self.listener.transformPoint("world", point)

        # Publish the 3D point as a marker for visualization in RViz
        self._publish_marker(point_world)

        return point_world
    
    def _initialPosition(self):
        self.moving_arm = True
        rospy.loginfo("Moving to initial position")
        self._moveit.go(self.SAFE_POSITION, wait=True)
        self._moveit.go(self.OBSERVATION_POSITION, wait=True)
        rospy.loginfo("Initial position reached")
        rospy.loginfo("Ready to take commands from camera image")
        self.moving_arm = False

    def _moveArmtoPoint(self, point_world):
        try:
            self.moving_arm = True
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = "world"
            goal_pose.header.stamp = rospy.Time.now()
            goal_pose.pose.position.x = point_world.point.x
            goal_pose.pose.position.y = point_world.point.y
            goal_pose.pose.position.z = point_world.point.z + self.Z_OFFSET
            # roll, pitch, yaw
            goal_roll = 0 # rotate in x
            goal_pitch = 3.14 # rotate in y
            goal_yaw = 3.14 # rotate in z
            # convert to quaternion
            quaternion = tf.transformations.quaternion_from_euler(goal_roll, goal_pitch, goal_yaw)
            goal_pose.pose.orientation.x = quaternion[0]
            goal_pose.pose.orientation.y = quaternion[1]
            goal_pose.pose.orientation.z = quaternion[2]
            goal_pose.pose.orientation.w = quaternion[3]

            rospy.loginfo(f"Goal pose: {goal_pose}")

            # move arm to goal pose
            self._moveit.set_pose_target(goal_pose)
            # increase planning time
            self._moveit.set_planning_time(25)
            self._moveit.go(wait=True)
            rospy.loginfo("Arm has reached the goal position!")
            self.moving_arm = False
        except:
            rospy.logwarn("Movement did not finish correctly")
            self.moving_arm = False


    
    # Function to publish the 3D point as a marker
    def _publish_marker(self, point_world):
        marker = Marker()
        marker.header.frame_id = "world"  # Replace with your actual world frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "my_namespace"
        marker.id = 0
        # marker as arrow pointing down
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = point_world.point.x
        marker.pose.position.y = point_world.point.y
        marker.pose.position.z = point_world.point.z + 0.1
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.707
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 0.707
        marker.scale.x = 0.1
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self._marker_pub.publish(marker)

    def _moveArmJoints(self, joints):
        try:
            self.moving_arm = True
            self._moveit.go(joints, wait=True)
            rospy.loginfo("Arm has reached the goal position!")
            self.moving_arm = False
        except:
            rospy.logwarn("Movement did not finish correctly")
            self.moving_arm = False
    
    def closeGripper(self):
        try:
            self.moving_gripper = True
            self._gripper.set_named_target("close")
            self._gripper.go(wait=True)
            rospy.loginfo("Gripper closed!")
            self.moving_gripper = False
        except:
            rospy.logwarn("Gripper did not move correctly")
            self.moving_gripper = False
    
    def openGripper(self):
        try:
            self.moving_gripper = True
            self._gripper.set_named_target("open")
            self._gripper.go(wait=True)
            rospy.loginfo("Gripper opened!")
            self.moving_gripper = False
        except:
            rospy.logwarn("Gripper did not move correctly")
            self.moving_gripper = False

    def main(self):
        rospy.loginfo("Camera ready")
        while not rospy.is_shutdown():
            # show image from camera
            if self._frame is not None:
                # on click listener for the image
                cv2.setMouseCallback('frame', self._click_callback)
                cv2.waitKey(1)
                # if key pressed is r, go to observation position
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    rospy.loginfo("Going to observation position")
                    if not self.moving_arm:
                        self.moveArmThread = threading.Thread(target=self._moveArmJoints, args=(self.OBSERVATION_POSITION,))
                        self.moveArmThread.start()
                    else:
                        rospy.logwarn("Arm is already moving, please wait")
                # if key pressed is c, close gripper
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    rospy.loginfo("Closing gripper")
                    # predetermined position name is close
                    if not self.moving_gripper:
                        self.moveGripperThread = threading.Thread(target=self.closeGripper)
                        self.moveGripperThread.start()
                    else:
                        rospy.logwarn("Gripper is already moving, please wait")
                # if key pressed is o, open gripper
                if cv2.waitKey(1) & 0xFF == ord('o'):
                    rospy.loginfo("Opening gripper")
                    if not self.moving_gripper:
                        self.moveGripperThread = threading.Thread(target=self.openGripper)
                        self.moveGripperThread.start()
                    else:
                        rospy.logwarn("Gripper is already moving, please wait")
                    

        pass

if __name__ == "__main__":
    demoReto()
