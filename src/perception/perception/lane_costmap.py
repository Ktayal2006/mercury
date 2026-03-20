import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

import tf2_ros
from tf2_ros import TransformException


class RoadLineCostmapNode(Node):

    def __init__(self):
        super().__init__('road_line_costmap_node')

        self.declare_parameter('image_topic','/camera/image_raw')
        self.declare_parameter('camera_info_topic','/camera/camera_info')
        self.declare_parameter('costmap_topic','/perception/road_costmap')
        self.declare_parameter('costmap_frame','map')  # FIXED

        self.declare_parameter('bev_width',800)
        self.declare_parameter('bev_height',600)

        self.declare_parameter('resolution',0.0051)

        self.declare_parameter('undistort',True)
        self.declare_parameter('show_debug',True)

        self._read_params()

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.homography = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(Image, self.image_topic, self.image_callback, sensor_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, sensor_qos)

        costmap_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.costmap_pub = self.create_publisher(OccupancyGrid, self.costmap_topic, costmap_qos)

        self.get_logger().info("Road line costmap node started")


    def _read_params(self):
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.costmap_topic = self.get_parameter('costmap_topic').value
        self.costmap_frame = self.get_parameter('costmap_frame').value

        self.bev_w = self.get_parameter('bev_width').value
        self.bev_h = self.get_parameter('bev_height').value

        self.resolution = self.get_parameter('resolution').value

        self.undistort = self.get_parameter('undistort').value
        self.show_debug = self.get_parameter('show_debug').value


    def camera_info_callback(self, msg):
        if self.camera_matrix is not None:
            return

        self.camera_matrix = np.array(msg.k).reshape(3,3)
        self.dist_coeffs = np.array(msg.d)

        self.get_logger().info("Camera calibration received")


    def compute_homography(self, frame):
        h, w = frame.shape[:2]

        src = np.float32([
            [w*0.1, h*0.85],
            [w*0.9, h*0.85],
            [w*0.325, h*0.65],
            [w*0.675, h*0.65]
        ])

        dst = np.float32([
            [0, self.bev_h],
            [self.bev_w, self.bev_h],
            [0, 0],
            [self.bev_w, 0]
        ])

        self.homography = cv2.getPerspectiveTransform(src, dst)

        if self.show_debug:
            dbg = frame.copy()
            for p in src:
                cv2.circle(dbg, tuple(p.astype(int)), 6, (0,0,255), -1)
            cv2.imshow("src_points", dbg)


    def image_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(str(e))
            return

        if self.undistort and self.camera_matrix is not None:
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

        if self.homography is None:
            self.compute_homography(frame)

        cv2.imshow("1_raw", frame)

        bev = cv2.warpPerspective(frame, self.homography, (self.bev_w, self.bev_h))
        cv2.imshow("2_bev", bev)

        mask = self.detect_white_lines(bev)
        cv2.imshow("3_mask", mask)

        grid = self.mask_to_grid(mask, msg.header.stamp)

        if grid is not None:
            self.costmap_pub.publish(grid)

        if self.show_debug:
            cv2.waitKey(1)


    def detect_white_lines(self, bev):

        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 80, 255])

        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_and(white_mask, bright)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        mask = cv2.dilate(mask, kernel2, iterations=1)

        return mask


    def mask_to_grid(self, mask, stamp):

        mask[-20:, :] = 0

        # rotate to align forward direction
        rot = np.rot90(mask,k=-1)
        rot = np.flipud(rot)

        # --- TF lookup ---
        try:
            t = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF error: {ex}")
            return None

        tx = t.transform.translation.x
        ty = t.transform.translation.y

        grid = OccupancyGrid()

        grid.header = Header()
        grid.header.stamp = stamp
        grid.header.frame_id = "map"  # FIXED

        grid.info.resolution = self.resolution
        grid.info.width = rot.shape[1]
        grid.info.height = rot.shape[0]

        origin = Pose()

        origin.position.x = tx
        origin.position.y = ty - (rot.shape[1] * self.resolution) / 2

        origin.orientation.x = 0.0
        origin.orientation.y = 0.0
        origin.orientation.z = 0.0
        origin.orientation.w = 1.0

        grid.info.origin = origin

        cost = np.where(rot > 0, 100, 0).astype(np.int8)
        grid.data = cost.flatten().tolist()

        return grid


def main():
    rclpy.init()
    node = RoadLineCostmapNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()