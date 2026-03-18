import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

points = []
frame = None
clone = None

BEV_W = 640
BEV_H = 480


def mouse_cb(event, x, y, flags, param):
    global points, clone, frame

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

        clone = frame.copy()

        for i, pt in enumerate(points):
            cv2.circle(clone, pt, 6, (0,255,0), -1)
            cv2.putText(clone, str(i+1), (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if len(points) >= 2:
            for i in range(len(points)-1):
                cv2.line(clone, points[i], points[i+1], (0,255,0), 2)

        if len(points) == 4:
            cv2.line(clone, points[3], points[0], (0,255,0), 2)


class HomographyCalibration(Node):

    def __init__(self):
        super().__init__('homography_calibration')

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        cv2.namedWindow("camera")
        cv2.setMouseCallback("camera", mouse_cb)

        self.get_logger().info("Click 4 points: TL → TR → BR → BL")

    def image_callback(self, msg):
        global frame, clone, points

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if clone is None:
            clone = frame.copy()

        cv2.imshow("camera", clone)

        key = cv2.waitKey(1)

        if key == ord('r'):
            points.clear()
            clone = frame.copy()
            print("reset points")

        if len(points) == 4:
            src = np.array(points, dtype=np.float32)

            dst = np.array([
                [0,0],
                [BEV_W-1,0],
                [BEV_W-1,BEV_H-1],
                [0,BEV_H-1]
            ], dtype=np.float32)

            H = cv2.getPerspectiveTransform(src, dst)

            bev = cv2.warpPerspective(frame, H, (BEV_W, BEV_H))
            cv2.imshow("BEV", bev)

            flat = H.flatten()

            print("\nHomography matrix:")
            print(flat.tolist())

            print("\nCopy into costmap node:")
            print(f"-p homography:=\"[{','.join(str(x) for x in flat)}]\"")

            points.clear()


def main(args=None):
    rclpy.init(args=args)

    node = HomographyCalibration()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()