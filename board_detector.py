import cv2
import numpy as np


class BoardDetector:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.board_corners = None
        self.transform_matrix = None

    def detect_board(self, frame):
        """检测棋盘"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # 边缘检测
            edges = cv2.Canny(blur, 50, 150)

            # 寻找轮廓
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return None

            # 找到最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)

            # 多边形逼近
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)

            if len(approx) == 4:
                self.board_corners = self.order_points(approx.reshape(4, 2))
                self.transform_matrix = self.get_transform_matrix()
                return self.board_corners

            return None

        except Exception as e:
            self.logger.error(f"棋盘检测失败: {str(e)}")
            return None

    def order_points(self, pts):
        """对四个角点进行排序"""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下

        return rect

    def get_transform_matrix(self):
        """获取透视变换矩阵"""
        if self.board_corners is None:
            return None

        width = height = self.config.board["size"] * 30  # 每个格子30像素
        dst_points = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )

        return cv2.getPerspectiveTransform(self.board_corners, dst_points)

    def transform_board(self, frame):
        """对棋盘图像进行透视变换"""
        if self.transform_matrix is None:
            return None

        try:
            width = height = self.config.board["size"] * 30
            return cv2.warpPerspective(frame, self.transform_matrix, (width, height))
        except Exception as e:
            self.logger.error(f"棋盘透视变换失败: {str(e)}")
            return None
