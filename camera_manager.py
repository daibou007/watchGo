import cv2
import numpy as np
import os


class CameraManager:
    def __init__(self):
        self.cap = None
        self.mapx = None
        self.mapy = None

    def init_camera(self):
        """初始化摄像头"""
        devices = cv2.videoio_registry.getBackends()
        for device in devices:
            if device == cv2.CAP_AVFOUNDATION:
                self.cap = cv2.VideoCapture(0 + cv2.CAP_AVFOUNDATION)
                self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
                break
        else:
            self.cap = cv2.VideoCapture(0)

    def load_calibration(self, filename):
        """加载相机校准数据"""
        if os.path.exists(filename):
            loaded = np.load(filename)
            self.mapx = loaded[0]
            self.mapy = loaded[1]

    def read_frame(self):
        """读取并校准图像帧"""
        if self.cap is None:
            return None

        success, img = self.cap.read()
        if not success:
            return None

        if self.mapx is not None and self.mapy is not None:
            return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
        return img

    def release(self):
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
