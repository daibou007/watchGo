import cv2
import numpy as np
from datetime import datetime

class Camera:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.cap = None
        self.initialize_camera()
        
    def initialize_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.config.camera['device_id'])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera['height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera['fps'])
            
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")
                
            self.logger.info("摄像头初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"摄像头初始化失败: {str(e)}")
            return False
            
    def read_frame(self):
        """读取一帧图像"""
        try:
            if self.cap is None:
                return None
                
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("读取摄像头帧失败")
                return None
                
            return frame
        except Exception as e:
            self.logger.error(f"读取摄像头帧异常: {str(e)}")
            return None
            
    def release(self):
        """释放摄像头资源"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.logger.info("摄像头资源已释放")
        except Exception as e:
            self.logger.error(f"释放摄像头资源失败: {str(e)}")