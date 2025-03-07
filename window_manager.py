import cv2
import os

class WindowManager:
    def __init__(self):
        self.instance_id = str(os.getpid())
        self.windows = {}
        
    def get_window_name(self, base_name):
        """获取唯一的窗口名称"""
        if base_name not in self.windows:
            self.windows[base_name] = f"{base_name}_{self.instance_id}"
        return self.windows[base_name]
        
    def show_image(self, name, image):
        """显示图像"""
        window_name = self.get_window_name(name)
        cv2.imshow(window_name, image)
        
    def close_window(self, name):
        """关闭窗口"""
        if name in self.windows:
            window_name = self.windows[name]
            cv2.destroyWindow(window_name)
            for _ in range(4):
                cv2.waitKey(1)
            del self.windows[name]
            
    def close_all(self):
        """关闭所有窗口"""
        for name in list(self.windows.keys()):
            self.close_window(name)