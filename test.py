from camera import Camera
from config import Config
from logger import Logger

def test_camera():
    config = Config()
    logger = Logger(config)
    camera = Camera(config, logger.get_logger())
    
    frame = camera.read_frame()
    if frame is not None:
        print("摄像头测试成功")
    else:
        print("摄像头测试失败")
    
    camera.release()

if __name__ == "__main__":
    test_camera()