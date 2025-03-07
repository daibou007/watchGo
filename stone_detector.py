import cv2
import numpy as np

class StoneDetector:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.grid_size = 30  # 与棋盘检测中的格子大小对应
        
    def detect_stones(self, board_image):
        """检测棋子位置和颜色"""
        try:
            board_size = self.config.board.size
            board_state = np.zeros((board_size, board_size), dtype=np.int8)
            
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(board_image, cv2.COLOR_BGR2HSV)
            
            # 检测黑子
            black_lower = np.array([0, 0, 0])
            black_upper = np.array([180, 255, 50])
            black_mask = cv2.inRange(hsv, black_lower, black_upper)
            
            # 检测白子
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            
            # 遍历棋盘交叉点
            for i in range(board_size):
                for j in range(board_size):
                    x = int((i + 0.5) * self.grid_size)
                    y = int((j + 0.5) * self.grid_size)
                    
                    # 检查该点周围区域
                    radius = int(self.grid_size * 0.4)
                    roi_black = black_mask[y-radius:y+radius, x-radius:x+radius]
                    roi_white = white_mask[y-radius:y+radius, x-radius:x+radius]
                    
                    black_ratio = np.sum(roi_black) / 255 / (4 * radius * radius)
                    white_ratio = np.sum(roi_white) / 255 / (4 * radius * radius)
                    
                    if black_ratio > 0.5:
                        board_state[i, j] = 1  # 黑子
                    elif white_ratio > 0.5:
                        board_state[i, j] = 2  # 白子
                        
            return board_state
            
        except Exception as e:
            self.logger.error(f"棋子检测失败: {str(e)}")
            return None
            
    def validate_move(self, prev_state, curr_state):
        """验证落子是否合法"""
        try:
            if prev_state is None:
                return True
                
            diff = curr_state - prev_state
            non_zero = np.nonzero(diff)
            
            # 检查是否只有一个变化
            if len(non_zero[0]) != 1:
                return False
                
            # 检查变化值是否合理
            x, y = non_zero[0][0], non_zero[1][0]
            if abs(diff[x, y]) > 2:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"落子验证失败: {str(e)}")
            return False
