import cv2
import numpy as np

class BoardManager:
    def __init__(self, config):
        self.config = config
        self.board = np.zeros((config.board_size, config.board_size), dtype="uint8")
        self.corners = None
        
        # 加载级联分类器
        self.empty_cascade = cv2.CascadeClassifier(config.empty_cascade_file)
        self.black_cascade = cv2.CascadeClassifier(config.black_cascade_file)
        self.white_cascade = cv2.CascadeClassifier(config.white_cascade_file)
        
    def create_blank_board(self):
        """创建空白棋盘"""
        yellow = [75, 215, 255]
        black = [0, 0, 0]
        board_side = self.config.board_block_size * self.config.board_size
        half_block = int(round(self.config.board_block_size / 2.0))
        
        # 创建底板
        board = np.zeros((board_side, board_side, 3), dtype="uint8")
        cv2.rectangle(board, (0, 0), (board_side, board_side), yellow, -1)
        
        # 绘制网格
        for i in range(self.config.board_size):
            spot = i * self.config.board_block_size + half_block
            cv2.line(board, (spot, half_block), 
                    (spot, board_side - half_block),
                    black, int(self.config.board_block_size / 10))
            cv2.line(board, (half_block, spot),
                    (board_side - half_block, spot),
                    black, int(self.config.board_block_size / 10))
        
        # 绘制星位
        if self.config.board_size == 19:
            spots = [[3, 3], [9, 3], [15, 3],
                    [3, 9], [9, 9], [15, 9],
                    [3, 15], [9, 15], [15, 15]]
            
            for s in spots:
                cv2.circle(board,
                          (s[0] * self.config.board_block_size + half_block,
                           s[1] * self.config.board_block_size + half_block),
                          int(self.config.board_block_size * 0.15),
                          black, -1)
        
        return board
        
    def analyze_image(self, image):
        """分析图像中的棋盘状态"""
        if image is None:
            return None, None
            
        # 使用级联分类器检测特征
        empties = self._detect_features(image, self.empty_cascade, 1.08)
        blacks = self._detect_features(image, self.black_cascade, 1.01)
        whites = self._detect_features(image, self.white_cascade, 1.01)
        
        # 更新棋盘状态
        self.board = np.zeros((self.config.board_size, self.config.board_size), dtype="uint8")
        self._update_board_state(empties, blacks, whites)
        
        return self.board, self.corners
        
    def _detect_features(self, image, cascade, scale_factor):
        """使用级联分类器检测特征"""
        if cascade is None:
            return []
            
        try:
            rectangles = cascade.detectMultiScale(image, scale_factor, 3)
            return [[x + w/2.0, y + w/2.0] for x, y, w, h in rectangles]
        except Exception as e:
            print(f"检测特征时出错: {str(e)}")
            return []
            
    def _update_board_state(self, empties, blacks, whites):
        """更新棋盘状态"""
        # ... 实现棋盘状态更新逻辑 ...