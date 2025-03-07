import cv2
import numpy as np
import os
from logger import Logger


class BoardManager:
    def __init__(self, config):
        self.config = config
        self.logger = Logger(self.config)
        self.board = np.zeros(
            (self.config.board.size, self.config.board.size), dtype="uint8"
        )
        self.corners = None
        self.move_history = []  # 记录落子历史

        # 创建初始棋盘图像
        self.board_image = self.create_blank_board()

        # 加载级联分类器
        cascade_path = os.path.join(self.config.base_path, "prefrences")
        empty_cascade_path = os.path.join(cascade_path, self.config.cascade_files.empty)
        self.empty_cascade = cv2.CascadeClassifier(empty_cascade_path)
        self.black_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, self.config.cascade_files.black)
        )
        self.white_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, self.config.cascade_files.white)
        )

    def create_blank_board(self):
        """创建空白棋盘"""
        self.logger.debug("开始创建空白棋盘")

        # BGR 格式的颜色值
        yellow = [32, 165, 218]  # 浅黄色背景
        black = [0, 0, 0]

        board_side = self.config.board.block_size * self.config.board.size
        half_block = int(round(self.config.board.block_size / 2.0))

        self.logger.debug(f"棋盘边长: {board_side}, 半格大小: {half_block}")

        # 创建底板
        board = np.zeros((board_side, board_side, 3), dtype="uint8")
        cv2.rectangle(board, (0, 0), (board_side, board_side), yellow, -1)
        self.logger.debug("创建底板完成")

        # 绘制网格
        self.logger.debug("开始绘制网格")
        for i in range(self.config.board.size):
            spot = i * self.config.board.block_size + half_block
            cv2.line(
                board,
                (spot, half_block),
                (spot, board_side - half_block),
                black,
                int(self.config.board.block_size / 10),
            )
            cv2.line(
                board,
                (half_block, spot),
                (board_side - half_block, spot),
                black,
                int(self.config.board.block_size / 10),
            )
        self.logger.debug("网格绘制完成")

        # 绘制星位
        if self.config.board.size == 19:
            self.logger.debug("开始绘制星位")
            spots = [
                [3, 3],
                [9, 3],
                [15, 3],
                [3, 9],
                [9, 9],
                [15, 9],
                [3, 15],
                [9, 15],
                [15, 15],
            ]

            for s in spots:
                cv2.circle(
                    board,
                    (
                        s[0] * self.config.board.block_size + half_block,
                        s[1] * self.config.board.block_size + half_block,
                    ),
                    int(self.config.board.block_size * 0.15),
                    black,
                    -1,
                )
            self.logger.debug("星位绘制完成")

        self.logger.info(f"空白棋盘创建完成，尺寸: {board_side}x{board_side}")
        return board

    def analyze_image(self, image):
        """分析图像中的棋盘状态"""
        if image is None:
            self.logger.error("输入图像为空")
            return None, None

        try:
            self.logger.info("开始分析棋盘图像")

            # 使用级联分类器检测特征
            self.logger.debug("开始检测空白交叉点")
            empties = self._detect_features(image, self.empty_cascade, 1.08)
            self.logger.debug(f"检测到 {len(empties)} 个空白交叉点")

            self.logger.debug("开始检测黑子")
            blacks = self._detect_features(image, self.black_cascade, 1.01)
            self.logger.debug(f"检测到 {len(blacks)} 个黑子")

            self.logger.debug("开始检测白子")
            whites = self._detect_features(image, self.white_cascade, 1.01)
            self.logger.debug(f"检测到 {len(whites)} 个白子")

            # 更新棋盘状态
            self.board = np.zeros(
                (self.config.board.size, self.config.board.size), dtype="uint8"
            )
            self._update_board_state(empties, blacks, whites)
            self.logger.info("棋盘状态分析完成")

            return self.board, self.corners

        except Exception as e:
            self.logger.error(f"棋盘分析失败: {str(e)}")
            return None, None

    def _detect_features(self, image, cascade, scale_factor):
        """使用级联分类器检测特征"""
        if cascade is None:
            return []

        try:
            rectangles = cascade.detectMultiScale(image, scale_factor, 3)
            return [[x + w / 2.0, y + w / 2.0] for x, y, w, h in rectangles]
        except Exception as e:
            print(f"检测特征时出错: {str(e)}")
            return []

    def _update_board_state(self, empties, blacks, whites):
        """更新棋盘状态"""
        try:
            # 计算棋盘网格
            height, width = self.board_image.shape[:2]
            grid_size = min(width, height) // (self.config.board.size + 1)
            offset_x = (width - grid_size * (self.config.board.size - 1)) // 2
            offset_y = (height - grid_size * (self.config.board.size - 1)) // 2

            # 清空当前棋盘状态
            self.board = np.zeros(
                (self.config.board.size, self.config.board.size), dtype="uint8"
            )

            # 更新黑子位置
            for point in blacks:
                x, y = point
                board_x = int(round((x - offset_x) / grid_size))
                board_y = int(round((y - offset_y) / grid_size))
                if (
                    0 <= board_x < self.config.board.size
                    and 1 <= board_y < self.config.board.size
                ):
                    self.board[board_x][board_y-1] = 1

            # 更新白子位置
            for point in whites:
                x, y = point
                board_x = int(round((x - offset_x) / grid_size))
                board_y = int(round((y - offset_y) / grid_size))
                if (
                    0 <= board_x < self.config.board.size
                    and 1 <= board_y < self.config.board.size
                ):
                    self.board[board_x][board_y-1] = 2

            # 记录角点信息
            if len(empties) >= 4:
                # 找到最外围的四个点作为角点
                xs = [p[0] for p in empties]
                ys = [p[1] for p in empties]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                self.corners = [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                ]

            self.logger.info(
                f"棋盘状态更新完成，检测到 {len(blacks)} 个黑子，{len(whites)} 个白子"
            )

        except Exception as e:
            self.logger.error(f"更新棋盘状态失败: {str(e)}")

    def draw_board(self, frame, board_state=None, start_index=1):
        """在图像上绘制棋盘和棋子"""
        try:
            if board_state is None:
                board_state = self.board

            # 添加更详细的日志输出
            self.logger.debug(f"绘制棋盘开始")
            self.logger.debug(
                f"输入frame类型: {type(frame)}, shape: {frame.shape if frame is not None else 'None'}"
            )
            self.logger.debug(
                f"board_state类型: {type(board_state)}, shape: {board_state.shape}"
            )

            # 使用预先创建的棋盘图像
            if frame is None:
                frame = self.board_image.copy()
            else:
                # 确保frame是BGR格式
                if len(frame.shape) == 2:  # 如果是灰度图
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # 调整棋盘图像大小以匹配输入帧
                resized_board = cv2.resize(
                    self.board_image, (frame.shape[1], frame.shape[0])
                )
                # 调整混合比例，增加原始图像的权重
                frame = cv2.addWeighted(frame, 0.7, resized_board, 0.3, 0)

            # 计算棋子位置
            height, width = frame.shape[:2]
            grid_size = self.config.board.block_size  # 使用配置中的格子大小
            offset_x = (width - grid_size * (self.config.board.size - 1)) // 2
            offset_y = (height - grid_size * (self.config.board.size - 1)) // 2

            # 绘制棋子
            for i in range(self.config.board.size):
                for j in range(self.config.board.size):
                    if board_state[i][j] > 0:
                        x = offset_x + i * grid_size
                        y = offset_y + j * grid_size
                        color = (0, 0, 0) if board_state[i][j] == 1 else (255, 255, 255)
                        stone_size = int(grid_size * 0.4)  # 调整棋子大小

                        # 为黑子添加高光效果
                        if board_state[i][j] == 1:
                            # 绘制主体
                            cv2.circle(
                                frame, (x, y), stone_size, color, -1, cv2.LINE_AA
                            )
                            # 添加高光
                            highlight_pos = (x - stone_size // 3, y - stone_size // 3)
                            cv2.circle(
                                frame,
                                highlight_pos,
                                stone_size // 4,
                                (40, 40, 40),
                                -1,
                                cv2.LINE_AA,
                            )
                        else:
                            # 绘制白子
                            cv2.circle(
                                frame, (x, y), stone_size, color, -1, cv2.LINE_AA
                            )
                            # 添加黑边
                            cv2.circle(
                                frame, (x, y), stone_size, (0, 0, 0), 1, cv2.LINE_AA
                            )

            # 添加落子顺序编号
            latest_moves = {}
            for idx, (move_x, move_y, color) in enumerate(self.move_history, start_index):
                # 记录每个位置的最新编号
                latest_moves[(move_x, move_y)] = idx

            for (move_x, move_y), idx in latest_moves.items():
                # 只有当该位置确实有棋子时才绘制编号
                if board_state[move_x][move_y] > 0:
                    x = offset_x + move_x * grid_size
                    y = offset_y + move_y * grid_size
                    
                    # 设置文本参数
                    text = str(idx)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # 根据编号长度动态调整字体大小
                    if len(text) == 1:
                        font_scale = grid_size * 0.015
                    elif len(text) == 2:
                        font_scale = grid_size * 0.012
                    elif len(text) == 3:
                        font_scale = grid_size * 0.010
                    elif len(text) == 4:
                        font_scale = grid_size * 0.008
                    else:
                        font_scale = grid_size * 0.006
                    thickness = 1
                    
                    # 获取文本大小以居中显示
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = x - text_width // 2
                    text_y = y + text_height // 2
                    
                    # 根据棋子颜色选择文字颜色
                    text_color = (255, 255, 255) if board_state[move_x][move_y] == 1 else (0, 0, 0)
                    
                    # 绘制文本
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
                    
            # 添加角点信息
            if self.corners:
                for corner in self.corners:
                    x, y = corner
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            # 打印棋盘状态
            self.print_board_state(board_state)


            self.logger.debug(f"绘制棋盘完成")

            return frame

        except Exception as e:
            self.logger.error(f"绘制棋盘失败: {str(e)}")
            return frame

    def print_board_state_old(self, board_state=None):
        if board_state is None:
            board_state = self.board
            
        self.logger.debug("开始打印棋盘状态")
        
        # 定义棋盘符号
        symbols = {
            0: "十",    # 空位交叉点
            1: "⚫",    # 黑子
            2: "⚪"     # 白子
        }
        
        # 打印列标题 (A-T，排除I)
        columns = "ABCDEFGHJKLMNOPQRST"[:self.config.board.size]
        col_header = "     " + "  ".join(columns)
        print(col_header)
        
        # 打印顶部边框
        print("   ┌" + "──" * self.config.board.size + "┐")
        
        # 打印棋盘内容
        for i in range(self.config.board.size):
            # 打印行号，保持对齐
            row_num = str(self.config.board.size - i).rjust(2)
            row = f"{row_num} │"
            
            # 添加棋子
            for j in range(self.config.board.size):
                row += symbols[board_state[j][i]]
            
            row += f"│ {row_num}"
            print(row)
        
        # 打印底部边框
        print("   └" + "──" * self.config.board.size + "┘")
        print(col_header)
        
        # 打印统计信息
        black_count = np.sum(board_state == 1)
        white_count = np.sum(board_state == 2)
        self.logger.debug(f"黑子: {black_count}, 白子: {white_count}")

    def print_board_state(self, board_state=None):
        if board_state is None:
            board_state = self.board
            
        self.logger.debug("开始打印棋盘状态")
        
        # 定义棋盘符号
        symbols = {
            0: "十",    # 空位交叉点
            1: "⚫",    # 黑子
            2: "⚪"     # 白子
        }
        
        # 打印列标题 (A-T，排除I)
        columns = "ABCDEFGHJKLMNOPQRST"[:self.config.board.size]
        col_header = "     " + " ".join(columns)  # 增加一个空格，使列标题向右偏移1/4格
        print(col_header)
        
        # 打印顶部边框
        print("   ┌" + "─" * (self.config.board.size * 2) + "┐")  # 增加边框长度
        
        # 打印棋盘内容
        for i in range(self.config.board.size):
            # 打印行号，保持对齐
            row_num = str(self.config.board.size - i).rjust(2)
            row = f"{row_num} │"
            
            # 添加棋子
            for j in range(self.config.board.size):
                row += symbols[board_state[j][i]]
                if j < self.config.board.size - 1:
                    row += ""  # 移除棋子之间的空格
            
            row += f"│ {row_num}"
            print(row)
        
        # 打印底部边框
        print("   └" + "─" * (self.config.board.size * 2) + "┘")  # 增加边框长度
        print(col_header)