import cv2
import numpy as np
import os
from board_manager import BoardManager


class InteractiveBoard(BoardManager):
    def __init__(self, config, start_index=1):
        super().__init__(config)
        self.last_ko_position = None  # 记录上一个劫争位置
        # self.move_history = []  # 记录落子历史
        self.dead_stones_history = []  # 记录死子历史
        self.start_index = start_index  # 添加起始编号支持

    def start(self, initial_image_path="resource/15.jpg"):
        """启动交互式棋盘"""
        try:
            # 加载并分析初始图片
            test_image = os.path.join(self.config.base_path, initial_image_path)
            img = cv2.imread(test_image)
            if img is None:
                self.logger.error(f"无法读取图像文件: {test_image}")
                return

            # 分析棋盘状态
            board_state, corners = self.analyze_image(img)
            if board_state is None:
                self.logger.error("无法识别棋盘状态")
                return

            # 创建窗口
            board_window = "Interactive Board"
            cv2.namedWindow(board_window)

            # 定义鼠标回调函数
            def mouse_callback(event, x, y, flags, param):
                if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
                    # 计算棋盘坐标
                    height, width = self.board_image.shape[:2]
                    grid_size = min(width, height) // (self.config.board.size + 1)
                    offset_x = (width - grid_size * (self.config.board.size - 1)) // 2
                    offset_y = (height - grid_size * (self.config.board.size - 1)) // 2

                    # 修正坐标计算方式
                    board_x = int((x - offset_x + grid_size / 2) / grid_size)
                    board_y = int((y - offset_y + grid_size / 2) / grid_size)

                    # 检查是否在有效范围内
                    if (
                        0 <= board_x < self.config.board.size
                        and 0 <= board_y < self.config.board.size
                    ):
                        # 检查是否已经有棋子
                        if self.board[board_x][board_y] != 0:
                            self.logger.info(f"位置 ({board_x}, {board_y}) 已有棋子")
                            return

                        # 检查是否是劫争点
                        if self._is_ko_point(board_x, board_y):
                            self.logger.info(f"位置 ({board_x}, {board_y}) 是劫争点")
                            return

                        # 左键放黑子，右键放白子
                        stone_color = 1 if event == cv2.EVENT_LBUTTONDOWN else 2

                        # 清空上一次提子记录
                        self.last_ko_position = None
                        self.dead_stones_history = []

                        # 临时落子
                        self.board[board_x][board_y] = stone_color

                        # 判断是否是禁手点（对黑白子都进行判断）
                        if self._is_forbidden_point(board_x, board_y, stone_color):
                            # 如果是禁手点，恢复棋盘状态
                            self.board[board_x][board_y] = 0
                            # 恢复被提走的子
                            if self.dead_stones_history:
                                dead_stones = self.dead_stones_history.pop()
                                for x, y, color in dead_stones:
                                    self.board[x][y] = color
                            self.logger.info(
                                f"位置 ({board_x}, {board_y}) 是禁手点，不能落子"
                            )
                            return

                        # 移除死子并检查劫争
                        self._remove_dead_stones()

                        # 记录这一手
                        self.move_history.append((board_x, board_y, stone_color))

                        # 重绘棋盘
                        board_img = self.draw_board(
                            None, self.board, self.start_index
                        )  # 传入起始编号
                        cv2.imshow(board_window, board_img)
                        # 打印更新后的棋盘状态
                        self.print_board_state()

            # 设置鼠标回调
            cv2.setMouseCallback(board_window, mouse_callback)

            # 初始化棋盘状态
            self.board = (
                board_state.copy()
                if board_state is not None
                else np.zeros(
                    (self.config.board.size, self.config.board.size), dtype=np.uint8
                )
            )

            # 显示初始棋盘
            board_img = self.draw_board(None, self.board)
            cv2.imshow(board_window, board_img)
            # 打印初始棋盘状态
            self.print_board_state()

            # 显示原始图像
            if img is not None:
                h, w = img.shape[:2]
                video_size = (int(round(w / 2.0)), int(round(h / 2.0)))
                cv2.imshow("Original Image", cv2.resize(img, video_size))

            # 等待用户操作
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键撤销上一手
                    if self._restore_last_move():
                        # 重绘棋盘
                        board_img = self.draw_board(
                            None, self.board, self.start_index
                        )  # 传入起始编号
                        cv2.imshow(board_window, board_img)
                        self.print_board_state()
                    continue
                elif key == ord("s"):  # 's'键保存当前状态
                    self._save_board_state()
                elif key == ord("p"):  # 'p'键打印当前状态
                    self.print_board_state()
                elif key == ord("c"):  # 'c'键清空棋盘
                    self.board = np.zeros(
                        (self.config.board.size, self.config.board.size), dtype=np.uint8
                    )
                    board_img = self.draw_board(None, self.board)
                    cv2.imshow(board_window, board_img)
                    self.print_board_state()
                elif key != 255:  # 任意其他键退出程序
                    break

            # 清理窗口
            cv2.destroyAllWindows()
            return self.board

        except Exception as e:
            self.logger.error(f"交互式棋盘创建失败: {str(e)}")
            return None

    def _save_board_state(self, filename="board_state.txt"):
        """保存棋盘状态到文件"""
        filepath = os.path.join(self.config.base_path, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for y in range(self.board.shape[1]):
                    row = []
                    for x in range(self.board.shape[0]):
                        if self.board[x][y] == 0:
                            row.append(".")
                        elif self.board[x][y] == 1:
                            row.append("B")
                        else:
                            row.append("W")
                    f.write("".join(row) + "\n")
            print(f"棋盘状态已保存到: {filepath}")
        except Exception as e:
            print(f"保存棋盘状态失败: {str(e)}")

    # def _is_forbidden_point(self, x, y, stone_color):
    #     """判断是否是禁手点"""
    #     # 临时落子以检查禁手
    #     self.board[x][y] = stone_color
    #     is_forbidden = False

    #     # 检查长连
    #     if self._has_overline(x, y, stone_color):
    #         is_forbidden = True

    #     # 检查三三禁手
    #     elif self._has_double_three(x, y, stone_color):
    #         is_forbidden = True

    #     # 检查四四禁手
    #     elif self._has_double_four(x, y, stone_color):
    #         is_forbidden = True

    #     # 恢复棋盘状态
    #     self.board[x][y] = 0
    #     return is_forbidden

    def _has_overline(self, x, y, stone_color):
        """检查是否有长连（超过5个子相连）"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # 向正方向检查
            tx, ty = x + dx, y + dy
            while (
                0 <= tx < self.config.board.size
                and 0 <= ty < self.config.board.size
                and self.board[tx][ty] == stone_color
            ):
                count += 1
                tx, ty = tx + dx, ty + dy
            # 向反方向检查
            tx, ty = x - dx, y - dy
            while (
                0 <= tx < self.config.board.size
                and 0 <= ty < self.config.board.size
                and self.board[tx][ty] == stone_color
            ):
                count += 1
                tx, ty = tx - dx, ty - dy
            if count > 5:
                return True
        return False

    def _has_double_three(self, x, y, stone_color):
        """检查是否有三三禁手"""
        # 八个方向：水平、垂直、两个对角线，每个方向都有两个方向
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        three_count = 0

        for dx, dy in directions:
            # 检查两个方向
            if self._check_open_three(
                x, y, dx, dy, stone_color
            ) or self._check_open_three(x, y, -dx, -dy, stone_color):
                three_count += 1

            # 如果找到两个活三，则为三三禁手
            if three_count >= 2:
                return True
        return False

    def _check_open_three(self, x, y, dx, dy, stone_color):
        """检查是否形成活三"""
        # 记录当前位置的原始值
        original_value = self.board[x][y]
        self.board[x][y] = stone_color

        # 在方向上统计连续的棋子
        count = 1
        space_count = 0
        line = []

        # 向一个方向延伸
        tx, ty = x + dx, y + dy
        while 0 <= tx < self.config.board.size and 0 <= ty < self.config.board.size:
            if self.board[tx][ty] == stone_color:
                count += 1
                line.append(1)
            elif self.board[tx][ty] == 0:
                space_count += 1
                line.append(0)
            else:
                break
            tx, ty = tx + dx, ty + dy

        # 向相反方向延伸
        tx, ty = x - dx, y - dy
        while 0 <= tx < self.config.board.size and 0 <= ty < self.config.board.size:
            if self.board[tx][ty] == stone_color:
                count += 1
                line.insert(0, 1)
            elif self.board[tx][ty] == 0:
                space_count += 1
                line.insert(0, 0)
            else:
                break
            tx, ty = tx - dx, ty - dy

        # 恢复原始值
        self.board[x][y] = original_value

        # 判断是否是活三
        # 活三的模式：
        # 1. "010110" 或 "011010" 或 "010101"（其中1表示己方棋子，0表示空位）
        patterns = ["010110", "011010", "010101"]
        line_str = "".join(map(str, line))

        for pattern in patterns:
            if pattern in line_str:
                return True

        return False

    def _has_double_four(self, x, y, stone_color):
        """检查是否有四四禁手"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        four_count = 0

        for dx, dy in directions:
            # 检查两个方向
            if self._check_four(x, y, dx, dy, stone_color) or self._check_four(
                x, y, -dx, -dy, stone_color
            ):
                four_count += 1

            # 如果找到两个四，则为四四禁手
            if four_count >= 2:
                return True
        return False

    def _check_four(self, x, y, dx, dy, stone_color):
        """检查是否形成四"""
        # 记录当前位置的原始值
        original_value = self.board[x][y]
        self.board[x][y] = stone_color

        # 在方向上统计连续的棋子
        count = 1
        space_count = 0
        line = []

        # 向一个方向延伸
        tx, ty = x + dx, y + dy
        while 0 <= tx < self.config.board.size and 0 <= ty < self.config.board.size:
            if self.board[tx][ty] == stone_color:
                count += 1
                line.append(1)
            elif self.board[tx][ty] == 0:
                space_count += 1
                line.append(0)
            else:
                break
            tx, ty = tx + dx, ty + dy

        # 向相反方向延伸
        tx, ty = x - dx, y - dy
        while 0 <= tx < self.config.board.size and 0 <= ty < self.config.board.size:
            if self.board[tx][ty] == stone_color:
                count += 1
                line.insert(0, 1)
            elif self.board[tx][ty] == 0:
                space_count += 1
                line.insert(0, 0)
            else:
                break
            tx, ty = tx - dx, ty - dy

        # 恢复原始值
        self.board[x][y] = original_value

        # 判断是否是四
        # 四的模式：
        # 1. "11110" 或 "01111"（冲四）
        # 2. "11011"（跳四）
        patterns = ["11110", "01111", "11011"]
        line_str = "".join(map(str, line))

        for pattern in patterns:
            if pattern in line_str:
                return True

        return False

    # def _remove_dead_stones(self):
    #     """移除死子"""
    #     dead_stones = []
    #     # 遍历整个棋盘
    #     for x in range(self.config.board.size):
    #         for y in range(self.config.board.size):
    #             if self.board[x][y] != 0:  # 如果有棋子
    #                 if not self._has_liberty(x, y):  # 如果没有气
    #                     dead_stones.append((x, y))

    #     # 移除死子
    #     for x, y in dead_stones:
    #         self.board[x][y] = 0

    def _has_liberty(self, x, y, visited=None):
        """检查棋子是否有气"""
        if visited is None:
            visited = set()

        color = self.board[x][y]
        visited.add((x, y))

        # 检查四个方向
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.config.board.size and 0 <= ny < self.config.board.size:
                # 如果找到空点，说明有气
                if self.board[nx][ny] == 0:
                    return True
                # 如果是同色棋子且未访问过，继续检查
                if self.board[nx][ny] == color and (nx, ny) not in visited:
                    if self._has_liberty(nx, ny, visited):
                        return True
        return False

    def _is_forbidden_point(self, x, y, stone_color):
        """判断是否是禁手点"""
        # 检查长连
        if self._has_overline(x, y, stone_color):
            return True
        # 检查三三禁手
        if self._has_double_three(x, y, stone_color):
            return True
        # 检查四四禁手
        if self._has_double_four(x, y, stone_color):
            return True

        return False

    # def _is_forbidden_point(self, x, y, stone_color):
    #     """判断是否是禁手点"""
    #     # 检查劫争
    #     if self._is_ko_point(x, y):
    #         return True

    #     # 临时落子以检查禁手
    #     self.board[x][y] = stone_color
    #     is_forbidden = False

    #     # 检查长连
    #     if self._has_overline(x, y, stone_color):
    #         is_forbidden = True
    #     # ... 其他禁手检查 ...

    #     # 恢复棋盘状态
    #     self.board[x][y] = 0
    #     return is_forbidden

    def _remove_dead_stones(self):
        """移除死子并更新劫争状态"""
        dead_stones = []
        last_move = self.move_history[-1] if self.move_history else None
        last_move_color = last_move[2] if last_move else None

        # 检查所有棋子的气
        for x in range(self.config.board.size):
            for y in range(self.config.board.size):
                if self.board[x][y] != 0:  # 对所有非空位置进行检查
                    if not self._has_liberty(x, y):
                        dead_stones.append((x, y, self.board[x][y]))

        # 如果有死子，记录到历史
        if dead_stones:
            self.dead_stones_history.append((dead_stones, self.board.copy()))
            # 记录当前棋盘镜像和吃子方颜色
            self.last_capture_mirror = self._generate_board_mirror()
            self.last_capture_color = last_move_color

        # 移除死子
        for x, y, _ in dead_stones:
            self.board[x][y] = 0

    def _is_ko_point(self, x, y):
        """判断是否是劫争点"""
        # 模拟落子并生成镜像
        self.board[x][y] = 3 - self.board[x][y]  # 模拟落子
        self._remove_dead_stones()  # 模拟吃子
        simulated_mirror = self._generate_board_mirror()
        self.board[x][y] = 0  # 恢复棋盘状态

        # 比较镜像
        if simulated_mirror == self.last_capture_mirror:
            # 判断上一次吃子是否为对方颜色
            if self.last_capture_color == 3 - self.board[x][y]:
                return True
        return False

    def _generate_board_mirror(self):
        """生成当前棋盘的镜像"""
        return np.flip(self.board, axis=0).tolist()

    def _restore_last_move(self):
        """恢复上一手棋"""
        if not self.move_history:
            return False

        # 恢复死子
        if self.dead_stones_history:
            dead_stones, _ = self.dead_stones_history.pop()
            for x, y, color in dead_stones:
                self.board[x][y] = color

        # 移除最后一手棋
        last_x, last_y, _ = self.move_history.pop()
        self.board[last_x][last_y] = 0
        return True

    def _is_potential_ko(self, x, y):
        """判断是否可能形成劫"""
        stone_color = self.board[x][y]
        opponent_color = 3 - stone_color  # 1->2, 2->1
        surrounded_count = 0

        # 检查四个方向
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.config.board.size and 0 <= ny < self.config.board.size:
                if self.board[nx][ny] == opponent_color:
                    surrounded_count += 1

        # 如果四周都是对方的子，则可能形成劫
        return surrounded_count >= 3
