from config import Config
from camera_manager import CameraManager
from window_manager import WindowManager
from board_manager import BoardManager
from lock_manager import LockManager
import cv2
import numpy as np
import sys
from os.path import exists
from logger import Logger
import os
from datetime import datetime
from interactive_board import InteractiveBoard


class WatchGo:
    def __init__(self):
        self.config = Config()
        self.lock = LockManager(self.config)
        self.camera = CameraManager()
        self.window = WindowManager()
        self.board = BoardManager(self.config)
        self.logger = Logger(self.config)
        self.history = []
        self.current_game_id = None
        self.interactive = InteractiveBoard(self.config, start_index=90)

    def start(self):
        """启动应用"""
        self.lock.acquire()
        try:
            if len(sys.argv) > 1:
                if sys.argv[1] == "--camera":
                    self.process_camera()
                elif sys.argv[1] == "--image":
                    if len(sys.argv) > 2:
                        self.process_local_image(sys.argv[2])
                    else:
                        self.test_local_image()
            else:
                # self.test_local_image()
                # 默认启动交互式棋盘
                self.interactive_board()
        finally:
            self.cleanup()

    def process_local_image(self, image_path):
        """处理本地图像"""
        if not exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return

        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像文件: {image_path}")
            return

        # 显示原始图像
        h, w = img.shape[:2]
        video_size = (int(round(w / 2.0)), int(round(h / 2.0)))
        self.window.show_image("camera", cv2.resize(img, video_size))

        # 分析棋盘状态
        board_state, corners = self.board.analyze_image(img)

        self.board.print_board_state()

        self.window.show_image("board", self.board.draw_board(None, board_state))

        

        # 在原图上标注识别结果
        if corners is not None:
            self._draw_annotations(img, corners, board_state)

        # 显示标注后的图像
        self.window.show_image("camera", cv2.resize(img, video_size))
        cv2.waitKey(0)

    def process_camera(self):
        """处理摄像头输入"""
        self.camera.init_camera()
        self.camera.load_calibration(self.config.calibration_file)

        # 初始化参数
        frame = self.camera.read_frame()
        if frame is None:
            print("无法读取摄像头图像")
            return

        h, w = frame.shape[:2]
        video_size = (int(round(w / 2.0)), int(round(h / 2.0)))

        # 初始化运动检测参数
        buf_size = 10
        buf = [self.camera.read_frame() for _ in range(buf_size)]
        i = 0
        roi = np.zeros((h, w), dtype="uint8")
        cv2.rectangle(roi, (0, 0), (w, h), 1, -1)
        movement_threshold = int(w * h * 0.1)
        still_frames = 0
        moving = True

        # 主循环
        while cv2.waitKey(1) == -1:
            frame = self.camera.read_frame()
            if frame is None:
                break

            self._process_video_frame(
                frame, buf, i, roi, movement_threshold, still_frames, moving, video_size
            )

    def _draw_annotations(self, img, corners, board_state):
        """在图像上绘制标注"""
        # 标注角点
        for corner in corners:
            cv2.circle(
                img, (int(round(corner[0])), int(round(corner[1]))), 6, (0, 0, 255), -1
            )

        # 标注棋子位置
        if board_state is not None:
            matrix = self._get_perspective_matrix(corners)
            if matrix is not None:
                self._draw_stones(img, board_state, matrix)

    def cleanup(self):
        """清理资源"""
        self.camera.release()
        self.window.close_all()

    def _process_video_frame(
        self, frame, buf, i, roi, movement_threshold, still_frames, moving, video_size
    ):
        """处理视频帧"""
        # 更新缓冲区
        buf[i] = frame
        background = self._average_frames(buf)
        i = (i + 1) % len(buf)

        # 检测运动
        motion = self._detect_motion(frame, background, roi)
        motion_sum = motion.sum()

        # 处理运动状态
        if motion_sum > movement_threshold:
            if not moving:
                board_state, corners = self.board.analyze_image(frame)
                self.window.show_image("board", self.board.draw_board(board_state))
                moving = True
                still_frames = 0
        else:
            moving = False
            still_frames += 1
            if still_frames == (len(buf) + 1):
                board_state, corners = self.board.analyze_image(background)
                self.window.show_image("board", self.board.draw_board(board_state))

                # 更新ROI区域
                if corners is not None and len(corners) > 3:
                    roi = self._update_roi(corners, frame.shape[:2])
                    movement_threshold = self._calculate_movement_threshold(corners)

        # 显示处理后的图像
        display_img = frame.copy()
        if corners is not None:
            self._draw_annotations(display_img, corners, board_state)
        self.window.show_image("camera", cv2.resize(display_img, video_size))

        return i, roi, movement_threshold, still_frames, moving

    def _average_frames(self, frames):
        """计算帧平均值"""
        avg = np.float32(frames[0])
        for i in range(1, len(frames)):
            if frames[i] is not None:
                avg = cv2.accumulateWeighted(frames[i], avg, 0.1)
        return cv2.convertScaleAbs(avg)

    def _detect_motion(self, frame, background, roi):
        """检测运动"""
        motion = cv2.absdiff(
            cv2.GaussianBlur(frame, (11, 11), 0),
            cv2.GaussianBlur(background, (11, 11), 0),
        )
        motion = cv2.cvtColor(motion, cv2.COLOR_BGR2GRAY)
        motion *= roi
        _, motion = cv2.threshold(motion, 32, 1, cv2.THRESH_BINARY)
        return motion

    def _update_roi(self, corners, shape):
        """更新感兴趣区域"""
        roi = np.zeros(shape, dtype="uint8")
        corners_array = np.array(corners).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillConvexPoly(roi, corners_array, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
        return cv2.dilate(roi, kernel)

    def _calculate_movement_threshold(self, corners):
        """计算运动检测阈值"""
        corners_array = np.array(corners).reshape((-1, 1, 2)).astype(np.int32)
        return int(cv2.contourArea(corners_array) / (self.config.board.size**2))

    def _get_perspective_matrix(self, corners):
        """获取透视变换矩阵"""
        src_points = np.array(corners, dtype="float32")
        dst_points = np.array(
            [
                [0, 0],
                [self.config.board.size - 1, 0],
                [self.config.board.size - 1, self.config.board.size - 1],
                [0, self.config.board.size - 1],
            ],
            dtype="float32",
        )
        return cv2.getPerspectiveTransform(dst_points, src_points)

    def _draw_stones(self, img, board_state, matrix):
        """绘制棋子标注"""
        for x in range(self.config.board.size):
            for y in range(self.config.board.size):
                if board_state[x][y] > 0:
                    point = np.array([[[x, y]]], dtype="float32")
                    transformed_point = cv2.perspectiveTransform(point, matrix)[0][0]
                    color = (0, 0, 0) if board_state[x][y] == 1 else (255, 255, 255)
                    cv2.circle(
                        img,
                        (
                            int(round(transformed_point[0])),
                            int(round(transformed_point[1])),
                        ),
                        10,
                        color,
                        -1,
                    )

    def test_local_image(self, image_path="resource/13.jpg"):
        """测试本地图像识别
        Args:
            image_path: 相对于 base_path 的图片路径
        """
        test_image = os.path.join(self.config.base_path, image_path)
        self.process_local_image(test_image)

    def test_camera(self):
        """测试摄像头实时识别"""
        self.process_camera()

    def print_board_state(self, board_state):
        """在控制台打印棋盘状态"""
        print("当前棋盘状态:")
        top_left, top_right = "┌", "┐"
        bottom_left, bottom_right = "└", "┘"
        horizontal, vertical = "─", "│"
        black, white, empty = "⚫", "⚪", "十"

        # 打印顶部边框
        print(top_left + horizontal * board_state.shape[0] * 2 + top_right)

        # 打印棋盘内容
        for y in range(board_state.shape[1]):
            row = vertical
            for x in range(board_state.shape[0]):
                if board_state[x][y] == 0:
                    row += empty
                elif board_state[x][y] == 1:
                    row += black
                else:
                    row += white
            row += vertical
            print(row)

        # 打印底部边框
        print(bottom_left + horizontal * board_state.shape[0] * 2 + bottom_right)
        print()

    def save_board_state(self, board_state, filename="board_state.txt"):
        """保存棋盘状态到文件"""
        filepath = os.path.join(self.config.base_path, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for y in range(board_state.shape[1]):
                    row = []
                    for x in range(board_state.shape[0]):
                        if board_state[x][y] == 0:
                            row.append(".")
                        elif board_state[x][y] == 1:
                            row.append("B")
                        else:
                            row.append("W")
                    f.write("".join(row) + "\n")
            print(f"棋盘状态已保存到: {filepath}")
        except Exception as e:
            print(f"保存棋盘状态失败: {str(e)}")

    def load_board_state(self, filename="board_state.txt"):
        """从文件加载棋盘状态"""
        filepath = os.path.join(self.config.base_path, filename)
        try:
            board_state = np.zeros(
                (self.config.board.size, self.config.board.size), dtype="uint8"
            )
            with open(filepath, "r", encoding="utf-8") as f:
                for y, line in enumerate(f):
                    for x, char in enumerate(line.strip()):
                        if char == "B":
                            board_state[x][y] = 1
                        elif char == "W":
                            board_state[x][y] = 2
            return board_state
        except Exception as e:
            print(f"加载棋盘状态失败: {str(e)}")
            return None

    def export_board_image(self, board_state, filename="board.png"):
        """导出棋盘图像"""
        try:
            filepath = os.path.join(self.config.base_path, "exports", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            board_img = self.board.draw_board(board_state)
            cv2.imwrite(filepath, board_img)
            self.logger.info(f"棋盘图像已导出到: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"导出棋盘图像失败: {str(e)}")
            return False

    def capture_screenshot(self, filename=None):
        """捕获当前画面"""
        try:
            if filename is None:
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            filepath = os.path.join(self.config.base_path, "screenshots", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            frame = self.camera.read_frame()
            if frame is not None:
                cv2.imwrite(filepath, frame)
                self.logger.info(f"截图已保存到: {filepath}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"截图失败: {str(e)}")
            return False

    def analyze_screenshot(self, filepath):
        """分析已保存的截图"""
        try:
            img = cv2.imread(filepath)
            if img is None:
                self.logger.error(f"无法读取图像: {filepath}")
                return None, None

            return self.board.analyze_image(img)
        except Exception as e:
            self.logger.error(f"分析截图失败: {str(e)}")
            return None, None

    def set_camera_parameters(self, brightness=None, contrast=None, exposure=None):
        """设置摄像头参数"""
        try:
            if brightness is not None:
                self.camera.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            if contrast is not None:
                self.camera.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
            if exposure is not None:
                self.camera.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.logger.info("摄像头参数设置成功")
            return True
        except Exception as e:
            self.logger.error(f"设置摄像头参数失败: {str(e)}")
            return False

    def calibrate_camera(self, num_frames=30):
        """相机校准"""
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros(
                (self.config.board.size * self.config.board.size, 3), np.float32
            )
            objp[:, :2] = np.mgrid[
                0 : self.config.board.size, 0 : self.config.board.size
            ].T.reshape(-1, 2)

            objpoints = []
            imgpoints = []

            for _ in range(num_frames):
                frame = self.camera.read_frame()
                if frame is None:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray, (self.config.board.size, self.config.board.size), None
                )

                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    imgpoints.append(corners2)

            if len(objpoints) > 0:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )
                np.save(self.config.calibration_file, [mtx, dist])
                self.logger.info("相机校准完成")
                return True

            self.logger.warning("未检测到足够的棋盘角点")
            return False

        except Exception as e:
            self.logger.error(f"相机校准失败: {str(e)}")
            return False

    def optimize_detection(self, frame):
        """优化图像检测"""
        try:
            # 图像预处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # 自适应阈值处理
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            return opening
        except Exception as e:
            self.logger.error(f"图像优化失败: {str(e)}")
            return frame

    def get_board_statistics(self, board_state):
        """获取棋盘统计信息"""
        try:
            black_count = np.sum(board_state == 1)
            white_count = np.sum(board_state == 2)
            empty_count = np.sum(board_state == 0)

            stats = {
                "黑子数量": black_count,
                "白子数量": white_count,
                "空位数量": empty_count,
                "总子数": black_count + white_count,
            }

            self.logger.info(f"棋盘统计: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"统计分析失败: {str(e)}")
            return None

    def start_new_game(self):
        """开始新的对局"""
        try:
            self.current_game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.history = []
            self.logger.info(f"开始新对局: {self.current_game_id}")
            return True
        except Exception as e:
            self.logger.error(f"开始新对局失败: {str(e)}")
            return False

    def save_move(self, board_state):
        """保存当前局面"""
        try:
            if self.current_game_id is None:
                self.start_new_game()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            move_data = {
                "timestamp": timestamp,
                "board_state": board_state.copy(),
                "statistics": self.get_board_statistics(board_state),
            }
            self.history.append(move_data)

            # 保存到文件
            game_dir = os.path.join(
                self.config.base_path, "games", self.current_game_id
            )
            os.makedirs(game_dir, exist_ok=True)

            # 保存棋盘图片
            self.export_board_image(
                board_state, os.path.join(game_dir, f"move_{timestamp}.png")
            )

            # 保存局面数据
            with open(
                os.path.join(game_dir, "game_record.json"), "w", encoding="utf-8"
            ) as f:
                import json
                json.dump(
                    {"game_id": self.current_game_id, "moves": self.history},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            self.logger.info(f"已保存当前局面: {timestamp}")
            return True
        except Exception as e:
            self.logger.error(f"保存局面失败: {str(e)}")
            return False

    def analyze_game(self):
        """分析当前对局"""
        try:
            if not self.history:
                self.logger.warning("没有可分析的对局数据")
                return None

            analysis = {
                "game_id": self.current_game_id,
                "total_moves": len(self.history),
                "move_timeline": [],
                "territory_changes": [],
            }

            for move in self.history:
                analysis["move_timeline"].append(
                    {"timestamp": move["timestamp"], "statistics": move["statistics"]}
                )

            self.logger.info(f"对局分析完成: {self.current_game_id}")
            return analysis
        except Exception as e:
            self.logger.error(f"对局分析失败: {str(e)}")
            return None

    def export_game_record(self, format="sgf"):
        """导出对局记录"""
        try:
            if not self.history:
                self.logger.warning("没有可导出的对局数据")
                return False

            export_dir = os.path.join(self.config.base_path, "exports")
            os.makedirs(export_dir, exist_ok=True)

            if format == "sgf":
                filename = f"game_{self.current_game_id}.sgf"
                filepath = os.path.join(export_dir, filename)
                self._export_sgf(filepath)
            else:
                filename = f"game_{self.current_game_id}.{format}"
                filepath = os.path.join(export_dir, filename)
                self._export_custom(filepath)

            self.logger.info(f"对局记录已导出: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"导出对局记录失败: {str(e)}")
            return False

    def _export_sgf(self, filepath):
        """导出SGF格式文件"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(
                    f"(;GM[1]FF[4]CA[UTF-8]AP[WatchGo]SZ[{self.config.board.size}]\n"
                )
                f.write(f"DT[{self.current_game_id}]\n")

                for move in self.history:
                    board = move["board_state"]
                    for x in range(self.config.board.size):
                        for y in range(self.config.board.size):
                            if board[x][y] == 1:
                                f.write(f";B[{chr(97+x)}{chr(97+y)}]")
                            elif board[x][y] == 2:
                                f.write(f";W[{chr(97+x)}{chr(97+y)}]")

                f.write(")")
        except Exception as e:
            raise Exception(f"导出SGF文件失败: {str(e)}")

    def _export_custom(self, filepath):
        """导出自定义格式文件"""
        try:
            data = {
                "game_id": self.current_game_id,
                "board_size": self.config.board.size,
                "total_moves": len(self.history),
                "moves": [],
            }

            for move in self.history:
                move_data = {
                    "timestamp": move["timestamp"],
                    "board": move["board_state"].tolist(),
                    "statistics": move["statistics"],
                }
                data["moves"].append(move_data)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            raise Exception(f"导出自定义格式文件失败: {str(e)}")

    def analyze_territory(self, board_state):
        """分析领地"""
        try:
            territory = np.zeros_like(board_state)
            visited = np.zeros_like(board_state, dtype=bool)

            def flood_fill(x, y, color):
                if (
                    x < 0
                    or x >= board_state.shape[0]
                    or y < 0
                    or y >= board_state.shape[1]
                    or visited[x, y]
                ):
                    return 0, set()

                visited[x, y] = True
                if board_state[x, y] != 0:
                    return 0, {board_state[x, y]}

                area = 1
                colors = set()
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    a, c = flood_fill(x + dx, y + dy, color)
                    area += a
                    colors.update(c)

                if len(colors) == 1:
                    territory[x, y] = list(colors)[0]

                return area, colors

            # 分析空白区域
            for x in range(board_state.shape[0]):
                for y in range(board_state.shape[1]):
                    if board_state[x, y] == 0 and not visited[x, y]:
                        flood_fill(x, y, 0)

            # 计算领地统计
            black_territory = np.sum(territory == 1)
            white_territory = np.sum(territory == 2)

            return {
                "territory_map": territory,
                "black_territory": int(black_territory),
                "white_territory": int(white_territory),
            }

        except Exception as e:
            self.logger.error(f"领地分析失败: {str(e)}")
            return None

    def detect_dead_stones(self, board_state):
        """检测死子"""
        try:
            dead_stones = []
            liberty_map = np.zeros_like(board_state)

            def count_liberties(x, y, color, visited):
                if (
                    x < 0
                    or x >= board_state.shape[0]
                    or y < 0
                    or y >= board_state.shape[1]
                ):
                    return 0, set()

                if visited[x, y]:
                    return 0, set()

                if board_state[x, y] == 0:
                    return 1, set()

                if board_state[x, y] != color:
                    return 0, set()

                visited[x, y] = True
                liberties = 0
                stones = {(x, y)}

                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    l, s = count_liberties(x + dx, y + dy, color, visited)
                    liberties += l
                    stones.update(s)

                return liberties, stones

            # 检查每个棋子
            for x in range(board_state.shape[0]):
                for y in range(board_state.shape[1]):
                    if board_state[x, y] > 0:
                        visited = np.zeros_like(board_state, dtype=bool)
                        liberties, stones = count_liberties(
                            x, y, board_state[x, y], visited
                        )

                        # 更新气数图
                        for sx, sy in stones:
                            liberty_map[sx, sy] = liberties

                        # 记录死子
                        if liberties <= 1:
                            dead_stones.extend(list(stones))

            return dead_stones, liberty_map

        except Exception as e:
            self.logger.error(f"死子检测失败: {str(e)}")
            return None, None

    def analyze_game_progress(self):
        """分析对局进展"""
        try:
            if not self.history:
                self.logger.warning("没有对局数据可分析")
                return None

            progress = {
                "total_time": 0,
                "average_time_per_move": 0,
                "territory_trend": [],
                "capture_stats": {"black": 0, "white": 0},
            }

            # 计算时间统计
            timestamps = [
                
                datetime.strptime(move["timestamp"], "%Y%m%d_%H%M%S")
                for move in self.history
            ]
            if len(timestamps) > 1:
                total_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
                progress["total_time"] = total_seconds
                progress["average_time_per_move"] = total_seconds / (
                    len(timestamps) - 1
                )

            # 分析领地变化趋势
            for move in self.history:
                territory = self.analyze_territory(move["board_state"])
                if territory:
                    progress["territory_trend"].append(
                        {
                            "timestamp": move["timestamp"],
                            "black": territory["black_territory"],
                            "white": territory["white_territory"],
                        }
                    )

            self.logger.info("对局进展分析完成")
            return progress
        except Exception as e:
            self.logger.error(f"分析对局进展失败: {str(e)}")
            return None

    def generate_game_report(self, output_format="html"):
        """生成对局报告"""
        try:
            if not self.history:
                self.logger.warning("没有对局数据可生成报告")
                return False

            progress = self.analyze_game_progress()
            if not progress:
                return False

            report_dir = os.path.join(self.config.base_path, "reports")
            os.makedirs(report_dir, exist_ok=True)

            filename = f"game_report_{self.current_game_id}.{output_format}"
            filepath = os.path.join(report_dir, filename)

            if output_format == "html":
                self._generate_html_report(filepath, progress)
            else:
                self._generate_text_report(filepath, progress)

            self.logger.info(f"对局报告已生成: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"生成对局报告失败: {str(e)}")
            return False

    def _generate_html_report(self, filepath, progress):
        """生成HTML格式报告"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(
                    f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>对局报告 - {self.current_game_id}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .stats {{ margin: 20px 0; }}
                        .chart {{ margin: 20px 0; }}
                    </style>
                </head>
                <body>
                    <h1>对局报告</h1>
                    <div class="stats">
                        <h2>基本统计</h2>
                        <p>总时长: {progress['total_time']/60:.1f} 分钟</p>
                        <p>平均每手用时: {progress['average_time_per_move']:.1f} 秒</p>
                    </div>
                    <div class="territory">
                        <h2>领地变化</h2>
                        <table border="1">
                            <tr><th>时间</th><th>黑方领地</th><th>白方领地</th></tr>
                """
                )

                for data in progress["territory_trend"]:
                    f.write(
                        f"""
                            <tr>
                                <td>{data['timestamp']}</td>
                                <td>{data['black']}</td>
                                <td>{data['white']}</td>
                            </tr>
                    """
                    )

                f.write(
                    """
                        </table>
                    </div>
                </body>
                </html>
                """
                )
        except Exception as e:
            raise Exception(f"生成HTML报告失败: {str(e)}")

    def _generate_text_report(self, filepath, progress):
        """生成文本格式报告"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"对局报告 - {self.current_game_id}\n")
                f.write("=" * 50 + "\n\n")

                f.write("基本统计\n")
                f.write("-" * 20 + "\n")
                f.write(f"总时长: {progress['total_time']/60:.1f} 分钟\n")
                f.write(f"平均每手用时: {progress['average_time_per_move']:.1f} 秒\n\n")

                f.write("领地变化\n")
                f.write("-" * 20 + "\n")
                for data in progress["territory_trend"]:
                    f.write(f"时间: {data['timestamp']}\n")
                    f.write(f"黑方领地: {data['black']}\n")
                    f.write(f"白方领地: {data['white']}\n")
                    f.write("-" * 20 + "\n")
        except Exception as e:
            raise Exception(f"生成文本报告失败: {str(e)}")

    def setup_ui_controls(self):
        """设置用户界面控制"""
        try:
            cv2.createTrackbar(
                "亮度",
                "camera",
                50,
                100,
                lambda x: self.set_camera_parameters(brightness=x),
            )
            cv2.createTrackbar(
                "对比度",
                "camera",
                50,
                100,
                lambda x: self.set_camera_parameters(contrast=x),
            )
            cv2.createTrackbar(
                "曝光",
                "camera",
                50,
                100,
                lambda x: self.set_camera_parameters(exposure=x),
            )
            self.logger.info("用户界面控制设置完成")
            return True
        except Exception as e:
            self.logger.error(f"设置用户界面控制失败: {str(e)}")
            return False

    def handle_keyboard_input(self, key):
        """处理键盘输入"""
        try:
            if key == ord("s"):  # 保存当前局面
                self.capture_screenshot()
            elif key == ord("n"):  # 开始新对局
                self.start_new_game()
            elif key == ord("a"):  # 分析当前局面
                board_state, _ = self.board.analyze_image(self.camera.read_frame())
                if board_state is not None:
                    self.analyze_territory(board_state)
                    self.detect_dead_stones(board_state)
            elif key == ord("r"):  # 生成报告
                self.generate_game_report()
            elif key == ord("e"):  # 导出对局记录
                self.export_game_record()
            elif key == ord("h"):  # 显示帮助信息
                self.show_help()

            return True
        except Exception as e:
            self.logger.error(f"处理键盘输入失败: {str(e)}")
            return False

    def show_help(self):
        """显示帮助信息"""
        help_text = """
        键盘快捷键:
        S - 保存当前局面截图
        N - 开始新对局
        A - 分析当前局面
        R - 生成对局报告
        E - 导出对局记录
        H - 显示此帮助信息
        Q - 退出程序
        
        滑动条控制:
        亮度 - 调节摄像头亮度
        对比度 - 调节摄像头对比度
        曝光 - 调节摄像头曝光
        """
        print(help_text)
        self.logger.info("显示帮助信息")

    def auto_adjust_camera(self):
        """自动调整摄像头参数"""
        try:
            frame = self.camera.read_frame()
            if frame is None:
                return False

            # 计算图像亮度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)

            # 计算对比度
            contrast = np.std(gray)

            # 自动调整参数
            if brightness < 100:
                self.set_camera_parameters(
                    brightness=min(
                        self.camera.cap.get(cv2.CAP_PROP_BRIGHTNESS) + 10, 100
                    )
                )
            elif brightness > 200:
                self.set_camera_parameters(
                    brightness=max(self.camera.cap.get(cv2.CAP_PROP_BRIGHTNESS) - 10, 0)
                )

            if contrast < 30:
                self.set_camera_parameters(
                    contrast=min(self.camera.cap.get(cv2.CAP_PROP_CONTRAST) + 10, 100)
                )

            self.logger.info("摄像头参数已自动调整")
            return True
        except Exception as e:
            self.logger.error(f"自动调整摄像头参数失败: {str(e)}")
            return False

    def analyze_move_quality(self, board_state):
        """分析落子质量"""
        try:
            quality_score = 0
            analysis = {"score": 0, "suggestions": [], "threats": []}

            # 分析棋子布局
            territory = self.analyze_territory(board_state)
            dead_stones, liberty_map = self.detect_dead_stones(board_state)

            if territory:
                # 评估领地优势
                territory_diff = (
                    territory["black_territory"] - territory["white_territory"]
                )
                quality_score += territory_diff * 0.5

                if abs(territory_diff) > 5:
                    analysis["suggestions"].append(
                        "黑方领先" if territory_diff > 0 else "白方领先"
                    )

            # 评估死子情况
            if dead_stones:
                quality_score -= len(dead_stones)
                analysis["threats"].extend(
                    [f"位置({x},{y})的棋子有危险" for x, y in dead_stones]
                )

            # 评估气数分布
            if liberty_map is not None:
                low_liberty_positions = np.where(liberty_map == 1)
                for x, y in zip(*low_liberty_positions):
                    analysis["threats"].append(f"位置({x},{y})的棋子气数不足")

            analysis["score"] = quality_score
            self.logger.info(f"落子质量分析完成，得分: {quality_score}")
            return analysis

        except Exception as e:
            self.logger.error(f"分析落子质量失败: {str(e)}")
            return None

    def suggest_next_move(self, board_state):
        """提供下一步走子建议"""
        try:
            suggestions = []

            # 分析当前局面
            territory = self.analyze_territory(board_state)
            dead_stones, liberty_map = self.detect_dead_stones(board_state)

            # 寻找潜在好点
            for x in range(board_state.shape[0]):
                for y in range(board_state.shape[1]):
                    if board_state[x][y] == 0:  # 空位
                        score = 0

                        # 检查周围是否有己方棋子
                        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < board_state.shape[0]
                                and 0 <= ny < board_state.shape[1]
                            ):
                                if board_state[nx][ny] > 0:
                                    score += 1

                        # 如果周围有死子，增加权重
                        if dead_stones and any(
                            (nx, ny) in dead_stones
                            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                            for nx, ny in [(x + dx, y + dy)]
                        ):
                            score += 3

                        if score > 0:
                            suggestions.append(
                                {
                                    "position": (x, y),
                                    "score": score,
                                    "reason": "可以提高棋子连接性",
                                }
                            )

            # 按分数排序
            suggestions.sort(key=lambda x: x["score"], reverse=True)

            self.logger.info(f"已生成{len(suggestions)}个走子建议")
            return suggestions[:3]  # 返回前三个建议

        except Exception as e:
            self.logger.error(f"生成走子建议失败: {str(e)}")
            return None

    def enhance_image_quality(self, frame):
        """增强图像质量"""
        try:
            # 图像去噪
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

            # 对比度增强
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            # 锐化处理
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)

            return sharpened
        except Exception as e:
            self.logger.error(f"图像增强失败: {str(e)}")
            return frame

    def optimize_performance(self):
        """优化性能"""
        try:
            # 设置OpenCV优化标志
            cv2.setUseOptimized(True)
            cv2.setNumThreads(4)  # 根据CPU核心数调整

            # 预分配内存
            self.frame_buffer = np.zeros(
                (self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8
            )

            # 设置相机缓冲区大小
            self.camera.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.logger.info("性能优化设置完成")
            return True
        except Exception as e:
            self.logger.error(f"性能优化设置失败: {str(e)}")
            return False

    def process_frame_async(self, frame):
        """异步处理帧"""
        try:
            import threading

            def process_worker():
                enhanced_frame = self.enhance_image_quality(frame)
                board_state, corners = self.board.analyze_image(enhanced_frame)
                if board_state is not None:
                    self.save_move(board_state)

            thread = threading.Thread(target=process_worker)
            thread.daemon = True
            thread.start()

            return True
        except Exception as e:
            self.logger.error(f"异步处理帧失败: {str(e)}")
            return False

    def analyze_game_patterns(self, board_state):
        """分析棋型特征"""
        try:
            patterns = {
                "connections": [],  # 连接关系
                "groups": [],  # 棋子群
                "influence": np.zeros_like(board_state, dtype=float),  # 影响力图
            }

            # 分析棋子群
            visited = np.zeros_like(board_state, dtype=bool)
            for x in range(board_state.shape[0]):
                for y in range(board_state.shape[1]):
                    if board_state[x][y] > 0 and not visited[x][y]:
                        group = self._find_group(board_state, x, y, visited)
                        if len(group) > 0:
                            patterns["groups"].append(
                                {
                                    "color": board_state[x][y],
                                    "stones": group,
                                    "size": len(group),
                                }
                            )

            # 计算影响力
            for group in patterns["groups"]:
                for x, y in group["stones"]:
                    # 向四周扩散影响力
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < board_state.shape[0]
                                and 0 <= ny < board_state.shape[1]
                            ):
                                dist = abs(dx) + abs(dy)
                                influence = 1.0 / (dist + 1)
                                if group["color"] == 1:  # 黑子
                                    patterns["influence"][nx][ny] += influence
                                else:  # 白子
                                    patterns["influence"][nx][ny] -= influence

            self.logger.info("棋型分析完成")
            return patterns
        except Exception as e:
            self.logger.error(f"棋型分析失败: {str(e)}")
            return None

    def _find_group(self, board_state, x, y, visited):
        """查找棋子群"""
        color = board_state[x][y]
        group = set()
        stack = [(x, y)]

        while stack:
            cx, cy = stack.pop()
            if visited[cx][cy]:
                continue

            visited[cx][cy] = True
            if board_state[cx][cy] == color:
                group.add((cx, cy))
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < board_state.shape[0]
                        and 0 <= ny < board_state.shape[1]
                        and not visited[nx][ny]
                    ):
                        stack.append((nx, ny))

        return group

    def visualize_analysis(self, board_state, analysis_type="influence"):
        """可视化分析结果"""
        try:
            if analysis_type == "influence":
                patterns = self.analyze_game_patterns(board_state)
                if patterns is None:
                    return None

                # 创建热力图
                plt.figure(figsize=(10, 10))
                plt.imshow(patterns["influence"], cmap="RdBu", interpolation="nearest")
                plt.colorbar(label="影响力")
                plt.title("棋子影响力分布")

                # 保存图像
                output_path = os.path.join(
                    self.config.base_path,
                    "analysis",
                    f'influence_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path)
                plt.close()

                self.logger.info(f"分析可视化已保存: {output_path}")
                return output_path

        except Exception as e:
            self.logger.error(f"分析可视化失败: {str(e)}")
            return None

    def interactive_board(self):
        """创建交互式棋盘窗口"""
        return self.interactive.start()


if __name__ == "__main__":
    app = WatchGo()
    app.start()
