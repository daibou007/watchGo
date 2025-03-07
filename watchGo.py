# 导入必要的库
try:
    import numpy as np
except ImportError:
    print("请先安装 numpy 库: pip install numpy")
    exit(1)
try:
    import cv2
except ImportError:
    print("请先安装 opencv-python 库: pip install opencv-python")
    exit(1)
from math import sqrt
from os.path import exists

# 全局配置
USE_CAMERA = False  # 是否使用摄像头

# 初始化摄像头
cap = None
if USE_CAMERA:
    devices = cv2.videoio_registry.getBackends()
    for device in devices:
        if device == cv2.CAP_AVFOUNDATION:  # MacOS 摄像头
            cap = cv2.VideoCapture(0 + cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
            break
    else:
        cap = cv2.VideoCapture(0)

# 棋盘大小设置
boardSize = 19  # 可以是 9, 13 或 19
frameSize = None  # 视频帧大小，由 watchBoard 设置

# 重要文件路径
calibrationFile = "./calibration.npy"  # 相机校准文件
blackCascadeFile = "./blackCascade.xml"  # 黑子检测器
whiteCascadeFile = "./whiteCascade.xml"  # 白子检测器
emptyCascadeFile = "./emptyCascade.xml"  # 空点检测器

# 相机校准参数
mapx = None
mapy = None


def loadCalibration(filename):
    """加载相机校准数据"""
    global mapx, mapy
    if exists(filename):
        loaded = np.load(filename)
        mapx = loaded[0]
        mapy = loaded[1]


# 加载校准数据
loadCalibration(calibrationFile)


def closeWindow(win="video"):
    """关闭 OpenCV 窗口"""
    cv2.destroyWindow(win)
    for i in range(4):
        cv2.waitKey(1)


def readImage():
    """从摄像头读取图像并应用校准"""
    # 检查摄像头对象是否有效
    if cap is None:
        return None
        
    # 读取图像
    success, img = cap.read()
    if not success:
        return None
        
    # 应用校准(如果有)
    if mapx is not None and mapy is not None:
        return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    else:
        return img


# 初始化级联分类器
empty_cascade = None
black_cascade = None
white_cascade = None

# 加载空点检测器
if exists(emptyCascadeFile):
    empty_cascade = cv2.CascadeClassifier()
    if not empty_cascade.load(emptyCascadeFile):
        print(f"无法加载空白交叉点检测文件: {emptyCascadeFile}")
        exit(1)
else:
    print(f"无法找到空白交叉点检测文件: {emptyCascadeFile}")
    exit(1)

# 加载黑子检测器
if exists(blackCascadeFile):
    black_cascade = cv2.CascadeClassifier()
    if not black_cascade.load(blackCascadeFile):
        print(f"无法加载黑子检测文件: {blackCascadeFile}")
        exit(1)
else:
    print(f"无法找到黑子检测文件: {blackCascadeFile}")
    exit(1)

# 加载白子检测器
if exists(whiteCascadeFile):
    white_cascade = cv2.CascadeClassifier()
    if not white_cascade.load(whiteCascadeFile):
        print(f"无法加载白子检测文件: {whiteCascadeFile}")
        exit(1)
else:
    print(f"无法找到白子检测文件: {whiteCascadeFile}")
    exit(1)


def avgImages(images):
    """对图像序列取平均值以减少噪声"""
    output = np.zeros(images[0].shape, dtype="float32")
    for i in images:
        output += i
    output /= len(images)
    return output.astype("uint8")

def readBoardC(image):
    """分析图像中的围棋棋盘"""
    if image is None:
        print("无效的图像数据")
        return np.zeros((boardSize, boardSize), dtype="uint8"), None

    output = np.zeros((boardSize, boardSize), dtype="uint8")
    
        # 检查级联分类器是否正确加载
    if empty_cascade is None or not isinstance(empty_cascade, cv2.CascadeClassifier):
        print("空白交叉点检测器未正确加载")
        return output, None
    if black_cascade is None or not isinstance(black_cascade, cv2.CascadeClassifier):
        print("黑子检测器未正确加载")
        return output, None
    if white_cascade is None or not isinstance(white_cascade, cv2.CascadeClassifier):
        print("白子检测器未正确加载")
        return output, None

    try:
        # 使用级联分类器检测特征
        emptyRectangles = empty_cascade.detectMultiScale(image, 1.08, 3)
        blackRectangles = black_cascade.detectMultiScale(image, 1.01, 3)
        whiteRectangles = white_cascade.detectMultiScale(image, 1.01, 3)
    except Exception as e:
        print(f"检测过程出错: {str(e)}")
        return output, None

    # 提取特征中心点
    empties = []
    blacks = []
    whites = []
    for ex, wy, w, h in emptyRectangles:
        empties.append([ex + w/2.0, wy + w/2.0])
    for ex, wy, w, h in blackRectangles:
        blacks.append([ex + w/2.0, wy + w/2.0])
    for ex, wy, w, h in whiteRectangles:
        whites.append([ex + w/2.0, wy + w/2.0])

    # 标记检测到的点
    for c in empties:
        cv2.circle(image, (int(round(c[0])), int(round(c[1]))), 3, (0,255,255), -1)
    for c in blacks:
        cv2.circle(image, (int(round(c[0])), int(round(c[1]))), 3, (0,255,0), -1)
    for c in whites:
        cv2.circle(image, (int(round(c[0])), int(round(c[1]))), 3, (0,0,255), -1)
    cv2.imshow("dots", image)

    # 寻找棋盘区域
    group = findGroup(empties + blacks + whites)
    if group is None:
        return output, None

    # 计算凸包和角点
    hull = cv2.convexHull(np.array(group, dtype="int32"))
    epsilon = 0.001 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    corners = findCorners(approx)
    
    if corners is None or len(corners) < 2:
        print("未检测到足够的棋盘角点")
        return output, None
        
    # 根据检测到的角点数量处理
    if len(corners) == 4:
        imgCorners = sortCorners(corners)
    elif len(corners) == 3:
        imgCorners = estimateCornerFrom3(corners)
    elif len(corners) == 2:
        imgCorners = estimateCornerFrom2(corners)
    else:
        return output, None

    # 透视变换
    if imgCorners is not None:
        flatCorners = np.array([
            [0, 0],
            [boardSize-1, 0],
            [boardSize-1, boardSize-1],
            [0, boardSize-1]
        ], dtype="float32")
        
        persp = cv2.getPerspectiveTransform(imgCorners, flatCorners)
        
        # 转换棋子坐标并对齐到左下
        if len(blacks) > 0:
            blacks = np.array([blacks], dtype="float32")
            blacksFlat = cv2.perspectiveTransform(blacks, persp)[0]
            for i in blacksFlat:
                x = int(round(i[0]))
                y = int(round(i[1]))
                if 0 <= x < boardSize and 0 <= y < boardSize:
                    # 对齐到左边和下边
                    aligned_x = min(x, boardSize-1)
                    aligned_y = max(y, 0)
                    output[aligned_x][aligned_y] = 1
                    
        if len(whites) > 0:
            whites = np.array([whites], dtype="float32")
            whitesFlat = cv2.perspectiveTransform(whites, persp)[0]
            for i in whitesFlat:
                x = int(round(i[0]))
                y = int(round(i[1]))
                if 0 <= x < boardSize and 0 <= y < boardSize:
                    # 对齐到左边和下边
                    aligned_x = min(x, boardSize-1)
                    aligned_y = max(y, 0)
                    output[aligned_x][aligned_y] = 2

    return output, imgCorners

def findCorners(approx):
    """从轮廓中提取角点
    参数：
    - approx: 近似多边形轮廓点
    返回：
    - 排序后的四个角点坐标，如果无法找到合适的角点则返回 None
    """
    if len(approx) < 4:
        return None

    # 将轮廓点转换为简单的坐标列表
    points = []
    for point in approx:
        points.append([point[0][0], point[0][1]])
    points = np.array(points)

    # 计算轮廓的边界矩形
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)

    # 计算轮廓面积和边界矩形面积
    contour_area = cv2.contourArea(points)
    rect_area = rect[1][0] * rect[1][1]

    # 如果轮廓面积与矩形面积比例不合适，说明可能不是棋盘
    if contour_area / rect_area < 0.7:
        return None

    # 对角点进行排序（左上、右上、右下、左下）
    center = np.mean(box, axis=0)
    corners = []
    
    for point in box:
        if point[0] < center[0] and point[1] < center[1]:
            corners.append(point)  # 左上
    for point in box:
        if point[0] > center[0] and point[1] < center[1]:
            corners.append(point)  # 右上
    for point in box:
        if point[0] > center[0] and point[1] > center[1]:
            corners.append(point)  # 右下
    for point in box:
        if point[0] < center[0] and point[1] > center[1]:
            corners.append(point)  # 左下

    if len(corners) != 4:
        return None

    return np.array(corners, dtype="float32")

def sortCorners(corners):
    """对四个角点进行排序：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # 计算重心
    center = np.mean(corners, axis=0)
    
    # 根据角点相对重心的位置进行分类
    top_left = []
    top_right = []
    bottom_right = []
    bottom_left = []
    
    for corner in corners:
        if corner[0] < center[0] and corner[1] < center[1]:
            top_left = corner
        elif corner[0] > center[0] and corner[1] < center[1]:
            top_right = corner
        elif corner[0] > center[0] and corner[1] > center[1]:
            bottom_right = corner
        else:
            bottom_left = corner
            
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

def estimateCornerFrom3(corners):
    """从3个角点估计第4个角点"""
    # 计算重心
    center = np.mean(corners, axis=0)
    
    # 找出缺失的角点位置
    missing_corner = None
    corners_list = corners.tolist()
    
    # 根据现有角点的位置关系估计缺失角点
    if all(c[0] < center[0] for c in corners_list):  # 缺少右侧角点
        right_points = sorted(corners_list, key=lambda x: x[1])
        missing_corner = [2 * center[0] - right_points[1][0], right_points[1][1]]
    elif all(c[0] > center[0] for c in corners_list):  # 缺少左侧角点
        left_points = sorted(corners_list, key=lambda x: x[1])
        missing_corner = [2 * center[0] - left_points[1][0], left_points[1][1]]
    elif all(c[1] < center[1] for c in corners_list):  # 缺少下方角点
        bottom_points = sorted(corners_list, key=lambda x: x[0])
        missing_corner = [bottom_points[1][0], 2 * center[1] - bottom_points[1][1]]
    elif all(c[1] > center[1] for c in corners_list):  # 缺少上方角点
        top_points = sorted(corners_list, key=lambda x: x[0])
        missing_corner = [top_points[1][0], 2 * center[1] - top_points[1][1]]
        
    if missing_corner:
        corners = np.vstack((corners, [missing_corner]))
        return sortCorners(corners)
    return None

def estimateCornerFrom2(corners):
    """从2个角点估计另外2个角点"""
    # 计算两点的中点作为棋盘中心估计
    center = np.mean(corners, axis=0)
    
    # 根据两个角点的位置关系估计缺失的角点
    p1, p2 = corners
    
    # 如果两点在对角线上
    if (p1[0] < center[0] and p1[1] < center[1] and 
        p2[0] > center[0] and p2[1] > center[1]) or \
       (p1[0] > center[0] and p1[1] > center[1] and 
        p2[0] < center[0] and p2[1] < center[1]):
        # 估计另外两个角点
        p3 = [p1[0], p2[1]]
        p4 = [p2[0], p1[1]]
        corners = np.array([p1, p4, p2, p3])
    else:
        # 如果两点在同一边，无法可靠估计其他角点
        return None
        
    return sortCorners(corners)

def readBoard(image):
    """
    分析图像中的围棋棋盘
    返回：
    - 棋盘状态数组（0=空，1=黑，2=白）
    - 棋盘四角坐标
    """
    if image is None:
        print("无效的图像数据")
        return np.zeros((boardSize, boardSize), dtype="uint8"), None

    output = np.zeros((boardSize, boardSize), dtype="uint8")
    imgCorners = None

    # 使用级联分类器检测棋盘上的特征
    # 检查 empty_cascade 是否为 None
    if empty_cascade is None:
        print("空白交叉点检测器未正确加载")
        return [], [], []
    emptyRectangles = empty_cascade.detectMultiScale(image, 1.08, 3)  # 空点
     # 检查 empty_cascade 是否为 None
    if black_cascade is None:
        print("黑色棋子检测器未正确加载")
        return [], [], []
    blackRectangles = black_cascade.detectMultiScale(image, 1.01, 3)  # 黑子
    # 检查 white_cascade 是否为 None
    if white_cascade is None:
        print("白色棋子检测器未正确加载")
        return [], [], []
    whiteRectangles = white_cascade.detectMultiScale(image, 1.01, 3)  # 白子

    # 提取特征中心点
    empties = []
    blacks = []
    whites = []
    for ex, wy, w, h in emptyRectangles:
        x = ex + w / 2.0
        y = wy + w / 2.0
        empties.append([x, y])
    for ex, wy, w, h in blackRectangles:
        x = ex + w / 2.0
        y = wy + w / 2.0
        blacks.append([x, y])
    for ex, wy, w, h in whiteRectangles:
        x = ex + w / 2.0
        y = wy + w / 2.0
        whites.append([x, y])

    # 在图像上标记检测到的点
    for c in empties:
        cv2.circle(image, (int(round(c[0])), int(round(c[1]))), 3, (0, 255, 255), -1)
    for c in blacks:
        cv2.circle(image, (int(round(c[0])), int(round(c[1]))), 3, (0, 255, 0), -1)
    for c in whites:
        cv2.circle(image, (int(round(c[0])), int(round(c[1]))), 3, (0, 0, 255), -1)
    cv2.imshow("dots", image)

    # 寻找棋盘区域
    group = findGroup(empties + blacks + whites)
    if group is None:
        return output, imgCorners

    # 计算凸包和近似多边形
    hull = cv2.convexHull(np.array(group, dtype="int32"))
    epsilon = 0.001 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    imgCorners = fourCorners(approx)

    # 如果找到四个角点，进行透视变换
    if imgCorners is not None and len(imgCorners) > 3:
        flatCorners = np.array(
            [
                [0, 0],
                [boardSize - 1, 0],
                [boardSize - 1, boardSize - 1],
                [0, boardSize - 1],
            ],
            dtype="float32",
        )
        
        # 确保输入点为float32类型
        src_points = np.array(imgCorners, dtype="float32")
        dst_points = np.array(flatCorners, dtype="float32")
        persp = cv2.getPerspectiveTransform(src_points, dst_points)

        # 转换黑子坐标
        if len(blacks) > 0:
            blacks = np.array(blacks, dtype="float32")
            blacks = np.array([blacks])
            blacksFlat = cv2.perspectiveTransform(blacks, persp)
            for i in blacksFlat[0]:
                x = int(round(i[0]))
                y = int(round(i[1]))
                if x >= 0 and x < boardSize and y >= 0 and y < boardSize:
                    output[x][y] = 1

        # 转换白子坐标
        if len(whites) > 0:
            whites = np.array(whites, dtype="float32")
            whites = np.array([whites])
            whitesFlat = cv2.perspectiveTransform(whites, persp)
            for i in whitesFlat[0]:
                x = int(round(i[0]))
                y = int(round(i[1]))
                if x >= 0 and x < boardSize and y >= 0 and y < boardSize:
                    output[x][y] = 2

    return output, imgCorners


def findGroupMembers(maxDistance, i, distances, group):
    """递归搜索距离足够近的点"""
    for j in range(len(group)):
        if group[j]:
            pass
        elif distances[i][j] < maxDistance:
            group[j] = True
            findGroupMembers(maxDistance, j, distances, group)


def findGroup(spots):
    """寻找紧密聚集的点群"""
    # 计算点之间的距离矩阵
    length = len(spots)
    distances = np.zeros((length, length), dtype="float32")
    distanceList = []
    for i in range(length):
        for j in range(length):
            d = sqrt(
                (spots[i][0] - spots[j][0]) ** 2 + (spots[i][1] - spots[j][1]) ** 2
            )
            distances[i][j] = d
            if d > 0:
                distanceList.append(d)

    # 计算最大允许距离
    distanceList.sort()
    numDistances = int((boardSize - 1) ** 2 * 1.8)
    maxDistance = np.mean(distanceList[0:numDistances]) * 1.75

    # 寻找足够大的群组
    minGroup = int(boardSize**2 * 0.75)
    group = np.zeros((length), dtype="bool_")
    for i in range(length):
        findGroupMembers(maxDistance, i, distances, group)
        if group.sum() >= minGroup:
            outPoints = []
            for k in range(length):
                if group[k]:
                    outPoints.append(spots[k])
            return outPoints
        else:
            group = np.zeros((length), dtype="bool_")


def sortPoints(box):
    """对四边形的四个点进行排序"""
    rect = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]
    diff = np.diff(box, axis=1)
    rect[1] = box[np.argmin(diff)]
    rect[3] = box[np.argmax(diff)]
    return rect


def fourCorners(hull):
    """从多边形轮廓中提取四个角点"""
    length = len(hull)
    if length < 4:
        return []

    # 计算所有线段及其长度
    allLines = []
    for i in range(length):
        if i == (length - 1):
            line = [[hull[i][0][0], hull[i][0][1]], [hull[0][0][0], hull[0][0][1]]]
        else:
            line = [
                [hull[i][0][0], hull[i][0][1]],
                [hull[i + 1][0][0], hull[i + 1][0][1]],
            ]
        d = sqrt((line[0][0] - line[1][0]) ** 2 + (line[0][1] - line[1][1]) ** 2)
        allLines.append([line, d])

    # 获取四条最长的线
    allLines.sort(key=lambda x: x[1], reverse=True)
    lines = []
    for i in range(4):
        lines.append(allLines[i][0])

    # 计算线段方程 y = mx + c
    equations = []
    for i in lines:
        x_coords, y_coords = zip(*i)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords)[0]
        equations.append([m, c])

    # 计算线段交点
    intersections = []
    for i in equations:
        for j in equations:
            # 检查方程是否有效
            if i is None or j is None:
                continue
                
            try:
                if i[0] == j[0]:
                    continue
                    
                # 计算交点
                a = np.array([[i[0] * -1, 1], [j[0] * -1, 1]])
                b = np.array([i[1], j[1]])
                solution = np.linalg.solve(a, b)
                
                # 检查frameSize是否有效
                if frameSize is None:
                    continue
                    
                # 验证交点是否在有效范围内
                if (solution[0] > 0 and 
                    solution[1] > 0 and 
                    solution[0] < frameSize[0] and 
                    solution[1] < frameSize[1]):
                    intersections.append([solution[0], solution[1]])
            except (IndexError, TypeError):
                continue

    intersections.sort()

    # 选择四个角点
    if len(intersections) > 6:
        output = [
            intersections[0],
            intersections[2],
            intersections[4],
            intersections[6],
        ]
        box = sortPoints(np.array(output, dtype="float32"))
        return box
    else:
        return []


def blankBoard(boardBlockSize):
    """创建空白棋盘"""
    yellow = [75, 215, 255]
    black = [0, 0, 0]
    white = [255, 255, 255]
    halfBoardBlock = int(round((boardBlockSize / 2.0)))
    boardSide = boardBlockSize * boardSize
    blankBoard = np.zeros((boardSide, boardSide, 3), dtype="uint8")

    # 绘制棋盘底色
    cv2.rectangle(blankBoard, (0, 0), (boardSide, boardSide), yellow, -1)

    # 绘制网格线
    for i in range(boardSize):
        spot = i * boardBlockSize + halfBoardBlock
        cv2.line(
            blankBoard,
            (spot, halfBoardBlock),
            (spot, boardSide - halfBoardBlock),
            black,
            int(boardBlockSize / 10),
        )
        cv2.line(
            blankBoard,
            (halfBoardBlock, spot),
            (boardSide - halfBoardBlock, spot),
            black,
            int(boardBlockSize / 10),
        )

    # 绘制天元和星位
    if boardSize == 19:
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
    else:
        spots = []

    for s in spots:
        cv2.circle(
            blankBoard,
            (
                s[0] * boardBlockSize + halfBoardBlock,
                s[1] * boardBlockSize + halfBoardBlock,
            ),
            int(boardBlockSize * 0.15),
            black,
            -1,
        )

    return blankBoard


def printBoard(board):
    """在控制台打印棋盘状态，保持正方形显示"""
    print("当前棋盘状态:")
    # 使用直线和直角字符
    top_left = "┌"     # 左上角
    top_right = "┐"    # 右上角
    bottom_left = "└"   # 左下角
    bottom_right = "┘"  # 右下角
    horizontal = "─"    # 水平线
    vertical = "│"      # 垂直线
    black = "⚫"        # 黑子
    white = "⚪"        # 白子
    empty = "十"        # 空位
    
    # 打印顶部边框（修正水平线长度）
    print(top_left + horizontal * board.shape[0]*2 + top_right)
    
    # 打印棋盘内容
    for y in range(board.shape[1]):
        row = vertical  # 行首
        for x in range(board.shape[0]):
            if board[x][y] == 0:
                row += empty
            elif board[x][y] == 1:
                row += black
            elif board[x][y] == 2:
                row += white
        row += vertical  # 行尾
        print(row)
    
    # 打印底部边框（修正水平线长度）
    print(bottom_left + horizontal * board.shape[0]*2 + bottom_right)
    print()

def drawBoardC(board, size=(500, 500)):
    """绘制棋盘界面"""
    black = [0, 0, 0]
    white = [255, 255, 255]

    # 打印文字版棋盘
    printBoard(board=board)

    # 设置棋盘格子大小
    boardBlockSize = 100
    halfBoardBlock = int(round(boardBlockSize / 2.0))
    output = blankBoard(100)

    # 获取棋盘角点
    corners = []
    for x in [0, boardSize-1]:
        for y in [0, boardSize-1]:
            if board[x][y] > 0:
                corners.append([x, y])

    # 根据检测到的角点数量调整棋盘显示
    if len(corners) >= 2:
        # 计算棋盘变换矩阵
        src_points = np.array([
            [0, 0],
            [boardSize-1, 0],
            [boardSize-1, boardSize-1],
            [0, boardSize-1]
        ], dtype="float32")
        
        if len(corners) == 4:
            dst_points = sortCorners(np.array(corners, dtype="float32"))
        elif len(corners) == 3:
            dst_points = estimateCornerFrom3(np.array(corners, dtype="float32"))
        elif len(corners) == 2:
            dst_points = estimateCornerFrom2(np.array(corners, dtype="float32"))
        
        # 检查dst_points是否存在且有效
        if 'dst_points' in locals() and dst_points is not None:
            # 应用透视变换
            # 检查dst_points是否存在且有效
            if 'dst_points' in locals() and dst_points is not None:
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            else:
                matrix = None
            
            # 绘制黑白棋子
            for x in range(boardSize):
                for y in range(boardSize):
                    if board[x][y] > 0:
                        # 计算棋子在变换后的位置
                        point = np.array([[[x, y]]], dtype="float32")
                        transformed_point = cv2.perspectiveTransform(point, matrix)[0][0]
                        
                        color = black if board[x][y] == 1 else white
                        cv2.circle(
                            output,
                            (
                                int(round(transformed_point[0] * boardBlockSize + halfBoardBlock)),
                                int(round(transformed_point[1] * boardBlockSize + halfBoardBlock))
                            ),
                            int(boardBlockSize / 2),
                            color,
                            -1,
                        )
    else:
        # 如果没有检测到足够的角点，使用原始方式绘制
        for x in range(boardSize):
            for y in range(boardSize):
                if board[x][y] > 0:
                    color = black if board[x][y] == 1 else white
                    cv2.circle(
                        output,
                        (
                            (x * boardBlockSize) + halfBoardBlock,
                            (y * boardBlockSize) + halfBoardBlock,
                        ),
                        int(boardBlockSize / 2),
                        color,
                        -1,
                    )

    # 调整输出图像大小
    output = cv2.resize(output, size, output, 0, 0, cv2.INTER_AREA)
    return output

def drawBoard(board, size=(500, 500)):
    """绘制棋盘界面
    参数：
    - board: 棋盘状态数组
    - size: 输出图像大小
    返回：渲染后的棋盘图像
    """
    black = [0, 0, 0]
    white = [255, 255, 255]

    # 打印文字版棋盘
    printBoard(board=board)

    # 设置棋盘格子大小
    boardBlockSize = 100
    halfBoardBlock = int(round(boardBlockSize / 2.0))
    output = blankBoard(100)

    # 绘制黑白棋子
    (w, h) = board.shape
    for x in range(w):
        for y in range(h):
            if board[x][y] == 1:  # 绘制黑子
                cv2.circle(
                    output,
                    (
                        (x * boardBlockSize) + halfBoardBlock,
                        (y * boardBlockSize) + halfBoardBlock,
                    ),
                    int(boardBlockSize / 2),
                    black,
                    -1,
                )
            elif board[x][y] == 2:  # 绘制白子
                cv2.circle(
                    output,
                    (
                        (x * boardBlockSize) + halfBoardBlock,
                        (y * boardBlockSize) + halfBoardBlock,
                    ),
                    int(boardBlockSize / 2),
                    white,
                    -1,
                )
    # 调整输出图像大小
    output = cv2.resize(output, size, output, 0, 0, cv2.INTER_AREA)
    return output


def processLocalImage(image_path):
    """处理本地图像文件
    参数：
    - image_path: 图像文件路径
    """
    # 检查文件是否存在
    if not exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return

    # 读取图像文件
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像文件: {image_path}")
        return

    # 设置全局帧大小
    global frameSize
    (h, w, d) = img.shape
    frameSize = (w, h)

    # 显示原始图像
    videoSize = (int(round(w / 2.0)), int(round(h / 2.0)))
    # 调整图像大小并显示
    resized_img = cv2.resize(img, videoSize, interpolation=cv2.INTER_AREA)
    cv2.imshow("camera", resized_img)

    # 分析棋盘状态
    board = np.zeros((boardSize, boardSize), dtype="uint8")
    board, imgCorners = readBoardC(img)
    cv2.imshow("board", drawBoardC(board))

        # 在原图上标注识别结果
    if imgCorners is not None:
        # 标注棋盘角点
        for c in imgCorners:
            cv2.circle(img, (int(round(c[0])), int(round(c[1]))), 6, (0, 0, 255), -1)

        # 计算棋盘上所有交叉点的坐标
        src_points = np.array(imgCorners, dtype="float32")
        dst_points = np.array([
            [0, 0],
            [boardSize - 1, 0],
            [boardSize - 1, boardSize - 1],
            [0, boardSize - 1]
        ], dtype="float32")
        
        # 获取透视变换矩阵
        matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        
        # 标注棋子位置
        if board is not None:
            for x in range(boardSize):
                for y in range(boardSize):
                    if board[x][y] > 0:  # 有棋子的位置
                        # 计算棋子在原图中的位置
                        point = np.array([[[x, y]]], dtype="float32")
                        transformed_point = cv2.perspectiveTransform(point, matrix)[0][0]
                        
                        # 绘制棋子
                        color = (0, 0, 0) if board[x][y] == 1 else (255, 255, 255)
                        cv2.circle(
                            img,
                            (int(round(transformed_point[0])), int(round(transformed_point[1]))),
                            10,
                            color,
                            -1,
                        )

    # 显示标注后的图像
    # 调整图像大小并显示
    resized_img = cv2.resize(img, videoSize, interpolation=cv2.INTER_AREA)
    cv2.imshow("camera", resized_img)

    # 等待按键并关闭窗口
    cv2.waitKey(0)
    closeWindow("camera")
    closeWindow("board")
    closeWindow("dots")

def processVideoFrame(frame, board, imgCorners, roi, movementThreshold, stillFrames, moving, buf, i):
    """处理视频帧并识别棋盘
    参数：
    - frame: 当前视频帧
    - board: 当前棋盘状态
    - imgCorners: 棋盘角点
    - roi: 感兴趣区域
    - movementThreshold: 运动检测阈值
    - stillFrames: 静止帧计数
    - moving: 是否在运动
    - buf: 帧缓冲区
    - i: 当前缓冲区索引
    返回：
    - board: 更新后的棋盘状态
    - imgCorners: 更新后的角点
    - roi: 更新后的感兴趣区域
    - movementThreshold: 更新后的运动阈值
    - stillFrames: 更新后的静止帧计数
    - moving: 更新后的运动状态
    - i: 更新后的缓冲区索引
    """
    h, w = frame.shape[:2]
    videoSize = (int(round(w / 2.0)), int(round(h / 2.0)))
    
    # 更新帧缓冲区
    buf[i] = frame
    bkg = avgImages(buf)
    i = (i + 1) % len(buf)

    # 检测运动
    motion = cv2.absdiff(
        cv2.GaussianBlur(frame, (11, 11), 0), 
        cv2.GaussianBlur(bkg, (11, 11), 0)
    )
    motion = cv2.cvtColor(motion, cv2.COLOR_BGR2GRAY)
    motion *= roi
    retval, motion = cv2.threshold(motion, 32, 1, cv2.THRESH_BINARY)
    motionSum = motion.sum()

    # 处理运动状态
    if motionSum > movementThreshold:
        if not moving:
            cv2.imshow("board", cv2.cvtColor(drawBoard(board), cv2.COLOR_BGR2GRAY))
            moving = True
            stillFrames = 0
    else:
        moving = False
        stillFrames += 1
        if stillFrames == (len(buf) + 1):
            # 使用readBoardC替代readBoard以确保返回值类型一致
            board, imgCorners = readBoardC(bkg)
            cv2.imshow("board", drawBoard(board))
            if imgCorners is not None and len(imgCorners) > 3:
                roi = np.zeros((h, w), dtype="uint8")
                # 将imgCorners转换为正确的形状
                corners_array = np.array(imgCorners).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillConvexPoly(roi, corners_array, 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
                roi = cv2.dilate(roi, kernel)
                movementThreshold = int(cv2.contourArea(corners_array) / (boardSize**2))

    # 显示标注后的图像
    image = frame.copy()
    if imgCorners is not None:
        for c in imgCorners:
            cv2.circle(image, (int(round(c[0])), int(round(c[1]))), 6, (0, 0, 255), -1)
    # 调整图像大小并显示
    resized_img = cv2.resize(image, videoSize, interpolation=cv2.INTER_AREA)
    cv2.imshow("camera", resized_img)

    return board, imgCorners, roi, movementThreshold, stillFrames, moving, i

def processCamera():
    """处理摄像头实时识别"""
    if not USE_CAMERA:
        print("摄像头功能已禁用")
        return

    # 初始化摄像头
    global frameSize
    if cap is not None:
        cap.open(0)

    # 初始化参数
    imgCorners = None
    img = readImage()
    if img is None:
        print("无法读取摄像头图像")
        return
        
    h, w = img.shape[:2]
    frameSize = (w, h)
    videoSize = (int(round(w / 2.0)), int(round(h / 2.0)))
    
    # 显示初始画面
    resized_img = cv2.resize(img, videoSize, interpolation=cv2.INTER_AREA)
    cv2.imshow("camera", resized_img)
    board = np.zeros((boardSize, boardSize), dtype="uint8")
    cv2.imshow("board", cv2.cvtColor(drawBoard(board), cv2.COLOR_BGR2GRAY))

    # 初始化缓冲区
    bufSize = 10
    buf = [readImage() for _ in range(bufSize)]
    i = 0

    # 初始化运动检测参数
    roi = np.zeros((h, w), dtype="uint8")
    cv2.rectangle(roi, (0, 0), (w, h), 1, -1)
    movementThreshold = int(w * h * 0.1)
    stillFrames = 0
    moving = True

    # 主循环
    while cv2.waitKey(1) == -1:
        frame = readImage()
        if frame is None:
            break
            
        board, imgCorners, roi, movementThreshold, stillFrames, moving, i = processVideoFrame(
            frame, board, imgCorners, roi, movementThreshold, stillFrames, moving, buf, i
        )

    # 清理资源
    # 检查cap是否为None并且是否有release方法
    if cap is not None and hasattr(cap, 'release'):
        cap.release()
    closeWindow("camera")
    closeWindow("board")

def watchBoard(image_path=None):
    """监控棋盘状态
    参数：
    - image_path: 可选的图像文件路径，如果提供则处理单张图像
    """
    if image_path:
        processLocalImage(image_path)
    else:
        processCamera()


def test_local_image():
    """测试本地图像识别"""
    test_image = "./13.jpg"  # 测试图像路径
    processLocalImage(test_image)

def test_camera():
    """测试摄像头实时识别"""
    global USE_CAMERA
    USE_CAMERA = True  # 启用摄像头
    watchBoard()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了命令行参数
        if sys.argv[1] == "--camera":
            test_camera()
        elif sys.argv[1] == "--image":
            if len(sys.argv) > 2:
                processLocalImage(sys.argv[2])
            else:
                test_local_image()
    else:
        # 默认测试本地图像
        test_local_image()