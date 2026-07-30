# ========== 相机硬件参数 ==========
# 焦距：像素单位，可修改做参数敏感性实验
FOCAL_LENGTH = 820

# ========== 物体真实高度（米）行业标准值 ==========
CAR_REAL_HEIGHT = 1.5    # 家用小轿车常规高度
PERSON_REAL_HEIGHT = 1.7 # 成年人平均身高

# ========== 距离告警阈值 ==========
URGENT_WARN_DIST = 3.0   # 小于3m：红色紧急告警
NORMAL_WARN_DIST = 5.0   # 3~5m：黄色提醒
SAFE_DIST = 5.0          # 大于5m：绿色安全

# ========== YOLOv10检测参数 ==========
YOLO_WEIGHT = "yolov10s.pt"
CONFIDENCE_THRESHOLD = 0.5  # 置信度过滤
IOU_THRESHOLD = 0.45

# ========== 创新优化开关（用于消融实验） ==========
ENABLE_FRAME_SMOOTH = True    # 帧平滑防抖
ENABLE_WH_DOUBLE_CAL = True   # 宽高双测距取平均

# ========== 平滑滤波配置 ==========
SMOOTH_FRAME_NUM = 5  # 用连续5帧做均值平滑