import cv2
import argparse
import logging 
import numpy as np
import math
from math import sqrt

logging.basicConfig(
    level = logging.INFO,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
    
rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

# COCO姿态关键点定义
# COCO pose keypoints definition
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 关键点连接关系用于绘制骨架
# Keypoint connections for skeleton drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # 头部 head
    (1, 3), (2, 4),  # 眼睛到耳朵 eyes to ears
    (5, 6),          # 肩膀 shoulders
    (5, 7), (7, 9),  # 左臂 left arm
    (6, 8), (8, 10), # 右臂 right arm
    (5, 11), (6, 12), # 肩膀到髋部 shoulders to hips
    (11, 12),        # 髋部 hips
    (11, 13), (13, 15), # 左腿 left leg
    (12, 14), (14, 16)  # 右腿 right leg
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='source/yolo11n_pose_bayese_640x640_nv12.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--img-save-path', type=str, default='py_result.jpg', help='Path to Save Result Image.')
    parser.add_argument('--classes-num', type=int, default=1, help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--nkpt', type=int, default=17, help='num of keypoints.')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5, help='keypoint confidence threshold.')
    parser.add_argument('--strides', type=lambda s: list(map(int, s.split(','))), 
                        default=[8, 16 ,32],
                        help='--strides 8, 16, 32')
    
    # 姿态检测相关参数
    # Pose detection related parameters
    parser.add_argument('--knee-angle-thres', type=float, default=165, 
                        help='Knee angle threshold for crouching detection (degrees, < threshold = crouching).')
    parser.add_argument('--hip-angle-thres', type=float, default=160, 
                        help='Hip angle threshold for crouching detection (degrees, < threshold = crouching).')
    parser.add_argument('--min-kpt-conf', type=float, default=0.15, 
                        help='Minimum keypoint confidence for pose analysis.')
    parser.add_argument('--min-valid-kpts', type=int, default=2, 
                        help='Minimum number of valid keypoints required for pose analysis.')
    parser.add_argument('--display-kpt-conf', type=float, default=0.1, 
                        help='Minimum keypoint confidence for display (lower than analysis threshold).')
    parser.add_argument('--show-angles', action='store_true', default=True,
                        help='Show angle indicators and measurements.')
    
    # 相机相关参数
    # Camera related parameters
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID (default: 0).')
    
    opt = parser.parse_args()
    return opt

def get_camera_properties(camera_id):
    """
    使用OpenCV获取相机的实际参数
    Get actual camera parameters using OpenCV
    
    Args:
        camera_id: 相机设备ID / Camera device ID
        
    Returns:
        tuple: (actual_width, actual_height, actual_fps, success)
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error(f"❌ 无法打开相机设备 {camera_id}")
        return None, None, None, False
    
    try:
        # 获取相机实际参数
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"📹 相机 {camera_id} 当前参数:")
        logger.info(f"   宽度: {actual_width}")
        logger.info(f"   高度: {actual_height}")
        logger.info(f"   帧率: {actual_fps}")
        
        return actual_width, actual_height, actual_fps, True
        
    except Exception as e:
        logger.error(f"❌ 获取相机参数时出错: {e}")
        return None, None, None, False
    finally:
        cap.release()

def calculate_angle(point1, point2, point3):
    """
    计算三点构成的角度 (point2为顶点)
    Calculate angle formed by three points (point2 is the vertex)
    
    Args:
        point1: (x1, y1) 第一个点
        point2: (x2, y2) 顶点
        point3: (x3, y3) 第三个点
    
    Returns:
        float: 角度 (度数)
    """
    # 计算两个向量
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])
    
    # 计算向量长度
    len1 = sqrt(vector1[0]**2 + vector1[1]**2)
    len2 = sqrt(vector2[0]**2 + vector2[1]**2)
    
    if len1 == 0 or len2 == 0:
        return 0
    
    # 计算余弦值
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    cos_angle = dot_product / (len1 * len2)
    
    # 防止数值误差
    cos_angle = max(-1, min(1, cos_angle))
    
    # 转换为角度
    angle = math.acos(cos_angle) * 180 / math.pi
    return angle

def get_valid_keypoints(kpts, min_conf=0.15):
    """
    获取置信度足够高的关键点
    Get keypoints with sufficient confidence
    
    Args:
        kpts: 关键点列表 [(x, y, conf), ...]
        min_conf: 最小置信度
    
    Returns:
        dict: {keypoint_id: (x, y, conf)}
    """
    valid_kpts = {}
    for i, (x, y, conf) in enumerate(kpts):
        if conf >= min_conf:
            valid_kpts[i] = (x, y, conf)
    return valid_kpts

def analyze_pose_status(kpts, knee_angle_thres=165, hip_angle_thres=160, min_conf=0.15):
    """
    基于关键点角度分析人体姿态：站立或蹲下
    Analyze human pose based on joint angles: standing or crouching
    
    Args:
        kpts: 17个关键点 [(x, y, conf), ...]
        knee_angle_thres: 膝盖角度阈值，小于此值判断为蹲下
        hip_angle_thres: 髋部角度阈值，小于此值判断为蹲下
        min_conf: 关键点最小置信度
    
    Returns:
        dict: 包含姿态分析结果的字典
    """
    # 获取有效关键点
    valid_kpts = get_valid_keypoints(kpts, min_conf)
    
    # 定义关键点索引 (COCO格式)
    # 5: left_shoulder, 6: right_shoulder
    # 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee
    # 15: left_ankle, 16: right_ankle
    
    result = {
        'status': 'unknown',
        'confidence': 0.0,
        'knee_angles': {'left': None, 'right': None},
        'hip_angles': {'left': None, 'right': None},
        'crouch_indicators': [],
        'valid_keypoints_count': len(valid_kpts),
        'analysis_valid': False
    }
    
    # 检查是否有足够的关键点进行分析
    if len(valid_kpts) < 3:
        logger.debug(f"关键点数量不足进行姿态分析: {len(valid_kpts)}")
        return result
    
    crouch_indicators = []
    angle_confidences = []
    
    # 计算左膝盖角度 (髋-膝-踝)
    if 11 in valid_kpts and 13 in valid_kpts and 15 in valid_kpts:
        left_hip = (valid_kpts[11][0], valid_kpts[11][1])
        left_knee = (valid_kpts[13][0], valid_kpts[13][1])
        left_ankle = (valid_kpts[15][0], valid_kpts[15][1])
        
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        result['knee_angles']['left'] = left_knee_angle
        
        if left_knee_angle < knee_angle_thres:
            crouch_indicators.append(f"左膝弯曲({left_knee_angle:.1f}°)")
            # 角度越小，置信度越高
            angle_conf = min(1.0, (knee_angle_thres - left_knee_angle) / knee_angle_thres + 0.3)
            angle_confidences.append(angle_conf)
        
        logger.debug(f"左膝盖角度: {left_knee_angle:.1f}°")
    
    # 计算右膝盖角度 (髋-膝-踝)
    if 12 in valid_kpts and 14 in valid_kpts and 16 in valid_kpts:
        right_hip = (valid_kpts[12][0], valid_kpts[12][1])
        right_knee = (valid_kpts[14][0], valid_kpts[14][1])
        right_ankle = (valid_kpts[16][0], valid_kpts[16][1])
        
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        result['knee_angles']['right'] = right_knee_angle
        
        if right_knee_angle < knee_angle_thres:
            crouch_indicators.append(f"右膝弯曲({right_knee_angle:.1f}°)")
            angle_conf = min(1.0, (knee_angle_thres - right_knee_angle) / knee_angle_thres + 0.3)
            angle_confidences.append(angle_conf)
        
        logger.debug(f"右膝盖角度: {right_knee_angle:.1f}°")
    
    # 计算左髋角度 (肩-髋-膝)
    if 5 in valid_kpts and 11 in valid_kpts and 13 in valid_kpts:
        left_shoulder = (valid_kpts[5][0], valid_kpts[5][1])
        left_hip = (valid_kpts[11][0], valid_kpts[11][1])
        left_knee = (valid_kpts[13][0], valid_kpts[13][1])
        
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        result['hip_angles']['left'] = left_hip_angle
        
        if left_hip_angle < hip_angle_thres:
            crouch_indicators.append(f"左髋弯曲({left_hip_angle:.1f}°)")
            angle_conf = min(1.0, (hip_angle_thres - left_hip_angle) / hip_angle_thres + 0.3)
            angle_confidences.append(angle_conf)
        
        logger.debug(f"左髋角度: {left_hip_angle:.1f}°")
    
    # 计算右髋角度 (肩-髋-膝)
    if 6 in valid_kpts and 12 in valid_kpts and 14 in valid_kpts:
        right_shoulder = (valid_kpts[6][0], valid_kpts[6][1])
        right_hip = (valid_kpts[12][0], valid_kpts[12][1])
        right_knee = (valid_kpts[14][0], valid_kpts[14][1])
        
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        result['hip_angles']['right'] = right_hip_angle
        
        if right_hip_angle < hip_angle_thres:
            crouch_indicators.append(f"右髋弯曲({right_hip_angle:.1f}°)")
            angle_conf = min(1.0, (hip_angle_thres - right_hip_angle) / hip_angle_thres + 0.3)
            angle_confidences.append(angle_conf)
        
        logger.debug(f"右髋角度: {right_hip_angle:.1f}°")
    
    # 判断姿态
    result['crouch_indicators'] = crouch_indicators
    
    if len(crouch_indicators) > 0:
        # 有弯曲指标，判断为蹲下
        status = 'crouching'
        # 置信度基于角度偏差程度和指标数量
        if angle_confidences:
            confidence = min(1.0, sum(angle_confidences) / len(angle_confidences) + len(crouch_indicators) * 0.1)
        else:
            confidence = 0.6
    else:
        # 没有明显弯曲，判断为站立
        status = 'standing'
        confidence = 0.8  # 基础置信度
    
    result.update({
        'status': status,
        'confidence': confidence,
        'analysis_valid': True
    })
    
    logger.debug(f"姿态分析: {status} (置信度: {confidence:.2f}, 指标: {crouch_indicators})")
    return result

def detect_pose_status(detection_results, opt):
    """
    从检测结果中分析人体姿态状态
    Analyze pose status from detection results
    
    Args:
        detection_results: YOLO检测结果列表，格式为 (class_id, score, x1, y1, x2, y2, kpts)
        opt: 命令行参数
    
    Returns:
        list: 姿态分析结果列表
    """
    pose_results = []
    person_class_id = 0  # COCO数据集中person的class_id为0
    
    for detection in detection_results:
        class_id, score, x1, y1, x2, y2, kpts = detection
        
        # 只处理人的检测结果，且置信度要高于阈值
        if class_id == person_class_id and score >= opt.score_thres:
            # 分析姿态
            pose_analysis = analyze_pose_status(
                kpts, 
                opt.knee_angle_thres, 
                opt.hip_angle_thres,
                opt.min_kpt_conf
            )
            
            # 创建结果，不管姿态分析是否有效都要显示
            pose_result = {
                'bbox': (x1, y1, x2, y2),
                'detection_score': score,
                'keypoints': kpts,
                'pose_analysis': pose_analysis,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2)
            }
            pose_results.append(pose_result)
            
            # 只有姿态分析有效时才输出姿态信息
            if pose_analysis['analysis_valid']:
                indicators_str = ", ".join(pose_analysis['crouch_indicators']) if pose_analysis['crouch_indicators'] else "无弯曲"
                logger.info(f"🧍 检测到人体姿态: {pose_analysis['status']} "
                           f"(检测置信度={score:.3f}, 姿态置信度={pose_analysis['confidence']:.3f}, "
                           f"指标={indicators_str})")
            else:
                logger.info(f"👤 检测到人体 (检测置信度={score:.3f}, 关键点={pose_analysis['valid_keypoints_count']}, 姿态分析无效)")
    
    return pose_results

def get_robot_command(pose_results, image_width, image_height, 
                      forward_zone_ratio=0.3, dead_zone_ratio=0.3, 
                      min_target_size_ratio=0.15):
    """
    根据姿态检测结果生成机器人控制指令
    Generate robot control commands based on pose detection results
    
    Args:
        pose_results: 姿态检测结果列表
        image_width: 图像宽度
        image_height: 图像高度
        forward_zone_ratio: 前进区域比例（中心区域范围）
        dead_zone_ratio: 水平死区比例（左右移动的死区）
        min_target_size_ratio: 最小目标大小比例（判断是否足够近）
    
    Returns:
        tuple: (command, target_person_info, command_reason)
        command: 'left', 'right', 'forward', 'rotate' 四种指令之一
        target_person_info: 目标人物信息，None表示无目标
        command_reason: 指令原因说明
    """
    # 筛选蹲着的人（只选择姿态分析有效的）
    crouching_persons = [
        person for person in pose_results 
        if person['pose_analysis']['analysis_valid'] and person['pose_analysis']['status'] == 'crouching'
    ]
    
    # 如果没有检测到蹲着的人，原地旋转找人
    if not crouching_persons:
        # 检查是否有其他检测到的人（但不是蹲着的）
        all_persons = [person for person in pose_results if person['detection_score'] >= 0.3]
        if all_persons:
            reason = f"检测到{len(all_persons)}个人，但都不是蹲着的，继续旋转寻找蹲着的目标"
        else:
            reason = "未检测到任何人，旋转寻找目标"
        
        logger.info(f"🔄 机器人指令: 原地旋转 ({reason})")
        return 'rotate', None, reason
    
    # 选择最佳目标：优先选择姿态置信度最高的蹲着的人
    if len(crouching_persons) > 1:
        target_person = max(crouching_persons, key=lambda x: x['pose_analysis']['confidence'])
        logger.info(f"🎯 检测到{len(crouching_persons)}个蹲着的目标，选择置信度最高的")
    else:
        target_person = crouching_persons[0]
    
    # 获取目标信息
    x1, y1, x2, y2 = target_person['bbox']
    target_center_x = target_person['center'][0]
    target_center_y = target_person['center'][1]
    
    # 计算目标大小（用于判断距离）
    target_width = x2 - x1
    target_height = y2 - y1
    target_size_ratio = (target_width * target_height) / (image_width * image_height)
    
    # 计算区域边界
    image_center_x = image_width // 2
    image_center_y = image_height // 2
    
    # 水平死区（左右移动的死区）
    horizontal_dead_zone = image_width * dead_zone_ratio / 2
    left_boundary = image_center_x - horizontal_dead_zone
    right_boundary = image_center_x + horizontal_dead_zone
    
    # 前进区域（中心区域）
    forward_zone = image_width * forward_zone_ratio / 2
    forward_left = image_center_x - forward_zone
    forward_right = image_center_x + forward_zone
    
    # 决策逻辑
    pose_conf = target_person['pose_analysis']['confidence']
    detection_conf = target_person['detection_score']
    
    # 1. 首先判断目标是否在前进区域内
    if forward_left <= target_center_x <= forward_right:
        # 目标在中心区域，检查是否足够近
        if target_size_ratio >= min_target_size_ratio:
            # 目标足够大（近），继续前进
            reason = f"目标在中心且足够近(大小比例:{target_size_ratio:.3f}), 继续前进"
            command = 'forward'
        else:
            # 目标在中心但还不够近，前进接近
            reason = f"目标在中心但距离较远(大小比例:{target_size_ratio:.3f}), 前进接近"
            command = 'forward'
    else:
        # 2. 目标不在前进区域，需要调整方向
        if target_center_x < left_boundary:
            # 目标在左侧，向左移动
            reason = f"目标在左侧(x={target_center_x}, 左边界={left_boundary:.0f}), 向左调整"
            command = 'left'
        elif target_center_x > right_boundary:
            # 目标在右侧，向右移动
            reason = f"目标在右侧(x={target_center_x}, 右边界={right_boundary:.0f}), 向右调整"
            command = 'right'
        else:
            # 目标在死区内，前进
            reason = f"目标在水平死区内(x={target_center_x}), 前进接近"
            command = 'forward'
    
    logger.info(f"🤖 机器人指令: {command.upper()} - {reason}")
    logger.debug(f"   目标位置: ({target_center_x}, {target_center_y})")
    logger.debug(f"   目标大小: {target_width}x{target_height} (比例:{target_size_ratio:.3f})")
    logger.debug(f"   姿态置信度: {pose_conf:.3f}, 检测置信度: {detection_conf:.3f}")
    
    return command, target_person, reason

def get_robot_direction_command(pose_results, image_width, target_status='crouching', dead_zone_ratio=0.3):
    """
    向后兼容的函数，调用新的get_robot_command函数
    Backward compatible function that calls the new get_robot_command function
    
    Args:
        pose_results: 姿态检测结果列表
        image_width: 图像宽度
        target_status: 目标姿态 (保持兼容性，实际只处理'crouching')
        dead_zone_ratio: 中心死区比例
    
    Returns:
        tuple: (direction, target_person_info) - 保持原有接口兼容
    """
    # 使用默认图像高度（如果没有提供）
    image_height = int(image_width * 0.75)  # 假设4:3比例
    
    command, target_person, reason = get_robot_command(
        pose_results, image_width, image_height, 
        dead_zone_ratio=dead_zone_ratio
    )
    
    # 映射新指令到旧指令格式
    if command == 'rotate':
        direction = 'stop'  # 旧版本用stop表示
    elif command == 'forward':
        direction = 'stop'  # 旧版本没有forward，映射为stop
    else:
        direction = command  # left, right保持不变
    
    return direction, target_person

def draw_detection(img: np.array, 
                   bbox: tuple[int, int, int, int],
                   score: float, 
                   class_id: int) -> None:
    """
    绘制检测框和标签
    Draw detection bounding box and label
    
    Args:
        img: 输入图像
        bbox: 检测框坐标 (x1, y1, x2, y2)
        score: 检测置信度
        class_id: 类别ID
    """
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id % 20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_keypoints_and_skeleton(img, kpts, min_conf=0.15):
    """
    绘制关键点和骨架
    Draw keypoints and skeleton
    
    Args:
        img: 输入图像
        kpts: 关键点列表 [(x, y, conf), ...]
        min_conf: 最小置信度阈值
    """
    valid_kpts = get_valid_keypoints(kpts, min_conf)
    logger.debug(f"绘制关键点: 总共{len(kpts)}个，有效{len(valid_kpts)}个 (阈值:{min_conf})")
    
    # 绘制骨架连接
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx in valid_kpts and end_idx in valid_kpts:
            start_point = (int(valid_kpts[start_idx][0]), int(valid_kpts[start_idx][1]))
            end_point = (int(valid_kpts[end_idx][0]), int(valid_kpts[end_idx][1]))
            cv2.line(img, start_point, end_point, (0, 255, 0), 2)
    
    # 绘制关键点
    for idx, (x, y, conf) in valid_kpts.items():
        x, y = int(x), int(y)
        # 根据关键点类型使用不同颜色
        if idx in [1, 2]:  # 头部 (眼睛)
            color = (255, 0, 0)  # 蓝色
        elif idx in [11, 12]:  # 腰部 (髋)
            color = (0, 255, 0)  # 绿色
        elif idx in [15, 16]:  # 脚部 (踝)
            color = (0, 0, 255)  # 红色
        else:
            color = (255, 255, 0)  # 黄色
        
        cv2.circle(img, (x, y), 4, color, -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)

def draw_angle_indicators(img, kpts, min_conf=0.15, knee_angle_thres=165, hip_angle_thres=160):
    """
    绘制角度指示器（辅助线）
    Draw angle indicators (helper lines)
    
    Args:
        img: 输入图像
        kpts: 关键点列表 [(x, y, conf), ...]
        min_conf: 最小置信度阈值
        knee_angle_thres: 膝盖角度阈值
        hip_angle_thres: 髋部角度阈值
    """
    valid_kpts = get_valid_keypoints(kpts, min_conf)
    
    # 绘制左膝盖角度辅助线
    if 11 in valid_kpts and 13 in valid_kpts and 15 in valid_kpts:
        left_hip = (int(valid_kpts[11][0]), int(valid_kpts[11][1]))
        left_knee = (int(valid_kpts[13][0]), int(valid_kpts[13][1]))
        left_ankle = (int(valid_kpts[15][0]), int(valid_kpts[15][1]))
        
        angle = calculate_angle(left_hip, left_knee, left_ankle)
        color = (0, 0, 255) if angle < knee_angle_thres else (0, 255, 0)  # 红色=弯曲，绿色=直立
        
        # 绘制角度线
        cv2.line(img, left_hip, left_knee, color, 2)
        cv2.line(img, left_knee, left_ankle, color, 2)
        
        # 在膝盖位置标注角度
        cv2.putText(img, f"{angle:.0f}°", 
                   (left_knee[0] + 10, left_knee[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 绘制右膝盖角度辅助线
    if 12 in valid_kpts and 14 in valid_kpts and 16 in valid_kpts:
        right_hip = (int(valid_kpts[12][0]), int(valid_kpts[12][1]))
        right_knee = (int(valid_kpts[14][0]), int(valid_kpts[14][1]))
        right_ankle = (int(valid_kpts[16][0]), int(valid_kpts[16][1]))
        
        angle = calculate_angle(right_hip, right_knee, right_ankle)
        color = (0, 0, 255) if angle < knee_angle_thres else (0, 255, 0)
        
        cv2.line(img, right_hip, right_knee, color, 2)
        cv2.line(img, right_knee, right_ankle, color, 2)
        cv2.putText(img, f"{angle:.0f}°", 
                   (right_knee[0] + 10, right_knee[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 绘制左髋角度辅助线
    if 5 in valid_kpts and 11 in valid_kpts and 13 in valid_kpts:
        left_shoulder = (int(valid_kpts[5][0]), int(valid_kpts[5][1]))
        left_hip = (int(valid_kpts[11][0]), int(valid_kpts[11][1]))
        left_knee = (int(valid_kpts[13][0]), int(valid_kpts[13][1]))
        
        angle = calculate_angle(left_shoulder, left_hip, left_knee)
        color = (255, 0, 255) if angle < hip_angle_thres else (0, 255, 255)  # 紫色=弯曲，青色=直立
        
        cv2.line(img, left_shoulder, left_hip, color, 1, cv2.LINE_4)
        cv2.line(img, left_hip, left_knee, color, 1, cv2.LINE_4)
        cv2.putText(img, f"{angle:.0f}°", 
                   (left_hip[0] - 30, left_hip[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 绘制右髋角度辅助线
    if 6 in valid_kpts and 12 in valid_kpts and 14 in valid_kpts:
        right_shoulder = (int(valid_kpts[6][0]), int(valid_kpts[6][1]))
        right_hip = (int(valid_kpts[12][0]), int(valid_kpts[12][1]))
        right_knee = (int(valid_kpts[14][0]), int(valid_kpts[14][1]))
        
        angle = calculate_angle(right_shoulder, right_hip, right_knee)
        color = (255, 0, 255) if angle < hip_angle_thres else (0, 255, 255)
        
        cv2.line(img, right_shoulder, right_hip, color, 1, cv2.LINE_4)
        cv2.line(img, right_hip, right_knee, color, 1, cv2.LINE_4)
        cv2.putText(img, f"{angle:.0f}°", 
                   (right_hip[0] + 10, right_hip[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def draw_pose_analysis(img, pose_result):
    """
    绘制姿态分析结果
    Draw pose analysis results
    
    Args:
        img: 输入图像
        pose_result: 姿态分析结果
    """
    bbox = pose_result['bbox']
    pose_analysis = pose_result['pose_analysis']
    
    x1, y1, x2, y2 = bbox
    
    # 根据姿态状态选择颜色
    if pose_analysis['status'] == 'standing':
        status_color = (0, 255, 0)  # 绿色
        status_text = "STANDING"
    elif pose_analysis['status'] == 'crouching':
        status_color = (0, 0, 255)  # 红色
        status_text = "CROUCHING"
    else:
        status_color = (128, 128, 128)  # 灰色
        status_text = "UNKNOWN"
    
    # 绘制姿态状态框
    cv2.rectangle(img, (x1, y1), (x2, y2), status_color, 3)
    
    # 绘制姿态信息
    info_lines = [
        f"{status_text}",
        f"Conf: {pose_analysis['confidence']:.2f}",
        f"KPts: {pose_analysis['valid_keypoints_count']}"
    ]
    
    # 添加角度信息
    if pose_analysis['knee_angles']['left'] is not None:
        info_lines.append(f"L_Knee: {pose_analysis['knee_angles']['left']:.0f}°")
    if pose_analysis['knee_angles']['right'] is not None:
        info_lines.append(f"R_Knee: {pose_analysis['knee_angles']['right']:.0f}°")
    if pose_analysis['hip_angles']['left'] is not None:
        info_lines.append(f"L_Hip: {pose_analysis['hip_angles']['left']:.0f}°")
    if pose_analysis['hip_angles']['right'] is not None:
        info_lines.append(f"R_Hip: {pose_analysis['hip_angles']['right']:.0f}°")
    
    for i, line in enumerate(info_lines):
        y_offset = y1 - 50 + i * 15
        if y_offset < 15:
            y_offset = y2 + 15 + i * 15
        cv2.putText(img, line, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

def detect_once(frame, model, opt):
    """
    执行一次检测并返回结果
    Perform one detection and return results
    
    Args:
        frame: 输入图像帧
        model: YOLO模型实例
        opt: 命令行参数
    
    Returns:
        tuple: (command, display_frame, pose_results, target_person, command_reason)
    """
    # 模型推理
    input_tensor = model.preprocess_yuv420sp(frame.copy())
    outputs = model.c2numpy(model.forward(input_tensor))
    results = model.postProcess(outputs)
    
    # 姿态分析
    pose_results = detect_pose_status(results, opt)
    
    # 生成机器人控制指令 (使用新的函数)
    command, target_person, command_reason = get_robot_command(
        pose_results, frame.shape[1], frame.shape[0],
        forward_zone_ratio=0.3, dead_zone_ratio=0.3, min_target_size_ratio=0.15
    )
    
    # 创建显示帧
    display_frame = frame.copy()
    
    # 绘制所有检测结果和关键点
    for detection in results:
        class_id, score, x1, y1, x2, y2, kpts = detection
        if class_id == 0:  # 只绘制人
            draw_detection(display_frame, (x1, y1, x2, y2), score, class_id)
            # 绘制所有检测到的关键点和骨架（不管姿态分析是否成功）
            draw_keypoints_and_skeleton(display_frame, kpts, opt.display_kpt_conf)
            # 绘制角度指示器
            if opt.show_angles:
                draw_angle_indicators(display_frame, kpts, opt.min_kpt_conf, opt.knee_angle_thres, opt.hip_angle_thres)
    
    # 绘制姿态分析结果
    standing_count = 0
    crouching_count = 0
    unknown_count = 0
    
    for pose_result in pose_results:
        # 绘制姿态分析（覆盖在关键点上方）
        draw_pose_analysis(display_frame, pose_result)
        
        # 统计姿态（只统计分析有效的）
        if pose_result['pose_analysis']['analysis_valid']:
            if pose_result['pose_analysis']['status'] == 'standing':
                standing_count += 1
            elif pose_result['pose_analysis']['status'] == 'crouching':
                crouching_count += 1
        else:
            unknown_count += 1
    
    # 计算无法分析姿态的人数
    total_persons = sum(1 for detection in results if detection[0] == 0 and detection[1] >= opt.score_thres)
    analyzed_persons = len(pose_results)
    unknown_count = total_persons - analyzed_persons
    
    # 特别标记目标人物
    if target_person:
        x1, y1, x2, y2 = target_person['bbox']
        center_x, center_y = target_person['center']
        
        # 黄色高亮边框
        cv2.rectangle(display_frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 255), 4)
        # 黄色中心点
        cv2.circle(display_frame, (center_x, center_y), 8, (0, 255, 255), -1)
        # 目标标签
        target_label = "TARGET"
        cv2.putText(display_frame, target_label, (x1, y1-70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # 显示统计信息
    info_y = 30
    line_height = 35
    
    # 姿态统计
    if unknown_count > 0:
        stats_text = f"Standing: {standing_count}, Crouching: {crouching_count}, Unknown: {unknown_count}"
    else:
        stats_text = f"Standing: {standing_count}, Crouching: {crouching_count}"
    cv2.putText(display_frame, stats_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 机器人指令 - 根据指令类型使用不同颜色
    info_y += line_height
    if command == 'forward':
        direction_color = (0, 255, 0)  # 绿色 - 前进
    elif command == 'rotate':
        direction_color = (255, 0, 255)  # 紫色 - 旋转
    elif command in ['left', 'right']:
        direction_color = (0, 165, 255)  # 橙色 - 左右移动
    else:
        direction_color = (128, 128, 128)  # 灰色 - 其他
    
    direction_text = f"Robot Command: {command.upper()}"
    cv2.putText(display_frame, direction_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, direction_color, 2)
    
    # 显示指令原因
    info_y += line_height - 10
    reason_text = f"Reason: {command_reason[:50]}..." if len(command_reason) > 50 else f"Reason: {command_reason}"
    cv2.putText(display_frame, reason_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # 绘制区域指示器
    image_width = frame.shape[1]
    image_height = frame.shape[0]
    image_center_x = image_width // 2
    
    # 绘制前进区域（中心区域）
    forward_zone = image_width * 0.3 / 2
    forward_left = int(image_center_x - forward_zone)
    forward_right = int(image_center_x + forward_zone)
    
    # 绘制水平死区
    dead_zone_width = image_width * 0.3 / 2
    left_boundary = int(image_center_x - dead_zone_width)
    right_boundary = int(image_center_x + dead_zone_width)
    
    # 绘制区域边界线
    cv2.line(display_frame, (forward_left, 0), (forward_left, image_height), (0, 255, 0), 2)  # 前进区域左边界（绿色）
    cv2.line(display_frame, (forward_right, 0), (forward_right, image_height), (0, 255, 0), 2)  # 前进区域右边界（绿色）
    cv2.line(display_frame, (left_boundary, 0), (left_boundary, image_height), (255, 255, 0), 1)  # 死区左边界（黄色）
    cv2.line(display_frame, (right_boundary, 0), (right_boundary, image_height), (255, 255, 0), 1)  # 死区右边界（黄色）
    cv2.line(display_frame, (image_center_x, 0), (image_center_x, image_height), (255, 255, 255), 1)  # 中心线（白色）
    
    # 添加区域标签
    cv2.putText(display_frame, "FORWARD ZONE", (forward_left + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(display_frame, "LEFT", (10, image_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(display_frame, "RIGHT", (image_width - 80, image_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # 添加操作提示和图例
    help_text = "Press 'q' to quit, 's' to save frame, 'i' for camera info"
    cv2.putText(display_frame, help_text, (10, image_height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 添加指令图例
    legend_text = f"Commands: GREEN=Forward, PURPLE=Rotate, ORANGE=Left/Right"
    cv2.putText(display_frame, legend_text, (10, image_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 添加角度颜色图例
    angle_legend = f"Angles: Red(<{opt.knee_angle_thres:.0f}°)=Bent, Green=Straight, Purple(<{opt.hip_angle_thres:.0f}°)=Hip Bent"
    cv2.putText(display_frame, angle_legend, (10, image_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return command, display_frame, pose_results, target_person, command_reason

def robot_interface(frame, model, opt):
    """
    简化的机器人接口函数 - 专门用于与机器人程序对接
    Simplified robot interface function - specifically for robot program integration
    
    Args:
        frame: 输入图像帧 (OpenCV format)
        model: YOLO模型实例
        opt: 命令行参数或配置对象
    
    Returns:
        dict: 机器人控制信息
        {
            'command': str,           # 'left', 'right', 'forward', 'rotate'
            'confidence': float,      # 指令置信度 (0-1)
            'target_detected': bool,  # 是否检测到蹲着的目标
            'target_info': dict,      # 目标信息 (如果有)
            'reason': str            # 指令原因
        }
    """
    try:
        # 执行检测
        command, _, pose_results, target_person, command_reason = detect_once(frame, model, opt)
        
        # 计算指令置信度
        if target_person and target_person['pose_analysis']['analysis_valid']:
            # 有有效目标时，置信度基于姿态分析置信度
            confidence = target_person['pose_analysis']['confidence']
            target_detected = True
            target_info = {
                'position': target_person['center'],
                'bbox': target_person['bbox'],
                'detection_confidence': target_person['detection_score'],
                'pose_confidence': target_person['pose_analysis']['confidence'],
                'pose_status': target_person['pose_analysis']['status']
            }
        else:
            # 无目标时，根据指令类型设定置信度
            if command == 'rotate':
                confidence = 0.8  # 旋转指令比较确定
            else:
                confidence = 0.5  # 其他情况保守估计
            target_detected = False
            target_info = None
        
        return {
            'command': command,
            'confidence': confidence,
            'target_detected': target_detected,
            'target_info': target_info,
            'reason': command_reason
        }
        
    except Exception as e:
        logger.error(f"❌ 机器人接口执行出错: {e}")
        # 返回安全的默认值
        return {
            'command': 'rotate',
            'confidence': 0.0,
            'target_detected': False,
            'target_info': None,
            'reason': f'检测出错: {str(e)}'
        }

def create_robot_controller(model_path, **kwargs):
    """
    创建机器人控制器的便捷函数
    Convenience function to create a robot controller
    
    Args:
        model_path: YOLO模型路径
        **kwargs: 其他配置参数
    
    Returns:
        tuple: (model, opt) - 可用于robot_interface函数
    """
    from argparse import Namespace
    
    # 默认配置
    default_config = {
        'classes_num': 1,
        'nms_thres': 0.7,
        'score_thres': 0.25,
        'reg': 16,
        'nkpt': 17,
        'kpt_conf_thres': 0.5,
        'strides': [8, 16, 32],
        'knee_angle_thres': 140,
        'hip_angle_thres': 140,
        'min_kpt_conf': 0.2,
        'min_valid_kpts': 3,
        'display_kpt_conf': 0.1,
        'show_angles': True
    }
    
    # 合并用户配置
    config = {**default_config, **kwargs}
    config['model_path'] = model_path
    
    # 创建命名空间对象
    opt = Namespace(**config)
    
    # 导入并创建模型
    from model.yolo.model import Ultralytics_YOLO_Pose_Bayese_YUV420SP
    model = Ultralytics_YOLO_Pose_Bayese_YUV420SP(
        model_path=opt.model_path,
        classes_num=opt.classes_num,
        nms_thres=opt.nms_thres,
        score_thres=opt.score_thres,
        reg=opt.reg,
        strides=opt.strides,
        nkpt=opt.nkpt
    )
    
    logger.info("🤖 机器人控制器创建成功")
    return model, opt

# 使用示例代码（注释掉，仅供参考）
"""
使用示例 / Usage Example:

# 1. 创建机器人控制器
model, opt = create_robot_controller('path/to/your/model.bin')

# 2. 在主循环中使用
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 获取机器人控制指令
    robot_cmd = robot_interface(frame, model, opt)
    
    # 根据指令控制机器人
    command = robot_cmd['command']
    confidence = robot_cmd['confidence']
    
    if confidence > 0.5:  # 只有在置信度足够高时才执行
        if command == 'left':
            # 控制机器人向左
            print("机器人向左转")
        elif command == 'right':
            # 控制机器人向右
            print("机器人向右转")
        elif command == 'forward':
            # 控制机器人向前
            print("机器人向前移动")
        elif command == 'rotate':
            # 控制机器人原地旋转
            print("机器人原地旋转")
    
    print(f"指令: {command}, 置信度: {confidence:.2f}, 原因: {robot_cmd['reason']}")

cap.release()
"""