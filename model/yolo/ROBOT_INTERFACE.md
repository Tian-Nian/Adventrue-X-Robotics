# 机器人控制接口使用说明

## 概述

这个项目提供了一个基于YOLO姿态检测的机器人控制接口，可以识别蹲着的人并生成相应的机器人控制指令。

## 机器人控制指令

系统支持四种控制指令：

1. **`left`** - 向左转：目标在机器人左侧，需要向左调整方向
2. **`right`** - 向右转：目标在机器人右侧，需要向右调整方向  
3. **`forward`** - 向前移动：目标在前方区域，向前靠近目标
4. **`rotate`** - 原地旋转：未检测到蹲着的目标，原地旋转寻找

## 控制逻辑

- **有蹲着的目标时**：根据目标在画面中的位置决定向左、向右或向前
- **无蹲着的目标时**：原地旋转寻找目标
- **多个目标时**：优先选择姿态置信度最高的目标

## 快速开始

### 1. 简单接口使用

```python
from process import robot_interface, create_robot_controller
import cv2

# 创建机器人控制器
model, opt = create_robot_controller('path/to/your/model.bin')

# 初始化相机
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 获取机器人控制指令
    robot_cmd = robot_interface(frame, model, opt)
    
    # 提取指令信息
    command = robot_cmd['command']        # 'left', 'right', 'forward', 'rotate'
    confidence = robot_cmd['confidence']  # 0.0 - 1.0
    target_detected = robot_cmd['target_detected']  # True/False
    reason = robot_cmd['reason']          # 指令原因说明
    
    # 根据指令控制机器人（需要根据实际机器人接口修改）
    if confidence > 0.5:  # 置信度阈值
        if command == 'left':
            print("🤖 向左转")
            # your_robot.turn_left()
        elif command == 'right':
            print("🤖 向右转")  
            # your_robot.turn_right()
        elif command == 'forward':
            print("🤖 向前移动")
            # your_robot.move_forward()
        elif command == 'rotate':
            print("🤖 原地旋转")
            # your_robot.rotate()
    
    print(f"指令: {command}, 置信度: {confidence:.2f}")

cap.release()
```

### 2. 详细接口使用

```python
from process import robot_interface, create_robot_controller
import cv2

# 创建机器人控制器（可自定义参数）
model, opt = create_robot_controller(
    model_path='path/to/your/model.bin',
    score_thres=0.3,        # 检测置信度阈值
    knee_angle_thres=165,   # 膝盖弯曲角度阈值
    hip_angle_thres=160     # 髋部弯曲角度阈值
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 获取完整的机器人控制信息
    robot_cmd = robot_interface(frame, model, opt)
    
    command = robot_cmd['command']
    confidence = robot_cmd['confidence'] 
    target_detected = robot_cmd['target_detected']
    target_info = robot_cmd['target_info']  # 目标详细信息
    reason = robot_cmd['reason']
    
    print(f"指令: {command}, 置信度: {confidence:.2f}, 原因: {reason}")
    
    if target_detected and target_info:
        pos = target_info['position']
        bbox = target_info['bbox']
        pose_conf = target_info['pose_confidence']
        print(f"目标位置: {pos}, 姿态置信度: {pose_conf:.2f}")
    
    # 控制机器人
    control_robot(command, confidence)

cap.release()

def control_robot(command, confidence):
    """根据指令控制机器人的示例函数"""
    if confidence < 0.5:
        return  # 置信度太低，不执行
        
    if command == 'left':
        # 向左转的实现
        pass
    elif command == 'right':
        # 向右转的实现 
        pass
    elif command == 'forward':
        # 向前移动的实现
        pass
    elif command == 'rotate':
        # 原地旋转的实现
        pass
```

## 返回值说明

`robot_interface()` 函数返回一个字典，包含以下字段：

```python
{
    'command': str,           # 控制指令：'left', 'right', 'forward', 'rotate'
    'confidence': float,      # 指令置信度 (0.0-1.0)
    'target_detected': bool,  # 是否检测到蹲着的目标
    'target_info': dict,      # 目标详细信息（如果有目标）
    'reason': str            # 指令生成原因说明
}
```

### target_info 详细字段

当检测到目标时，`target_info` 包含：

```python
{
    'position': (x, y),              # 目标中心坐标
    'bbox': (x1, y1, x2, y2),       # 目标边界框
    'detection_confidence': float,   # 目标检测置信度
    'pose_confidence': float,        # 姿态分析置信度  
    'pose_status': str              # 姿态状态：'standing' 或 'crouching'
}
```

## 配置参数

可以通过 `create_robot_controller()` 的参数调整检测行为：

```python
model, opt = create_robot_controller(
    model_path='model.bin',
    score_thres=0.25,        # 目标检测置信度阈值
    knee_angle_thres=165,    # 膝盖角度阈值（小于此值认为蹲下）
    hip_angle_thres=160,     # 髋部角度阈值（小于此值认为蹲下）
    min_kpt_conf=0.15,       # 关键点最小置信度
    min_valid_kpts=2         # 最少有效关键点数
)
```

## 注意事项

1. **模型文件**：需要提供合适的YOLO姿态检测模型文件
2. **置信度控制**：建议设置置信度阈值（如0.5）避免误操作
3. **实时性**：根据硬件性能调整检测频率
4. **安全性**：在实际机器人上使用时要考虑安全措施

## 测试与调试

运行原有的可视化程序进行测试：

```bash
python model.py --model-path your_model.bin --camera-id 0
```

按键功能：
- `q`: 退出
- `s`: 保存当前帧  
- `i`: 显示相机信息
- `c`: 显示当前指令状态
- `t`: 显示控制说明 