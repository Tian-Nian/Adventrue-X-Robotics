import cv2
import numpy as np
from scipy.special import softmax
# from scipy.special import expit as sigmoid
from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API

from time import time
import argparse
import logging 

from argparse import Namespace

# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

class Ultralytics_YOLO_Pose_Bayese_YUV420SP():
    def __init__(self, model_path, classes_num, nms_thres, score_thres, reg, strides, nkpt):
        # 加载BPU的bin模型, 打印相关参数
        # Load the quantized *.bin model and print its parameters
        try:
            begin_time = time()
            self.quantize_model = dnn.load(model_path)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(model_path))
            logger.error("You can download the model file from the following docs: ./models/download.md") 
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        # init
        self.REG = reg
        self.CLASSES_NUM = classes_num
        self.SCORE_THRESHOLD = score_thres
        self.NMS_THRESHOLD = nms_thres
        self.CONF_THRES_RAW = -np.log(1/self.SCORE_THRESHOLD - 1)
        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[2:4]
        self.strides = strides
        self.nkpt = nkpt
        logger.info(f"{self.REG = }, {self.CLASSES_NUM = }")
        logger.info("SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f"%(self.SCORE_THRESHOLD, self.NMS_THRESHOLD))
        logger.info("CONF_THRES_RAW = %.2f"%self.CONF_THRES_RAW)
        logger.info(f"{self.input_H = }, {self.input_W = }")
        logger.info(f"{self.strides = }")
        logger.info(f"{self.nkpt = }")

        # DFL求期望的系数, 只需要生成一次
        # DFL calculates the expected coefficients, which only needs to be generated once.
        self.weights_static = np.array([i for i in range(reg)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        logger.info(f"{self.weights_static.shape = }")

        # anchors, 只需要生成一次
        self.grids = []
        for stride in self.strides:
            assert self.input_H % stride == 0, f"{stride=}, {self.input_H=}: input_H % stride != 0"
            assert self.input_W % stride == 0, f"{stride=}, {self.input_W=}: input_W % stride != 0"
            grid_H, grid_W = self.input_H // stride, self.input_W // stride
            self.grids.append(np.stack([np.tile(np.linspace(0.5, grid_H-0.5, grid_H), reps=grid_H), 
                            np.repeat(np.arange(0.5, grid_W+0.5, 1), grid_W)], axis=0).transpose(1,0))
            logger.info(f"{self.grids[-1].shape = }")

    def preprocess_yuv420sp(self, img):
        RESIZE_TYPE = 0
        LETTERBOX_TYPE = 1
        PREPROCESS_TYPE = LETTERBOX_TYPE
        logger.info(f"PREPROCESS_TYPE = {PREPROCESS_TYPE}")

        begin_time = time()
        self.img_h, self.img_w = img.shape[0:2]
        if PREPROCESS_TYPE == RESIZE_TYPE:
            # 利用resize的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST) # 利用resize重新开辟内存节约一次
            input_tensor = self.bgr2nv12(input_tensor)
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            logger.info("\033[1;31m" + f"pre process(resize) time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        elif PREPROCESS_TYPE == LETTERBOX_TYPE:
            # 利用 letter box 的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            self.x_scale = min(1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w)
            self.y_scale = self.x_scale
            
            if self.x_scale <= 0 or self.y_scale <= 0:
                raise ValueError("Invalid scale factor.")
            
            new_w = int(self.img_w * self.x_scale)
            self.x_shift = (self.input_W - new_w) // 2
            x_other = self.input_W - new_w - self.x_shift
            
            new_h = int(self.img_h * self.y_scale)
            self.y_shift = (self.input_H - new_h) // 2
            y_other = self.input_H - new_h - self.y_shift
            
            input_tensor = cv2.resize(img, (new_w, new_h))
            input_tensor = cv2.copyMakeBorder(input_tensor, self.y_shift, y_other, self.x_shift, x_other, cv2.BORDER_CONSTANT, value=[127, 127, 127])
            input_tensor = self.bgr2nv12(input_tensor)
            logger.info("\033[1;31m" + f"pre process(letter box) time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        else:
            logger.error(f"illegal PREPROCESS_TYPE = {PREPROCESS_TYPE}")
            exit(-1)

        logger.debug("\033[1;31m" + f"pre process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        logger.info(f"y_scale = {self.y_scale:.2f}, x_scale = {self.x_scale:.2f}")
        logger.info(f"y_shift = {self.y_shift:.2f}, x_shift = {self.x_shift:.2f}")
        return input_tensor

    def bgr2nv12(self, bgr_img):
        begin_time = time()
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        logger.debug("\033[1;31m" + f"bgr8 to nv12 time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return nv12

    def forward(self, input_tensor):
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs

    def c2numpy(self, outputs):
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

    def postProcess(self, outputs):
        begin_time = time()
        # reshape
        clses = [outputs[0].reshape(-1, self.CLASSES_NUM), outputs[3].reshape(-1, self.CLASSES_NUM), outputs[6].reshape(-1, self.CLASSES_NUM)]
        bboxes = [outputs[1].reshape(-1, self.REG * 4), outputs[4].reshape(-1, self.REG * 4), outputs[7].reshape(-1, self.REG * 4)]
        kpts = [outputs[2].reshape(-1, self.nkpt * 3), outputs[5].reshape(-1, self.nkpt * 3), outputs[8].reshape(-1, self.nkpt * 3)]

        
        dbboxes, ids, scores, kpts_xy, kpts_score = [], [], [], [], []
        for cls, bbox, stride, grid, kpt in zip(clses, bboxes, self.strides, self.grids, kpts):    
            # score 筛选
            max_scores = np.max(cls, axis=1)
            bbox_selected = np.flatnonzero(max_scores >= self.CONF_THRES_RAW)
            ids.append(np.argmax(cls[bbox_selected, : ], axis=1))
            # 3个Classify分类分支：Sigmoid计算 
            scores.append(1 / (1 + np.exp(-max_scores[bbox_selected])))
            # dist2bbox (ltrb2xyxy)
            ltrb_selected = np.sum(softmax(bbox[bbox_selected,:].reshape(-1, 4, self.REG), axis=2) * self.weights_static, axis=2)
            grid_selected = grid[bbox_selected, :]
            x1y1 = grid_selected - ltrb_selected[:, 0:2]
            x2y2 = grid_selected + ltrb_selected[:, 2:4]
            dbboxes.append(np.hstack([x1y1, x2y2]) * stride)
            # kpts
            kpt = kpt[bbox_selected,:].reshape(-1, 17, 3)
            kpts_xy.append((kpt[:, :, :2] * 2.0 + (grid[bbox_selected,:][:,np.newaxis,:] - 0.5)) * stride)
            kpts_score.append(kpt[:, :, 2:3])

        dbboxes = np.concatenate((dbboxes), axis=0)
        scores = np.concatenate((scores), axis=0)
        ids = np.concatenate((ids), axis=0)
        hw = (dbboxes[:,2:4] - dbboxes[:,0:2])
        xyhw2 = np.hstack([dbboxes[:,0:2], hw])

        kpts_xy = np.concatenate((kpts_xy), axis=0)
        kpts_score  = np.concatenate((kpts_score), axis=0)

        # 分类别nms
        results = []
        for i in range(self.CLASSES_NUM):
            id_indices = ids==i
            indices = cv2.dnn.NMSBoxes(xyhw2[id_indices,:], scores[id_indices], self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
            if len(indices) == 0:
                continue
            for indic in indices:
                x1, y1, x2, y2 = dbboxes[id_indices,:][indic]
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)

                x1 = x1 if x1 > 0 else 0
                x2 = x2 if x2 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                y2 = y2 if y2 > 0 else 0
                x1 = x1 if x1 < self.img_w else self.img_w
                x2 = x2 if x2 < self.img_w else self.img_w
                y1 = y1 if y1 < self.img_h else self.img_h
                y2 = y2 if y2 < self.img_h else self.img_h

                kpts_ = []
                for j in range(self.nkpt):
                    kpt_x = kpts_xy[id_indices,:][indic][j,0]
                    kpt_y = kpts_xy[id_indices,:][indic][j,1]
                    kpt_score = kpts_score[id_indices,:][indic][j,0]

                    kpt_x = int((kpt_x - self.x_shift) / self.x_scale)
                    kpt_y = int((kpt_y - self.y_shift) / self.y_scale)
                    kpt_x = kpt_x if kpt_x > 0 else 0
                    kpt_y = kpt_y if kpt_y > 0 else 0
                    kpt_x = kpt_x if kpt_x < self.img_w else self.img_w
                    kpt_y = kpt_y if kpt_y < self.img_h else self.img_h

                    kpts_.append((kpt_x, kpt_y, kpt_score))

                results.append((i, scores[id_indices][indic], x1, y1, x2, y2, kpts_))
        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return results

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

if __name__ == "__main__":
    import sys
    sys.path.append('./')

    from model.yolo.process import get_args, get_camera_properties, detect_once

    opt = get_args()

    logger.info(opt)

    # 实例化
    model = Ultralytics_YOLO_Pose_Bayese_YUV420SP(
        model_path=opt.model_path,
        classes_num=opt.classes_num,   # default: 1
        nms_thres=opt.nms_thres,       # default: 0.7
        score_thres=opt.score_thres,   # default: 0.25
        reg=opt.reg,                   # default: 16
        strides=opt.strides,           # default: [8, 16, 32]
        nkpt=opt.nkpt                  # default: 17
        )
    
    # 初始化USB相机
    cap = cv2.VideoCapture(opt.camera_id)

    if not cap.isOpened():
        logger.error(f"❌ 无法打开相机设备 {opt.camera_id}")
        exit(1)
        
    actual_width, actual_height, actual_fps, success = get_camera_properties(opt.camera_id)
    
    # 获取实际设置后的参数
    final_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    final_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    final_fps = cap.get(cv2.CAP_PROP_FPS)

    target_width = final_width
    target_height = final_height
    target_fps = final_fps

    # 设置相机参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    logger.info(f"📹 相机设备 {opt.camera_id} 已启动")
    logger.info(f"📐 目标分辨率: {target_width}x{target_height}, 目标帧率: {target_fps}fps")
    logger.info(f"📐 实际分辨率: {final_width}x{final_height}, 实际帧率: {final_fps}fps")
    
    # 检查设置是否成功
    if final_width != target_width or final_height != target_height:
        logger.warning(f"⚠️  相机不支持目标分辨率，使用实际分辨率 {final_width}x{final_height}")
    
    if abs(final_fps - target_fps) > 1:
        logger.warning(f"⚠️  相机不支持目标帧率，使用实际帧率 {final_fps}fps")
    
    logger.info("按 'q' 键退出，按 's' 键保存当前帧，按 'i' 键查看相机信息，按 'c' 键查看当前指令，按 't' 键查看控制说明")
    
    frame_count = 0
    total_inference_time = 0
    last_command = 'rotate'
    
    while True:
        ret, img = cap.read()
        if not ret or img is None:
            logger.error("❌ 无法读取相机画面")
            break
            
        frame_count += 1
        
        # 使用重构后的检测函数
        command, display_frame, pose_results, target_person, command_reason = detect_once(img, model, opt)
        
        # 记录指令变化
        if command != last_command:
            logger.info(f"🔄 指令改变: {last_command} -> {command}")
            logger.info(f"   原因: {command_reason}")
            last_command = command
        
        # 显示结果
        cv2.imshow('YOLO Pose Detection - Standing/Crouching', display_frame)
            
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("🛑 用户退出检测")
            break
        elif key == ord('s'):
            # 保存当前帧
            save_filename = f"captured_frame_{int(time.time())}.jpg"
            cv2.imwrite(save_filename, display_frame)
            logger.info(f"💾 保存当前帧: {save_filename}")
        elif key == ord('i'):
            # 显示详细相机信息
            logger.info("📹 相机详细信息:")
            logger.info(f"   设备ID: {opt.camera_id}")
            logger.info(f"   分辨率: {final_width}x{final_height}")
            logger.info(f"   帧率: {final_fps}fps")
            logger.info(f"   亮度: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
            logger.info(f"   对比度: {cap.get(cv2.CAP_PROP_CONTRAST)}")
            logger.info(f"   饱和度: {cap.get(cv2.CAP_PROP_SATURATION)}")
            logger.info(f"   曝光: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
        elif key == ord('c'):
            # 显示当前指令状态
            logger.info(f"🤖 当前机器人指令: {command.upper()}")
            logger.info(f"   指令原因: {command_reason}")
            if target_person:
                logger.info(f"   目标位置: {target_person['center']}")
                logger.info(f"   目标置信度: 检测={target_person['detection_score']:.3f}, 姿态={target_person['pose_analysis']['confidence']:.3f}")
            else:
                logger.info("   当前无目标")
        elif key == ord('t'):
            # 显示机器人控制说明
            logger.info("🤖 机器人控制说明:")
            logger.info("   FORWARD: 检测到蹲着的人在前进区域，向前移动靠近")
            logger.info("   LEFT: 目标在左侧，向左转动调整方向")
            logger.info("   RIGHT: 目标在右侧，向右转动调整方向")
            logger.info("   ROTATE: 未检测到蹲着的人，原地旋转寻找目标")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 保存最后一帧
    if 'display_frame' in locals():
        cv2.imwrite(opt.img_save_path, display_frame)
    logger.info("\033[1;32m" + f"saved in path: \"./{opt.img_save_path}\"" + "\033[0m")



    """

  python -m model.yolo.model  --model-path source/yolo11n_pose_bayese_640x640_nv12.bin     --camera-id 0     --score-thres 0.3     --nms-thres 0.7     --crouch-ratio-thres 1.8     --min-kpt-conf 0.3     --min-valid-kpts 4     --kpt-conf-thres 0.5
    """

    """
    [RDK_YOLO] [15:17:42.145] [DEBUG] forward time = 10.33 ms
[RDK_YOLO] [15:17:42.146] [DEBUG] c to numpy time = 1.10 ms
[RDK_YOLO] [15:17:42.153] [DEBUG] Post Process time = 6.49 ms
[RDK_YOLO] [15:17:42.154] [DEBUG] 关键点数量不足进行姿态分析: 2
[RDK_YOLO] [15:17:42.155] [DEBUG] 关键点数量不足进行姿态分析: 1
[RDK_YOLO] [15:17:42.175] [INFO] PREPROCESS_TYPE = 1
[RDK_YOLO] [15:17:42.182] [DEBUG] bgr8 to nv12 time = 5.61 ms
[RDK_YOLO] [15:17:42.183] [INFO] pre process(letter box) time = 6.90 ms
[RDK_YOLO] [15:17:42.183] [DEBUG] pre process time = 7.21 ms
[RDK_YOLO] [15:17:42.183] [INFO] y_scale = 1.00, x_scale = 1.00
[RDK_YOLO] [15:17:42.183] [INFO] y_shift = 80.00, x_shift = 0.00
"""