import sys
sys.path.append("./")
import os
import cv2
import time

import rclpy
from rclpy.node import Node

from controller.Bunker_controller import BunkerController
from controller.Piper_controller import PiperController

from model.yolo.process import get_args, get_camera_properties, robot_interface, create_robot_controller

import logging 

logging.basicConfig(
    level = logging.INFO,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

class Robot:
    def __init__(self):
        self.bunker = BunkerController("bunker_mini")
        self.bunker.set_up()
        self.bunker.set_collect_info(["move_velocity"])

        self.piper = PiperController("piper_arm")
        self.piper.set_up("can2")
        self.piper.set_collect_info(["gripper","qpos","joint"])
    
    # def open_gripper(self):
        # self.piper.move({"gripper": 1.0})
        time.sleep(5)
        self.piper.move({"gripper": 0.0})

    def arm_start(self):
        self.piper.start()
    
    def arm_stop(self):
        self.piper.stop()

    def move(self, command):
        if command == 'left':
            print("向左转")
            self.bunker.move({"move_velocity": [0., 0., 0., 0., 0., 0.1]})
            # time.sleep(0.05)
        elif command == 'right':
            print("向右转")  
            self.bunker.move({"move_velocity": [0., 0., 0., 0., 0., -0.1]})
            # time.sleep(0.05)
        elif command == 'forward':
            print("向前移动")
            self.bunker.move({"move_velocity": [0.05, 0., 0., 0., 0., 0.]})
        elif command == 'rotate':
            print("rotate")
            self.bunker.move({"move_velocity": [0., 0., 0., 0., 0., 0.1]})
            # time.sleep(0.05)


def main():
    rclpy.init()

    opt = get_args()
    actual_width, actual_height, actual_fps, success = get_camera_properties(opt.camera_id)

    if not success:
        logger.error(f"❌ 无法获取相机 {opt.camera_id} 的参数")
        return
    
    # 初始化USB相机
    cap = cv2.VideoCapture(opt.camera_id)

    if not cap.isOpened():
        logger.error(f"❌ 无法打开相机设备 {opt.camera_id}")
        return
    
    target_width = actual_width
    target_height = actual_height
    target_fps = actual_fps
    
    # 设置相机参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    # 获取实际设置后的参数
    final_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    final_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    final_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"相机设备 {opt.camera_id} 已启动")
    logger.info(f"目标分辨率: {target_width}x{target_height}, 目标帧率: {target_fps}fps")
    logger.info(f"实际分辨率: {final_width}x{final_height}, 实际帧率: {final_fps}fps")
    
    # 检查设置是否成功
    if final_width != target_width or final_height != target_height:
        logger.warning(f"⚠️  相机不支持目标分辨率，使用实际分辨率 {final_width}x{final_height}")
    
    if abs(final_fps - target_fps) > 1:
        logger.warning(f"⚠️  相机不支持目标帧率，使用实际帧率 {final_fps}fps")
    
    logger.info("按 'q' 键退出，按 's' 键保存当前帧，按 'i' 键查看相机信息")
    
    frame_count = 0
    total_inference_time = 0
    last_direction = 'stop'

    model, opt = create_robot_controller('source/yolo11n_pose_bayese_640x640_nv12.bin')

    robot = Robot()
    robot.open_gripper()
    robot.arm_start()
    try:
        while True:
            # 读取相机帧
            ret, frame = cap.read()
            if not ret:
                logger.error("❌ 无法读取相机帧")
                break
            
            infer_info = robot_interface(frame, model, opt)

            # cv2.imshow('Real-time Crouching Detection', frame)

            # 根据指令控制机器人（需要根据实际机器人接口修改）
            robot.move(infer_info["command"])
            print(infer_info)
            # exit()

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("🛑 用户退出检测")
                break
            elif key == ord('s'):
                # 保存当前帧
                save_filename = f"captured_frame_{int(time())}.jpg"
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

    except KeyboardInterrupt:
        logger.info("检测被中断")
    except Exception as e:
        logger.error(f"检测过程中出现错误: {e}")
    finally:
        # 清理资源
        robot.arm_stop()

        cap.release()
        cv2.destroyAllWindows()
        logger.info("相机资源已释放")
    
if __name__ == "__main__":
    main()