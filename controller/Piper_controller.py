import sys
sys.path.append("./")

from controller.arm_controller import ArmController

from multiprocessing import Process, Event

from piper_sdk import *
import numpy as np
import time

'''
Piper base code from:
https://github.com/agilexrobotics/piper_sdk.git
'''

class PiperController(ArmController):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.controller_type = "user_controller"
        self.controller = None
    
    def set_up(self, can:str):
        piper = C_PiperInterface_V2(can)
        piper.ConnectPort()
        piper.EnableArm(7)
        enable_fun(piper=piper)
        self.controller = piper

    def reset(self, start_state):
        try:
            self.set_joint(start_state)
        except e:
            print(f"reset error: {e}")
        return

    # 返回单位为米
    def get_state(self):
        state = {}
        eef = self.controller.GetArmEndPoseMsgs()
        joint = self.controller.GetArmJointMsgs()
        
        state["joint"] = np.array([joint.joint_state.joint_1, joint.joint_state.joint_2, joint.joint_state.joint_3,\
                                   joint.joint_state.joint_4, joint.joint_state.joint_5, joint.joint_state.joint_6]) * 0.001 / 180 * 3.1415926
        state["qpos"] = np.array([eef.end_pose.X_axis, eef.end_pose.Y_axis, eef.end_pose.Z_axis, \
                                  eef.end_pose.RX_axis, eef.end_pose.RY_axis, eef.end_pose.RZ_axis]) * 0.001 / 1000
        state["gripper"] = self.controller.GetArmGripperMsgs().gripper_state.grippers_angle * 0.001 / 70
        return state

    # All returned values are expressed in meters,if the value represents an angle, it is returned in radians
    def set_position(self, position):
        x, y, z, rx, ry, rz = position*1000*1000
        x, y, z, rx, ry, rz = int(x), int(y), int(z), int(rx), int(ry), int(rz)

        self.controller.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.controller.EndPoseCtrl(x, y, z, rx, ry, rz)
    
    def set_joint(self, joint):
        factor = 57295.7795 #1000*180/3.1415926
        joint_0 = round(joint[0]*factor)
        joint_1 = round(joint[1]*factor)
        joint_2 = round(joint[2]*factor)
        joint_3 = round(joint[3]*factor)
        joint_4 = round(joint[4]*factor)
        joint_5 = round(joint[5]*factor)
        self.controller.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.controller.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

    # The input gripper value is in the range [0, 1], representing the degree of opening.
    def set_gripper(self, gripper):
        gripper = int(gripper * 70 * 1000)
        self.controller.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.controller.GripperCtrl(gripper, 1000, 0x01, 0)

    def __del__(self):
        try:
            if hasattr(self, 'controller'):
                # Add any necessary cleanup for the arm controller
                pass
        except:
            pass
        
    def _loop(self, stop_event):
        time_interval = 1
        while not stop_event.is_set():
            self.move({"joint": [0., 0.4, -0.85, 0.0, 0.6, 0.0]})
            time.sleep(time_interval)
            
            self.move({"joint": [0., 0.4, -0.85, 0.0, 1.5, 0.0]})
            time.sleep(time_interval)
    
    def start(self):
        self.stop_event = Event()
        self.process = Process(target=self._loop, args=(self.stop_event,))
        self.process.start()
    
    def stop(self):
        if self.process is not None:
            self.stop_event.set()
            self.process.join()
            self.process.close()
        
def enable_fun(piper:C_PiperInterface_V2):
    enable_flag = False
    timeout = 5
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("enable flag:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,1000,0x01, 0)

        print("--------------------")
        if elapsed_time > timeout:
            print("time out....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("time out, exit!")
        exit(0)

if __name__=="__main__":
    controller = PiperController("test_piper")
    controller.set_up("can1")
    controller.set_collect_info(["gripper","qpos","joint"])

    # while True:
    #     print(controller.get()["joint"])
    #     time.sleep(0.1)
    controller.move({"joint": [ 0.,0., 0., 0., 0.,  0.], 
                        "gripper":0.0})
    time.sleep(5)

    # controller.move({"gripper":0.0})
    
    # print("start")
    # controller.start()

    # controller.stop()

