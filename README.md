# 机器人举大锤

该项目旨在于观察那些不遵守公共卫生规矩的人, 对他们进行各种物理层面的规劝.

## 使用设备说明
### 移动式巡检机器人:
| **设备类型** | **设备提供方** | **设备型号** |
|------------|---------|----------|
| 算力平台 | 地瓜机器人 | X5 4G |
| 移动底盘 | 松灵机器人 | BUNKER MINI |
| 机械臂 | 松灵机器人 | PIPER |
| 视觉 | seeed robot| 海康usb摄像头 |

### 桌面监管机器人

pass...

## 开始使用
### 配置环境
1. **移动式巡检机器人**
```
配置底盘:
参考https://github.com/agilexrobotics/bunker_ros2/
我们使用的是是humble分支, 因为我们的系统是ubuntu22.04
由于该代码是用于编译ros2的, 所以请根据指南, 我们默认编译目录为~/ros_ws/

# 配置机械臂
参考https://github.com/agilexrobotics/piper_sdk/
我们将其放置于: ~/projects下
由于地瓜板子默认走了can0通讯, 所以我们机械臂和底盘的得分别走can1与can2
```
2. **桌面监管机器人**
```

```

## 启动项目

**需要注意, can线不能接入拓展坞!**

``` bash
# 确认can接口(piper与bunker各自会占用一个can口, RDK X5默认占据can0口, 你需要记住你的对应can信息)
cd ~/projects/piper_sdk/piper_sdk/
bash find_all_can_port.sh 

# 启动小车, 这里小车的can口被设置为can1
cd ~/ros2_ws/
source install/setup.bash
cd src/ugv_sdk/scripts/
bash bringup_can2usb_500k.bash
ros2 launch bunker_base bunker_base.launch.py port_name:="can1"

# 启动机械臂, 我们的机械臂can口信息: Interface can2 is connected to USB port 1-1.4:1.0
cd ~/projects/piper_sdk/piper_sdk/
bash can_activate.sh can2 1000000 1-1.4:1.0
```
如果一切顺利, 您将可以使用`ifconfig`查看到有can1与can2通讯协议.

