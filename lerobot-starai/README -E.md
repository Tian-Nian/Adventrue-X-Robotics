# How to use the LeRobot-starai robotic arm in Lerobot

 [LeRobot](https://github.com/huggingface/lerobot/tree/main) is committed to providing models, datasets and tools for real-world robotics in PyTorch. Its aim is to reduce the entry barrier of robotics, enabling everyone to contribute and benefit from sharing datasets and pretrained models. LeRobot integrates cutting-edge methodologies validated for real-world application, centering on imitation learning. It has furnished a suite of pre-trained models, datasets featuring human-gathered demonstrations, and simulation environments, enabling users to commence without the necessity of robot assembly. In the forthcoming weeks, the intention is to augment support for real-world robotics on the most cost-effective and competent robots presently accessible.



##  Products Introduction

1. **Open-Source & Developer-Friendly**
   It is an open-source, developer-friendly 6+1 DoF robotic arm solution from [Fishion Star Technology Limited](https://fashionrobo.com/).
2. **Integration with LeRobot**
   Designed for integration with [LeRobot Platform](https://github.com/huggingface/lerobot) , which provides PyTorch models, datasets, and tools for imitation learning in real-world robotic tasks — including data collection, simulation, training, and deployment.
3. **Comprehensive Learning Resources**
   Provides comprehensive open-source learning resources like assembly and calibration guides, and example custom grasping tasks to assist users in quickly getting started and developing robotic applications.
4. **Compatible with Nvidia**
   Supports deployment on the reComputer Mini J4012 Orin NX 16GB platform.

## Main Features

- Ready to Go — No Assembly Required. Just Unbox and Dive into the World of AI.
- 6+1 Degrees of Freedom and a 470mm Reach — Built for Versatility and Precision.
- Powered by Dual Brushless Bus Servos — Smooth, Silent, and Strong with up to 300g Payload.
- Parallel Gripper with 66mm Maximum Opening — Modular Fingertips for Quick-Replace Flexibility.
- Exclusive Hover Lock Technology — Instantly Freeze Leader Arm at Any Position with a Single Press.



## Specifications

![image-20250709072845215](media/starai/image-20250709072845215.png)

| Item                 | Follower Arm \| Viola                             | Leder Arm \|Violin                                |
| -------------------- | ------------------------------------------------- | ------------------------------------------------- |
| Degrees of Freedom   | 6                                                 | 6+1                                               |
| Reach                | 470mm                                             | 470mm                                             |
| Span                 | 940mm                                             | 940mm                                             |
| Repeatability        | 2mm                                               | -                                                 |
| Working Payload      | 300g (with 70% Reach)                            | -                                                 |
| Servos               | RX8-U50H-M x2<br/>RA8-U25H-M x4<br/>RA8-U26H-M x1 | RX8-U50H-M x2<br/>RA8-U25H-M x4<br/>RA8-U26H-M x1 |
| Parallel Gripper Ki  | √                                                 | -                                                 |
| Wrist Rotate         | Yes                                               | Yes                                               |
| Hold at any Position | Yes                                               | Yes (with handle button)                          |
| Wrist Camera Mount   | √                                                 | -                                                 |
| Works with LeRobot   | √                                                 | √                                                 |
| Works with ROS 2     | √                                                 | /                                                 |
| Works with MoveIt    | √                                                 | /                                                 |
| Works with Gazebo    | √                                                 | /                                                 |
| Communication Hub    | UC-01                                             | UC-01                                             |
| Power Supply         | 12v/120w                                          | 12v/120w                                          |

For more information about the servo, please refer to the link below.

[RA8-U25H-M](https://fashionrobo.com/actuator-u25/23396/)

[RX18-U100H-M](https://fashionrobo.com/actuator-u100/22853/)

[RX8-U50H-M](https://fashionrobo.com/actuator-u50/136/)







## Initial Environment Setup

For Ubuntu X86:

- Ubuntu 22.04
- CUDA 12+
- Python 3.13
- Troch 2.6



## Installation and Debugging

### Install LeRobot

Environments such as pytorch and torchvision need to be installed based on your CUDA.

1. Install Miniconda: For Jetson:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
chmod +x Miniconda3-latest-Linux-aarch64.sh
./Miniconda3-latest-Linux-aarch64.sh
source ~/.bashrc
```

Or, For X86 Ubuntu 22.04:

```bash
mkdir -p ~/miniconda3
cd miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

2.Create and activate a fresh conda environment for lerobot

```bash
conda create -y -n lerobot python=3.10 && conda activate lerobot
```

3.Clone Lerobot:

```bash
https://github.com/servodevelop/lerobot-starai.git
```

4.When using miniconda, install ffmpeg in your environment:

```bash
conda install ffmpeg -c conda-forge
```
This usually installs ffmpeg 7.X for your platform compiled with the libsvtav1 encoder. If libsvtav1 is not supported (check supported encoders with ffmpeg -encoders), you can:

- [On any platform] Explicitly install ffmpeg 7.X using:

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```

5.Install LeRobot with dependencies for the fashionstar  motors:

```bash
cd ~/lerobot && pip install -e ".[starai]"
```
6.Check Pytorch and Torchvision

Since installing the lerobot environment via pip will uninstall the original Pytorch and Torchvision and install the CPU versions of Pytorch and Torchvision, you need to perform a check in Python.

```python
import torch
print(torch.cuda.is_available())
```

If the result is False, you need to reinstall Pytorch and Torchvision according to the [official website tutorial](https://fashionrobo.com/).

### Robotic Arm Unboxing

The robotic arm kit includes:

- Leader arm
- Follower arm
- Controller (handle)
- Parallel gripper
- Installation tools (screws, hex wrench)
- Power supply ×2
- C-clamp ×2
- UC-01 debuging board ×2



UC-01 debuging board switch：

<img src="media/image-20250722141339815.png" alt="image-20250722141339815" style="zoom:25%;" />




https://github.com/user-attachments/assets/56130bd9-21ee-4ae4-9cac-3817ac4d659f






### Configure arm port

Run the following command in the terminal to find USB ports associated to your arms：

```bash
python lerobot/scripts/find_motors_bus_port.py
```

For example：

1. Example output when identifying the leader arm's port (e.g., `/dev/tty.usbmodem575E0031751` on Mac, or possibly `/dev/ttyACM0` on Linux):
2. Example output when identifying the follower arm's port (e.g., `/dev/tty.usbmodem575E0032081`on Mac, or possibly `/dev/ttyACM1` on Linux):

Open-file

```bash
lerobot\lerobot\common\robot_devices\robots\configs.py
```

Use the Ctrl+F to search for starai and locate the following code. Then, you need to modify the port settings of follower_arms and leader_arms to match the actual port settings.





```py
@RobotConfig.register_subclass("starai")
@dataclass
class StaraiRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/starai"
    max_relative_target: int | None = None
    
    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": StaraiMotorsBusConfig(
                port="/dev/ttyUSB1",##### UPDATE HEARE
                interval = 100,								
                motors={
                    # name: (index, model)
                    "joint1": [0, "rx8-u50"],
                    "joint2": [1, "rx8-u50"],
                    "joint3": [2, "rx8-u50"],
                    "joint4": [3, "rx8-u50"],
                    "joint5": [4, "rx8-u50"],
                    "joint6": [5, "rx8-u50"],
                    "gripper": [6, "rx8-u50"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": StaraiMotorsBusConfig(
                port="/dev/ttyUSB0",##### UPDATE HEARE
                interval = 100,								
                motors={
                    # name: (index, model)
                    "joint1": [0, "rx8-u50"],
                    "joint2": [1, "rx8-u50"],
                    "joint3": [2, "rx8-u50"],
                    "joint4": [3, "rx8-u50"],
                    "joint5": [4, "rx8-u50"],
                    "joint6": [5, "rx8-u50"],
                    "gripper": [6, "rx8-u50"],
                },
            ),
        }
    )
```

### Set runtime parameters

Open-file

```bash
lerobot\lerobot\common\robot_devices\robots\configs.py
```

Use the Ctrl + F to search for starai and locate the following code. Then, you need modify the interval setting of follower_arms.

- Description: The faster the follower responds when the time interval becomes smaller, and more stable the follower runs when the time interval becomes larger.
- Value Range: Integer, greater than 50 and less than 2000.

It is recommended to set the interval to 100 (default value) during teleoperation for better responsiveness, and to 1000 during autonomous execution in evaluation phases to ensure more stable motion.

```PY
@RobotConfig.register_subclass("starai")
@dataclass
class StaraiRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/starai"
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": StaraiMotorsBusConfig(
                port="/dev/ttyUSB1",
                interval = 100,								
                motors={
                    # name: (index, model)
                    "joint1": [0, "rx8-u50"],
                    "joint2": [1, "rx8-u50"],
                    "joint3": [2, "rx8-u50"],
                    "joint4": [3, "rx8-u50"],
                    "joint5": [4, "rx8-u50"],
                    "joint6": [5, "rx8-u50"],
                    "gripper": [6, "rx8-u50"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": StaraiMotorsBusConfig(
                port="/dev/ttyUSB0",
                interval = 100,								##### UPDATE HEARE
                motors={
                    # name: (index, model)
                    "joint1": [0, "rx8-u50"],
                    "joint2": [1, "rx8-u50"],
                    "joint3": [2, "rx8-u50"],
                    "joint4": [3, "rx8-u50"],
                    "joint5": [4, "rx8-u50"],
                    "joint6": [5, "rx8-u50"],
                    "gripper": [6, "rx8-u50"],
                },
            ),
        }
    )

```



### Calibrate

Normally, the robotic arm is pre-calibrated in factory and does not require recalibration. If a joint motor is found to remain at a limit position for a long period, please contact us to obtain the calibration file and perform recalibration again.

> [!NOTE]
>
> If the ttyUSB0 serial port cannot be identified, try the following solutions:
>
> List all USB ports.
>
> ```sh
> lsusb
> ```
>
> ![image-20241230112928879-1749511998299-1](./media/starai/image-20241230112928879-1749511998299-1.png)
>
> Once identified, check the information of the ttyusb.
>
> ```sh
> sudo dmesg | grep ttyUSB
> ```
>
> ![image-20241230113058856](./media/starai/image-20241230113058856-1749512093309-2.png)
>
> The last line indicates a disconnection because brltty is occupying the USB. Removing brltty will resolve the issue.
>
> ```sh
> sudo apt remove brltty
> ```
>
> ![image-20241230113211143](./media/starai/image-20241230113211143-1749512102599-4.png)
>
> Finally，use chmod command.
>
> ```sh
> sudo chmod 666 /dev/ttyUSB0
> ```
>
> 



## Teleoperate

https://github.com/user-attachments/assets/23b3aa00-9889-48d3-ae2c-00ad50595e0a

Move the arm to the position shown in the diagram and set it to standby.

![image-20250717064511074](media/image-20250717064511074.png)



Then you are ready to teleoperate your robot (It won't display the cameras)! Run this simple script :

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --robot.cameras='{}' \
  --control.type=teleoperate
```

After the program starts, the Hover Lock Technology remains functional.



## Add cameras

https://github.com/user-attachments/assets/82650b56-96be-4151-9260-2ed6ab8b133f


After inserting your two USB cameras, run the following script to check the port numbers of the cameras. It is important to remember that the camera must not be connected to a USB Hub; instead, it should be plugged directly into the device. The slower speed of a USB Hub may result in the inability to read image data.



```bash
python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras
```

The terminal will print out the following information. For example, the laptop camera is `index 0`, and the USB camera is `index 2`.

```markdown
Mac or X86 Ubuntu detected. Finding available camera indices through scanning all indices from 0 to 60
[...]
Camera found at index 0
Camera found at index 2
[...]
Connecting cameras
OpenCVCamera(0, fps=30.0, width=640, height=480, color_mode=rgb)
OpenCVCamera(2, fps=30.0, width=640, height=480, color_mode=rgb)
Saving images to outputs/images_from_opencv_cameras
Frame: 0000 Latency (ms): 39.52
[...]
Frame: 0046 Latency (ms): 40.07
Images have been saved to outputs/images_from_opencv_cameras
```

You can find the pictures taken by each camera in the `outputs/images_from_opencv_cameras` directory, and confirm the port index information corresponding to the cameras at different positions. Then complete the alignment of the camera parameters in the `lerobot/lerobot/common/robot_devices/robots/configs.py` file.

<img src="./media/starai/image-20250625094612644.png" alt="image-20250625094612644" style="zoom:50%;" /> 

```
@RobotConfig.register_subclass("starai")
@dataclass
class StaraiRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/starai"

    cameras: dict[str, CameraConfig] = field(
        default_factory=*lambda*: {
            "laptop": OpenCVCameraConfig(
                camera_index=2,             ##### UPDATE HEARE
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=0,             ##### UPDATE HEARE
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

​    mock: bool = False
  
```

Then you will be able to display the cameras on your computer while you are teleoperating：

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=teleoperate \
  --control.display_data=true
```

## Record the dataset


https://github.com/user-attachments/assets/8bb25714-783a-4f29-83dd-58b457aed80c




Once you're familiar with teleoperation, you can record your first dataset.

If you want to use the Hugging Face hub features for uploading your dataset and you haven't previously done it, make sure you've logged in using a write-access token, which can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens):

```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Store your Hugging Face repository name in a variable to run these commands:

```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

Record 2 episodes and upload your dataset to the hub:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/starai \
  --control.tags='["starai","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=20 \
  --control.display_data=true \
  --control.push_to_hub=false
```



> [!TIP]
>
> - "If you want to save the data locally (`--control.push_to_hub=false`), replace `--control.repo_id=${HF_USER}/starai` with a custom local folder name, such as `--control.repo_id=starai/starai`. It will then be stored in the system's home directory at `~/.cache/huggingface/lerobot`."
> - If you uploaded your dataset to the hub with `--control.push_to_hub=true`, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset) by copy and paste your repo id.
> - Note: You can resume recording by adding `--control.resume=true` . Also if you didn't push your dataset yet, add `--control.local_files_only=true`. 
> - Press right arrow -> at any time during episode recording to stop early and go to resetting. During resetting, you can also stop early and go to the next episode recording.
> - Press left arrow <- at any time during episode recording or resetting to an earlier stage, you can cancel the current episode, and re-record it.
> - Press ESCAPE ESC at any time during episode recording to end the session early and go straight to video encoding and dataset uploading.
> - Once you're comfortable with data recording, you can create a larger dataset for training. A good starting task is grasping an object at different locations and placing it in a bin. We suggest recording at least 50 episodes, with 10 episodes per location. Keep the cameras fixed and maintain consistent grasping behavior throughout the recordings. Also make sure the object you are manipulating is visible on the camera's. A good rule of thumb is you should be able to do the task yourself by only looking at the camera images.
> - In the following sections, you’ll train your neural network. After achieving reliable grasping performance, you can start introducing more variations during data collection, such as additional grasp locations, different grasping techniques, and altering camera positions.
> - Avoid adding too much variation too quickly, as it may hinder your results.
> - On Linux, if the left and right arrow keys and escape key don't have any effect during data recording, make sure you've set the $DISPLAY environment variable. See [pynput limitations](https://pynput.readthedocs.io/en/latest/limitations.html#linux) for more details.





## Visualize the dataset

The dataset is saved locally. If you upload with `--control.push_to_hub=false` , you can also visualize it locally with:

```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id starai/starai_test \
```

Here, `starai` is the custom `repo_id` name defined when collecting data.



## Replay an episode

```
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=starai/starai \
  --control.episode=0
  --control.local_files_only=true
```



## Train a policy

To train a policy to control your robot, use the `python lerobot/scripts/train.py` script. A few arguments are required. Here is an example command:

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/starai \
  --policy.type=act \
  --output_dir=outputs/train/act_starai \
  --job_name=act_starai \
  --policy.device=cuda \
  --wandb.enable=false
```

If you want to train on a local dataset, make sure the `repo_id` matches the one used during data collection. Training should take several hours. You will find checkpoints in`outputs/train/act_starai_test/checkpoints` .

To resume training from a checkpoint, below is an example command to resume from last checkpoint of the `act_starai` :

```bash
python lerobot/scripts/train.py \
  --config_path=outputs/train/act_starai_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## Evaluate your policy

You can use the `record` function from [`lerobot/scripts/control_robot.py`](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/control_robot.py) , but with a policy checkpoint as input. For instance, run this command to record 10 evaluation episodes:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=starai/eval_act_starai \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.policy.path=outputs/train/act_so101_test/checkpoints/last/pretrained_model
```

As you can see, this is almost identical to the command previously used to record the training dataset. There are only two differences:

1. There is an additional `--control.policy.path` argument which indicates the path to your policy checkpoint (e.g. `outputs/train/eval_act_starai/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `${HF_USER}/eval_act_starai`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `${HF_USER}/eval_act_starai`).

## FAQ

- If you are following this documentation/tutorial, please git clone the recommended GitHub repository `https://github.com/servodevelop/lerobot-starai.git`.

- If you encounter the following error, you need to check whether the robotic arm connected to the corresponding port is powered on and whether the bus servos have any loose or disconnected cables.

  ```bash
  ConnectionError: Read failed due to comunication eror on port /dev/ttyACM0 for group key Present_Position_Shoulder_pan_Shoulder_lift_elbow_flex_wrist_flex_wrist_roll_griper: [TxRxResult] There is no status packet!
  ```

- If the remote control functions normally but the remote control with Camera fails to display the image interface, you can refer to [here](https://github.com/huggingface/lerobot/pull/757/files)

- If you encounter libtiff issues during dataset remote operation, please update the libtiff version.

  ```bash
  conda install libtiff==4.5.0  #for Ubuntu 22.04 is libtiff==4.5.1
  ```

- After executing the [Lerobot Installation](https://wiki.seeedstudio.com/cn/lerobot_so100m/#安装lerobot), the GPU version of pytorch may be automatically uninstalled, so you need to manually install torch-gpu.

- For Jetson, please first install [Pytorch and Torchvsion](https://github.com/Seeed-Projects/reComputer-Jetson-for-Beginners/blob/main/3-Basic-Tools-and-Getting-Started/3.3-Pytorch-and-Tensorflow/README.md#installing-pytorch-on-recomputer-nvidia-jetson) before executing `conda install -y -c conda-forge ffmpeg`, otherwise, when compiling torchvision, an ffmpeg version mismatch issue may occur.

- If the following problem occurs, it means that your computer does not support this video codec format. You need to modify line 134 in the file `lerobot/lerobot/common/datasets /video_utils.py` by changing the value of `vcodec: str = "libsvtav1"` to `libx264` or `libopenh264`. Different computers may require different parameters, so you can try various options. [Issues 705](https://github.com/huggingface/lerobot/issues/705)

  ```bash
  [vost#0:0 @ 0x13207240] Unknown encoder 'libsvtav1' [vost#0:0 @ 0x13207240] Error selecting an encoder Error opening output file /home/han/.cache/huggingface/lerobot/lyhhan/so100_test/videos/chunk-000/observation.images.laptop/episode_000000.mp4. Error opening output files: Encoder not found
  ```

  

- Training 50 sets of ACT data on an 8G 3060 laptop takes approximately 6 hours, while on a 4090 or A100 computer, training 50 sets of data takes about 2–3 hours.

- During data collection, ensure that the camera's position, angle, and environmental lighting remain stable, and minimize capturing excessive unstable backgrounds and pedestrians; otherwise, significant environmental changes during deployment may cause the robotic arm to fail to grasp properly.

- Ensure that the `num-episodes` parameter in the data collection command is set to collect sufficient data, and do not manually pause midway. This is because the mean and variance of the data are calculated only after data collection is complete, which is necessary for training.

- If the program prompts that it cannot read the USB camera image data, please ensure that the USB camera is not connected to a hub. The USB camera must be directly connected to the device to ensure a fast image transmission rate.

## Reference Documentation

seeed Wiki：[How to use the SO10xArm robotic arm in Lerobot | Seeed Studio Wiki]([How to use the SO-ARM100 and SO-ARM101 robotic arm in Lerobot](https://wiki.seeedstudio.com/lerobot_so100m/))

Huggingface Project:[Lerobot](https://github.com/huggingface/lerobot/tree/main)

ACT or ALOHA:[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/)

VQ-BeT:[VQ-BeT: Behavior Generation with Latent Actions](https://sjlee.cc/vq-bet/)

Diffusion Policy:[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)

TD-MPC:[TD-MPC](https://www.nicklashansen.com/td-mpc/)
