# SO-101 Robot Arm Hardware Setup Guide

## 硬件连接与初始化完整指南

---

## 第一部分：硬件检查与准备

### 1.1 组件清单检查

在开始之前，确认您有以下组件：

**必需硬件：**
- [ ] SO-101 机械臂本体（6个舵机已安装）
- [ ] USB转TTL串口适配器（CH340、CP2102或FT232等）
- [ ] 12V 电源适配器（推荐5A以上）
- [ ] 电源线和连接线
- [ ] USB数据线（连接电脑）

**可选硬件：**
- [ ] 舵机调试板（用于单独测试舵机）
- [ ] 万用表（检查电源电压）
- [ ] 备用舵机（防止损坏）

### 1.2 舵机类型识别

SO-101通常使用以下舵机之一：
- **Feetech SCS系列**（如SCS0009、SCS15）
- **Dynamixel兼容舵机**（如XL330）
- **其他串行总线舵机**

**检查方法：**
1. 查看舵机侧面的型号标签
2. 检查舵机连接线颜色：
   - 红色：VCC (电源+)
   - 黑色/棕色：GND (地)
   - 黄色/白色：DATA (数据)

---

## 第二部分：电气连接

### 2.1 接线拓扑

SO-101通常采用**串行总线连接**：

```
[电脑USB] ←→ [USB转TTL] ←→ [控制板] ←→ [舵机1] ←→ [舵机2] ←→ ... ←→ [舵机6]
                                  ↑
                            [12V电源]
```

### 2.2 详细接线步骤

**步骤1：连接电源**
1. 确认电源适配器输出为12V DC
2. 检查极性：中心正极(+)，外圈负极(-)
3. **暂时不要接通电源**

**步骤2：连接USB转TTL适配器**

常见接线方式：
```
USB转TTL          控制板/舵机总线
---------        ----------------
VCC (5V)    →    不连接（舵机由12V供电）
GND         →    GND (黑线)
TXD         →    DATA (黄/白线)
RXD         →    DATA (黄/白线) [某些协议需要短接]
```

**重要说明：**
- 某些舵机协议（如Feetech SCS）使用**半双工通信**，TXD和RXD需要**通过电阻短接**
- Dynamixel协议可能需要专用适配器（如U2D2）

**步骤3：连接舵机串联线**
1. 按照从底座到末端的顺序连接：
   - 舵机1：Shoulder Pan (底座旋转)
   - 舵机2：Shoulder Lift (肩部抬升)
   - 舵机3：Elbow Flex (肘部弯曲)
   - 舵机4：Wrist Flex (腕部弯曲)
   - 舵机5：Wrist Roll (腕部旋转)
   - 舵机6：Gripper (夹爪)

2. 每个舵机有两个接口（IN和OUT），形成菊花链

### 2.3 安全检查清单

**⚠️ 接通电源前必须检查：**
- [ ] 电源电压正确（12V）
- [ ] 极性正确（正负不能接反）
- [ ] 所有舵机连接牢固
- [ ] 没有短路或裸露的线头
- [ ] 机械臂周围没有障碍物
- [ ] 人员远离机械臂运动范围

---

## 第三部分：软件环境配置

### 3.1 Windows系统配置

**安装驱动：**
1. 插入USB转TTL适配器
2. Windows应自动识别（如果不行，手动安装驱动）：
   - CH340: [下载地址](http://www.wch.cn/downloads/CH341SER_EXE.html)
   - CP2102: [下载地址](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers)
   - FT232: [下载地址](https://ftdichip.com/drivers/vcp-drivers/)

**识别COM端口：**
```powershell
# 方法1：设备管理器
# Win+X → 设备管理器 → 端口(COM和LPT)

# 方法2：PowerShell命令
Get-WmiObject Win32_SerialPort | Select-Object Name,DeviceID
```

记录COM端口号（如 `COM3`）

### 3.2 安装LeRobot

**方法1：使用pip安装（推荐）**
```bash
# 确保已安装Python 3.8+
python --version

# 安装LeRobot
pip install lerobot

# 安装额外依赖（用于SO-101）
pip install pyserial
pip install dynamixel-sdk  # 如果使用Dynamixel舵机
```

**方法2：从源码安装（开发版）**
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

### 3.3 安装舵机SDK

根据您的舵机型号安装对应SDK：

**Feetech SCS系列：**
```bash
pip install scservo-sdk
# 或从源码：https://github.com/feetech-rc/scservo_sdk
```

**Dynamixel系列：**
```bash
pip install dynamixel-sdk
```

---

## 第四部分：初始连接测试

### 4.1 串口通信测试

创建测试脚本 `test_connection.py`：

```python
import serial
import serial.tools.list_ports

# 列出所有可用串口
def list_ports():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"Port: {port.device}, Description: {port.description}")

# 测试串口连接
def test_serial(port_name, baudrate=1000000):
    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=baudrate,
            timeout=1
        )
        print(f"✓ 成功连接到 {port_name} at {baudrate} baud")
        ser.close()
        return True
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False

if __name__ == "__main__":
    print("=== 扫描可用串口 ===")
    list_ports()
    
    print("\n=== 测试连接 ===")
    port = input("输入COM端口号 (例如 COM3 或 /dev/ttyUSB0): ")
    test_serial(port)
```

运行测试：
```bash
python test_connection.py
```

### 4.2 舵机ID扫描

创建扫描脚本 `scan_servos.py`：

```python
import serial
import time

def scan_servos_feetech(port, baudrate=1000000):
    """扫描Feetech SCS舵机ID"""
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        time.sleep(0.1)
        
        print("扫描舵机ID (1-253)...")
        found_servos = []
        
        for servo_id in range(1, 254):
            # Feetech SCS PING指令
            # 帧格式: [0xFF, 0xFF, ID, Length, Instruction, Checksum]
            ping_packet = [0xFF, 0xFF, servo_id, 0x02, 0x01]  # PING
            checksum = (~(sum(ping_packet[2:]) % 256)) & 0xFF
            ping_packet.append(checksum)
            
            ser.write(bytes(ping_packet))
            response = ser.read(6)  # 等待响应
            
            if len(response) >= 6 and response[0:2] == b'\xff\xff':
                print(f"✓ 发现舵机 ID: {servo_id}")
                found_servos.append(servo_id)
            
            if servo_id % 50 == 0:
                print(f"  已扫描: {servo_id}/253")
        
        ser.close()
        
        print(f"\n总共发现 {len(found_servos)} 个舵机")
        print(f"ID列表: {found_servos}")
        return found_servos
        
    except Exception as e:
        print(f"扫描失败: {e}")
        return []

if __name__ == "__main__":
    port = input("输入COM端口 (如 COM3): ")
    scan_servos_feetech(port)
```

**预期结果：**
- 应该扫描到6个舵机（ID通常为1-6）
- 如果扫描不到，检查：
  1. 电源是否接通
  2. 波特率是否正确（常见：115200, 1000000）
  3. 接线是否正确

### 4.3 舵机ID配置

如果舵机ID未配置或重复，需要逐个设置：

```python
# set_servo_id.py
import serial
import time

def set_servo_id_feetech(port, old_id, new_id, baudrate=1000000):
    """修改Feetech舵机ID"""
    try:
        ser = serial.Serial(port, baudrate, timeout=0.5)
        time.sleep(0.1)
        
        # 写入ID指令
        # [0xFF, 0xFF, ID, Length, Instruction, Address, Data, Checksum]
        packet = [0xFF, 0xFF, old_id, 0x04, 0x03, 0x05, new_id]  # 0x05是ID地址
        checksum = (~(sum(packet[2:]) % 256)) & 0xFF
        packet.append(checksum)
        
        ser.write(bytes(packet))
        response = ser.read(6)
        
        if len(response) >= 6:
            print(f"✓ 成功将舵机 {old_id} 修改为 {new_id}")
            return True
        else:
            print(f"✗ 修改失败")
            return False
            
        ser.close()
        
    except Exception as e:
        print(f"错误: {e}")
        return False

# 使用方法：
# 1. 只连接一个舵机
# 2. 如果不知道旧ID，使用扫描脚本
# 3. 逐个设置ID为1-6
```

**建议的ID分配：**
```
ID 1: Shoulder Pan (底座)
ID 2: Shoulder Lift (肩部)
ID 3: Elbow Flex (肘部)
ID 4: Wrist Flex (腕部)
ID 5: Wrist Roll (腕旋)
ID 6: Gripper (夹爪)
```

---

## 第五部分：LeRobot初始化

### 5.1 创建机器人配置

创建 `src/002/robot_control/so101_interface.py`：

```python
"""SO-101 robot interface using LeRobot."""

import numpy as np
from typing import Optional, List
import serial
import time


class SO101Robot:
    """Interface for SO-101 robot arm."""
    
    def __init__(self, 
                 port: str,
                 baudrate: int = 1000000,
                 servo_ids: List[int] = [1, 2, 3, 4, 5, 6]):
        """Initialize SO-101 robot.
        
        Args:
            port: Serial port (e.g., 'COM3' or '/dev/ttyUSB0')
            baudrate: Communication baud rate
            servo_ids: List of servo IDs [1-6]
        """
        self.port = port
        self.baudrate = baudrate
        self.servo_ids = servo_ids
        
        # Connect
        self.serial = serial.Serial(port, baudrate, timeout=0.5)
        time.sleep(0.5)
        
        print(f"✓ Connected to SO-101 on {port}")
        
        # Initialize servo parameters
        self._init_servos()
    
    def _init_servos(self):
        """Initialize servo parameters."""
        # Enable torque for all servos
        for servo_id in self.servo_ids:
            self.set_torque_enable(servo_id, True)
        
        print("✓ All servos initialized")
    
    def set_torque_enable(self, servo_id: int, enable: bool):
        """Enable/disable servo torque."""
        # Implementation depends on servo protocol
        pass
    
    def read_position(self, servo_id: int) -> float:
        """Read current position from servo.
        
        Returns:
            Position in radians
        """
        # Send read command
        # Parse response
        # Convert to radians
        pass
    
    def write_position(self, servo_id: int, position: float):
        """Write target position to servo.
        
        Args:
            position: Target position in radians
        """
        # Convert radians to servo units
        # Send write command
        pass
    
    def read_joint_states(self) -> np.ndarray:
        """Read all joint positions.
        
        Returns:
            Array of 6 joint angles in radians
        """
        positions = np.zeros(6)
        for i, servo_id in enumerate(self.servo_ids):
            positions[i] = self.read_position(servo_id)
        return positions
    
    def write_joint_targets(self, positions: np.ndarray):
        """Write target positions to all joints.
        
        Args:
            positions: Array of 6 target angles in radians
        """
        if len(positions) != 6:
            raise ValueError("Expected 6 joint positions")
        
        for i, servo_id in enumerate(self.servo_ids):
            self.write_position(servo_id, positions[i])
    
    def go_to_home(self):
        """Move to home position (all zeros)."""
        print("Moving to home position...")
        home_position = np.zeros(6)
        self.write_joint_targets(home_position)
        time.sleep(2)
        print("✓ At home position")
    
    def close(self):
        """Close connection."""
        # Disable torque
        for servo_id in self.servo_ids:
            self.set_torque_enable(servo_id, False)
        
        self.serial.close()
        print("✓ Connection closed")


# 使用示例
if __name__ == "__main__":
    robot = SO101Robot(port="COM3")
    
    # 读取当前位置
    current_pos = robot.read_joint_states()
    print(f"Current position: {np.rad2deg(current_pos)} degrees")
    
    # 回到零位
    robot.go_to_home()
    
    # 关闭连接
    robot.close()
```

### 5.2 校准和测试

创建校准脚本 `calibrate_robot.py`：

```python
"""SO-101 calibration script."""

import numpy as np
import time
from robot_control.so101_interface import SO101Robot


def calibrate_zero_positions(robot: SO101Robot):
    """Calibrate zero positions for all joints."""
    print("\n=== 零位校准 ===")
    print("请手动将机械臂移动到零位姿态：")
    print("  - 所有关节伸直")
    print("  - 机械臂垂直向上")
    print("  - 夹爪居中")
    
    input("完成后按Enter继续...")
    
    # 读取当前位置作为零位
    zero_positions = robot.read_joint_states()
    print(f"零位读数: {zero_positions}")
    
    # 保存到配置文件
    np.save('configs/zero_positions.npy', zero_positions)
    print("✓ 零位已保存")


def test_single_joint(robot: SO101Robot, joint_id: int):
    """Test single joint movement."""
    print(f"\n=== 测试关节 {joint_id} ===")
    
    # 读取初始位置
    initial_pos = robot.read_position(joint_id)
    print(f"初始位置: {np.rad2deg(initial_pos):.2f}°")
    
    # 小幅度运动
    print("正向运动 10°...")
    robot.write_position(joint_id, initial_pos + np.deg2rad(10))
    time.sleep(1)
    
    print("负向运动 10°...")
    robot.write_position(joint_id, initial_pos - np.deg2rad(10))
    time.sleep(1)
    
    print("返回初始位置...")
    robot.write_position(joint_id, initial_pos)
    time.sleep(1)
    
    print("✓ 测试完成")


def find_joint_limits(robot: SO101Robot, joint_id: int):
    """Find mechanical joint limits."""
    print(f"\n=== 测定关节 {joint_id} 限位 ===")
    print("⚠️ 请注意观察，如有异常立即断电！")
    
    input("按Enter开始...")
    
    # TODO: 实现安全的限位探测
    # 1. 缓慢向一个方向运动
    # 2. 监控电流/负载
    # 3. 检测到阻力增大时停止
    # 4. 记录位置
    
    print("此功能需要谨慎实现，建议参考机械图纸")


if __name__ == "__main__":
    port = input("输入COM端口 (如 COM3): ")
    robot = SO101Robot(port=port)
    
    try:
        # 校准零位
        calibrate_zero_positions(robot)
        
        # 测试每个关节
        for joint_id in range(1, 7):
            test_single_joint(robot, joint_id)
            input(f"关节{joint_id}测试完成，按Enter继续下一个...")
        
    finally:
        robot.close()
```

---

## 第六部分：安全注意事项

### ⚠️ 重要安全规则

**启动前：**
1. 确保机械臂周围至少1米内无障碍物
2. 确保所有人员远离运动范围
3. 准备好紧急断电开关（拔掉电源插头）

**运行中：**
1. 首次测试时速度设置为最低
2. 随时准备按紧急停止
3. 不要在关节限位附近运行
4. 听到异常声音立即停止

**禁止操作：**
1. ❌ 不要在通电时手动移动关节（扭力大时）
2. ❌ 不要超过关节限位运行
3. ❌ 不要突然发送大幅度运动指令
4. ❌ 不要在机械臂运动时触摸

### 6.1 添加安全限制

在您的控制代码中加入：

```python
class SafetyController:
    """Safety wrapper for robot control."""
    
    def __init__(self, robot, max_velocity=0.5, max_acceleration=1.0):
        self.robot = robot
        self.max_velocity = max_velocity  # rad/s
        self.max_acceleration = max_acceleration  # rad/s^2
        self.last_command_time = time.time()
        self.last_positions = robot.read_joint_states()
    
    def safe_move(self, target_positions: np.ndarray):
        """Execute movement with safety checks."""
        current_positions = self.robot.read_joint_states()
        
        # Check joint limits
        if not self._check_limits(target_positions):
            print("⚠️ 目标超出关节限位！")
            return False
        
        # Check maximum step size
        max_step = np.max(np.abs(target_positions - current_positions))
        if max_step > np.deg2rad(30):  # 单步最大30度
            print(f"⚠️ 单步变化过大: {np.rad2deg(max_step):.1f}°")
            return False
        
        # Execute movement
        self.robot.write_joint_targets(target_positions)
        self.last_positions = target_positions
        return True
    
    def _check_limits(self, positions: np.ndarray) -> bool:
        """Check if positions are within limits."""
        # Load limits from config
        lower_limits = np.array([-3.14, -1.57, -2.35, -1.57, -3.14, -1.57])
        upper_limits = np.array([3.14, 1.57, 2.35, 1.57, 3.14, 1.57])
        
        return np.all(positions >= lower_limits) and np.all(positions <= upper_limits)
```

---

## 第七部分：常见问题排查

### 问题1：找不到COM端口
**解决方法：**
- 检查USB线是否插好
- 重新安装驱动程序
- 尝试其他USB端口
- 在设备管理器中查看是否有黄色感叹号

### 问题2：扫描不到舵机
**可能原因：**
- 电源未接通或电压不足
- 波特率不匹配（尝试115200, 57600, 1000000）
- 接线错误（检查DATA线）
- 舵机损坏

### 问题3：舵机抖动或不响应
**解决方法：**
- 检查电源供电能力（至少5A）
- 降低运动速度
- 检查是否有多个ID冲突
- 降低扭力限制

### 问题4：机械臂突然失控
**紧急处理：**
1. 立即拔掉电源插头
2. 检查代码中是否有错误的角度转换
3. 确认关节限位设置正确
4. 降低初始速度重新测试

---

## 第八部分：下一步

完成以上步骤后，您应该能够：
- ✓ 连接并识别机械臂
- ✓ 读取关节位置
- ✓ 发送简单的运动指令
- ✓ 理解安全操作规范

**后续集成到项目：**
1. 将校准数据更新到 `configs/robot_config.yaml`
2. 在 `robot_control/so101_interface.py` 中完善协议实现
3. 创建正运动学验证脚本，对比实际位置与计算位置
4. 测试逆运动学解在真机上的效果

---

## 附录：参考资源

**LeRobot文档：**
- GitHub: https://github.com/huggingface/lerobot
- 文档: https://github.com/huggingface/lerobot/tree/main/docs

**舵机协议文档：**
- Feetech SCS: https://www.feetechrc.com/
- Dynamixel: https://emanual.robotis.com/

**串口调试工具：**
- Windows: Serial Port Monitor, PuTTY
- Linux: minicom, screen

**社区支持：**
- LeRobot Discord/论坛
- ROS/Robotics Stack Exchange

---

**创建日期：** 2026-07-30  
**作者：** Student 002  
**项目：** SO-101 IK Research

如有问题，请参考LeRobot官方文档或提issue到项目仓库。
