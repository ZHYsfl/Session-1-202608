# SO-101 机械臂硬件连接 - 当前状态检查

## 📋 当前检测结果

**日期：** 2026-07-30  
**检测到的串口：** 12个（全部为蓝牙设备）  
**USB转TTL设备：** ❌ 未检测到

---

## ⚠️ 问题诊断

您的系统当前只检测到蓝牙串口设备，没有发现USB转TTL适配器。这通常意味着：

### 可能原因1：机械臂尚未连接
- USB转TTL适配器还没有插入电脑
- USB线松动或损坏
- 使用的USB端口有问题

**解决方法：**
1. 将USB转TTL适配器插入电脑USB端口
2. 等待5-10秒让系统识别
3. 重新运行测试程序

### 可能原因2：驱动程序未安装
USB转TTL芯片常见型号及驱动下载：

**CH340/CH341芯片**
- 官方驱动：http://www.wch.cn/downloads/CH341SER_EXE.html
- 识别特征：设备名称包含"CH340"或"CH341"

**CP2102芯片**
- 官方驱动：https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers
- 识别特征：设备名称包含"Silicon Labs CP210x"

**FT232芯片**
- 官方驱动：https://ftdichip.com/drivers/vcp-drivers/
- 识别特征：设备名称包含"FTDI"或"FT232"

**安装步骤：**
1. 下载对应芯片的驱动程序
2. 以管理员身份运行安装程序
3. 重启电脑
4. 重新插入USB设备
5. 检查设备管理器

### 可能原因3：SO-101使用特殊连接方式
某些SO-101版本可能：
- 直接使用蓝牙连接（不需要USB转TTL）
- 使用专用控制器（需要特定驱动）
- 通过网络连接（需要配置IP）

---

## 🔍 下一步检查步骤

### 步骤1：检查设备管理器

**Windows操作：**
1. 按 `Win + X`，选择"设备管理器"
2. 展开"端口(COM和LPT)"部分
3. 查找以下设备：
   - USB Serial Port (COMx)
   - USB-SERIAL CH340 (COMx)
   - Silicon Labs CP210x (COMx)
   - USB Serial Converter

**预期结果：**
- 如果看到上述设备 → 驱动已安装，记下COM端口号
- 如果看到黄色感叹号 → 驱动未安装或有问题
- 如果什么都没有 → USB设备未连接

### 步骤2：物理连接检查

**检查清单：**
```
硬件连接：
[ ] USB转TTL适配器已插入电脑USB口
[ ] 适配器的LED灯是否亮起
[ ] USB线完好无损
[ ] 机械臂控制板已连接到适配器
[ ] 12V电源线已连接（但先不要通电）

接线正确性：
[ ] GND（黑线）→ 控制板GND
[ ] TXD（绿/白线）→ 控制板RX或DATA
[ ] RXD（白/黄线）→ 控制板TX或DATA
[ ] VCC（红线）→ 不连接（舵机由12V供电）
```

### 步骤3：确认机械臂型号

**SO-101可能的连接方式：**

**类型A：串行总线舵机**
- 需要USB转TTL适配器
- 舵机通过串行总线连接
- 通信协议：Feetech SCS或Dynamixel

**类型B：蓝牙控制**
- 使用内置蓝牙模块
- 可能使用COM3/COM12等蓝牙端口
- 需要先配对蓝牙设备

**类型C：专用控制器**
- 使用LeRobot官方控制板
- 可能需要专用驱动或软件

**请确认您的SO-101具体型号和连接方式！**

---

## 🚀 根据您的情况选择操作

### 情况1：我还没有连接USB设备

**立即操作：**
1. 找到USB转TTL适配器
2. 插入电脑USB端口
3. 等待驱动自动安装（约10-30秒）
4. 重新运行测试：
   ```bash
   python robot_control/test_connection.py
   ```

### 情况2：已经插入但没有检测到

**故障排查：**
1. 尝试不同的USB端口
2. 检查设备管理器（Win+X → 设备管理器）
3. 查看是否有黄色感叹号
4. 下载并安装对应的驱动程序
5. 重启电脑后重试

### 情况3：SO-101使用蓝牙连接

如果您的机械臂通过蓝牙连接，可以尝试使用检测到的蓝牙端口：

**可用的蓝牙COM端口：**
```
COM3  - 蓝牙设备 (E8EECCDB1569)
COM4  - 蓝牙设备 (通用)
COM6  - 蓝牙设备 (EC538207EFCE)
COM7  - 蓝牙设备 (EC538207EFCE)
COM8  - 蓝牙设备 (通用)
COM9  - 蓝牙设备 (通用)
COM11 - 蓝牙设备 (通用)
COM12 - 蓝牙设备 (4142B0B10284)
COM13 - 蓝牙设备 (通用)
COM14 - 蓝牙设备 (414200FDA6F5)
COM15 - 蓝牙设备 (通用)
COM16 - 蓝牙设备 (41428C56C856)
```

**测试蓝牙连接：**
```bash
# 手动指定端口测试（非交互模式）
python -c "
import serial
import time

port = 'COM3'  # 选择一个端口
try:
    ser = serial.Serial(port, 1000000, timeout=1)
    print(f'✓ 成功连接到 {port}')
    ser.close()
except Exception as e:
    print(f'✗ 连接失败: {e}')
"
```

### 情况4：需要查阅机械臂文档

**建议：**
1. 查看SO-101的用户手册或快速入门指南
2. 确认机械臂的具体型号和连接方式
3. 访问LeRobot官方文档：https://github.com/huggingface/lerobot
4. 查看机械臂包装盒或控制板上的型号标识

---

## 📖 参考文档

完成硬件连接后，请参考：
- **HARDWARE_SETUP.md** - 完整的硬件连接教程
- **QUICK_REFERENCE.md** - 快速参考指南

---

## ❓ 常见问题

**Q: 驱动安装后还是找不到设备？**
A: 尝试重启电脑，或者使用不同的USB端口

**Q: 蓝牙端口可以用吗？**
A: 如果SO-101支持蓝牙，可以。需要先配对蓝牙设备

**Q: 没有USB转TTL适配器怎么办？**
A: 需要购买一个（推荐CH340或CP2102，价格约10-20元）

**Q: 如何确认驱动安装成功？**
A: 设备管理器中应该看到"USB Serial Port (COMx)"，无黄色感叹号

---

## 📞 下一步

完成硬件连接后：

1. **确认连接成功：**
   ```bash
   python robot_control/test_connection.py
   ```

2. **扫描舵机：**
   ```bash
   python robot_control/scan_servos.py
   ```

3. **查看详细教程：**
   ```bash
   # Windows
   notepad HARDWARE_SETUP.md
   
   # 或用浏览器打开
   start HARDWARE_SETUP.md
   ```

---

**请先完成硬件连接，然后再继续后续步骤。**

如有任何疑问，请参考LeRobot官方文档或查看机械臂的用户手册。
