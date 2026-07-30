"""Helper script to identify robot arm connection."""

import serial.tools.list_ports
import sys


def find_robot_ports():
    """Find USB serial ports (excluding Bluetooth)."""
    print("\n" + "="*70)
    print("SO-101 机械臂端口识别工具")
    print("="*70)

    all_ports = serial.tools.list_ports.comports()

    # Separate USB and Bluetooth ports
    usb_ports = []
    bluetooth_ports = []

    for port in all_ports:
        desc_upper = port.description.upper()
        hwid_upper = port.hwid.upper()

        # Check if it's a USB serial device
        if any(keyword in desc_upper or keyword in hwid_upper
               for keyword in ['USB', 'CH340', 'CH341', 'CP210', 'FTDI', 'SERIAL']):
            # Exclude Bluetooth
            if 'BLUETOOTH' not in desc_upper and 'BTHENUM' not in hwid_upper:
                usb_ports.append(port)
        elif 'BLUETOOTH' in desc_upper or 'BTHENUM' in hwid_upper:
            bluetooth_ports.append(port)

    print("\n【USB串口设备】（可能是机械臂）")
    print("-" * 70)

    if usb_ports:
        for i, port in enumerate(usb_ports, 1):
            print(f"\n[{i}] {port.device}")
            print(f"    描述: {port.description}")
            print(f"    硬件ID: {port.hwid}")

            # Identify chip type
            if 'CH340' in port.description or 'CH340' in port.hwid:
                print(f"    芯片: CH340 (常见USB转串口芯片)")
            elif 'CP210' in port.description or 'CP210' in port.hwid:
                print(f"    芯片: CP2102 (Silicon Labs)")
            elif 'FTDI' in port.description or 'FT232' in port.hwid:
                print(f"    芯片: FT232 (FTDI)")

        print("\n" + "="*70)
        print(f"✓ 发现 {len(usb_ports)} 个USB串口设备")

        if len(usb_ports) == 1:
            print(f"\n推荐使用: {usb_ports[0].device}")
            print("\n下一步：")
            print(f"  python robot_control/scan_servos.py")
            print(f"  然后输入端口: {usb_ports[0].device}")
        elif len(usb_ports) == 2:
            print("\n⚠️ 发现2个USB串口（可能是主从双机械臂）")
            print("\n建议：")
            print("  1. 如果只需要一个机械臂，断开其中一个USB")
            print("  2. 如果需要双机械臂，分别记录两个端口号")
            print("  3. 查看 DUAL_ARM_SETUP.md 了解详细配置")
        else:
            print(f"\n⚠️ 发现{len(usb_ports)}个USB串口")
            print("  建议断开不需要的设备，只保留机械臂")

    else:
        print("\n✗ 未发现USB串口设备！")
        print("\n可能的原因：")
        print("  1. USB线未连接")
        print("  2. 驱动未安装（CH340/CP2102/FT232）")
        print("  3. 控制板未上电")
        print("\n请检查：")
        print("  - USB线是否插紧（控制板和电脑）")
        print("  - 设备管理器中是否有黄色感叹号")
        print("  - 参考 HARDWARE_CONNECTION_STATUS.md 安装驱动")

    # Show Bluetooth ports for reference
    if bluetooth_ports:
        print(f"\n【蓝牙串口】（{len(bluetooth_ports)}个，通常不是机械臂）")
        print("-" * 70)
        for port in bluetooth_ports[:3]:  # Only show first 3
            print(f"  {port.device} - {port.description[:50]}")
        if len(bluetooth_ports) > 3:
            print(f"  ... 还有 {len(bluetooth_ports) - 3} 个蓝牙设备")

    print("\n" + "="*70)

    return [p.device for p in usb_ports]


def test_port_connection(port_name):
    """Quick test if a port is accessible."""
    try:
        import serial
        ser = serial.Serial(port_name, 1000000, timeout=0.5)
        ser.close()
        return True
    except:
        return False


if __name__ == "__main__":
    ports = find_robot_ports()

    if ports:
        print("\n准备测试连接...")
        for port in ports:
            if test_port_connection(port):
                print(f"  ✓ {port} - 可访问")
            else:
                print(f"  ✗ {port} - 无法访问（可能被占用）")

    sys.exit(0 if ports else 1)
