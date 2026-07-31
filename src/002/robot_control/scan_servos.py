"""Servo scanning utility for SO-101 robot arm."""

import serial
import time
import sys
from typing import List, Tuple


def calculate_checksum_feetech(data: List[int]) -> int:
    """Calculate checksum for Feetech SCS protocol.

    Args:
        data: Packet data [ID, Length, Instruction, ...]

    Returns:
        Checksum byte
    """
    return (~(sum(data) % 256)) & 0xFF


def ping_servo_feetech(ser: serial.Serial, servo_id: int, timeout: float = 0.05) -> bool:
    """Ping a servo using Feetech SCS protocol.

    Args:
        ser: Serial connection
        servo_id: Servo ID to ping
        timeout: Response timeout in seconds

    Returns:
        True if servo responds, False otherwise
    """
    # Feetech SCS PING packet format:
    # [0xFF, 0xFF, ID, Length, Instruction, Checksum]
    packet_data = [servo_id, 0x02, 0x01]  # ID, Length=2, Instruction=PING
    checksum = calculate_checksum_feetech(packet_data)

    packet = bytes([0xFF, 0xFF] + packet_data + [checksum])

    # Clear buffer
    ser.reset_input_buffer()

    # Send packet
    ser.write(packet)

    # Wait for response
    start_time = time.time()
    response = b''

    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            response += ser.read(ser.in_waiting)

            # Check if we have a complete response (at least 6 bytes)
            if len(response) >= 6:
                # Check header
                if response[0:2] == b'\xff\xff':
                    # Check if ID matches
                    if response[2] == servo_id:
                        return True
                break

        time.sleep(0.001)

    return False


def scan_servos_feetech(port: str,
                        baudrate: int = 1000000,
                        id_range: Tuple[int, int] = (1, 253),
                        fast_mode: bool = False) -> List[int]:
    """Scan for Feetech SCS servos on the bus.

    Args:
        port: Serial port name
        baudrate: Communication baud rate
        id_range: Tuple of (min_id, max_id) to scan
        fast_mode: If True, use faster scanning with less retries

    Returns:
        List of found servo IDs
    """
    print("\n" + "="*60)
    print(f"扫描 Feetech SCS 舵机")
    print("="*60)
    print(f"端口: {port}")
    print(f"波特率: {baudrate}")
    print(f"扫描范围: ID {id_range[0]} - {id_range[1]}")
    print("="*60)

    try:
        # Open serial port
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=0.1
        )
        time.sleep(0.1)

        print("\n开始扫描...\n")

        found_servos = []
        min_id, max_id = id_range

        for servo_id in range(min_id, max_id + 1):
            # Ping servo (with retries if not in fast mode)
            retries = 1 if fast_mode else 2
            found = False

            for attempt in range(retries):
                if ping_servo_feetech(ser, servo_id):
                    found = True
                    break
                time.sleep(0.01)

            if found:
                print(f"✓ 发现舵机 ID: {servo_id:3d}")
                found_servos.append(servo_id)

            # Progress indicator
            if servo_id % 50 == 0:
                progress = (servo_id - min_id) / (max_id - min_id) * 100
                print(f"  进度: {progress:.0f}% (已扫描 {servo_id}/{max_id})")

        ser.close()

        # Summary
        print("\n" + "="*60)
        print(f"扫描完成！")
        print("="*60)
        print(f"发现 {len(found_servos)} 个舵机")

        if found_servos:
            print(f"ID列表: {found_servos}")
            print("\n建议的ID分配：")
            servo_names = [
                "Shoulder Pan (底座旋转)",
                "Shoulder Lift (肩部抬升)",
                "Elbow Flex (肘部弯曲)",
                "Wrist Flex (腕部弯曲)",
                "Wrist Roll (腕部旋转)",
                "Gripper (夹爪)"
            ]
            for i, servo_id in enumerate(found_servos[:6]):
                if i < len(servo_names):
                    print(f"  ID {servo_id}: {servo_names[i]}")
        else:
            print("\n⚠️ 未发现任何舵机！")
            print("\n请检查：")
            print("  1. 12V电源是否已接通")
            print("  2. 舵机连接线是否正确")
            print("  3. 波特率是否匹配（常见：115200, 1000000）")
            print("  4. 是否需要短接TXD和RXD（半双工通信）")

        print("="*60 + "\n")

        return found_servos

    except serial.SerialException as e:
        print(f"\n✗ 串口错误: {e}")
        return []
    except Exception as e:
        print(f"\n✗ 未知错误: {e}")
        return []


def scan_servos_dynamixel(port: str, baudrate: int = 1000000) -> List[int]:
    """Scan for Dynamixel servos (requires dynamixel-sdk).

    Args:
        port: Serial port name
        baudrate: Communication baud rate

    Returns:
        List of found servo IDs
    """
    try:
        import dynamixel_sdk
    except ImportError:
        print("⚠️ dynamixel-sdk 未安装")
        print("请运行: pip install dynamixel-sdk")
        return []

    print("\n" + "="*60)
    print(f"扫描 Dynamixel 舵机")
    print("="*60)
    print(f"端口: {port}")
    print(f"波特率: {baudrate}")
    print("="*60)

    # Initialize PortHandler and PacketHandler
    portHandler = dynamixel_sdk.PortHandler(port)
    packetHandler = dynamixel_sdk.PacketHandler(2.0)  # Protocol version 2.0

    # Open port
    if not portHandler.openPort():
        print(f"✗ 无法打开端口 {port}")
        return []

    # Set baudrate
    if not portHandler.setBaudRate(baudrate):
        print(f"✗ 无法设置波特率 {baudrate}")
        return []

    print("\n开始扫描...\n")

    found_servos = []

    for servo_id in range(1, 254):
        # Try to ping
        dxl_model_number, dxl_comm_result, dxl_error = packetHandler.ping(portHandler, servo_id)

        if dxl_comm_result == COMM_SUCCESS:
            print(f"✓ 发现舵机 ID: {servo_id:3d}, 型号: {dxl_model_number}")
            found_servos.append(servo_id)

        if servo_id % 50 == 0:
            print(f"  进度: {servo_id}/253")

    # Close port
    portHandler.closePort()

    print("\n" + "="*60)
    print(f"扫描完成！发现 {len(found_servos)} 个舵机")
    if found_servos:
        print(f"ID列表: {found_servos}")
    print("="*60 + "\n")

    return found_servos


def interactive_scan():
    """Interactive servo scanning."""
    print("\n" + "="*60)
    print("SO-101 舵机扫描工具")
    print("="*60)

    # Get port
    port = input("\n输入COM端口 (例如 COM3 或 /dev/ttyUSB0): ").strip()

    # Get protocol
    print("\n选择舵机协议：")
    print("  1. Feetech SCS")
    print("  2. Dynamixel")

    while True:
        try:
            choice = input("输入选择 [1/2]: ").strip()
            if choice in ['1', '2']:
                break
            print("无效选择，请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n\n用户取消")
            return

    # Get baudrate
    print("\n选择波特率：")
    print("  1. 1000000 (常用)")
    print("  2. 115200")
    print("  3. 57600")
    print("  4. 自定义")

    baudrate_map = {
        '1': 1000000,
        '2': 115200,
        '3': 57600
    }

    while True:
        try:
            baud_choice = input("输入选择 [1/2/3/4]: ").strip()
            if baud_choice in ['1', '2', '3']:
                baudrate = baudrate_map[baud_choice]
                break
            elif baud_choice == '4':
                baudrate = int(input("输入波特率: ").strip())
                break
            print("无效选择")
        except (ValueError, KeyboardInterrupt):
            print("\n\n用户取消")
            return

    # Scan
    if choice == '1':
        found_servos = scan_servos_feetech(port, baudrate)
    else:
        found_servos = scan_servos_dynamixel(port, baudrate)

    # Next steps
    if found_servos:
        print("\n下一步：")
        if len(found_servos) == 6:
            print("  ✓ 发现6个舵机，数量正确！")
            print("  → 运行校准脚本: python calibrate_robot.py")
        elif len(found_servos) < 6:
            print(f"  ⚠️ 只发现{len(found_servos)}个舵机，预期6个")
            print("  → 检查未连接的舵机")
        else:
            print(f"  ⚠️ 发现{len(found_servos)}个舵机，多于预期")
            print("  → 检查是否有ID冲突或多余设备")

        # Check ID continuity
        if found_servos == list(range(1, 7)):
            print("  ✓ ID分配完美 (1-6)，可以直接使用")
        else:
            print(f"  ⚠️ ID不连续: {found_servos}")
            print("  → 考虑重新分配ID (使用 set_servo_id.py)")


if __name__ == "__main__":
    try:
        interactive_scan()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
