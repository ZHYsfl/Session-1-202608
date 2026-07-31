"""Serial port testing utilities for SO-101."""

import serial
import serial.tools.list_ports
import time
import sys


def list_available_ports():
    """List all available serial ports."""
    print("\n" + "="*60)
    print("可用串口列表")
    print("="*60)

    ports = serial.tools.list_ports.comports()

    if not ports:
        print("未发现任何串口设备！")
        print("\n请检查：")
        print("  1. USB转TTL适配器是否已插入")
        print("  2. 驱动程序是否已安装")
        print("  3. 设备管理器中是否有黄色感叹号")
        return []

    available_ports = []
    for i, port in enumerate(ports, 1):
        print(f"\n[{i}] {port.device}")
        print(f"    描述: {port.description}")
        print(f"    硬件ID: {port.hwid}")
        available_ports.append(port.device)

    print("\n" + "="*60)
    return available_ports


def test_serial_connection(port_name, baudrate=1000000):
    """Test serial port connection.

    Args:
        port_name: Port name (e.g., 'COM3')
        baudrate: Baud rate (common: 115200, 1000000)

    Returns:
        True if connection successful, False otherwise
    """
    print(f"\n尝试连接到 {port_name} (波特率: {baudrate})...")

    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )

        time.sleep(0.5)  # 等待连接稳定

        print("✓ 串口打开成功")
        print(f"  端口: {ser.port}")
        print(f"  波特率: {ser.baudrate}")
        print(f"  超时: {ser.timeout}s")

        # 清空缓冲区
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        ser.close()
        print("✓ 测试完成，连接正常")
        return True

    except serial.SerialException as e:
        print(f"✗ 连接失败: {e}")
        print("\n可能的原因：")
        print("  1. 端口已被其他程序占用")
        print("  2. 权限不足（Linux需要加入dialout组）")
        print("  3. 端口名称错误")
        return False
    except Exception as e:
        print(f"✗ 未知错误: {e}")
        return False


def test_multiple_baudrates(port_name):
    """Test connection with multiple baud rates.

    Args:
        port_name: Port name to test
    """
    common_baudrates = [
        9600,
        57600,
        115200,
        1000000,
        2000000
    ]

    print(f"\n测试不同波特率...")
    print("="*60)

    successful_rates = []

    for baudrate in common_baudrates:
        try:
            ser = serial.Serial(port_name, baudrate, timeout=0.5)
            time.sleep(0.1)
            ser.close()
            print(f"✓ {baudrate:>8} baud - 连接成功")
            successful_rates.append(baudrate)
        except:
            print(f"✗ {baudrate:>8} baud - 连接失败")

    print("="*60)

    if successful_rates:
        print(f"\n建议使用的波特率: {successful_rates}")
    else:
        print("\n⚠️ 所有波特率测试均失败，请检查硬件连接")

    return successful_rates


def interactive_test():
    """Interactive serial port testing."""
    print("\n" + "="*60)
    print("SO-101 串口连接测试工具")
    print("="*60)

    # Step 1: List ports
    available_ports = list_available_ports()

    if not available_ports:
        return

    # Step 2: Select port
    print("\n请选择要测试的串口：")
    if len(available_ports) == 1:
        port = available_ports[0]
        print(f"自动选择唯一端口: {port}")
    else:
        while True:
            try:
                choice = input(f"输入端口名称 (例如 {available_ports[0]}): ").strip()
                if choice in available_ports:
                    port = choice
                    break
                else:
                    print(f"无效的端口名称，请从以下列表选择: {available_ports}")
            except KeyboardInterrupt:
                print("\n\n用户取消")
                return

    # Step 3: Test connection with default baud rate
    print("\n" + "="*60)
    print("测试 1: 默认波特率 (1000000)")
    print("="*60)
    test_serial_connection(port, 1000000)

    # Step 4: Test multiple baud rates
    print("\n" + "="*60)
    print("测试 2: 多波特率扫描")
    print("="*60)
    test_multiple_baudrates(port)

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print("\n下一步：")
    print("  1. 如果连接成功，运行 scan_servos.py 扫描舵机")
    print("  2. 如果连接失败，检查 HARDWARE_SETUP.md 中的故障排除部分")


if __name__ == "__main__":
    try:
        interactive_test()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
