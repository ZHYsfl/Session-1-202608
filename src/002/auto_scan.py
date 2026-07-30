"""Auto-scan servos on COM23."""

import sys
sys.path.insert(0, '.')

from robot_control.scan_servos import scan_servos_feetech

# Automatically scan COM23 with Feetech protocol
print("自动扫描 COM23 端口...")
found_servos = scan_servos_feetech(port="COM23", baudrate=1000000, id_range=(1, 20), fast_mode=False)

if found_servos:
    print("\n" + "="*60)
    print("✓ 扫描成功！")
    print("="*60)
    print(f"发现的舵机ID: {found_servos}")
    print(f"舵机数量: {len(found_servos)}")

    if len(found_servos) == 6:
        print("\n✓ 完美！发现6个舵机，符合SO-101配置")
    elif len(found_servos) < 6:
        print(f"\n⚠️ 只发现{len(found_servos)}个舵机，预期6个")
        print("可能原因：")
        print("  - 12V电源未接通")
        print("  - 某些舵机连接线松动")
    else:
        print(f"\n⚠️ 发现{len(found_servos)}个舵机，超过预期")

    print("\n下一步：")
    print("  1. 如果发现6个舵机 → 可以开始测试单关节运动")
    print("  2. 如果舵机数量不对 → 检查连接和电源")

else:
    print("\n" + "="*60)
    print("✗ 未发现舵机")
    print("="*60)
    print("\n故障排查：")
    print("  1. 检查12V电源是否接通")
    print("     - 控制板上LED是否亮起")
    print("     - 电源适配器指示灯是否正常")
    print("  2. 检查舵机连接线")
    print("     - 确认所有舵机都串联连接")
    print("     - 检查连接器是否插紧")
    print("  3. 尝试不同波特率")
    print("     - 115200")
    print("     - 57600")
    print("\n如需尝试其他波特率，运行：")
    print("  python test_scan_115200.py")
