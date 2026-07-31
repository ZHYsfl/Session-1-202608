"""关节空间闭环控制 - 正确的机械误差补偿

之前的FK闭环失败原因: 末端位置也是FK算的, 观测不到真实物理偏差。
正确做法: 用编码器反馈做关节空间闭环。

原理:
  IK给出目标关节角 q_target
  机械臂开环命令后, 实际到达 q_actual (因重力下垂等偏离 q_target)
  迭代: 命令 q_cmd += (q_target - q_actual), 直到 q_actual ≈ q_target
  编码器提供真实关节角反馈, 所以这个闭环能真正消除稳态偏差

这样能让"真机实际关节角"逼近"IK目标关节角",
从而真机末端位置逼近软件预测位置(把真机误差压向软件误差)。
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/002

import serial, time, json
import numpy as np
from kinematics.forward_kinematics_simple import ForwardKinematics
from kinematics.numerical_ik import NumericalIK


class Ctrl:
    def __init__(self, port, baud=1000000):
        self.serial=serial.Serial(port,baud,timeout=0.5); time.sleep(0.5); print(f"[OK] {port}")
    def _ck(self,d): return (~(sum(d)%256))&0xFF
    def _send(self,sid,inst,p=None):
        p=p or []; pk=[sid,len(p)+2,inst]+p; self.serial.reset_input_buffer()
        self.serial.write(bytes([0xFF,0xFF]+pk+[self._ck(pk)])); time.sleep(0.01)
    def enable(self,sid): self._send(sid,0x03,[0x28,0x01])
    def disable(self,sid): self._send(sid,0x03,[0x28,0x00])
    def read_pos(self,sid):
        self._send(sid,0x02,[0x38,0x02]); r=self.serial.read(8)
        if len(r)>=8 and r[0:2]==b'\xff\xff': return r[5]+(r[6]<<8)
        return None
    def write_pos(self,sid,pos,sp=80):
        pos=int(pos); self._send(sid,0x03,[0x2A,pos&0xFF,(pos>>8)&0xFF,sp&0xFF,(sp>>8)&0xFF])
    def read_all(self): return [self.read_pos(i) for i in [1,2,3,4,5]]
    def move_all(self,servos,sp=80,wait=2.5):
        for i in [1,2,3,4,5]: self.enable(i)
        time.sleep(0.2)
        for sid,p in enumerate(servos,1): self.write_pos(sid,p,sp)
        time.sleep(wait)
    def close(self):
        for i in [1,2,3,4,5]: self.disable(i)
        self.serial.close()

def s2r(s): return (np.array(s)-2048)/4096.0*(240.0*np.pi/180.0)
def r2s(r): return np.clip(np.array(r)/(240.0*np.pi/180.0)*4096.0+2048,100,3995).astype(int)
HOME=[2048]*5; Z_FLOOR=0.192

def load_cfg(cfg='configs/robot_config.yaml'):
    import yaml; c=yaml.safe_load(open(cfg)); jl=c['robot']['joint_limits']
    names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
    return (np.array([float(jl[n][0]) for n in names]),np.array([float(jl[n][1]) for n in names]),c['robot']['link_lengths'])


def joint_closed_loop(ctrl, fk, servo_target, target_pos, max_iter=5, tol_steps=8):
    """关节空间闭环: 迭代命令直到实际到达的舵机值≈目标舵机值。

    Returns: (开环误差mm, 闭环误差mm, 每轮末端误差列表, 每轮关节偏差列表)
    """
    servo_target = np.array(servo_target, dtype=float)
    servo_cmd = servo_target.copy()
    open_err = None
    pos_errs = []; joint_devs = []

    ctrl.move_all(HOME, sp=80, wait=1.2)
    for it in range(max_iter):
        ctrl.move_all(r2s(s2r(servo_cmd.astype(int))) if False else servo_cmd.astype(int), sp=80, wait=2.6)
        actual = ctrl.read_all()
        if None in actual:
            break
        actual = np.array(actual, dtype=float)
        # 末端位置误差(相对目标)
        p_act, _ = fk.compute(s2r(actual))
        pe = np.linalg.norm(p_act - target_pos)*1000
        pos_errs.append(pe)
        # 关节偏差(实际 vs 我们真正想到达的 servo_target)
        dev = servo_target - actual
        joint_devs.append(dev.tolist())
        if open_err is None:
            open_err = pe  # 第一次就是开环误差
        # 收敛判据: 实际关节角接近目标
        if np.max(np.abs(dev)) < tol_steps:
            break
        # 关节空间修正: 命令 += 缺口(带增益避免震荡)
        servo_cmd = servo_cmd + 0.8 * dev
        servo_cmd = np.clip(servo_cmd, 100, 3995)
    closed_err = pos_errs[-1] if pos_errs else None
    return open_err, closed_err, pos_errs, joint_devs


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--port',default='COM23'); ap.add_argument('--urdf',default='models/so101_new_calib.urdf')
    ap.add_argument('--points',type=int,default=5); ap.add_argument('--max_iter',type=int,default=5)
    args=ap.parse_args()

    fk=ForwardKinematics(args.urdf); lo,hi,ll=load_cfg()
    numerical=NumericalIK(fk,(lo,hi))

    safe={1:[-45,45],2:[-10,45],3:[-40,40],4:[-40,40],5:[-80,80]}
    def rand_safe():
        while True:
            s=[np.random.randint(int(safe[j][0]/240.*4096+2048)+50,int(safe[j][1]/240.*4096+2048)-50) for j in [1,2,3,4,5]]
            p,_=fk.compute(s2r(s))
            if p[2]>=Z_FLOOR and np.sqrt(p[0]**2+p[1]**2)>=0.10 and np.linalg.norm(p)>=0.20: return s
    np.random.seed(100)  # 同benchmark, 可比
    targets=[]
    for _ in range(args.points):
        st=rand_safe(); th=s2r(st); pos,alpha,psi=fk.compute_task_space(th)
        targets.append({'st':st,'pos':pos,'alpha':alpha,'psi':psi})

    print("⚠️ 机械臂将移动做关节空间闭环, 请确保安全")
    ctrl=Ctrl(args.port); ctrl.move_all(HOME,sp=60,wait=3.0)
    results=[]
    try:
        for i,t in enumerate(targets):
            # 数值IK求目标关节角(软件最准)
            q=numerical.solve(t['pos'],t['alpha'],t['psi'],initial_guess=np.array(s2r(HOME)))
            if q is None:
                print(f"点{i+1}: IK无解, 跳过"); continue
            p_pred,_=fk.compute(np.asarray(q))
            if p_pred[2]<Z_FLOOR:
                print(f"点{i+1}: 预测姿态低于安全线, 跳过"); continue
            servo_target=r2s(q)
            oe,ce,pes,devs=joint_closed_loop(ctrl,fk,servo_target,t['pos'],max_iter=args.max_iter)
            results.append({'point':i+1,'open_mm':round(oe,1) if oe else None,
                            'closed_mm':round(ce,1) if ce else None,
                            'iters':len(pes),'pos_err_seq':[round(x,1) for x in pes]})
            print(f"点{i+1} 目标({t['pos'][0]*1000:.0f},{t['pos'][1]*1000:.0f},{t['pos'][2]*1000:.0f}): "
                  f"开环{oe:.1f}mm → 闭环{ce:.1f}mm (迭代{len(pes)}次) 轨迹{[round(x,1) for x in pes]}")
        ctrl.move_all(HOME,sp=60,wait=3.0)
    finally:
        ctrl.close()

    if results:
        oes=[r['open_mm'] for r in results if r['open_mm']]
        ces=[r['closed_mm'] for r in results if r['closed_mm']]
        print("\n"+"="*60)
        print(f"关节空间闭环结果 (N={len(results)})")
        print("="*60)
        print(f"  开环平均误差: {np.mean(oes):.1f}mm")
        print(f"  闭环平均误差: {np.mean(ces):.1f}mm")
        print(f"  改善: {(1-np.mean(ces)/np.mean(oes))*100:.0f}%")
        json.dump({'results':results,'open_mean':float(np.mean(oes)),'closed_mean':float(np.mean(ces))},
                  open('closed_loop_joint_results.json','w'),indent=2,ensure_ascii=False)
        print("✓ 已保存: closed_loop_joint_results.json")


if __name__=='__main__':
    main()
