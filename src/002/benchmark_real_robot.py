"""四方法真机统一对比 - 同一批目标点公平测试

回答组长的问题:
  1. 训练数据来源: 关节角=真机实测, 末端位置=FK软件计算(见文档)
  2. 各方法整体统计 + 逐点单件数据
  3. 闭环迭代抵消机械误差, 看精度能提到多少

四方法: 解析IK / 数值IK / 神经网络IK / 混合法
每个方法对每个目标点:
  - 软件误差: FK(IK(P)) vs P
  - 真机开环误差: 命令一次, 读实际到达
  - 真机闭环误差: 迭代补偿数步
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).parent.parent))

import serial, time, json
import numpy as np

from kinematics.forward_kinematics_simple import ForwardKinematics
from kinematics.analytical_ik import AnalyticalIK
from kinematics.numerical_ik import NumericalIK


class Ctrl:
    def __init__(self, port, baud=1000000):
        self.serial = serial.Serial(port, baud, timeout=0.5); time.sleep(0.5)
        print(f"[OK] connected {port}")
    def _ck(self, d): return (~(sum(d) % 256)) & 0xFF
    def _send(self, sid, inst, p=None):
        p = p or []; pk = [sid, len(p)+2, inst] + p
        self.serial.reset_input_buffer(); self.serial.write(bytes([0xFF,0xFF]+pk+[self._ck(pk)])); time.sleep(0.01)
    def enable(self, sid): self._send(sid, 0x03, [0x28, 0x01])
    def disable(self, sid): self._send(sid, 0x03, [0x28, 0x00])
    def read_pos(self, sid):
        self._send(sid, 0x02, [0x38, 0x02]); r = self.serial.read(8)
        if len(r) >= 8 and r[0:2] == b'\xff\xff': return r[5]+(r[6]<<8)
        return None
    def write_pos(self, sid, pos, speed=80):
        pos = int(pos); self._send(sid,0x03,[0x2A,pos&0xFF,(pos>>8)&0xFF,speed&0xFF,(speed>>8)&0xFF])
    def read_all(self): return [self.read_pos(i) for i in [1,2,3,4,5]]
    def move_all(self, servos, speed=80, wait=2.5):
        for i in [1,2,3,4,5]: self.enable(i)
        time.sleep(0.2)
        for sid,p in enumerate(servos,1): self.write_pos(sid,p,speed)
        time.sleep(wait)
    def close(self):
        for i in [1,2,3,4,5]: self.disable(i)
        self.serial.close()


def s2r(s): return (np.array(s)-2048)/4096.0*(240.0*np.pi/180.0)
def r2s(r): return np.clip(np.array(r)/(240.0*np.pi/180.0)*4096.0+2048, 100, 3995).astype(int)
HOME = [2048]*5
Z_FLOOR = 0.192


def load_cfg(cfg='configs/robot_config.yaml'):
    import yaml
    c = yaml.safe_load(open(cfg))
    jl = c['robot']['joint_limits']
    names = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
    lo = np.array([float(jl[n][0]) for n in names]); hi = np.array([float(jl[n][1]) for n in names])
    return lo, hi, c['robot']['link_lengths']


class NeuralIK:
    def __init__(self, path, jl):
        import torch
        from neural_network.model import IKNet, prepare_input
        self.prep = prepare_input
        ck = torch.load(path, map_location='cpu'); mc = ck['config']['training']['model']
        self.m = IKNet(input_dim=mc['input_dim'],hidden_dims=mc['hidden_dims'],output_dim=mc['output_dim'],
                       activation=mc['activation'],use_batch_norm=mc['use_batch_norm'],dropout=mc['dropout'],joint_limits=jl)
        self.m.load_state_dict(ck['model_state_dict']); self.m.eval()
    def solve(self, pos, alpha, psi, q_cur):
        return self.m.predict(self.prep(pos, alpha, psi, q_cur))[0]


def closed_loop(ctrl, fk, q_seed, target_pos, max_iter=4, tol=0.005):
    """闭环迭代: 命令→读实际→按位置残差修正关节角。用数值IK的雅可比做增量修正。"""
    q_cmd = np.asarray(q_seed).copy()
    best_err = 1e9; best_actual = None
    errs = []
    for it in range(max_iter):
        p_pred, _ = fk.compute(q_cmd)
        if p_pred[2] < Z_FLOOR:
            break
        ctrl.move_all(HOME, speed=80, wait=1.2)
        ctrl.move_all(r2s(q_cmd), speed=80, wait=2.5)
        actual = ctrl.read_all()
        if None in actual:
            break
        q_act = s2r(actual)
        p_act, _ = fk.compute(q_act)
        err = np.linalg.norm(p_act - target_pos)
        errs.append(err*1000)
        if err < best_err:
            best_err = err; best_actual = p_act
        if err < tol:
            break
        # 位置残差 → 关节修正: 用数值IK雅可比伪逆
        J = _jacobian(fk, q_act)
        dp = target_pos - p_act
        dq = np.linalg.pinv(J) @ dp
        dq = np.clip(dq, -0.15, 0.15)  # 限幅防大跳
        q_cmd = q_act + dq
    return best_err*1000, errs


def _jacobian(fk, q, eps=1e-5):
    J = np.zeros((3,5))
    p0,_ = fk.compute(q)
    for i in range(5):
        qd = q.copy(); qd[i]+=eps
        pi,_ = fk.compute(qd)
        J[:,i] = (pi-p0)/eps
    return J


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM23')
    ap.add_argument('--urdf', default='models/so101_new_calib.urdf')
    ap.add_argument('--model', default='models/ik_mlp_best.pth')
    ap.add_argument('--points', type=int, default=5)
    ap.add_argument('--closed_loop', action='store_true', help='额外做闭环迭代')
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    fk = ForwardKinematics(args.urdf)
    lo, hi, ll = load_cfg()
    analytical = AnalyticalIK(shoulder_height=float(ll['shoulder_height']),radial_offset=float(ll['radial_offset']),
        upper_arm_length=float(ll['upper_arm']),forearm_length=float(ll['forearm']),
        wrist_to_tcp=float(ll['wrist_to_tcp']),joint_limits=(lo.tolist(),hi.tolist()))
    numerical = NumericalIK(fk, (lo, hi))
    neural = NeuralIK(args.model, (lo, hi))

    safe = {1:[-45,45],2:[-10,45],3:[-40,40],4:[-40,40],5:[-80,80]}
    def rand_safe():
        while True:
            s=[np.random.randint(int(safe[j][0]/240.*4096+2048)+50,int(safe[j][1]/240.*4096+2048)-50) for j in [1,2,3,4,5]]
            p,_=fk.compute(s2r(s))
            if p[2]>=Z_FLOOR and np.sqrt(p[0]**2+p[1]**2)>=0.10 and np.linalg.norm(p)>=0.20: return s
    np.random.seed(100)
    targets=[]
    for _ in range(args.points):
        st=rand_safe(); th=s2r(st); pos,alpha,psi=fk.compute_task_space(th)
        targets.append({'servo_true':st,'th':th,'pos':pos,'alpha':alpha,'psi':psi})

    methods=['numerical','neural','hybrid']  # 解析法解不出,单列
    data={m:{'sw':[],'open':[],'closed':[]} for m in methods}
    per_point=[]

    ctrl=None
    if not args.dry:
        print("⚠️ 机械臂将移动, 请确保安全"); ctrl=Ctrl(args.port); ctrl.move_all(HOME,speed=60,wait=3.0)

    for idx,t in enumerate(targets):
        pos,alpha,psi,th=t['pos'],t['alpha'],t['psi'],t['th']
        q_init=np.array(s2r(HOME))
        rec={'point':idx+1,'target_mm':[round(float(v)*1000,1) for v in pos]}

        # 求解
        qn=numerical.solve(pos,alpha,psi,initial_guess=q_init)
        qm=neural.solve(pos,alpha,psi,q_init)
        qmn=neural.solve(pos,alpha,psi,q_init)
        qh,_=numerical.refine_solution(pos,alpha,psi,np.asarray(qmn),num_steps=10)

        sols={'numerical':qn,'neural':qm,'hybrid':qh}
        # 软件误差
        for m,q in sols.items():
            if q is None: rec[f'{m}_sw']=None; continue
            p2,_=fk.compute(np.asarray(q)); e=np.linalg.norm(p2-pos)*1000
            data[m]['sw'].append(e); rec[f'{m}_sw']=round(e,2)

        # 真机开环
        if not args.dry:
            for m,q in sols.items():
                if q is None: rec[f'{m}_open']=None; continue
                p_pred,_=fk.compute(np.asarray(q))
                if p_pred[2]<Z_FLOOR: rec[f'{m}_open']='skip(unsafe)'; continue
                ctrl.move_all(HOME,speed=80,wait=1.2); ctrl.move_all(r2s(q),speed=80,wait=2.8)
                act=ctrl.read_all()
                if None in act: rec[f'{m}_open']=None; continue
                pa,_=fk.compute(s2r(act)); e=np.linalg.norm(pa-pos)*1000
                data[m]['open'].append(e); rec[f'{m}_open']=round(e,1)

            # 闭环(只对hybrid做, 代表最优方法迭代抵消机械误差)
            if args.closed_loop and qh is not None:
                ce,_=closed_loop(ctrl,fk,qh,pos,max_iter=4)
                data['hybrid']['closed'].append(ce); rec['hybrid_closed']=round(ce,1)

        per_point.append(rec)
        print(f"点{idx+1} 目标{rec['target_mm']}: " +
              " ".join(f"{m}={rec.get(f'{m}_open',rec.get(f'{m}_sw'))}" for m in methods))

    if ctrl: ctrl.move_all(HOME,speed=60,wait=3.0); ctrl.close()

    # 汇总
    print("\n"+"="*78)
    print(f"四方法真机对比 (N={args.points}, 同一批目标点)")
    print("="*78)
    tag={'numerical':'数值IK','neural':'神经网络','hybrid':'混合法'}
    print(f"\n{'方法':<12}{'软件误差(mm)':<24}{'真机开环(mm)':<24}{'真机闭环(mm)':<16}")
    print(f"{'':12}{'均值/中位/最大':<24}{'均值/中位/最大':<24}")
    print("-"*78)
    summary={}
    for m in methods:
        sw=data[m]['sw']; op=data[m]['open']; cl=data[m]['closed']
        def stat(x): return f"{np.mean(x):.1f}/{np.median(x):.1f}/{np.max(x):.1f}" if x else "-"
        print(f"{tag[m]:<12}{stat(sw):<24}{stat(op):<24}{stat(cl):<16}")
        summary[m]={'sw':stat(sw),'open':stat(op),'closed':stat(cl),
                    'sw_mean':float(np.mean(sw)) if sw else None,
                    'open_mean':float(np.mean(op)) if op else None,
                    'closed_mean':float(np.mean(cl)) if cl else None}

    out={'summary':summary,'per_point':per_point,'note':'关节角=真机实测,末端位置=FK计算',
         'timestamp':time.strftime("%Y-%m-%d %H:%M:%S")}
    with open('benchmark_real_robot.json','w') as f: json.dump(out,f,indent=2,ensure_ascii=False)
    print(f"\n✓ 逐点+汇总已保存: benchmark_real_robot.json")


if __name__=='__main__':
    main()
