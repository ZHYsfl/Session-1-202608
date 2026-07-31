import sys,os,time,math,json
sys.path.insert(0,os.path.expanduser('~/BreezySLAM/python'))
sys.path.insert(0,os.path.expanduser('~/BreezySLAM/examples'))
from breezyslam.algorithms import RMHC_SLAM
from mines import MinesLaser,Rover,load_data
from pgm_utils import pgm_save
import numpy as np
from PIL import Image
MP=800;MM=32
def mm2pix(mm): return int(mm/(MM*1000/MP))
def run(ds,odom,seed=9999):
    rover=Rover() if odom else None
    _,lidars,odoms=load_data(os.path.expanduser('~/BreezySLAM/examples'),ds)
    slam=RMHC_SLAM(MinesLaser(),MP,MM,random_seed=seed)
    traj=[]; t0=time.time()
    for i in range(len(lidars)):
        if odom: slam.update(lidars[i],rover.computePoseChange(odoms[i]))
        else: slam.update(lidars[i],(0,0.1,0))
        x,y,_=slam.getpos(); traj.append((x,y))
    t=time.time()-t0
    mb=bytearray(MP*MP); slam.getmap(mb)
    for x,y in traj:
        xp,yp=mm2pix(x),mm2pix(y)
        if 0<=xp<MP and 0<=yp<MP: mb[yp*MP+xp]=0
    return mb,traj,t,len(lidars)

os.makedirs('results',exist_ok=True); results=[]
for ds in ['exp1','exp2']:
    for odom in [False,True]:
        lb=f'{ds} odom={odom}'; print(f'Running {lb}...')
        mb,traj,et,n=run(ds,odom)
        fn=f'{ds}{"_odom" if odom else ""}'
        pgm_save(f'results/{fn}.pgm',mb,(MP,MP))
        Image.fromarray(np.frombuffer(mb,dtype=np.uint8).reshape(MP,MP),'L').save(f'results/{fn}.png')
        tm=[(x/1000,y/1000) for x,y in traj]
        pl=sum(math.sqrt((tm[i][0]-tm[i-1][0])**2+(tm[i][1]-tm[i-1][1])**2) for i in range(1,len(tm)))
        xs=[p[0] for p in tm];ys=[p[1] for p in tm]
        a=(max(xs)-min(xs))*(max(ys)-min(ys))
        results.append({'dataset':ds,'odom':odom,'scans':n,'time_s':round(et,2),'path_m':round(pl,2),'area_m2':round(a,2)})
        print(f'  {n} scans, {et:.1f}s')

with open('results/comparison.json','w') as f: json.dump(results,f,indent=2)
imgs=[]; order=['exp1','exp1_odom','exp2','exp2_odom']
for fn in order:
    p=f'results/{fn}.png'
    if os.path.exists(p): imgs.append(Image.open(p))
if len(imgs)==4:
    w,h=imgs[0].size; g=Image.new('RGB',(w*2+10,h*2+10),(255,255,255))
    for i,im in enumerate(imgs): g.paste(im,((i%2)*(w+5),(i//2)*(h+5)))
    g.save('results/comparison_grid.png')
    print('Grid saved')
print(json.dumps(results,indent=2))
print('Done!')
