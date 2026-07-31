import math
def analyze_trajectory(trajectory_mm):
    if len(trajectory_mm) < 2: return {}
    traj = [(x/1000, y/1000) for x,y in trajectory_mm]
    total = sum(math.sqrt((traj[i][0]-traj[i-1][0])**2 + (traj[i][1]-traj[i-1][1])**2) for i in range(1,len(traj)))
    xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
    area = (max(xs)-min(xs)) * (max(ys)-min(ys))
    return {'poses':len(traj), 'path_m':round(total,2), 'area_m2':round(area,2),
            'x_range_m':round(max(xs)-min(xs),2), 'y_range_m':round(max(ys)-min(ys),2)}
