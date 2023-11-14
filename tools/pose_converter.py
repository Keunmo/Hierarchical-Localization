'''convert Geunmo into multipleye.
the last colume was the frame idx. move this to the first column and put 0s at the last column.
save the txt file with proper format'''
import numpy as np
traj=np.loadtxt('/data2/chaerin/nerf_hyu_shared/box_nerf/trajectory_train.txt')
ts = traj[:,-1]
ts = ts[:,None]
ts_traj = np.concatenate([ts,traj[:,1:]],1)
ts_traj = ts_traj[:,:-1]
ts_0 = np.zeros_like(ts)
ts_traj = np.concatenate([ts_traj,ts_0],1)
fmts = ['%d']+['%.16f']*6+['%d']
np.savetxt('/data2/chaerin/nerf_hyu_shared/box_nerf/trajectory_train_fix.txt',ts_traj, fmt=fmts)
# not yet sorted
'''For trajectory_train.txt,
    w2c to c2w,
    and sort by fidx
'''
# from .pyscripts.utils.init import *
from .pyscripts.utils.geometry import *
from .pyscripts.utils.file_io import *
import numpy as np
fidx, poses, timestamps = load_pose_text_file('/data2/chaerin/nerf_hyu_shared/box_nerf/trajectory_train.txt')
idx = np.argsort(fidx)
poses = poses[:,idx]
nposes = len(fidx)
timestamps = timestamps[idx]
timestamps = timestamps[:,None]
fidx = np.sort(fidx)
fidx = fidx[:,None]
for n in range(nposes):
    poses[:,n] = inverse_transform(poses[:,n]).flatten()
poses = np.transpose(poses,[1,0])
traj = np.concatenate([fidx,poses,timestamps],1)
fmts = ['%d']+['%.16f']*6+['%d']
np.savetxt('/data2/chaerin/nerf_hyu_shared/box_nerf/trajectory_train_c2w.txt',traj, fmt=fmts)