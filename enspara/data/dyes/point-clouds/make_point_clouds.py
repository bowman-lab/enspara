import numpy as np
import mdtraj as md
import yaml
import sys

"""
Scheduler parallelized version of point-cloud creation
usage: 
python make_point_clouds.py $SLURM_ARRAY_TASK_ID
"""

print(sys.argv)
task_id=int(sys.argv[1])

#Open libraries
with open('../libraries.yml') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
keys = [key for key in data]
dye_dir='../'

dyename=data[keys[task_id]]["filename"].rsplit('_',1)[0]

#Load files
dye_top=md.load(f'{dye_dir}/{dyename}.pdb')
dye_traj=md.load(f'{dye_dir}/{dyename}_cutoff10.dcd', top=dye_top)
dye_weights=np.loadtxt(f'{dye_dir}/{dyename}_cutoff10_weights.txt')

#Slice just the dye emission center
atom_sele='name ' + data[keys[task_id]]['r'][0]
dye_traj=dye_traj.atom_slice(dye_traj.top.select(atom_sele))
dye_top=dye_top.atom_slice(dye_top.top.select(atom_sele))

#Stack all frames into one and save
for frame in range(len(dye_traj)):
	#Save multiple points according to weights

	#First frame is already saved once on object creation.
	if frame==0:
		for n in range(int(dye_weights[frame])-1):
			dye_top=dye_top.stack(dye_traj[frame])
	else:
		for n in range(int(dye_weights[frame])):
			dye_top=dye_top.stack(dye_traj[frame])
dye_top.save(f'./{dyename}-pc.pdb')
