import rebound 
import numpy as np
import sys
import shlex

# from PyAstronomy import pyasl
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import subprocess
import rebound

from latex_exp import latex_exp
from rebound_gc import bin_analysis
import argparse
import json


@np.vectorize
def menc_dp(dist, rho_rb, rb, alpha, beta):
    '''
    Mass enclosed 
    '''
    if dist<rb:
        menc_rad=(dist/rb)**alpha/(alpha)
    else:
        menc_rad=1/(alpha)+(1/beta)*((dist/rb)**beta-1)
    menc=4*np.pi*rb**3.*rho_rb*menc_rad
    return menc

@np.vectorize
def stellar_pot_dp(dist, rho_rb, rb, alpha, beta):
    if dist<rb:
        rho_int=4.*np.pi*rho_rb*rb**2.*(1-(dist/rb)**(alpha-1))/(alpha-1)
    else:
        rho_int=4.*np.pi*rho_rb*rb**2.*(1-(dist/rb)**(beta-1))/(beta-1)
    return (-menc_dp(dist, rho_rb, rb, alpha, beta)/dist-rho_int)


def bash_command(cmd):
	'''Run command from the bash shell'''
	process=subprocess.Popen(['/bin/bash', '-c',cmd],  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	return process.communicate()[0]

parser=argparse.ArgumentParser(description='Set up a rebound run')
parser.add_argument('loc',help='Data location')
parser.add_argument('--xlim', type=float, default=1)
parser.add_argument('--ylim', type=float, default=1)
args=parser.parse_args()

##Location of simulation data...
loc=args.loc
xlim=args.xlim
ylim=args.ylim

##You will have to modify the names of the data files (the archive_out part) here. 
sa=rebound.SimulationArchive('{0}/archive.bin'.format(loc))
frames=len(sa)
##and here
samp=np.genfromtxt(loc+'/archive_out_0.dat')

pts=200
x=np.empty([frames, len(samp), pts+1])
y=np.empty([frames, len(samp), pts+1])
z=np.empty([frames, len(samp), pts+1])

fig,ax=plt.subplots(figsize=(10,9))
dis=[]
col='0.5'
frac=np.zeros(frames)
en0=0

for idx in range(frames):
	dis_indics=[]
	sim=sa[idx]
	parts=sim.particles
	orbs=sim.calculate_orbits(primary=sim.particles[0])
	hashes=[str(pp.hash) for pp in parts[1:]]
	for i in range(len(orbs)):
		if((orbs[i].a*(1-orbs[i].e)<=3e-4) & (~np.in1d(hashes[i], dis))[0]):
			print('Dis!', orbs[i].a, orbs[i].e)
			dis.append(hashes[i])
			dis_indics.append(i)

		# rr=np.linalg.norm(sim.particles[-1].xyz)

		p=sim.particles[i+1]
		if orbs[i].a>0:
			pos=np.array(p.sample_orbit(pts, primary=sim.particles[0]))
			# sim.remove(1)

			x[idx,i,:-1] = pos[:,0]
			x[idx,i][-1]=np.nan
			y[idx,i,:-1] = pos[:,1]
			y[idx,i][-1]=np.nan
			z[idx,i,:-1] = pos[:,2]
			z[idx,i][-1]=np.nan

			ax.plot(x[idx,i,:-1], y[idx,i,:-1], '-', color=col)

	# tt=bin_analysis.bin_find_sim(sim)
	##Add stellar potential constribution here
	en=sim.calculate_energy()

	if idx==0:
		en0=en
	print('parts:'+str(len(sim.particles)))

	for tmp in dis_indics:
		ax.plot(x[idx,tmp,:-1], y[idx,tmp,:-1], '--', color='b')
			
 
	ax.annotate('t={0} yr\nBlue, Dashed=Disruption'.format(latex_exp.latex_exp(sim.t*1.5e7)), (0.01, 0.99),\
		xycoords='axes fraction', va='top', fontsize=16)
	ax.set_xlabel('x [pc]')
	ax.set_ylabel('y [pc]')
	ax.set_xlim(-xlim, xlim)
	ax.set_ylim(-ylim, ylim)
	# # ax.set_xticks([-1,-0.5,0,0.5,1])
	# ax.set_ylim(-1,1)
	# ax.set_yticks([-1,-0.5,0,0.5,1])
	fig.savefig(loc+'/sim_movie_man_sa_{0:03d}.png'.format(idx), ppi=72)
	plt.cla()

# np.savez(loc+'/x.npz', x)
# np.savez(loc+'/y.npz', y)
# np.savez(loc+'/z.npz', z)
