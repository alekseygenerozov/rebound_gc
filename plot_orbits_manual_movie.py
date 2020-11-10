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
parser.add_argument('--Mbh', '-M', type=float, default=4e6,help='Mass of central body')
parser.add_argument('--xlim', type=float, default=1)
parser.add_argument('--ylim', type=float, default=1)
args=parser.parse_args()

##Location of simulation data...
loc=args.loc
Mbh=args.Mbh
xlim=args.xlim
ylim=args.ylim

##You will have to modify the names of the data files (the archive_out part) here. 
frames=bash_command('echo {0}/archive_out_*.dat'.format(loc)).decode('utf-8')
frames=len(shlex.split(frames))
##and here
samp=np.genfromtxt(loc+'/archive_out_0.dat')
init_disk=np.genfromtxt(loc+'/init_disk')

pts=200
x=np.empty([frames, len(samp), pts+1])
y=np.empty([frames, len(samp), pts+1])
z=np.empty([frames, len(samp), pts+1])
sim=rebound.Simulation()
sim.add(m=Mbh)

fig,ax=plt.subplots(figsize=(10,9))
dis=[]
col='0.5'
frac=np.zeros(frames)
en0=0

for idx in range(frames):
	print(idx)
	##and here.
	dat=np.genfromtxt(loc+'/archive_out_{0}.dat'.format(idx))
	hashes=np.genfromtxt(loc+'/archive_out_{0}.hash'.format(idx), dtype=str)[1:]
	print(len(dat), len(hashes))
	dis_indics=[]
	frac[idx]=len(dat[dat[:,2]>np.pi/2.])/len(dat)

	dat_filt=dat[dat[:,0]>=0]
	unbound=len(dat[dat[:,0]<0])
	# print(len(dat_filt))
	en_stellar=np.zeros(len(dat))
	for i in range(len(dat)):
		# if dat[i,0]<0:
		# 	continue
		##Better use hash...
		if((dat[i,0]*(1-dat[i,1])<=3e-4) & (~np.in1d(hashes[i], dis))[0]):
			print('Dis!', dat[i,0], dat[i,1])
			dis.append(hashes[i])
			dis_indics.append(i)

		sim.add(a=dat[i,0], e=dat[i,1], inc=dat[i,2], Omega=dat[i, 3], omega=dat[i,4], f=dat[i, 5], m=init_disk[i+1, -1], primary=sim.particles[0])
		rr=np.linalg.norm(sim.particles[-1].xyz)
		en_stellar[i]=init_disk[i+1,-1]*stellar_pot_dp(rr, 4.93e4, 3., 1.86, 0.1)

		p=sim.particles[-1]
		if dat[i,0]>0:
			pos=np.array(p.sample_orbit(pts, primary=sim.particles[0]))
			# sim.remove(1)

			x[idx,i,:-1] = pos[:,0]
			x[idx,i][-1]=np.nan
			y[idx,i,:-1] = pos[:,1]
			y[idx,i][-1]=np.nan
			z[idx,i,:-1] = pos[:,2]
			z[idx,i][-1]=np.nan

			ax.plot(x[idx,i,:-1], y[idx,i,:-1], '-', color=col)

	tt=bin_analysis.bin_find_sim(sim)
	print(tt)
	##Add stellar potential constribution here
	en=sim.calculate_energy()+np.sum(en_stellar)
	enb=sim.calculate_energy()

	if idx==0:
		en0=en
		en0b=enb
	for jj in range(len(dat)):
		sim.remove(len(dat)-jj)
	print('parts:'+str(len(sim.particles)))

	for tmp in dis_indics:
		ax.plot(x[idx,tmp,:-1], y[idx,tmp,:-1], '--', color='b')
			
 
	ax.annotate('unbound: {0}'.format(unbound)+', particles:{0}, '.format(len(dat))+'energy error={0:.2g}\n'.format((en-en0)/en0, (enb-en0b)/en0b)+'retro frac={0:.2f}, '.format(frac[idx])+r'$M_d$'+'={0}'.format(latex_exp.latex_exp(np.sum(init_disk[1:,-1])))+r' $M_{\odot}$'+', t={0} yr\nBlue, Dashed=Disruption'.format(latex_exp.latex_exp(idx*0.004*1.5e7)), (0.01, 0.99),\
		xycoords='axes fraction', va='top', fontsize=16)
	ax.set_xlabel('x [pc]')
	ax.set_ylabel('y [pc]')
	ax.set_xlim(-xlim, xlim)
	ax.set_ylim(-ylim, ylim)
	# # ax.set_xticks([-1,-0.5,0,0.5,1])
	# ax.set_ylim(-1,1)
	# ax.set_yticks([-1,-0.5,0,0.5,1])
	fig.savefig(loc+'/sim_movie_man_{0:03d}.png'.format(idx), ppi=72)
	plt.cla()

# np.savez(loc+'/x.npz', x)
# np.savez(loc+'/y.npz', y)
# np.savez(loc+'/z.npz', z)
