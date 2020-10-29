#!/usr/bin/env python

import configparser
import argparse
import uuid
# import sys

import numpy as np
from collections import OrderedDict
# sys.path.append('/usr/local/lib/python2.7/dist-packages/')
import rebound
import random as rand
from bin_analysis import bin_find_sim
from bash_command import bash_command as bc
import math
import reboundx
import os
import shlex
import cgs_const as cgs

from scipy.interpolate import interp1d
from extrap import extrap


from numpy.random import SeedSequence, default_rng
from numpy.random import RandomState
from numpy.random import MT19937



def get_last_index(loc="."):
	trials=bc.bash_command("echo trial_*").decode('utf-8')
	trials=shlex.split(trials)
	if (trials[0]=='trial_*'):
		return 1
	else:
		trials=np.array([tt.replace('trial_','') for tt in trials]).astype(int)
		end=np.sort(trials)[-1]
		return end+1


def rotate_vec(angle,axis,vec):    
	'''
	Rotate vector vec by angle around axis (couter-clockwise)
	'''
	vRot = vec*math.cos(angle) + np.cross(axis,vec)*math.sin(angle) + axis*np.dot(axis,vec)*(1 -math.cos(angle))
	return vRot	


def heartbeat(sim):
	print(sim.contents.dt, sim.contents.t)

# sim is a pointer to the simulation object,
# thus use contents to access object data.
# See ctypes documentation for details.
	# print(sim.contents.dt)

def delete_bins(sim, nparts, sections):
	##Integrate forward a small amount time to initialize accelerations.
	sim.move_to_com()
	##Integrate forward to ensure tidal forces are initialized
	# sim.integrate(sim.t+1.0e-30)
	##Look for binaries
	bins=bin_find_sim(sim)
	print(len(bins))
	np.savetxt('bins', bins)
	while len(bins)>0:
		# bins=np.array(bins)
		##Delete in reverse order (else the indices would become messed up)
		to_del=(np.sort(np.unique(bins[:,1]))[::-1]).astype(int)
		#print "deleting",len(to_del)
		for idx in to_del:
			print(type(idx), idx)
			sim.remove(index=int(idx))
		bins=bin_find_sim(sim)
		print(len(bins))
		N0=1
		##Update indices for each section after binary deletion
		for ss in sections:
			del1=len(np.intersect1d(range(nparts[ss][0],nparts[ss][-1]+1), to_del))
			tot1=nparts[ss][-1]-nparts[ss][0]+1
			nparts[ss]=(N0, N0+tot1-del1-1)
			N0=N0+tot1-del1

def get_tde_no_delR(sim, reb_coll):
	orbits = sim[0].calculate_orbits(primary=sim[0].particles[0])
	p1,p2 = reb_coll.p1, reb_coll.p2
	idx, idx0 = max(p1, p2), min(p1, p2)
	if idx0==0:
		##idx decremented by 1 because there is no orbit 0
		rp=orbits[idx-1].a*(1-orbits[idx-1].e)
		rg=sim[0].particles[0].m*(cgs.G*cgs.M_sun/cgs.c**2.0/cgs.pc)
		name=sim[0].simulationarchive_filename.decode('utf-8')
		f=open(name.replace('.bin', '_tde'), 'a+')
		f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(sim[0].t, orbits[idx-1].a, orbits[idx-1].e, orbits[idx-1].inc,\
			orbits[idx-1].omega, orbits[idx-1].Omega, sim[0].particles[idx].hash, sim[0].particles[idx].m))
		f.close()
		sim[0].move_to_com()

	return 0

def get_tde(sim, reb_coll):
	orbits = sim[0].calculate_orbits(primary=sim[0].particles[0])
	p1,p2 = reb_coll.p1, reb_coll.p2
	idx, idx0 = max(p1, p2), min(p1, p2)
	if idx0==0:
		##idx decremented by 1 because there is no orbit 0
		rp=orbits[idx-1].a*(1-orbits[idx-1].e)
		rg=sim[0].particles[0].m*(cgs.G*cgs.M_sun/cgs.c**2.0/cgs.pc)
		name=sim[0].simulationarchive_filename.decode('utf-8')
		f=open(name.replace('.bin', '_tde'), 'a+')

		if rp<10.0*rg:
			m1=sim[0].particles[0].m
			m2=sim[0].particles[idx].m
			sim[0].particles[0]=(m1*sim[0].particles[0]+m2*sim[0].particles[idx])/(m1+m2)
			sim[0].particles[0].m=(m1+m2)
			sim[0].remove(idx)

			sim[0].move_to_com()
			return 0
		f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(sim[0].t, orbits[idx-1].a, orbits[idx-1].e, orbits[idx-1].inc,\
			orbits[idx-1].omega, orbits[idx-1].Omega, sim[0].particles[idx].hash, sim[0].particles[idx].m))
		f.close()

		pp=sim[0].particles[idx]
		ppc=sim[0].particles[0]
		with open(name.replace('.bin', '_tde2'), 'a+') as f2:
			f2.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(sim[0].t, pp.x-ppc.x, pp.y-ppc.y, pp.z-ppc.z,\
				pp.vx-ppc.vx, pp.vy-ppc.vy, pp.vz-ppc.vz, pp.hash, pp.m))
		sim[0].move_to_com()

	return 0

def main():
	parser=argparse.ArgumentParser(
		description='Set up a rebound run')
	parser.add_argument('--config', nargs=1, default='config',
		help='File containing simulation parameters')
	parser.add_argument('--index', '-i', type=int, default=1,
		help='Index of output file')
	# parser.add_argument('--keep_bins', action='store_true',
	# 	help="Don't delete bins from simulation")


	##Parsing command line arguments.
	args=parser.parse_args()
	config_file=args.config
	##Unique tag for output file.
	# tag=str(uuid.uuid4())
	# print(get_last_index())

	loc="trial_{0}/".format(args.index)
	bc.bash_command('mkdir {0}'.format(loc))
	##Default stellar parameters 
	config=configparser.SafeConfigParser(defaults={'name': 'archive', 'N':'100', 
		'gravity':'basic', 'integrator':'ias15', 'dt':'0', 'keep_bins':'False', 'coll':'line', 'pRun':'0.1', 'pOut':'0.1', \
		'p':'1', 'frac':'2.5e-3', 'outDir':'./', 'gr':'True', 'rinf':'4.0', 'alpha':'1.5', 'beta':'1.5', 'rb':'3',\
		'rho_rb':'0','rt':'1e-4', 'merge':'False', 'menc_comp':'False', 'Mbh':'4e6',\
		'c':'4571304.57795483', 'delR':'True', 'epsilon':'1e-9', 'buff':'1.5', 'min_dt':'0', 'seed':'false', 'cen':'False'}, dict_type=OrderedDict)
	# config.optionxform=str
	config.read(config_file)

	##Name of output file 
	name=config.get('params', 'name')
	init_file=config.get('params', 'init_file')
	name=name+".bin"
	name=config.get('params', 'outDir')+'/'+name
	##Length of simulation and interval between snapshots
	pRun=config.getfloat('params', 'pRun')
	pOut=config.getfloat('params', 'pOut')
	keep_bins=config.getboolean('params', 'keep_bins')
	# rt=config.getfloat('params', 'rt')
	coll=config.get('params', 'coll')
	gr=config.getboolean('params', 'gr')
	rinf=config.getfloat('params', 'rinf')
	alpha=config.getfloat('params', 'alpha')
	beta=config.getfloat('params', 'beta')
	rb=config.getfloat('params', 'rb')
	rho_rb=config.getfloat('params', 'rho_rb')
	# menc_comp=config.getboolean('params', 'menc_comp')
	Mbh=config.getfloat('params', 'Mbh')

	#print pRun, pOut, rt, coll
	sections=config.sections()
	sections=sections[1:]
	##Initialized the rebound simulation
	sim = rebound.Simulation()
	sim.G = 1.	
	##Central object
	rt=config.getfloat('params', 'rt')	
	# sim.add(m = Mbh, r=rt, hash="smbh") 
	sim.gravity=config.get('params', 'gravity')
	sim.integrator=config.get('params', 'integrator')
	epsilon=config.getfloat('params', 'epsilon')
	min_dt=config.getfloat('params', 'min_dt')

	sim.ri_ias15.epsilon=epsilon
	sim.ri_ias15.min_dt=min_dt
	dt=config.getfloat('params', 'dt')
	if dt:
		sim.dt=dt

	buff=config.getfloat('params', 'buff')
	# mbar=6.0
	nparts={}
	num={}
	seed=config.getboolean('params', 'seed')
	##Set up a bunch of random states for random number generators
	rs=RandomState()
	if seed:
		print('test')
		ss = SeedSequence(12345)
		# Spawn off 10 child SeedSequences to pass to child processes.
		child_seeds = ss.spawn(100)
		# Spawn off 10 child SeedSequences to pass to child processes.
		rs=RandomState(MT19937(child_seeds[args.index]))


	init_dat=np.genfromtxt(init_file)
	order=np.argsort(init_dat[:,0])[::-1]
	init_dat=init_dat[order]
	sim.add(m=init_dat[0,0], x=init_dat[0, 1], y=init_dat[0, 2], z=init_dat[0, 3], vx=init_dat[0, 4], vy=init_dat[0, 5], vz=init_dat[0, 6], r=rt, hash='smbh')
	print(sim.particles[0].m)
	##Add particles; Can have different sections with different types of particles (e.g. heavy and light)
	##see the example config file in repository. Only require section is params which defines global parameters 
	##for the simulation (pRun and pOut).
	for ss in sections:
		num[ss]=int(config.get(ss, 'N'))
		N=int(buff*num[ss])
		##rescale stellar masses so they are a fixed fraction of the central mass.
		frac=config.getfloat(ss, 'frac')
		#Read data and setup rebound simulation to get orbital elements
		mbar=sim.particles[0].m*frac/num[ss]
		N0=len(sim.particles)
		samp=list(range(1,len(init_dat)))
		rs.shuffle(samp)
		samp=samp[:N]
		init_dat=init_dat[samp]

		# print(init_dat[:10], samp[:10])
		for l in range(0,N): # Adds stars
			# m=mbar
			sim.add(m = mbar, x=init_dat[l, 1], y=init_dat[l, 2], z=init_dat[l, 3],\
				vx=init_dat[l, 4], vy=init_dat[l, 5], vz=init_dat[l, 6], r=0, hash=str(l))
		##Indices of each component
		nparts[ss]=(N0,N0+N-1)
		print(nparts)


	##Deleting unbound particles
	orbs=sim.calculate_orbits(primary=sim.particles[0])
	smas=np.array([oo.a for oo in orbs])
	to_del=(np.sort(np.where(smas<0)[0])[::-1]).astype(int)+1
	print(len(to_del))
	for idx in to_del:
		sim.remove(index=int(idx))
	N0=1
	##Update indices for each section after deletion
	for ss in sections:
		del1=len(np.intersect1d(range(nparts[ss][0],nparts[ss][-1]+1), to_del))
		tot1=nparts[ss][-1]-nparts[ss][0]+1
		nparts[ss]=(N0, N0+tot1-del1-1)
		N0=N0+tot1-del1
	print(len(sim.particles), nparts)

	if not keep_bins:
		delete_bins(sim, nparts, sections)
	print(len(sim.particles), nparts)

	##Delete all of the excess particles
	for ss in sections[::-1]:
		to_del=range(nparts[ss][0]+num[ss], nparts[ss][-1]+1)[::-1]
		for idx in to_del:
			sim.remove(index=idx)
	orbs=sim.calculate_orbits(primary=sim.particles[0])
	smas=np.array([oo.a for oo in orbs])
	print(len(sim.particles), len(smas[smas<0]))

	sim.move_to_com()
	##Energy data (may be too slow?)
	fen=open(loc+name.replace('.bin', '_en'), 'w')
	fen.write(sim.gravity+'_'+sim.integrator+'_'+'{0}'.format(sim.dt))
	##Masses 
	ms=np.array([pp.m for pp in sim.particles[1:]])
	##Collisions
	sim.collision=coll
	sim.collision_resolve=get_tde
	delR=config.getboolean('params', 'delR')
	if not delR:
		print('delR:', delR)
		sim.collision_resolve=get_tde_no_delR

	##Stellar potential
	rebx = reboundx.Extras(sim)

	cen=config.getboolean('params', 'cen')
	cen='_cen' if cen else ''
	print(cen)
	if rinf>0:
		menc=rebx.add("menc")
		menc.params["rinf"]=rinf
		menc.params["alpha"]=alpha
	elif rho_rb>0:
		menc=rebx.add("menc_dp"+cen)
		menc.params["rb"]=rb
		menc.params["rho_rb"]=rho_rb
		menc.params["alpha"]=alpha
		menc.params["beta"]=beta
	##GR effects
	if gr:
		gr=rebx.add("gr")
		##Speed of light in simulation units.
		gr.params["c"]=config.getfloat('params', 'c')

	##Set up simulation archive for output
	# sa = rebound.SimulationArchive(loc+name, rebxfilename='rebx.bin')
	sim.automateSimulationArchive(loc+name,interval=pOut*pRun,deletefile=True)
	# sim.heartbeat=heartbeat
	sim.move_to_com()
	sim.simulationarchive_snapshot(loc+name)
	bc.bash_command('cp {0} {1}'.format(config_file, loc))


	en=sim.calculate_energy()
	print(sim.N, rebound.__version__)
	t=0.0
	delta_t=0.01*pRun
	orb_idx=0
	# print(delta_t, pRun)
	f=open(loc+'init_disk', 'w')
	for ii in range(len(sim.particles)):
		f.write('{0:.16e} {1:.16e} {2:.16e} {3:.16e} {4:.16e} {5:.16e} {6:.16e}\n'.format(sim.particles[ii].x-sim.particles[0].x, sim.particles[ii].y-sim.particles[0].y, sim.particles[ii].z-sim.particles[0].z,\
			sim.particles[ii].vx-sim.particles[0].vx, sim.particles[ii].vy-sim.particles[0].vy, sim.particles[ii].vz-sim.particles[0].vz, sim.particles[ii].m))
	f.close()

	print(sim.particles[1].hash)
	print(sim.integrator, sim.dt)
	##Period at the inner edge of the disk (hard-code former to 0.05 for now)
	a_min=0.05
	p_in=2.0*np.pi*(a_min**3.0/Mbh)**0.5
	my_step=0.1*p_in
	print(my_step)
	while(t<pRun):
		fen.write(str(sim.calculate_energy())+'\n')
		if t>=orb_idx*delta_t:
			orbits=sim.calculate_orbits(primary=sim.particles[0])
			np.savetxt(loc+name.replace('.bin', '_out_{0}.dat'.format(orb_idx)), [[oo.a, oo.e, oo.inc, oo.Omega, oo.omega, oo.f] for oo in orbits])
			np.savetxt(loc+name.replace('.bin', '_out_{0}.hash'.format(orb_idx)),\
			 np.array([str(sim.particles[i].hash) for i in range(len(sim.particles))]).astype(str), fmt='%s')
			orb_idx+=1
		sim.move_to_com()
		sim.integrate(sim.t+my_step)
		orbits=sim.calculate_orbits(primary=sim.particles[0])
		rps=np.array([oo.a*(1-oo.e) for oo in orbits])
		if np.any(rps<rt):
			##Add 1 since central black hole is not included in rps
			indics=np.array(range(len(rps)))[rps<rt]+1
			print(indics)
			with open(loc+'tmp_tde', 'a+') as f2:
				for idx in indics:
					pp=sim.particles[int(idx)]
					f2.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(sim.t, pp.x, pp.y, pp.z,\
						pp.vx, pp.vy, pp.vz, pp.hash, pp.m))

		##Should increment line below by pin not deltat
		t+=my_step
	fen.close()




if __name__ == '__main__':
	main()








