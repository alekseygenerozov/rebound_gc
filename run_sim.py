#!/usr/bin/env python

import configparser
import argparse
import uuid
# import sys

import numpy as np
from collections import OrderedDict
# sys.path.append('/usr/local/lib/python2.7/dist-packages/')
import rebound
# import random as rand
from bin_analysis import bin_find_sim
from bash_command import bash_command as bc
import math
import reboundx
import os
import shlex
import cgs_const as cgs

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

def gen_disk(ang1, ang1_mean, ang2, ang2_mean, ang3, ang3_mean, rs):
	'''
	This is from some old code that starts with perfectly aligned e and j vectors and then rotates them by a small amount
	'''
	ehat = np.array([1,0,0])
	jhat = np.array([0,0,1])
	bhat = np.cross(jhat,ehat)    # rotate jhat by angle1 over major axis and angle 2 minor axis
	# rotate ehat by angle2 over minor axis (for consistency) and angle3 about jhat
	angle1 = rs.normal(ang1_mean, ang1, 1)
	angle2 = rs.normal(ang2_mean, ang2, 1)
	angle3 = rs.normal(ang3_mean, ang3, 1)    
	jhat = rotate_vec(angle1,ehat,jhat)
	jhat = rotate_vec(angle2,bhat,jhat)
	ehat = rotate_vec(angle2,bhat,ehat)
	ehat = rotate_vec(angle3,jhat,ehat)    
	n = np.cross(np.array([0,0,1]), jhat)
	n = n / np.linalg.norm(n)   
	Omega = math.atan2(n[1], n[0])
	omega = math.acos(np.dot(n, ehat))
	if ehat[2] < 0:
		omega = 2*np.pi - omega    
	inc=math.acos(jhat[2])    
	return inc, Omega, omega


def density(min1, max1, p, rs):
	'''
	Generate a random from a truncated power law PDF with power law index p. 
	min1 and max1

	'''
	r=rs.random(1)[0]
	if p==1:
		return min1*np.exp(r*np.log(max1/min1))
	else:
		return (r*(max1**(1.-p)-min1**(1.-p))+min1**(1.-p))**(1./(1-p))

def mpow_gc(mbar, rs):
	return density(1., 60., 1.7, rs)*(mbar/6.0)


def mfixed(mbar, rs):
	return mbar

def heartbeat(sim):
	name=sim[0].simulationarchive_filename.decode('utf-8')
	with open(name.replace('.bin', 'time_log'), 'a') as ff:
		ff.write('{0} {1}\n'.format(sim.contents.dt, sim.contents.t))
		ff.flush()
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
	config=configparser.SafeConfigParser(defaults={'name': 'archive', 'N':'100', 'e':'0.7', 'eslope':'0',
		'gravity':'basic', 'integrator':'ias15', 'dt':'0', \
		'a_min':'0.05', 'a_max':'0.5', 'ang1_mean':'0', 'ang2_mean':'0', 'ang3_mean':'0', 'ang1':'2.',\
		'ang2':'2.', 'ang3':'2.', 'keep_bins':'False', 'coll':'line', 'pRun':'0.1', 'pOut':'0.1', \
		'p':'1', 'frac':'2.5e-3', 'outDir':'./', 'gr':'True', 'rinf':'4.0', 'alpha':'1.5', 'beta':'1.5', 'rb':'3',\
		'rho_rb':'0','rt':'1e-4', 'mf':"mfixed", 'merge':'False', 'menc_comp':'False', 'Mbh':'4e6',\
		'c':'4571304.57795483', 'delR':'True', 'epsilon':'1e-9', 'twist':'0', 'seed':'False'}, dict_type=OrderedDict)
	# config.optionxform=str
	config.read(config_file)

	##Name of our put file 
	name=config.get('params', 'name')
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

	#print pRun, pOut, rt, coll
	sections=config.sections()
	sections=sections[1:]
	##Initialized the rebound simulation
	sim = rebound.Simulation()
	sim.G = 1.	
	##Central object
	rt=config.getfloat('params', 'rt')	
	sim.add(m = Mbh, r=rt, hash="smbh") 
	sim.gravity=config.get('params', 'gravity')
	sim.integrator=config.get('params', 'integrator')
	epsilon=config.getfloat('params', 'epsilon')
	sim.ri_ias15.epsilon=epsilon
	dt=config.getfloat('params', 'dt')
	if dt:
		sim.dt=dt

	buff=1.5
	mbar=6.0
	nparts={}
	num={}
	##Add particles; Can have different sections with different types of particles (e.g. heavy and light)
	##see the example config file in repository. Only require section is params which defines global parameters 
	##for the simulation (pRun and pOut).
	for ss in sections:
		num[ss]=int(config.get(ss, 'N'))
		N=int(buff*num[ss])
		e=config.getfloat(ss, 'e')
		eslope=config.getfloat(ss, 'eslope')
		##rescale stellar masses so they are a fixed fraction of the central mass.
		frac=config.getfloat(ss, 'frac')
		mbar=sim.particles[0].m*frac/num[ss]
		print(mbar)
		a_min=config.getfloat(ss, 'a_min')
		a_max=config.getfloat(ss, 'a_max')
		p=config.getfloat(ss, 'p')
		ang1_mean=config.getfloat(ss, 'ang1_mean')
		ang1=config.getfloat(ss, 'ang1')
		ang2_mean=config.getfloat(ss, 'ang2_mean')
		ang2=config.getfloat(ss, 'ang2')
		ang3_mean=config.getfloat(ss, 'ang3_mean')
		ang3=config.getfloat(ss, 'ang3')
		twist=config.getfloat(ss, 'twist')

		##We can generalize this to be a function?
		# rt=config.getfloat(ss, 'rt')

		N0=len(sim.particles)
		for l in range(0,N): # Adds stars
			##Use AM's code to generate disk with aligned eccentricity vectors, but a small scatter in i and both omegas...
			a0=density(a_min, a_max, p, rs)
			tt=np.interp(a0, [a_min, a_max], [0, twist])
			inc, Omega, omega=gen_disk(ang1*np.pi/180., (ang1_mean)*np.pi/180., ang2*np.pi/180., (ang2_mean)*np.pi/180., ang3*np.pi/180., (ang3_mean+tt)*np.pi/180.0, rs)
			##Better way to include the mass spectrum...name of function define MF as a parameter.
			m=globals()[config.get(ss, "mf")](mbar, rs)
			M = rs.uniform(0., 2.*np.pi)
			# print(m, (sim.particles[0].m/m)**(1./3.)*0.1*cgs.au/cgs.pc)
			sim.add(m = m, a = a0, e = e*(a0/a_min)**-eslope, inc=inc, omega=omega, Omega=Omega, M = M, primary=sim.particles[0],\
				r=0, hash=str(l))
		##Indices of each component
		nparts[ss]=(N0,N0+N-1)
	
	sim.move_to_com()


	fen=open(loc+name.replace('.bin', '_en'), 'w')
	fen.write(sim.gravity+'_'+sim.integrator+'_'+'{0}'.format(sim.dt))
	if not keep_bins:
		print('deleting binaries')
		delete_bins(sim, nparts, sections)
	print(nparts)

	##Delete all of the excess particles
	for ss in sections[::-1]:
		to_del=range(nparts[ss][0]+num[ss], nparts[ss][-1]+1)[::-1]
		for idx in to_del:
			# print(type(idx))
			sim.remove(index=idx)
	print(len(sim.particles))

	ms=np.array([pp.m for pp in sim.particles[1:]])
	print(np.sum(ms))
	sim.collision=coll
	sim.collision_resolve=get_tde
	delR=config.getboolean('params', 'delR')
	##We cannot get merges in restart !!!
	# merge=config.getboolean('params', 'merge')
	# if merge:
	# 	sim.collision_resolve='merge'
	##Move this above the previo
	if not delR:
		print('delR:', delR)
		sim.collision_resolve=get_tde_no_delR
	# print("gr:", gr, "rinf:", rinf, "alpha:", alpha, "merge:", merge)


	


	##Stellar potential
	rebx = reboundx.Extras(sim)
	if rinf>0:
		print('menc1')
		menc=rebx.add("menc")
		menc.params["rinf"]=rinf
		menc.params["alpha"]=alpha
	elif rho_rb>0:
		print('menc2')
		menc=rebx.add("menc_dp")
		menc.params["rb"]=rb
		menc.params["rho_rb"]=rho_rb
		menc.params["alpha"]=alpha
		menc.params["beta"]=beta
	##GR effects
	if gr:
		print('gr')
		gr=rebx.add("gr")
		##Speed of light in simulation units.
		gr.params["c"]=config.getfloat('params', 'c')

	##Set up simulation archive for output
	# sa = rebound.SimulationArchive(loc+name, rebxfilename='rebx.bin')
	sim.automateSimulationArchive(loc+name,interval=pOut*pRun, deletefile=True)
	sim.heartbeat=heartbeat
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
	##Period at the inner edge of the disk

	p_in=min(2.0*np.pi*(a_min**3.0/Mbh)**0.5, pRun)
	my_step=0.1*p_in
	# print(2.0*np.pi*(a_min**3.0/Mbh)**0.5)
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








