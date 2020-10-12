import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages/')
import rebound
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def get_com(ps):
	'''
	Get center of mass for a collection of particles
	'''
	com=ps[0].m*ps[0]
	ms=ps[0].m
	for pp in ps[1:]:
		com=com+pp.m*pp
		ms+=pp.m
	return com/ms


def orb_plane_proj(p1, p2, p3):
	'''
	Project p3 into orbital plane of p1 and p2
	'''
	##Vector going from p2 to p1
	dp = p1 - p2  
	##Unit vector going from p2 to p1
	rhat = np.array(dp.xyz)/np.linalg.norm(dp.xyz)
	v2 = p2.vxyz
	vhat2 = v2/np.linalg.norm(v2)
	norm2=np.cross(rhat, vhat2)
	##Unit vector perpendicular to the orbital plane of p1 and p2
	norm2=norm2/np.linalg.norm(norm2)

	pos=np.array(p3.xyz)
	print(np.linalg.norm(p3.xyz), p3.xyz, norm2)
	##Projection of p3's position to the orbital plane
	# pos=pos-np.dot(pos, norm2)*norm2
	# print np.linalg.norm(p3.xyz)

	##x-component; note that xhat point from particle 1 to particle 2.
	x=-np.dot(pos, rhat)
	##Define y using normal to plan and rhat
	yhat=np.cross(norm2, -rhat)
	y=np.dot(pos, yhat)
	return x,y


def bin_props(p1, p2):
	'''
	Auxiliary function to get binary properties for two particles. 

	p1 and p2 -- Two particles from a rebound simulation.
	'''
	dp = p1 - p2   
	d2 = dp.x*dp.x+dp.y*dp.y+dp.z*dp.z
	# print(d2)

	##Calculate com velocity of two particles...
	##Masses
	m1=p1.m
	m2=p2.m
	com = (m1*p1+m2*p2)/(m1+m2)
	##Particle velocities in com frame
	p1_com = p1 - com
	p2_com = p2 - com
	v12 = (p1_com.vx**2.+p1_com.vy**2.+p1_com.vz**2.)
	v22 = (p2_com.vx**2.+p2_com.vy**2.+p2_com.vz**2.)

	d12 = (p1_com.x**2.+p1_com.y**2.+p1_com.z**2.)
	d22 = (p2_com.vx**2.+p2_com.y**2.+p2_com.z**2.)	
	##Difference in the forces acting on the two particles;
	ft = np.array([m2*(p2.ax)-m2*(com.ax), m2*(p2.ay)-m2*(com.ay), m2*(p2.az)-m2*com.az])
	##Unit vector pointing from particle 2 to particle 1
	rhat = np.array(dp.xyz)/d2**0.5
	f12 = m1*m2/d2*rhat 
	##Tidal force that star 2 experiences
	ft = ft - f12
	ft = np.linalg.norm(ft)

	##Kinetic and potential energies
	ke = 0.5*m1*v12+0.5*m2*v22
	##Potential energy; Assumes G = 1
	pe = (m1*m2)/d2**0.5

	##Distance of binary center of mass from COM of system (should be near central SMBH)
	com_d=(com.x**2.+com.y**2.+com.z**2.)**0.5
	a_bin=(m1*m2)/(2.*(pe-ke))
	##Angular momentum in binary com
	j_bin=m1*np.cross(p1_com.xyz, p1_com.vxyz)+m2*np.cross(p2_com.xyz, p2_com.vxyz)
	##Angular momentum of binary com
	j_com=(m1+m2)*np.cross(com.xyz, com.vxyz)

	#Inclination of star's orbit wrt the binary's orbit win the disk 
	inc=np.arccos(np.dot(j_bin, j_com)/np.linalg.norm(j_bin)/np.linalg.norm(j_com))*180./np.pi
	mu=m1*m2/(m1+m2)
	##Eccentricity of the binary
	e_bin=(1.-np.linalg.norm(j_bin)**2./((m1+m2)*a_bin)/(mu**2.))

	return com_d, a_bin, e_bin, p1_com, p2_com, d2, inc, ft


def bin_find(loc):
	'''
	Find all binaries for a given sim and time. 

	loc should be a tuple containing the time and 
	simulation name. (bin_find_sim below 
	performs the same function but accepts
	a simulation object; this version is useful 
	because it can be used w rebound's InterruptiblePool)

	Return a numpy array. 1st columnn is time, 2nd and 3rd columns
	are indices of the binary stars. 

	Next columns are binary separation, sma, ratio of sma to the hill radius, 
	and the binary eccentricity. 

	'''
	t,name=loc
	sat = rebound.SimulationArchive(name)
	sim = sat.getSimulation(t)
	##Ensure we are in the com frame of the simulation.
	sim.move_to_com()
	##Integrate forward for a small time to ensure that the accelerations
	##are in sync with the rest of the simulation (this is important for
	##calculating tidal forces...
	sim.integrate(sim.t+sim.t*1.0e-14)

	ps = sim.particles
	##Mass of central SMBH
	m0 = ps[0].m
	bin_indics=[]
	for i1, i2 in combinations(list(range(1, sim.N)),2): # get all pairs of indices/start at 1 to exclude SMBH
		##Call bin_props above to extract binary properties
		com_d, a_bin, e_bin, p1_com, p2_com, d2, inc, ft = bin_props(ps[i1], ps[i2])
		m1,m2 =ps[i1].m, ps[i2].m
		##Hill sphere condition.
		inside_hill=(a_bin<((m1+m2)/m0)**(1./3.)*com_d)
		tidal_2 = (m1*m2/d2>ft)

		##If the kinetic energy is less than the potential energy 
		if ((a_bin>0) and (inside_hill) and (tidal_2)):
		#if ((a_bin>0) and (inside_hill)):
			rh=(((m1+m2)/m0)**(1./3.)*com_d)
			vh=rh*(m0/com_d**3.)**0.5
			bin_indics.append([sim.t, i1, i2, d2**0.5, a_bin, a_bin/(((m1+m2)/m0)**(1./3.)*com_d), e_bin, rh, vh])

	return np.array(bin_indics)
	#return Table(bin_indics, names=['t', 'i1', 'i2', 'bin_sep', 'a_bin', 'a_bin/r_h', 'e_bin', 'rh', 'vh'])

 
def bin_find_sim(sim):
	##Ensure we are in the com frame of the simulation.
	sim.move_to_com()
	##Integrate forward for a small time to ensure that the accelerations
	##are in sync with 
	sim.integrate(sim.t+sim.t*1.0e-14)
	
	ps = sim.particles
	##mass of of primary 
	m0 = ps[0].m
	bin_indics=[]
	for i1, i2 in combinations(list(range(1, sim.N)),2): # get all pairs of indices/start at 1 to exclude SMBH
		com_d, a_bin, e_bin, p1_com, p2_com, d2, inc, ft = bin_props(ps[i1], ps[i2])
		m1,m2 =ps[i1].m, ps[i2].m
		##Hill sphere condition.
		inside_hill=(a_bin<((m1+m2)/m0)**(1./3.)*com_d)
		tidal_2 = (m1*m2/d2>ft)

		##If the kinetic energy is less than the potential energy 
		if ((a_bin>0) and (inside_hill) and (tidal_2)):
			rh=(((m1+m2)/m0)**(1./3.)*com_d)
			vh=rh*(m0/com_d**3.)**0.5
			bin_indics.append([sim.t, i1, i2, d2**0.5, a_bin, a_bin/(((m1+m2)/m0)**(1./3.)*com_d), e_bin, rh, vh])

	return np.array(bin_indics)
	#return Table(bin_indics, names=['t', 'i1', 'i2', 'bin_sep', 'a_bin', 'a_bin/r_h', 'e_bin', 'rh', 'vh'])

def p_dist(loc, idx):
	t,name=loc
	sat = rebound.SimulationArchive(name)
	sim = sat.getSimulation(t)
	ps=sim.particles
	ds=np.empty(len(ps))
	for ii in range(0,len(ps)):
		dp=ps[ii]-ps[idx]
		ds[ii] = (dp.x*dp.x+dp.y*dp.y+dp.z*dp.z)**0.5

	order=np.argsort(ds)
	return order, ds[order]

def com_plot(sa_name, i1, i2, extras=[], name='', cols=['r', 'g', 'k'], idx_min=0, idx_max=None, lim=0.1):
	planes = [['x', 'y'], ['x', 'z'], ['y','z']]
	fig,ax=plt.subplots(nrows=1, ncols=3, figsize=(10*len(planes),10))
	#fig.patch.set_visible(False)

	for kk in range(3):
		ax[kk].set_xlim(-lim,  lim)
		ax[kk].set_ylim(-lim, lim)
		ax[kk].set_xlabel(planes[kk][0])
		ax[kk].set_ylabel(planes[kk][1])
		#ax[kk].patch.set_visible(False)

	sa = rebound.SimulationArchive(sa_name)
	if not idx_max:
		idx_max=len(sa)
	m0=sa[0].particles[0].m
	for ii in range(idx_min, idx_max):
		for kk, plane in enumerate(planes):
			sim=sa[ii]
			p1,p2=sim.particles[i1],sim.particles[i2]
			m1,m2=p1.m,p2.m
			com_d, a_bin, e_bin, p1_com, p2_com, d2, inc, ft = bin_props(p1,p2)
			p1_pos=getattr(p1_com, plane[0]), getattr(p1_com, plane[1])
			p2_pos=getattr(p2_com, plane[0]), getattr(p2_com, plane[1])
			com=get_com([p1, p2])

			ax[kk].plot(p1_pos[0], p1_pos[1], 'o', markersize=2, color=cols[0])
			ax[kk].plot(p2_pos[0], p2_pos[1], 'o', markersize=2, color=cols[1])
			for jj, extra in enumerate(extras):
				ax[kk].plot(getattr(sim.particles[extra]-com, plane[0]), getattr(sim.particles[extra]-com, plane[1]), 'o', markersize=2, color=cols[(2+jj)%len(cols)])
			# if inset:
			# 	ax2.set_xlim(-(1+e_bin**0.5)*a_bin, (1+e_bin**0.5)*a_bin)
			# 	ax2.set_ylim(-(1+e_bin**0.5)*a_bin, (1+e_bin**0.5)*a_bin)
			# 	ax2.plot(p1_pos[0], p1_pos[1], 'o', markersize=2, color=cols[0])
			# 	ax2.plot(p2_pos[0], p2_pos[1], 'o', markersize=2, color=cols[1])

				# for jj,extra in enumerate(extras):
				# 	ax.plot(sim.particles[extra].x-com.x, sim.particles[extra].y-com.y, 'o', markersize=2, color=cols[(2+jj)%len(cols)])
			# print name+'com2_{0:03d}.png'.format(ii)
		ann=ax[1].annotate('a={0:2.2g}, a,/rt={1:2.2g}, r={2:2.2g}\n e^2={3:2.2g}, 1-e^2={4:2.2g}\n i={5:2.2g}'\
			.format(a_bin, a_bin/(((m1+m2)/m0)**(1./3.)*com_d), com_d, e_bin, 1-e_bin, inc),\
			(0.9*lim, 0.9*lim), horizontalalignment='right',verticalalignment='top', fontsize=20)
		#fig.canvas.print_png(open(sa_name.replace('.bin', '')+name+'_com_{0:03d}.png'.format(ii), 'w'))
		fig.savefig(sa_name.replace('.bin', '')+name+'_com_{0:03d}.png'.format(ii), bbox_inches='tight', pad_inches=0)
		ann.remove()


def lag_simple_plot(sa_name, i1, i2, extras=[], name='', cols=['r', 'g', 'k'], idx_min=0, idx_max=None, lim=0.1, ms=2, interval=1):
	'''
	Project particles to co-rotating frame ; this is function is only accurate if the orbital plane is the xy plane.
	'''

	planes = [['x', 'y'], ['x', 'z'], ['y','z']]
	#fig.patch.set_visible(False)


	sa = rebound.SimulationArchive(sa_name)
	if not idx_max:
		idx_max=len(sa)
	m0=sa[0].particles[0].m
	for ii in range(idx_min, idx_max, interval):
		fig,ax=plt.subplots(nrows=1, ncols=3, figsize=(10*len(planes),10))
		for kk, plane in enumerate(planes):
			ax[kk].set_xlim(-lim,  lim)
			ax[kk].set_ylim(-lim, lim)
			ax[kk].set_xlabel(planes[kk][0])
			ax[kk].set_ylabel(planes[kk][1])

			sim=sa[ii]
			p1,p2=sim.particles[i1],sim.particles[i2]
			m1,m2=p1.m,p2.m
			com_d, a_bin, e_bin, p1_com, p2_com, d2, inc, ft = bin_props(p1,p2)
			theta=-np.arctan2(p2_com.y,p2_com.x)
			ct,st=np.cos(theta), np.sin(theta)
			p1_xyz=np.dot([[ct, -st,0], [st, ct,0], [0,0,1]], p1_com.xyz)
			p2_xyz=np.dot([[ct, -st,0], [st, ct,0], [0,0,1]], p2_com.xyz)
			p1_xyz={'x':p1_xyz[0], 'y':p1_xyz[1], 'z':p1_xyz[2]}
			p2_xyz={'x':p2_xyz[0], 'y':p2_xyz[1], 'z':p2_xyz[2]}


			p1_pos=p1_xyz[plane[0]], p1_xyz[plane[1]]
			p2_pos=p2_xyz[plane[0]], p2_xyz[plane[1]]
			com=get_com([p1, p2])

			ax[kk].plot(p1_pos[0], p1_pos[1], 'o', markersize=ms, color=cols[0])
			ax[kk].plot(p2_pos[0], p2_pos[1], 'o', markersize=ms, color=cols[1])
			for jj, extra in enumerate(extras):
				xyz=np.dot([[ct, -st,0], [st, ct,0], [0,0,1]], (sim.particles[extra]-com).xyz)
				xyz={'x':xyz[0], 'y':xyz[1], 'z':xyz[2]}
				ax[kk].plot(xyz[plane[0]], xyz[plane[1]], 'o', markersize=ms, color=cols[(2+jj)%len(cols)])

		alpha=m2/(m1+m2)
		l1=(d2**0.5*(1.-(alpha/3.)**(1./3.)))
		l2=(d2**0.5*(1.+(alpha/3.)**(1./3.)))
		l3=(-d2**0.5*(1.+5./12.*alpha))
		ax[0].plot(l1, 0, 'gv')
		ax[0].plot(l2, 0, 'gv')
		ax[0].plot(l3, 0, 'gv')
		ax[0].plot(d2**0.5*np.cos(np.pi/3.), d2**0.5*np.sin(np.pi/3.), 'gv')
		ax[0].plot(d2**0.5*np.cos(np.pi/3.), -d2**0.5*np.sin(np.pi/3.), 'gv')

		fig.savefig(sa_name.replace('.bin', '')+name+'_com_{0:03d}.png'.format(ii), bbox_inches='tight', pad_inches=0)
		plt.clf()
		#ann.remove()



def lag_plot(sa_name, i1, i2, extras=[], name='',  idx_min=0, idx_max=None, lim=0.1, ms=2, interval=1, cols=['r']):
	'''
	Project particles to co-rotating frame only accurate if the orbital plane is the xy plane.
	'''
	sa = rebound.SimulationArchive(sa_name)
	if not idx_max:
		idx_max=len(sa)
	m0=sa[0].particles[0].m
	for ii in range(idx_min, idx_max, interval):
		fig,ax=plt.subplots(figsize=(10,10))
		ax.set_ylim(-lim, lim)
		ax.set_xlim(-lim, lim)

		sim=sa[ii]
		p1,p2=sim.particles[i1],sim.particles[i2]
		m1,m2=p1.m,p2.m
		com_d, a_bin, e_bin, p1_com, p2_com, d2, inc, ft = bin_props(p1,p2)
		x1,x2=np.linalg.norm(p1_com.xyz), np.linalg.norm(p2_com.xyz)
		ax.plot(x2, 0, 'ko', markersize=ms)
		ax.plot(-x1, 0, 'ko', markersize=ms)
		for jj, extra in enumerate(extras):
			p3=sim.particles[extra]
			x,y=orb_plane_proj(p1, p2, p3)
			ax.plot(x,y, 'o', color=cols[jj%len(cols)], markersize=ms)
		##Plotting Lagrange points
		alpha=m2/(m1+m2)
		l1=(d2**0.5*(1.-(alpha/3.)**(1./3.)))
		l2=(d2**0.5*(1.+(alpha/3.)**(1./3.)))
		l3=(-d2**0.5*(1.+5./12.*alpha))
		ax.plot(l1, 0, 'gv')
		ax.plot(l2, 0, 'gv')
		ax.plot(l3, 0, 'gv')
		ax.plot(d2**0.5*np.cos(np.pi/3.), d2**0.5*np.sin(np.pi/3.), 'gv')
		ax.plot(d2**0.5*np.cos(np.pi/3.), -d2**0.5*np.sin(np.pi/3.), 'gv')

		fig.savefig(sa_name.replace('.bin', '')+name+'_lag_{0:03d}.png'.format(ii), bbox_inches='tight', pad_inches=0)
		plt.clf()

		

class BinAnalysis(object):
	def __init__(self, sa_name):
		'''
		Getting properties of all of the binaries in a rebound simulation run.
		'''
		self.sa_name=sa_name
		#sa=rebound.SimulationArchive(sa_name)
		#self.m0=sa[0].particles[0].m
		# self.tords=np.arange(0., 500.1*2.*np.pi, 0.2*np.pi)
		
		try:
			self.ts= np.genfromtxt(sa_name.replace('.bin', '_times'))
			self.bins=np.genfromtxt(sa_name.replace('.bin','_bins.csv'), delimiter=',')
			self.masses=np.genfromtxt(sa_name.replace('.bin', '_masses'))
		except:
			print("Generating bin table")
			self.__bin_init__()
		self.delta_t=np.diff(self.ts)[0]
		#self.locs = [[tt, sa_name] for tt in self.ts]

		self.pairs_arr=self.bins[:,[1,2]].astype(int)
		self.times_arr=self.bins[:,0]
		self.pairs=np.array([{int(self.bins[i,1]),int(self.bins[i,2])} for i in range(len(self.bins))])
		tmp,idx=np.unique(self.pairs.astype(str), return_index=True)
		self.pairs_u=self.pairs[idx]
		##Filter pairs by mass
		mh_thres=np.median(self.masses)
		pairs_u_arr=np.array([list(pp) for pp in self.pairs_u])
		filt=np.array([((self.masses[pp[0]-1]<=mh_thres) & (self.masses[pp[1]-1]<=mh_thres)) for pp in pairs_u_arr])
		pairs_u_arr_light=pairs_u_arr[filt]
		self.pairs_u_light=np.array([set(pp) for pp in pairs_u_arr_light])
		filt=np.array([((self.masses[pp[0]-1]>mh_thres) & (self.masses[pp[1]-1]>mh_thres)) for pp in pairs_u_arr])
		pairs_u_arr_heavy=pairs_u_arr[filt]
		self.pairs_u_heavy=np.array([set(pp) for pp in pairs_u_arr_heavy])
		filt=np.array([((self.masses[pp[0]-1]>mh_thres) ^ (self.masses[pp[1]-1]>mh_thres)) for pp in pairs_u_arr])
		pairs_u_arr_mixed=pairs_u_arr[filt]
		self.pairs_u_mixed=np.array([set(pp) for pp in pairs_u_arr_mixed])


		# pair_u_arr_heavy=[self.masses[pp[0]>1.0e-4] & self.masses[pp[1]>1.0e-4] for pp in pairs_u_arr]
		# self.pairs_u_heavy=np.array([set(pp) for pp in pair_u_arr_heavy])


	def __bin_init__(self):
		sa = rebound.SimulationArchive(self.sa_name)
		##Range of snapshots to get -- sometime bin files contain too many densely spaced snapshots.
		##Make sure we get get data from only every 0.1 orbits...Replace the last number with snapshot interval
		##read directly from the simulation.
		self.tords=np.arange(0, sa[-1].t+0.01*np.pi, 0.2*np.pi)
		self.tords[0]=2.0e-15

		sims= sa.getSimulations(self.tords)
		self.ts=[sim.t for sim in sims] 
		np.savetxt(self.sa_name.replace('.bin', '_times'), self.ts)
		locs = [[tt, self.sa_name] for tt in self.ts]
		#pool = rebound.InterruptiblePool(processes=3)
		bins = list(map(bin_find,locs))
		#bins=np.array(bins)
		filt=np.array([len(bins[i])>0 for i in range(len(bins))])
		bins2=np.array(bins)[filt]
		bins2=np.concatenate(bins2)
		self.bins=bins2

		np.savetxt(self.sa_name.replace('.bin','_bins.csv'), self.bins,delimiter=',')
		self.masses = np.array([pp.m for pp in sa[0].particles[1:]])
		np.savetxt(self.sa_name.replace('.bin', '_masses'), self.masses)


	def sigs(self, ii):
		sa = rebound.SimulationArchive(self.sa_name)
		sigs = np.std([np.array(pp.vxyz) for pp in sa[ii].particles[1:]],axis=0)
		return sigs

	def N_bin_analytic(self, ii, rh, omega):
		vh=rh*omega
		return ((4.*np.pi)/3.)*rh**2.*(vh/self.sigs(ii)[2])**4.

	def exotica(self):
		'''
		Identifying triples and exchange interactions.
		'''
		for ns in range(np.min(self.pairs_arr), np.max(self.pairs_arr)+1):
			last=[]
			tlast=0.
			for tt in np.unique(self.times_arr):
				filt=(self.times_arr==tt)
				filt2=[(ns in pp) for pp in self.pairs[filt]]
				tmp=self.pairs[filt][filt2]

				if len(tmp)>1:
					print("star {0}, {1} bound stars!, tt={2}".format(ns, len(tmp)+1, tt/(2.*np.pi)))
				elif (len(tmp)==1) and (len(last)==1) and (list(tmp)!=list(last)) and (tt-tlast<1.5*self.delta_t):
					print("star {0}, exchange!, tt={1}, {2}->{3}, {4}".format(ns, tt/(2.*np.pi), last, tmp, (tt-tlast)/self.delta_t))
				last=np.copy(tmp)
				tlast=tt

	def num_bins(self):
		'''
		Number of binaries for each snapshot of the simulation.
		'''
		num_bins=[len(self.times_arr[np.isclose(self.times_arr,tt, atol=0., rtol=1.0e-12)]) for tt in self.ts]
		return num_bins

	def num_bins_filt(self):
		'''
		Count number of binaries but only include binaries that complete at 
		least one orbit...
		'''
		times=self.bin_times()
		filt=times>1.		
		bins2=self.bins[filt]
		times_arr=bins2[:,0]

		num_bins=[len(times_arr[np.isclose(times_arr,tt, atol=0., rtol=1.0e-12)]) for tt in self.ts]
		return num_bins

	def bin_times(self, norm=True, extra='', total=False):
		pairs=self.pairs
		pairs_u=getattr(self, 'pairs_u'+extra)

		t_survs=np.zeros(len(pairs_u))
		##For each binary identify how long it survives
		for ii,pp in enumerate(pairs_u):
			##Identify all times where each binary pair exists.
			t_bin=self.times_arr[pairs==pp]
			t_surv=t_bin[-1]-t_bin[0]
			#Index of one of the stars in the pair
			idx=self.pairs_arr[pairs==pp][0,1]
			idx2=self.pairs_arr[pairs==pp][0,0]

			##Edge case: Binary splits up and forms again. See if the binary has skipped any snapshots.
			##Note that snapshots may not be exactly evenly spaced in time...
			# if np.any(lens>1):
			# 	print self.sa_name,np.array([self.ts[np.where(np.abs(self.ts-t_bin[jj])/t_bin[jj]<1.0e-12)[0]] for jj in range(len(t_bin))])[lens>1]
			diffs=np.diff([np.where(np.abs(self.ts-t_bin[jj])/t_bin[jj]<1.0e-12)[0][0] for jj in range(len(t_bin))])
			## This second line is necessary in cases for which we have duplicate snapshots.
			diffs1=np.diff([np.where(np.abs(self.ts-t_bin[jj])/t_bin[jj]<1.0e-12)[0][-1] for jj in range(len(t_bin))])


			if np.any((diffs>1.01) & (diffs1>1.01)):
				tmp=np.split(t_bin, (np.where((diffs>1.01) & (diffs1>1.01)))[0]+1)
				tmp2=[tmp[i][-1]-tmp[i][0] for i in range(len(tmp))]
				order=np.argsort(tmp2)
				# t_bin=tmp[order[-1]]
				t_surv=tmp2[order[-1]] if not total else np.sum(tmp2)

			##Survival time of the binary normalized to the binary orbital period
			#sim=sa.getSimulation(t_bin[-1])
			#sim.move_to_com()
			#t_orb=sim.particles[idx].P
			##Binary orbital period -- not this is not a constant--take the minimum orbital period
			if norm:
				m1=self.masses[idx-1]
				m2=self.masses[idx2-1]
				t_orb = 2.*np.pi*np.min((self.bins[self.pairs==pp][:,4]**3./(m1+m2))**0.5)
				t_surv=t_surv/t_orb 
			t_survs[ii]=t_surv

		return t_survs

	# def bin_times_fast(self, norm=True):
	# 	pairs=self.pairs
	# 	pairs_u=self.pairs_u
	# 	t_survs=np.zeros(len(pairs_u))
	# 	##For each binary identify how long it survives
	# 	for ii,pp in enumerate(pairs_u):
	# 		##Identify all times where each binary pair exists.
	# 		t_bin=self.times_arr[pairs==pp]
	# 		#Index of one of the stars in the pair
	# 		idx=self.pairs_arr[pairs==pp][0,1]
	# 		idx2=self.pairs_arr[pairs==pp][0,0]
	# 		##snapshots are more or less regularly spaced: interval between 
	# 		##snapshots typically does not vary by more than 5%.
	# 		indics=np.where(np.diff(t_bin)>1.5*(self.ts[2]-self.ts[1]))[0]+1
	# 		t_bin=np.split(t_bin, indics)
	# 		##Binary survival time (take the longest segment)
	# 		t_surv=max([tt[-1]-tt[0] for tt in t_bin])
	# 		##Binary orbital period -- not this is not a constant--take the minimum orbital period 
	# 		if norm:
	# 			sa = rebound.SimulationArchive(self.sa_name)
	# 			m1=sa[0].particles[idx].m
	# 			m2=sa[0].particles[idx2].m
	# 			t_orb = 2.*np.pi*np.min((self.bins[self.pairs==pp][:,4]**3./(m1+m2))**0.5)
	# 			t_surv=t_surv/t_orb
	# 		t_survs[ii]=t_surv

	# 	return t_survs

	def bin_pairs_sort(self, extra=''):
		pairs_u=getattr(self, 'pairs_u'+extra)
		return pairs_u[np.argsort(self.bin_times(norm=False, extra=extra))]







                
