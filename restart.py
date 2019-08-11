import rebound
import sys
import numpy as np
from run_sim import get_tde, get_tde_no_delR, heartbeat


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
	loc="trial_{0}/".format(args.index)

	config=configparser.SafeConfigParser(defaults={'name': 'archive', 'N':'100', 'e':'0.7',
		'gravity':'compensated', 'integrator':'ias15', 'dt':'0', \
		'a_min':'0.05', 'a_max':'0.5', 'ang1_mean':'0', 'ang2_mean':'0', 'ang3_mean':'0', 'ang1':'2.',\
		 'ang2':'2.', 'ang3':'2.', 'keep_bins':'False', 'coll':'line', 'pRun':'0.1', 'pOut':'0.1', 
		'p':'1', 'frac':'2.5e-3', 'outDir':'./', 'gr':'True', 'rinf':'4.0', 'alpha':'1.5',
		'rt':'1e-4', 'mf':"mfixed", 'merge':'False', 'menc_comp':'False', 'Mbh':'4e6',
		'c':'4571304.57795483', 'delR':'True'}, dict_type=OrderedDict)

	##Name of our put file 
	name=config.get('params', 'name')
	name=name+".bin"
	name=config.get('params', 'outDir')+'/'+name
	##Length of simulation and interval between snapshots
	pRun=config.getfloat('params', 'pRun')
	pOut=config.getfloat('params', 'pOut')
	coll=config.get('params', 'coll')
	gr=config.getboolean('params', 'gr')
	rinf=config.getfloat('params', 'rinf')
	alpha=config.getfloat('params', 'alpha')
	menc_comp=config.getboolean('params', 'menc_comp')
	Mbh=config.getfloat('params', 'Mbh')
	a_min=config.getfloat(ss, 'a_min')


	sim=rebound.from_archive(loc+'/archive.bin')
	sim.heartbeat=heartbeat
	sim.collision=coll
	sim.collision_resolve=get_tde
	delR=config.getboolean('params', 'delR')
	merge=config.getboolean('params', 'merge')
	if merge:
		sim.collision_resolve='merge'
	if not delR:
		sim.collision_resolve=get_tde_no_delR

	##Stellar potential
	rebx = reboundx.Extras(sim)
	if rinf>0:
		if menc_comp:
			print("test!")
			menc=rebx.add("menc_comp")
		else:
			menc=rebx.add("menc")
		menc.params["rinf"]=rinf
		menc.params["alpha"]=alpha
	##GR effects
	if gr:
		gr=rebx.add("gr")
		##Speed of light in simulation units.
		gr.params["c"]=config.getfloat('params', 'c')
	sim.automateSimulationArchive(loc+name,interval=pOut*pRun,deletefile=False)
	p_in=2.0*np.pi*(a_min**3.0/Mbh)**0.5
	while(t<pRun):
		if t>=orb_idx*delta_t:
			orbits=sim.calculate_orbits(primary=sim.particles[0])
			np.savetxt(loc+name.replace('.bin', '_out_{0}.dat'.format(orb_idx)), [[oo.a, oo.e, oo.inc, oo.Omega, oo.omega, oo.f] for oo in orbits])
			orb_idx+=1
		sim.move_to_com()
		sim.integrate(sim.t+p_in)
		##Should increment line below by pin not deltat
		t+=p_in



