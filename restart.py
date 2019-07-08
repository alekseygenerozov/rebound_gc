import rebound
import sys
import numpy as np

def heartbeat(sim):
	print(sim.contents.dt, sim.contents.t)
# sim is a pointer to the simulation object,
# thus use contents to access object data.
# See ctypes documentation for details.
	# print(sim.contents.dt)

def get_tde(sim, reb_coll):
	orbits = sim[0].calculate_orbits(primary=sim[0].particles[0])
	p1,p2 = reb_coll.p1, reb_coll.p2
	idx, idx0 = max(p1, p2), min(p1, p2)
	if idx0==0:
		##idx decremented by 1 because there is no orbit 0
		print sim[0].t, orbits[idx-1].e, idx, 'TDE!'

	return 0

tmax = 500.
sim = rebound.Simulation.from_archive(sys.argv[1])
sim.automateSimulationArchive(sys.argv[1],interval=np.pi*0.2,deletefile=False)
if sim.t>=tmax*2*np.pi:
	sys.exit(0)
sim.simulationarchive_next=sim.t+0.2*np.pi	
sim.collision_resolve=get_tde
sim.integrate(tmax*2*np.pi)
