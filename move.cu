

#ifdef __CUDACC__
 __host__ __device__
 #endif
#ifdef VIRTUAL_FUNCTIONS
virtual
#endif
Particle Move(unsigned int i,int *cells,CurrentTensor *t1,CurrentTensor *t2,double mass,double q_mass,
		 CellDouble *Ex1,CellDouble *Ey1,CellDouble *Ez1,
		 CellDouble *Hx1,CellDouble *Hy1,CellDouble *Hz1)
{
     double3 x,x1;
     double  m,q_m;
//     int flag;
     Particle p;
     Field fd;

     if(i >= number_of_particles) return p;
     p = readParticleFromSurfaceDevice(i);
//	 x = p.GetX();
	 fd = GetField(&p,Ex1,Ey1,Ez1,Hx1,Hy1,Hz1);

	 p.Move(fd.E,fd.H,tau);

	 x = p.GetX();
	 x1 = p.GetX1();
	 CurrentToMesh(x,x1,p.m,p.q_m,tau,cells,t1,t2,&p);
	 p.x = p.x1;
	 p.y = p.y1;
	 p.z = p.z1;

	 Reflect(&p);



     writeParticleToSurface(i,&p);

     return p;
}
