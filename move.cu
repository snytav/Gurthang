

#ifdef __CUDACC__
 __host__ __device__
 #endif
#ifdef VIRTUAL_FUNCTIONS
virtual
#endif
void Move(unsigned int i,int *cells,CurrentTensor *t1,CurrentTensor *t2,double mass,double q_mass,
		 CellDouble *Ex1,CellDouble *Ey1,CellDouble *Ez1,
		 CellDouble *Hx1,CellDouble *Hy1,CellDouble *Hz1)
{
     double3 x,x1;
     double  m,q_m;
//     int flag;
     Particle p;
     Field fd;

     if(i >= number_of_particles) return;
     p = readParticleFromSurfaceDevice(i);
//	 x = p.GetX();
	 fd = GetField(&p,Ex1,Ey1,Ez1,Hx1,Hy1,Hz1);

	 p.Move(fd.E,fd.H,tau);
	 writeParticleToSurface(i,&p);

	 Particle p1;
	 p1 = p;//readParticleFromSurfaceDevice(i);

	 CurrentToMesh(tau,cells,t1,t2,&p1);

	 writeParticleToSurface(i,&p1);
//	 AccCurrent(i,cells,t1,t2,mass,q_mass,Ex1,Ey1,Ez1,Hx1,Hy1,Hz1);
}

#ifdef __CUDACC__
 __host__ __device__
 #endif
 void AccCurrent(unsigned int i,int *cells,CurrentTensor *t1,CurrentTensor *t2,double mass,double q_mass,
 		 CellDouble *Ex1,CellDouble *Ey1,CellDouble *Ez1,
 		 CellDouble *Hx1,CellDouble *Hy1,CellDouble *Hz1)
 {
	 Particle p;
	 if(i >= number_of_particles) return;

	 p = readParticleFromSurfaceDevice(i);
	 CurrentToMesh(tau,cells,t1,t2,&p);




     writeParticleToSurface(i,&p);

//     return p;
}
