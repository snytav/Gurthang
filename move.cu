

#ifdef __CUDACC__
 __host__ __device__
 #endif
#ifdef VIRTUAL_FUNCTIONS
virtual
#endif
Particle *Move(unsigned int i,int *cells,CurrentTensor *t1,CurrentTensor *t2,double mass,double q_mass,
		 CellDouble *Ex1,CellDouble *Ey1,CellDouble *Ez1,
		 CellDouble *Hx1,CellDouble *Hy1,CellDouble *Hz1)
{

     double3 x,x1;
     Field fd;
     double  m,q_m;
//     int flag;
     Particle p;

     if(i >= number_of_particles) return 0;
     p = readParticleFromSurfaceDevice(i);
     x = p.GetX();
     fd = GetField(x,&p,Ex1,Ey1,Ez1,Hx1,Hy1,Hz1);
    		 p.Move(fd.E,fd.H,tau);
    		 m = p.GetMass();

    		 x = p.GetX();
   		     x1 = p.GetX1();
    		 q_m = p.GetQ2M();
    		 CurrentToMesh(x,x1,m,q_m,tau,cells,t1,t2,&p);
//    		     		 p.x = x1.x;
//    		     		 p.y = x1.y;
//    		     		 p.z = x1.z;
//
//	         Reflect(&p);
//
//
//
//     writeParticleToSurface(i,&p);

     return (&p);
}
