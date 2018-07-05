


__device__ double cuda_atomicAdd(double *address, double val)
{
    double assumed,old=*address;
    do {
        assumed=old;
        old= __longlong_as_double(atomicCAS((unsigned long long int*)address,
                    __double_as_longlong(assumed),
                    __double_as_longlong(val+assumed)));
    }while (assumed!=old);

    //printf("NEW ATOMIC ADD\n");

    return old;
}

template <template <class Particle> class Cell >
__global__
void GPU_getCellEnergy(
		Cell<Particle>  **cells,double *d_ee,
		double *d_Ex,double *d_Ey,double *d_Ez)
{
	unsigned int i = blockIdx.x;
	unsigned int l= blockIdx.y;
	unsigned int k = blockIdx.z;
	//int i,l,k;
	Cell<Particle>  *c0 = cells[0],nc;
	double t,ex,ey,ez;
	__shared__ extern CellDouble fd[9];
	//double *src;//,*dst;
	int n  = c0->getGlobalCellNumber(i,l,k);

	ex = d_Ex[n];
	ey = d_Ey[n];
	ez = d_Ez[n];

	t = ex*ex+ey*ey+ez*ez;

	cuda_atomicAdd(d_ee,t);
}

template <template <class Particle> class Cell >
global_for_CUDA
void printParticle(Cell<Particle>  **cells,int num,int sort)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
//	int i,l,k;
	Cell<Particle>  *c,*c0 = cells[0],nc;
	//double t;
	__shared__ extern CellDouble fd[9];
	//double *src,*dst;
	//int pqr2;
	//CurrentTensor t1,t2;
	Particle p;

	c = cells[ c0->getGlobalCellNumber(nx,ny,nz)];
//	c = cells[ n ];

	nc = *c;
    if(nc.number_of_particles < threadIdx.x) return;

	nc.readParticleFromSurfaceDevice(threadIdx.x,&p);

		if(p.fortran_number == num && (int)p.sort == sort)
		{
//#ifdef PARTICLE_CELL_DEBUG_PRINTS
			printf("particle-print %5d thread %3d cell (%d,%d,%d) sort %d  %25.15e,%25.15e,%25.15e \n",p.fortran_number,threadIdx.x,c->i,c->l,c->k,(int)p.sort,p.x,p.y,p.z);
//#endif
		}
}

template <template <class Particle> class Cell >
global_for_CUDA
void GPU_SetAllCurrentsToZero(Cell<Particle>  **cells)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	//int i,l,k;
	Cell<Particle>  *c,*c0 = cells[0],nc;
	//double t;
	__shared__ extern CellDouble fd[9];
	//double *src;//,*dst;

	c = cells[ c0->getGlobalCellNumber(nx,ny,nz)];

	nc = *c;

	nc.SetAllCurrentsToZero(threadIdx);
}




template <template <class Particle> class Cell >
global_for_CUDA
void GPU_SetFieldsToCells(Cell<Particle>  **cells,
        double *Ex,double *Ey,double *Ez,
        double *Hx,double *Hy,double *Hz
		)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	//int i,l,k;
	Cell<Particle>  *c,*c0 = cells[0];
	//double t;

	c = cells[ c0->getGlobalCellNumber(nx,ny,nz)];


	c->readFieldsFromArrays(Ex,Ey,Ez,Hx,Hy,Hz,threadIdx);
}

hostdevice_for_CUDA
double CheckArraySize(double* a, double* dbg_a,int size)
	{
	//    Cell<Particle> c = (*AllCells)[0];
	    int wrong = 0;
#ifdef CHECK_ARRAY_SIZE_DEBUG_PRINTS
	    printf("begin array checking1=============================\n");
#endif
	    for(int n = 0;n < size;n++)
	    {
	        //double t  = a[n];
	//	double dt = dbg_a[n];

	        if(fabs(a[n] - dbg_a[n]) > SIZE_TOLERANCE)
		{

		   //int3 i = c.getCellTripletNumber(n);
#ifdef CHECK_ARRAY_SIZE_DEBUG_PRINTS
		   printf("n %5d %15.5e dbg %15.5e diff %15.5e wrong %10d \n",
				   n,a[n],dbg_a[n],fabs(a[n] - dbg_a[n]),wrong++);
#endif
		}
	    }
#ifdef CHECK_ARRAY_SIZE_DEBUG_PRINTS
	    printf("  end array checking=============================\n");
#endif

	    return (1.0-((double)wrong/(size)));
	}


template <template <class Particle> class Cell >
global_for_CUDA void GPU_WriteAllCurrents(Cell<Particle>  **cells,int n0,
		      double *jx,double *jy,double *jz,double *rho)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	Cell<Particle>  *c,*c0 = cells[0];
	__shared__ extern CellDouble fd[9];

	c = cells[ c0->getGlobalCellNumber(nx,ny,nz)];

	int i1,l1,k1;
	        	 i1 = threadIdx.x;
	        	 l1 = threadIdx.y;
	        	 k1 = threadIdx.z;
    	         int n = c->getFortranCellNumber(c->i+i1-1,c->l+l1-1,c->k+k1-1);

    	         if (n < 0 ) n = -n;
        		 double t,t_x,t_y;
		         t_x = c->Jx->M[i1][l1][k1];
		         int3 i3 = c->getCellTripletNumber(n);


		         cuda_atomicAdd(&(jx[n]),t_x);
		         t_y= c->Jy->M[i1][l1][k1];
		         cuda_atomicAdd(&(jy[n]),t_y);
		         t = c->Jz->M[i1][l1][k1];
		         cuda_atomicAdd(&(jz[n]),t);

}

template <template <class Particle> class Cell >
global_for_CUDA void GPU_WriteControlSystem(Cell<Particle>  **cells)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
//	int i,l,k;
	Cell<Particle>  *c,*c0 = cells[0],nc;
	//double t;
	__shared__ extern CellDouble fd[9];
	//double *src; //,*dst;
//	int pqr2;
	//CurrentTensor t1,t2;

	c = cells[ c0->getGlobalCellNumber(nx,ny,nz)];
//	c = cells[ n ];

	 nc = *c;

	 nc.SetControlSystemToParticles();

}

//TODO : 1. 3 separate kernels :
//            A. form 3x3x3 array with number how many to fly and departure list with start positions in 3x3x3 array
//            B. func to get 3x3x3 indexes from a pair of cell numbers, to and from function
//            C. 2nd kernel to write arrival 3x3x3 matrices
///           D. 3rd kernel to form arrival positions in the particle list
//            E. 4th to write arriving particles



template <template <class Particle> class Cell >
global_for_CUDA void GPU_MakeDepartureLists(Cell<Particle>  **cells,int nt,int *d_stage)
{
	    unsigned int nx = blockIdx.x;
		unsigned int ny = blockIdx.y;
		unsigned int nz = blockIdx.z;
		int ix,iy,iz;//,n;

		Particle p;
		Cell<Particle>  *c,*c0 = cells[0],nc;//,*new_c;
		c = cells[c0->getGlobalCellNumber(nx,ny,nz)];

#ifdef FLY_PRINTS

//		printf("GPU_MakeDepartureDim gridDim %3u %3u %3u  nx %3u ny %3u nz %3u \n",
//				gridDim.x,gridDim.y,gridDim.z,
//				nx,ny,nz);
//		d_stage[n] = 1;


#endif



#ifdef FLY_PRINTS

//		printf("GPU_MakeDepartureLists %5d %3d %3d nx %3u ny %3u nz %3u n %5d\n",c->i,c->l,c->k,nx,ny,nz,n);

//		c->printCellParticles("Make-BEGIN",nt);
#endif


		c->departureListLength = 0;
		for(ix = 0;ix < 3;ix++)
		{
			for(iy = 0;iy < 3;iy++)
			{
				for(iz = 0;iz < 3;iz++)
				{
					c->departure[ix][iy][iz]      = 0;
				//	c->departureIndex[ix][iy][iz] = 0;
				}
			}
		}
//        d_stage[n] = 2;
		c->departureListLength  = 0;
        //printf("GPU_MakeDepartureLists nt %d number %d \n",nt,c->number_of_particles);
        //return;
		for(int num = 0;num < c->number_of_particles; num++)
			{
			c->readParticleFromSurfaceDevice(num,&p);
#ifdef FLY_PRINTS

//					printf("resident-begin step %3d %5d %3d %3d p %10d sort %2d num %5d size %5d nx %3u ny %3u nz %3u\n",nt,
//							                                                       c->i,c->l,c->k,
//							                                                       p.fortran_number,
//							                                                       (int)p.sort,
//							                                                       num,
//							                           							   c->number_of_particles,
//							                           							   nx,ny,nz
//							                           							   );

#endif
//				c->readParticleFromSurfaceDevice(num,&p);

#ifdef FLY_PRINTS

//					printf("RESIDENT STEP %d %5d %3d %3d p %10d sort %2d num %5d size %5d %15.5e < %15.5e < %15.5e\n",
//							nt,
//							c->i,c->l,c->k,
//							                                                       p.fortran_number,
//							                                                       (int)p.sort,
//							                                                       num,
//							                           							   c->number_of_particles,
//
//							c->x0,p.x,c->x0+c->hx
//							                           							   );

#endif

				if(!c->isPointInCell(p.GetX()))   //check Paricle = operator !!!!!!!!!!!!!!!!!!!!!!!!!!!
				{
					c->removeParticleFromSurfaceDevice(num,&p,&(c->number_of_particles));
					c->flyDirection(&p,&ix,&iy,&iz);
					if(p.fortran_number == 325041 && p.sort == 2) {
						d_stage[0] = ix;
						d_stage[1] = iy;
						d_stage[2] = iz;
					}
#ifdef FLY_PRINTS
//					printf("%5d %3d %3d resident-remove step %3d fn %10d sort %d x %25.15e count %3d length %3d num %3d number %5d ix %2d iy %2d iz %2d\n",
//							c->i,c->l,c->k,
//							nt,
//							p.fortran_number,
//							p.sort,
//							p.x,
//							c->departure[ix][iy][iz],
//					//		c->departureIndex[ix][iy][iz],
//							c->departureListLength,
//							num,c->number_of_particles,
//							ix,iy,iz
//							);

#endif
//TODO: mke FINAL print at STRAY function.
//					Make 3x3x3x20(50) particle fly array at each cell

					//departureList[departureListLength++] = p;


#ifdef FLY_PRINTS
#endif

//					if(c->i == 1 && c->l == 1 && c->k == 1)
//					{
#ifdef FLY_PRINTS
//					   printf("cell %5d %2d %2d part %3d fn %10d fly %d%d%d x:%15.5e < %15.5e < %15.5e y:%15.5e < %15.5e < %15.5e z::%15.5e < %15.5e < %15.5e\n",
//							   c->i,c->l,c->k,num,p.fortran_number,ix,iy,iz,
//							   c->x0,p.x,c->x0+c->hx,
//							   c->y0,p.y,c->y0+c->hy,
//							   c->z0,p.z,c->z0+c->hz
//							   );
#endif
//					}

					//if(c->departure[ix][iy][iz] == 0) c->departureIndex[ix][iy][iz] = c->departureListLength;


                    if(c->departureListLength == PARTICLES_FLYING_ONE_DIRECTION)
                    {
                    	d_stage[0] = TOO_MANY_PARTICLES;
                    	d_stage[1] = c->i;
                    	d_stage[2] = c->l;
                    	d_stage[3] = c->k;
                    	d_stage[1] = ix;
                    	d_stage[2] = iy;
                    	d_stage[3] = iz;
                    	return;
                    }
					c->departureListLength++;
					int num1 = c->departure[ix][iy][iz];

					c->departureList[ix][iy][iz][num1] = p;
					if(p.fortran_number == 325041 && p.sort == 2) {
						d_stage[4] = num1;
						d_stage[5] = c->departureList[ix][iy][iz][num1].fortran_number;

					}

					c->departure[ix][iy][iz] += 1;
#ifdef FLY_PRINTS
//					Particle new_p;
//
//					new_p = c->departureList[ix][iy][iz][num1];
//					printf(" %5d %2d %2d departureC fn %10d count %3d length %3d num %3d number %5d\n",
//							c->i,c->l,c->k,new_p.fortran_number,c->departure[ix][iy][iz],
//							//c->departureIndex[ix][iy][iz],
//							c->departureListLength,
//							num1,c->number_of_particles);
#endif
					num--;
				}
#ifdef FLY_PRINTS
//					printf("resident-final %5d %3d %3d fn %10d length %3d num %3d number %5d\n",
//							c->i,c->l,c->k,p.fortran_number,
//							c->departureListLength,
//							num,c->number_of_particles);
//
#endif
			}
//		d_stage[n] = 3;
#ifdef FLY_PRINTS
					//printf("resident-end\n");
#endif
		//char s[1000],s1[10];
		//printf(s,"DEPARTURE cell %5d %2d %2d ",c->i,c->l,c->k);
		for(ix = 0;ix < 3;ix++)
			for(iy = 0;iy < 3;iy++)
				for(iz = 0;iz < 3;iz++)
				{
#ifdef FLY_PRINTS
				//	printf("DEPARTURE step %d cell %5d %2d %2d %d%d%d num %3d \n",nt,c->i,c->l,c->k,ix,iy,iz,
				//			c->departure[ix][iy][iz]);//,c->departureIndex[ix][iy][iz]);
//					int num = c->departure[ix][iy][iz];
//					for(int ip = 0;ip < num;ip++)
//					{
//						p = c->departureList[ix][iy][iz][ip];
//						printf("%10d ",p.fortran_number);
//					}
//					printf("\n");
#endif
//					strcat(s,s1);
				}
		//printf("%s \n",s);
//        d_stage[n] = 5;
}

template <template <class Particle> class Cell >
global_for_CUDA void GPU_ArrangeFlights(Cell<Particle>  **cells,int nt, int *d_stage)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	int ix,iy,iz,snd_ix,snd_iy,snd_iz,num,n;
	Particle p;

	Cell<Particle>  *c,*c0 = cells[0],nc,*snd_c;
		//int first = 1;
		//cuPrintf("stray %3d %3d %3d \n",nx,ny,nz);

	//printf("GPU_ArrangeFlights \n");
//	return;

		c = cells[ n = c0->getGlobalCellNumber(nx,ny,nz)];

		for(ix = 0;ix < 3;ix++)
			for(iy = 0;iy < 3;iy++)
				for(iz = 0;iz < 3;iz++)
				{
					int index = ix*9 +iy*3 +iz;
					n = c0->getWrapCellNumber(nx+ix-1,ny+iy-1,nz+iz-1);

		            snd_c  = cells[ n ];
					if(nx == 24 && ny == 2 && nz == 2)
					{

						d_stage[index*4]   = snd_c->i;
						d_stage[index*4+1] = snd_c->l;
						d_stage[index*4+2] = snd_c->k;
						d_stage[index*4+3] = snd_c->departureListLength;
					}

//#ifdef FLY_PRINTS
//					printf("arrange %5d %2d %2d  %d%d%d nx %5d %3d %3d n %5d snd %p \n",c->i,c->l,c->k,ix,iy,iz,nx+ix-1,ny+iy-1,nz+iz-1,n,snd_c);
//#endif



					//printf("arrangge %5d %2d %2d  %d%d%d snd_c %p \n",c->i,c->l,c->k,ix,iy,iz,snd_c);

	//				continue;
					snd_ix = ix;
					snd_iy = iy;
					snd_iz = iz;
					//printf("arrannge %5d %2d %2d  %d%d%d \n",c->i,c->l,c->k,ix,iy,iz);


					//printf("arrrange %5d %2d %2d  %d%d%d \n",c->i,c->l,c->k,ix,iy,iz);
					c->inverseDirection(&snd_ix,&snd_iy,&snd_iz);
//					printf("inverse %5d %2d %2d  %d%d%d \n",c->i,c->l,c->k,ix,iy,iz);
//#ifdef FLY_PRINTS
//
//					printf("cell %5d %2d %2d direction %d%d%d from %5d %2d %2d snd dir %d%d%d \n", //);from %5d %2d %2d snd dir %d%d%d\n",
//													 c->i,c->l,c->k,ix,iy,iz,
//                                                     snd_c->i,snd_c->l,snd_c->k,snd_ix,snd_iy,snd_iz);
//#endif
					//continue;

					num = snd_c->departure[snd_ix][snd_iy][snd_iz];
//					if(nx == 24 && ny == 2 && nz == 2)
//					{
//						d_stage[index*4+3] = num;
//					}
					//pos = snd_c->departureIndex[ix][iy][iz];
//#ifdef FLY_PRINTS
//					printf("BEFORE_ARR step %d cell %5d %2d %2d num %3d direction %d%d%d from %d%d%d donor %5d %2d %2d\n",nt,
//							c->i,c->l,c->k,num,ix,iy,iz,snd_ix,snd_iy,snd_iz,
//							snd_c->i,snd_c->l,snd_c->k
//							);
//#endif

					for(int i = 0;i < num;i++)
					{
						p = snd_c->departureList[snd_ix][snd_iy][snd_iz][i];
//#ifdef FLY_PRINTS
                      if(nx == 24 && ny == 2 && nz == 2)
						{
//                    	    if(snd_c->i == 23 && snd_c->k == 2 && snd_c->l == 1)
//                    	    {
// 							   d_stage[0] = num;
//							   d_stage[1] = p.fortran_number;
//                    	    }
						}
//						printf("step %3d sort %d ARRIVAL cell %5d %2d %2d part %3d (total %3d ) pos %d fn %10d to %d%d%d from %d%d%d numInCell %3d x %15.5e %15.5e %15.5e m %15.5e q_m %15.5e px %15.5e %15.5e %15.5e\n",
//								                       nt,p.sort,
//													   c->i,c->l,c->k,i,snd_c->departure[snd_ix][snd_iy][snd_iz],
//													   pos,p.fortran_number,snd_ix,snd_iy,snd_iz,ix,iy,iz,
//													   c->number_of_particles,
//													   p.x,p.y,p.z,p.m,p.q_m,p.pu,p.pv,p.pw
//													   );
//#endif
						c->Insert(p);

					}


					//new_c->arrival[new_ix][new_iy][new_iz] = c->departure[ix][iy][iz];
				}
//#ifdef FLY_PRINTS
//		c->printCellParticles("FINAL",nt);
//#endif

}


template <template <class Particle> class Cell >
global_for_CUDA void GPU_CollectStrayParticles(Cell<Particle>  **cells,int nt
//		                         int n,
//		                         int i,
//		                         double mass,
//		                         double q_mass,
//		                         double *p_control,
//		                         int jmp
		                         )
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;

//	int busy;
	Particle p;
	int n;
//	int i,l,k;
	Cell<Particle>  *c,*c0 = cells[0],nc,*new_c;
	//int first = 1;
	//cuPrintf("stray %3d %3d %3d \n",nx,ny,nz);

	c = cells[ n = c0->getGlobalCellNumber(nx,ny,nz)];

	for(int i = 0;i < c->number_of_particles; i++)
	{
		c->readParticleFromSurfaceDevice(i,&p);
#ifdef STRAY_DEBUG_PRINTS
		//if((p.fortran_number == 2498) && (p.sort == BEAM_ELECTRON))
	//	{
		    printf("STRAY-BASIC step %d cell %3d %d %d sort %d particle %d FORTRAN %5d X: %15.5e < %15.5e < %15.5e Y: %15.5e < %15.5e < %15.5e Z: %15.5e < %15.5e < %15.5e \n",
		    		nt,c->i,c->l,c->k,(int)p.sort,i,p.fortran_number,
		    		c->x0,p.x,c->x0+c->hx,
		    		c->y0,p.y,c->y0+c->hy,
		    		c->z0,p.z,c->z0+c->hz
		    		);
		//}
#endif
		if(!c->isPointInCell(p.GetX()))// || (p.fortran_number == 753) )//|| (p.fortran_number == 10572))
		{
#ifdef STRAY_DEBUG_PRINTS

   			    printf("STRAY-OUT step %3d cell %3d %d %d sort %d particle %d FORTRAN %5d X: %15.5e < %25.17e < %15.5e \n",
   			    		nt,c->i,c->l,c->k,(int)p.sort,i,p.fortran_number,c->x0,p.x,c->x0+c->hx);
#endif
            int new_n = c->getPointCell(p.GetX());
            new_c = cells[new_n];


            if(c->i == 99 && c->l == 0 && c->k == 3)
            {
         //      c->printCellParticles();

//               if(c->i >= c->Nx-1)
//               {
#ifdef STRAY_DEBUG_PRINTS
//       		if((p.fortran_number == 2498) && (p.sort == BEAM_ELECTRON))
//       		{
//       		   printf("c %3d (%d,%d,%d)->(%d,%d,%d) s %d p %d FN %5d X: %15.5e<%23.16e<%15.5e \n",n,c->i,c->l,c->k,new_c->i,new_c->l,new_c->k,(int)p.sort,i,p.fortran_number,c->x0,p.x,c->x0+c->hx);
//   			   printf("c %3d (%d,%d,%d)->(%d,%d,%d) s %d p %d FN %5d Y: %15.5e<%23.16e<%15.5e \n",n,c->i,c->l,c->k,new_c->i,new_c->l,new_c->k,(int)p.sort,i,p.fortran_number,c->y0,p.y,c->y0+c->hy);
//   			   printf("c %3d (%d,%d,%d)->(%d,%d,%d) s %d p %d FN %5d Z: %15.5e<%23.16e<%15.5e \n",n,c->i,c->l,c->k,new_c->i,new_c->l,new_c->k,(int)p.sort,i,p.fortran_number,c->z0,p.z,c->z0+c->hz);
//       		}
#endif
//               }
            }
//            if(first == 1)
//            {

  //          	do{
  //          	    	busy = atomicCAS(&(c->busyParticleArray),0,1);
            	    	// busy = ((c->busyParticleArra == 0 )? 1: c->busyParticleArra)

            	    	// busy = ((c->busyParticleArra == 1 )? 0: c->busyParticleArra)

   //         	  }while(busy == 1);

            while (atomicCAS(&(c->busyParticleArray),0,1)) {}
               c->removeParticleFromSurfaceDevice(i,&p,&(c->number_of_particles));
               //c->busyParticleArray = 0;
              atomicExch(&(c->busyParticleArray),0u);
               i--;
//               first = 0;
//            }
//            if(c->i == 99 && c->l == 0 && c->k == 3)
//            {
//               c->printCellParticles();
//            }
              //do{
               // 	busy = atomicCAS(&(new_c->busyParticleArray),0,1);
              //}while(busy == 1);

               while (atomicCAS(&(new_c->busyParticleArray),0,1)) {}

              new_c->Insert(p);
#ifdef STRAY_DEBUG_PRINTS

   			    printf("STRAY-INSERT step %d %3d %d %d sort %d particle %d FORTRAN %5d X: %15.5e < %25.17e < %15.5e \n",
   			    		nt,
   			    		new_c->i,new_c->l,new_c->k,(int)p.sort,i,p.fortran_number,new_c->x0,p.x,new_c->x0+new_c->hx);
#endif
              //new_c->busyParticleArray = 0;
              atomicExch(&(new_c->busyParticleArray),0u);
       		if((p.fortran_number == 2498) && (p.sort == BEAM_ELECTRON))
      		{

          //    new_c->printCellParticles();
      		}
            if(c->i == 99 && c->l == 0 && c->k == 3)
            {
            //   new_c->printCellParticles();
            }
		}
	}
	c->printCellParticles("STRAY-FINAL",nt);

}

__device__ void writeCurrentComponent(CellDouble *J,CurrentTensorComponent *t1,CurrentTensorComponent *t2,int pqr2)
{
    cuda_atomicAdd(&(J->M[t1->i11][t1->i12][t1->i13]),t1->t[0]);
    cuda_atomicAdd(&(J->M[t1->i21][t1->i22][t1->i23]),t1->t[1]);
    cuda_atomicAdd(&(J->M[t1->i31][t1->i32][t1->i33]),t1->t[2]);
    cuda_atomicAdd(&(J->M[t1->i41][t1->i42][t1->i43]),t1->t[3]);

    if(pqr2 == 2)
    {
        cuda_atomicAdd(&(J->M[t2->i11][t2->i12][t2->i13]),t2->t[0]);
        cuda_atomicAdd(&(J->M[t2->i21][t2->i22][t2->i23]),t2->t[1]);
        cuda_atomicAdd(&(J->M[t2->i31][t2->i32][t2->i33]),t2->t[2]);
        cuda_atomicAdd(&(J->M[t2->i41][t2->i42][t2->i43]),t2->t[3]);
    }

}

__device__ void copyCellDouble(CellDouble *dst,CellDouble *src,unsigned int n,uint3 block)
{
	if(n < CellExtent*CellExtent*CellExtent)
	{
		double *d_dst,*d_src;//,t;

		d_dst = (double *)(dst->M);
		d_src = (double *)(src->M);

//		t = d_dst[n];
//
//		if(fabs(d_dst[n] - d_src[n]) > 1e-15)
//		{
//     		printf("block %5d %3d %3d thread %5d CCD t %15.5e dst %15.5e src %15.5e dst %p src %p d_dst[n] %p d_src[n] %p \n",
//     				block.x,block.y,block.z,threadIdx.x,t,d_dst[n],d_src[n],dst,src,&(d_dst[n]),&(d_src[n]));
//		}
		d_dst[n] = d_src[n];
	}
}

template <template <class Particle> class Cell >
global_for_CUDA void GPU_GetCellNumbers(Cell<Particle>  **cells,
		                         int *numbers)
{
		Cell<Particle>  *c;//,nc;
		c = cells[blockIdx.x];

		numbers[blockIdx.x] = (*c).number_of_particles;
}

__device__ void assignSharedWithLocal(
		                         CellDouble **c_jx,
		                         CellDouble **c_jy,
		                         CellDouble **c_jz,
		                         CellDouble **c_ex,
		                         CellDouble **c_ey,
		                         CellDouble **c_ez,
		                         CellDouble **c_hx,
		                         CellDouble **c_hy,
		                         CellDouble **c_hz,
		                         CellDouble *fd)
{
	*c_ex = &(fd[0]);
	*c_ey = &(fd[1]);
	*c_ez = &(fd[2]);

	*c_hx = &(fd[3]);
	*c_hy = &(fd[4]);
	*c_hz = &(fd[5]);

	*c_jx = &(fd[6]);
	*c_jy = &(fd[7]);
	*c_jz = &(fd[8]);
}

__device__ void copyFieldsToSharedMemory(
		 CellDouble *c_jx,
		 CellDouble *c_jy,
		 CellDouble *c_jz,
		 CellDouble *c_ex,
		 CellDouble *c_ey,
		 CellDouble *c_ez,
		 CellDouble *c_hx,
		 CellDouble *c_hy,
		 CellDouble *c_hz,
		 Cell<Particle>  *c,
		 int index,
		 dim3 blockId,
		 int blockDimX
		)
{
	//int index  = threadIdx.x;


	while(index < CellExtent*CellExtent*CellExtent)
	{
//		if(index < 125) {

		copyCellDouble(c_ex,c->Ex,index,blockId);
		copyCellDouble(c_ey,c->Ey,index,blockId);
		copyCellDouble(c_ez,c->Ez,index,blockId);

		copyCellDouble(c_hx,c->Hx,index,blockId);
		copyCellDouble(c_hy,c->Hy,index,blockId);
		copyCellDouble(c_hz,c->Hz,index,blockId);

		copyCellDouble(c_jx,c->Jx,index,blockId);
		copyCellDouble(c_jy,c->Jy,index,blockId);
		copyCellDouble(c_jz,c->Jz,index,blockId);
		//}
		index += blockDimX;
	}

	__syncthreads();

}

__device__ void set_cell_double_array_to_zero(CurrentTensorComponent *ca,int length)
{
     for(int i = 0; i<= 100;i++)
     {
    	ca[i].t[0] = 0.0;


     }
}

__device__ void Move(
									 CellDouble *c_ex,
									 CellDouble *c_ey,
									 CellDouble *c_ez,
									 CellDouble *c_hx,
									 CellDouble *c_hy,
									 CellDouble *c_hz,
									 CellDouble *c_jx,
									 CellDouble *c_jy,
									 CellDouble *c_jz,
									 Cell<Particle>  *c,
		                             int index,
		                             int blockDimX,
		                             double mass,
		                             double q_mass
		                             )
{
	CurrentTensor t1,t2;
    int pqr2;
	Particle p;

    while(index < c->number_of_particles)
    {

        c->Move (index,&pqr2,&t1,&t2,mass,q_mass,c_ex,c_ey,c_ez,c_hx,c_hy,c_hz);

        writeCurrentComponent(c_jx,&(t1.Jx),&(t2.Jx),pqr2);
        writeCurrentComponent(c_jy,&(t1.Jy),&(t2.Jy),pqr2);
        writeCurrentComponent(c_jz,&(t1.Jz),&(t2.Jz),pqr2);

        index += blockDimX;
    }



    __syncthreads();
}

__device__ void MoveAndWriteCurrents(
									 CellDouble *c_ex,
									 CellDouble *c_ey,
									 CellDouble *c_ez,
									 CellDouble *c_hx,
									 CellDouble *c_hy,
									 CellDouble *c_hz,
									 CellDouble *c_jx,
									 CellDouble *c_jy,
									 CellDouble *c_jz,
									 Cell<Particle>  *c,
		                             int index,
		                             int blockDimX,
		                             double mass,
		                             double q_mass
		                             )
{
	CurrentTensor t1,t2;
    int pqr2;
	Particle p;

    while(index < c->number_of_particles)
    {

        c->Move (index,&pqr2,&t1,&t2,mass,q_mass,c_ex,c_ey,c_ez,c_hx,c_hy,c_hz);

        writeCurrentComponent(c_jx,&(t1.Jx),&(t2.Jx),pqr2);
        writeCurrentComponent(c_jy,&(t1.Jy),&(t2.Jy),pqr2);
        writeCurrentComponent(c_jz,&(t1.Jz),&(t2.Jz),pqr2);

        index += blockDimX;
    }



    __syncthreads();
}



__device__ void copyFromSharedMemoryToCell(
		                                     CellDouble *c_jx,
											 CellDouble *c_jy,
											 CellDouble *c_jz,
											 Cell<Particle>  *c,
				                             int index,
				                             int blockDimX,
				                             dim3 blockId
		)
{
	while(index < CellExtent*CellExtent*CellExtent)
	{
      	copyCellDouble(c->Jx,c_jx,index,blockIdx);
    	copyCellDouble(c->Jy,c_jy,index,blockIdx);
    	copyCellDouble(c->Jz,c_jz,index,blockIdx);

    	index += blockDim.x;
    }
    c->busyParticleArray = 0;
}

template <template <class Particle> class Cell >
global_for_CUDA void GPU_StepAllCells(Cell<Particle>  **cells,
		                         int i,
		                         double *global_jx,
		                         double mass,
		                         double q_mass
		                         )
{
	Cell<Particle>  *c,*c0 = cells[0];
	__shared__ extern CellDouble fd[9];
	CellDouble *c_jx,*c_jy,*c_jz,*c_ex,*c_ey,*c_ez,*c_hx,*c_hy,*c_hz;
//	CurrentTensor t1,t2;
//	int pqr2;
	Particle p;

	c = cells[ c0->getGlobalCellNumber(blockIdx.x,blockIdx.y,blockIdx.z)];

	assignSharedWithLocal(&c_jx,&c_jy,&c_jz,&c_ex,&c_ey,&c_ez,&c_hx,&c_hy,&c_hz,fd);




	copyFieldsToSharedMemory(c_jx,c_jy,c_jz,c_ex,c_ey,c_ez,c_hx,c_hy,c_hz,c,
			threadIdx.x,blockIdx,blockDim.x);


	Move(c_ex,c_ey,c_ez,c_hx,c_hy,c_hz,c_jx,c_jy,c_jz,
						 c,threadIdx.x,blockDim.x,mass,q_mass);

//    WriteCurrents(c_jx,c_jy,c_jz,c_jx,c_jy,c_jz,
//						 c,threadIdx.x,blockDim.x,mass,q_mass);



    copyFromSharedMemoryToCell(c_jx,c_jy,c_jz,c,threadIdx.x,blockDim.x,blockIdx);

}

template <template <class Particle> class Cell >
global_for_CUDA void GPU_ControlAllCellsCurrents(Cell<Particle>  **cells,int n,int i,CellDouble *jx,CellDouble *jy,CellDouble *jz)
{
//	unsigned int nx = blockIdx.x;
//	unsigned int ny = blockIdx.y;
//	unsigned int nz = blockIdx.z;
//	int i,l,k;
	Cell<Particle>  *c,*c0 = cells[0],nc;
	//double t;
	__shared__ extern CellDouble fd[9];
	//double *src;
	//int pqr2;
//	CurrentTensor t1,t2;

//	c = cells[ c0->getGlobalCellNumber(nx,ny,nz)];
	c = cells[ n ];

	nc = *c;

	// double cjx,cjy,cjz;

//	              cjx = CheckArraySize((double *)jx,(double *)(nc.Jx),sizeof(CellDouble)/sizeof(double));
//	              cjy = CheckArraySize((double *)jy,(double *)(nc.Jy),sizeof(CellDouble)/sizeof(double));
//	              cjz = CheckArraySize((double *)jz,(double *)(nc.Jz),sizeof(CellDouble)/sizeof(double));
#ifdef GPU_CONTROL_ALL_CELLS_CURRENTS_PRINT
//	              printf("cell (%d,%d,%d) particle %d currents %.5f %.5f %.5f \n",nc.i,nc.l,nc.k,i,cjx,cjy,cjz);
#endif


}

__host__ __device__
void emh2_Element(
		Cell<Particle> *c,
		int i,int l,int k,
		double *Q,double *H)
{
	int n  = c->getGlobalCellNumber(i,l,k);

	H[n] += Q[n];
}

template <template <class Particle> class Cell >
global_for_CUDA
void GPU_emh2(
		 Cell<Particle>  **cells,
				            int i_s,int l_s,int k_s,
							double *Q,double *H
		)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	Cell<Particle>  *c0 = cells[0];

	emh2_Element(c0,i_s+nx,l_s+ny,k_s+nz,Q,H);
}


__host__ __device__
void emh1_Element(
		Cell<Particle> *c,
		int3 i,
		double *Q,double *H,double *E1, double *E2,
		double c1,double c2,
		int3 d1,int3 d2)
{

    int n  = c->getGlobalCellNumber(i.x,i.y,i.z);
	int n1 = c->getGlobalCellNumber(i.x+d1.x,i.y+d1.y,i.z+d1.z);
	int n2 = c->getGlobalCellNumber(i.x+d2.x,i.y+d2.y,i.z+d2.z);

	double e1_n1 = E1[n1];
	double e1_n  = E1[n];
	double e2_n2 = E2[n2];
	double e2_n  = E2[n];

	double t  = 0.5*(c1*(e1_n1 - e1_n)- c2*(e2_n2 - e2_n));
    Q[n] = t;
    H[n] += Q[n];
}

template <template <class Particle> class Cell >
global_for_CUDA
void GPU_emh1(
		 Cell<Particle>  **cells,
							double *Q,double *H,double *E1, double *E2,
							double c1,double c2,
							int3 d1,int3 d2
		)
{

	int3 i3 = make_int3(blockIdx.x,blockIdx.y,blockIdx.z);
	Cell<Particle>  *c0 = cells[0];

	emh1_Element(c0,i3,Q,H,E1,E2,c1,c2,d1,d2);
}

__host__ __device__
	void emeElement(Cell<Particle> *c,int3 i,double *E,double *H1, double *H2,
			double *J,double c1,double c2, double tau,
			int3 d1,int3 d2
			)
	{
	   int n  = c->getGlobalCellNumber(i.x,i.y,i.z);
	  int n1 = c->getGlobalCellNumber(i.x+d1.x,i.y+d1.y,i.z+d1.z);
	  int n2 = c->getGlobalCellNumber(i.x+d2.x,i.y+d2.y,i.z+d2.z);

	  E[n] += c1*(H1[n] - H1[n1]) - c2*(H2[n] - H2[n2]) - tau*J[n];
	}

__host__ __device__
void periodicElement(Cell<Particle> *c,int i,int k,double *E,int dir, int to,int from)
{
    int n   = c->getGlobalBoundaryCellNumber(i,k,dir,to);
	int n1  = c->getGlobalBoundaryCellNumber(i,k,dir,from);
	E[n]    = E[n1];
}

template <template <class Particle> class Cell >
global_for_CUDA void GPU_periodic(Cell<Particle>  **cells,
                             int i_s,int k_s,
                             double *E,int dir, int to,int from)
{
	unsigned int nx = blockIdx.x;
	//unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	Cell<Particle>  *c0 = cells[0];

	periodicElement(c0,nx+i_s,nz+k_s,E, dir,to,from);
}

__host__ __device__
void periodicCurrentElement(Cell<Particle> *c,int i,int k,double *E,int dir, int dirE,int N)
{
    int n1    = c->getGlobalBoundaryCellNumber(i,k,dir,1);
    int n_Nm1 = c->getGlobalBoundaryCellNumber(i,k,dir,N-1);
    if(dir != dirE)
    {
       E[n1] += E[n_Nm1];
    }
    if(dir != 1 || dirE != 1)
    {
       E[n_Nm1] =  E[n1];
    }

    int n_Nm2 = c->getGlobalBoundaryCellNumber(i,k,dir,N-2);
    int n0    = c->getGlobalBoundaryCellNumber(i,k,dir,0);

#ifdef PERIODIC_CURRENT_PRINTS
    printf("%e %e \n",E[n0],E[n_Nm2]);
#endif
    E[n0] += E[n_Nm2];
    E[n_Nm2] = E[n0];
}

template <template <class Particle> class Cell >
global_for_CUDA void GPU_CurrentPeriodic(Cell<Particle>  **cells,double *E,int dirE, int dir,
                             int i_s,int k_s,int N)
{
	unsigned int nx = blockIdx.x;
	//unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	Cell<Particle>  *c0 = cells[0];

//	cuPrintf("i %d k %d N %d dir % dirE %d\n",nx+i_s,nz+k_s,N,dir,dirE);
	periodicCurrentElement(c0,nx+i_s,nz+k_s,E, dir,dirE,N);
}

template <template <class Particle> class Cell >
global_for_CUDA void GPU_eme(

		            Cell<Particle>  **cells,
		            int3 s,
					double *E,double *H1, double *H2,
					double *J,double c1,double c2, double tau,
					int3 d1,int3 d2
		)
{
	unsigned int nx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ny = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int nz = blockIdx.z*blockDim.z + threadIdx.z;
	Cell<Particle>  *c0 = cells[0];

    s.x += nx;
    s.y += ny;
    s.z += nz;

    emeElement(c0,s,E,H1,H2,J,c1,c2,tau,d1,d2);
}

template <template <class Particle> class Cell >
global_for_CUDA void copy_pointers(Cell<Particle>  **cells,int *d_flags,double_pointer *d_pointers)
{
	Cell<Particle>  *c = cells[blockIdx.x];

	c->flag_wrong_current_cell = d_flags[blockIdx.x];
	c->d_wrong_current_particle_attributes = d_pointers[blockIdx.x];

}
