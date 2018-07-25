/*
 * wrap_kernel.cu
 *
 *  Created on: Jul 23, 2018
 *      Author: snytav
 */

#include "gpu_plasma.h"


global_for_CUDA void GPU_CollectStrayParticles(Cell **cells,int nt
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
	Cell  *c,*c0 = cells[0],nc,*new_c;
	//int first = 1;

	c = cells[ n = c0->getGlobalCellNumber(nx,ny,nz)];

	for(int i = 0;i < c->number_of_particles; i++)
	{
		p = c->readParticleFromSurfaceDevice(i);
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


global_for_CUDA
void GPU_SetFieldsToCells(GPUCell  **cells,
        double *Ex,double *Ey,double *Ez,
        double *Hx,double *Hy,double *Hz
		)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	//int i,l,k;
	Cell  *c,*c0 = cells[0];
	//double t;

	c = cells[ c0->getGlobalCellNumber(nx,ny,nz)];


	c->readFieldsFromArrays(Ex,Ey,Ez,Hx,Hy,Hz,threadIdx);
}

global_for_CUDA void copy_pointers(Cell  **cells,int *d_flags,double_pointer *d_pointers)
{
	Cell  *c = cells[blockIdx.x];

	c->flag_wrong_current_cell = d_flags[blockIdx.x];
	c->d_wrong_current_particle_attributes = d_pointers[blockIdx.x];

}

global_for_CUDA void GPU_ControlAllCellsCurrents(Cell  **cells,int n,int i,CellDouble *jx,CellDouble *jy,CellDouble *jz)
{
//	unsigned int nx = blockIdx.x;
//	unsigned int ny = blockIdx.y;
//	unsigned int nz = blockIdx.z;
//	int i,l,k;
	Cell  *c,*c0 = cells[0],nc;
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


global_for_CUDA
void GPU_SetAllCurrentsToZero(GPUCell  **cells)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	//int i,l,k;
	Cell *c,*c0 = cells[0],nc;
	//double t;
	__shared__ extern CellDouble fd[9];
	//double *src;//,*dst;

	c = cells[ c0->getGlobalCellNumber(nx,ny,nz)];

	nc = *c;

	nc.SetAllCurrentsToZero(threadIdx);
}

global_for_CUDA void GPU_ArrangeFlights(GPUCell  **cells,int nt, int *d_stage)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	int ix,iy,iz,snd_ix,snd_iy,snd_iz,num,n;
	Particle p;

	Cell  *c,*c0 = cells[0],nc,*snd_c;
		//int first = 1;


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

int Plasma::reallyPassParticlesToAnotherCells(int nt,int *stage1,int *d_stage1)
{
    int err;
    dim3 dimGridBulk(Nx,Ny,Nz),dimBlockOne(1,1,1);
	cudaMemset(d_stage1,0,sizeof(int)*(Nx+2)*(Ny+2)*(Nz+2));


//	    GPU_ArrangeFlights//<<<dimGridBulk, dimBlockOne>>>(d_CellArray,nt,d_stage1);
	    void* args[] = {
	    		          (void* )&d_CellArray,
	    		          (void *)&nt,
	    		          (void*)&d_stage1,
	    		          0};

	        cudaError_t cudaStatus = cudaLaunchKernel(
	                     (const void*)GPU_ArrangeFlights, // pointer to kernel func.
	                     dimGridBulk,                       // grid
	                     dimBlockOne,                   // block
	                     args,                          // arguments
	                     16000,
	                     0
	                 );


#ifdef BALANCING_PRINTS
	    CUDA_Errot_t after_ArrangeFlights = cudaGetLastError();


    printf("after_ArrangeFlights %d %s\n",after_ArrangeFlights,cudaGetErrorString(after_ArrangeFlights));
            cudaDeviceSynchronize();
#endif

	err = MemoryCopy(stage1,d_stage1,sizeof(int)*(Nx+2)*(Ny+2)*(Nz+2),DEVICE_TO_HOST);
	if(err != cudaSuccess)
	{
	   puts("copy error");
	   exit(0);
	}
	ListAllParticles(nt,"aArrangeFlights");


	memory_monitor("CellOrder_StepAllCells7",nt);
	return (int)err;

}




global_for_CUDA void GPU_MakeDepartureLists(GPUCell  **cells,int nt,int *d_stage)
{
	    unsigned int nx = blockIdx.x;
		unsigned int ny = blockIdx.y;
		unsigned int nz = blockIdx.z;
		int ix,iy,iz;//,n;

		Particle p;
		Cell  *c,*c0 = cells[0],nc;//,*new_c;
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
			p = c->readParticleFromSurfaceDevice(num);
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



__device__ void set_cell_double_array_to_zero(CurrentTensorComponent *ca,int length)
{
     for(int i = 0; i<= 100;i++)
     {
    	ca[i].t[0] = 0.0;


     }
}

__device__ void MoveParticlesInCell(
									 CellDouble *c_ex,
									 CellDouble *c_ey,
									 CellDouble *c_ez,
									 CellDouble *c_hx,
									 CellDouble *c_hy,
									 CellDouble *c_hz,
									 Cell  *c,
		                             int index,
		                             int blockDimX//,
//		                             double mass,
//		                             double q_mass
		                             )
{
//	CurrentTensor t1,t2;
    int pqr2;
//	Particle p;
    CellTotalField cf;

    while(index < c->number_of_particles)
    {
    	cf.Ex = c->Ex;
    	cf.Ey = c->Ey;
    	cf.Ez = c->Ez;
    	cf.Hx = c->Hx;
    	cf.Hy = c->Hy;
    	cf.Hz = c->Hz;

        c->MoveSingleParticle(index,cf);


        index += blockDimX;
    }



    __syncthreads();
}






__device__ void copyFromSharedMemoryToCell(
		                                     CellDouble *c_jx,
											 CellDouble *c_jy,
											 CellDouble *c_jz,
											 Cell  *c,
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





__global__
void GPU_getCellEnergy(
		GPUCell **cells,double *d_ee,
		double *d_Ex,double *d_Ey,double *d_Ez)
{
	unsigned int i = blockIdx.x;
	unsigned int l= blockIdx.y;
	unsigned int k = blockIdx.z;
	//int i,l,k;
	Cell *c0 = cells[0],nc;
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

global_for_CUDA void GPU_WriteAllCurrents(GPUCell **cells,int n0,
		      double *jx,double *jy,double *jz,double *rho)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	Cell  *c,*c0 = cells[0];
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

__device__ void AccumulateCurrentWithParticlesInCell(
                                                                         CellDouble *c_jx,
                                                                         CellDouble *c_jy,
                                                                         CellDouble *c_jz,
                                                                         Cell  *c,
                                             int index,
                                             int blockDimX
                                             )
{
        CurrentTensor t1,t2;
        DoubleCurrentTensor dt,dt1;;
    int pqr2;


    while(index < c->number_of_particles)
    {
        c->AccumulateCurrentSingleParticle    (index,&pqr2,&dt);

        dt1 = dt;

        writeCurrentComponent(c_jx,&(dt.t1.Jx),&(dt.t2.Jx),pqr2);
        writeCurrentComponent(c_jy,&(dt.t1.Jy),&(dt.t2.Jy),pqr2);
        writeCurrentComponent(c_jz,&(dt.t1.Jz),&(dt.t2.Jz),pqr2);

        index += blockDimX;
    }
    __syncthreads();
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
		 Cell *c,
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


global_for_CUDA void GPU_CurrentsAllCells(GPUCell  **cells
//		                         int i,
//		                         double *global_jx,
//		                         double mass,
//		                         double q_mass
		                         )
{
	Cell  *c,*c0 = cells[0];
	__shared__ extern CellDouble fd[9];
	CellDouble *c_jx,*c_jy,*c_jz,*c_ex,*c_ey,*c_ez,*c_hx,*c_hy,*c_hz;
//	CurrentTensor t1,t2;
//	int pqr2;
//	Particle p;

	c = cells[ c0->getGlobalCellNumber(blockIdx.x,blockIdx.y,blockIdx.z)];

	assignSharedWithLocal(&c_jx,&c_jy,&c_jz,&c_ex,&c_ey,&c_ez,&c_hx,&c_hy,&c_hz,fd);




	copyFieldsToSharedMemory(c_jx,c_jy,c_jz,c_ex,c_ey,c_ez,c_hx,c_hy,c_hz,c,
			threadIdx.x,blockIdx,blockDim.x);

	AccumulateCurrentWithParticlesInCell(c_jx,c_jy,c_jz,
							 c,threadIdx.x,blockDim.x);


    copyFromSharedMemoryToCell(c_jx,c_jy,c_jz,c,threadIdx.x,blockDim.x,blockIdx);

}




double Plasma::getElectricEnergy()
	{
		dim3 dimGrid(Nx+2,Ny+2,Nz+2),dimGridOne(1,1,1),dimBlock(MAX_particles_per_cell/2,1,1),
    		 dimBlockOne(1,1,1),dimBlockGrow(1,1,1),dimBlockExt(CellExtent,CellExtent,CellExtent);
		static int first = 1;
		static double *d_ee;
		double ee;

		if(first == 1)
		{
			cudaMalloc(&d_ee,sizeof(double));
			first = 0;
		}
		cudaMemset(d_ee,0,sizeof(double));

//		GPU_getCellEnergy//<<<dimGrid, dimBlockOne,16000>>>(d_CellArray,d_ee,d_Ex,d_Ey,d_Ez);
		void* args[] = { (void* )&d_CellArray,
		        		           (void *)&d_ee,
		        		           (void *)&d_Ex,
		        		           (void *)&d_Ey,
		        		           (void *)&d_Ez,
		        		           0};
		          cudaError_t cudaStatus = cudaLaunchKernel(
		          	            (const void*)GPU_getCellEnergy, // pointer to kernel func.
		          	            dimGrid,                       // grid
		          	            dimBlockOne,                      // block
		          	            args,                          // arguments
		          	            0,
		          	            0
		          	            );


        MemoryCopy(&ee,d_ee,sizeof(double),DEVICE_TO_HOST);

        return ee;

	}

int Plasma::WriteCurrentsFromCellsToArrays(int nt)
{
	char name[100];
	dim3 dimGrid(Nx+2,Ny+2,Nz+2);

	sprintf(name,"before_write_currents_%03d.dat",nt);
	write3D_GPUArray(name,d_Jx);

    dim3 dimExt(CellExtent,CellExtent,CellExtent);
//    GPU_WriteAllCurrents//<<<dimGrid, dimExt,16000>>>(d_CellArray,0,d_Jx,d_Jy,d_Jz,d_Rho);
    int zero = 0;
    void* args[] = { (void *)&d_CellArray,
    		         (void *)&zero,
    		         (void *)&d_Jx,
    		         (void *)&d_Jy,
    		         (void *)&d_Jz,
    		         (void *)&d_Rho,
    		         0};
        cudaError_t cudaStatus = cudaLaunchKernel(
                     (const void*)GPU_WriteAllCurrents, // pointer to kernel func.
                     dimGrid,                       // grid
                     dimExt,                   // block
                     args,                          // arguments
                     16000,
                     0
                 );

    memory_monitor("CellOrder_StepAllCells5",nt);

    sprintf(name,"after_write_currents_%03d.dat",nt);
	write3D_GPUArray(name,d_Jx);

	memory_monitor("CellOrder_StepAllCells6",nt);

	return 0;
}

global_for_CUDA void GPU_StepAllCells(GPUCell  **cells//,
//		                         int i,
//		                         double *global_jx
//		                         double mass,
//		                         double q_mass
		                         )
{
	Cell  *c,*c0 = cells[0];
	__shared__ extern CellDouble fd[9];
	CellDouble *c_jx,*c_jy,*c_jz,*c_ex,*c_ey,*c_ez,*c_hx,*c_hy,*c_hz;
//	CurrentTensor t1,t2;
//	int pqr2;
	Particle p;

	c = cells[ c0->getGlobalCellNumber(blockIdx.x,blockIdx.y,blockIdx.z)];

	assignSharedWithLocal(&c_jx,&c_jy,&c_jz,&c_ex,&c_ey,&c_ez,&c_hx,&c_hy,&c_hz,fd);




	copyFieldsToSharedMemory(c_jx,c_jy,c_jz,c_ex,c_ey,c_ez,c_hx,c_hy,c_hz,c,
			threadIdx.x,blockIdx,blockDim.x);


	MoveParticlesInCell(c_ex,c_ey,c_ez,c_hx,c_hy,c_hz,
						 c,threadIdx.x,blockDim.x);//,mass,q_mass);
//	MoveAccCurrent(c_ex,c_ey,c_ez,c_hx,c_hy,c_hz,c_jx,c_jy,c_jz,
//							 c,threadIdx.x,blockDim.x,mass,q_mass);


//    WriteCurrents(c_jx,c_jy,c_jz,c_jx,c_jy,c_jz,
//						 c,threadIdx.x,blockDim.x,mass,q_mass);



    copyFromSharedMemoryToCell(c_jx,c_jy,c_jz,c,threadIdx.x,blockDim.x,blockIdx);

}


int Plasma::StepAllCells(int nt,double mass,double q_mass)
{
	   dim3 dimGrid(Nx+2,Ny+2,Nz+2),dimBlock(512,1,1);
	   cudaDeviceSynchronize();
       puts("begin step");



//	   GPU_StepAllCells//<<<dimGrid, dimBlock,16000>>>(d_CellArray);//,d_Jx);
//	            		     		                 mass,q_mass);

	   void* args[] = { (void* )&d_CellArray,0};
//	   void *d_args;
//	   cudaError_t err = cudaMalloc(&d_args,sizeof(d_CellArray)+sizeof(d_Jx));
//	   cudaError_t err1 = cudaMemcpy(d_args,args,sizeof(d_CellArray)+sizeof(d_Jx),cudaMemcpyHostToDevice);
	   cudaError_t cudaStatus = cudaLaunchKernel(
	                                            (const void*)GPU_StepAllCells, // pointer to kernel func.
	                                            dimGrid,                       // grid
	                                            dimBlock,                      // block
	                                            args,                          // arguments
	                                            16000,
	                                            0
	                                           );
//                mass,q_mass };





//	   GPU_CurrentsAllCells//<<<dimGrid, dimBlock,16000>>>(d_CellArray);//,0,d_Jx,
	   cudaStatus = cudaLaunchKernel(
	                                            (const void*)GPU_CurrentsAllCells, // pointer to kernel func.
	                                            dimGrid,                       // grid
	                                            dimBlock,                      // block
	                                            args,                          // arguments
	                                            16000,
	                                            0
	                                           );
	            		     		                 //mass,q_mass);
	   puts("end step");
	   cudaDeviceSynchronize();

	   puts("end step-12");

	   return 0;
}

__host__ __device__
void emh2_Element(
		Cell *c,
		int i,int l,int k,
		double *Q,double *H)
{
	int n  = c->getGlobalCellNumber(i,l,k);

	H[n] += Q[n];
}


global_for_CUDA
void GPU_emh2(
		 GPUCell  **cells,
				            int i_s,int l_s,int k_s,
							double *Q,double *H
		)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	Cell  *c0 = cells[0];

	emh2_Element(c0,i_s+nx,l_s+ny,k_s+nz,Q,H);
}


__host__ __device__
void emh1_Element(
		Cell *c,
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


global_for_CUDA
void GPU_emh1(
		 GPUCell  **cells,
							double *Q,double *H,double *E1, double *E2,
							double c1,double c2,
							int3 d1,int3 d2
		)
{

	int3 i3 = make_int3(blockIdx.x,blockIdx.y,blockIdx.z);
	Cell  *c0 = cells[0];

	emh1_Element(c0,i3,Q,H,E1,E2,c1,c2,d1,d2);
}

__host__ __device__
	void emeElement(Cell *c,int3 i,double *E,double *H1, double *H2,
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
void periodicElement(Cell *c,int i,int k,double *E,int dir, int to,int from)
{
    int n   = c->getGlobalBoundaryCellNumber(i,k,dir,to);
	int n1  = c->getGlobalBoundaryCellNumber(i,k,dir,from);
	E[n]    = E[n1];
}

global_for_CUDA void GPU_periodic(GPUCell  **cells,
                             int i_s,int k_s,
                             double *E,int dir, int to,int from)
{
	unsigned int nx = blockIdx.x;
	//unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	Cell  *c0 = cells[0];

	periodicElement(c0,nx+i_s,nz+k_s,E, dir,to,from);
}

__host__ __device__
void periodicCurrentElement(Cell *c,int i,int k,double *E,int dir, int dirE,int N)
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


global_for_CUDA void GPU_CurrentPeriodic(GPUCell  **cells,double *E,int dirE, int dir,
                             int i_s,int k_s,int N)
{
	unsigned int nx = blockIdx.x;
	//unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	Cell  *c0 = cells[0];


	periodicCurrentElement(c0,nx+i_s,nz+k_s,E, dir,dirE,N);
}


global_for_CUDA void GPU_eme(

		            GPUCell  **cells,
		            int3 s,
					double *E,double *H1, double *H2,
					double *J,double c1,double c2, double tau,
					int3 d1,int3 d2
		)
{
	unsigned int nx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ny = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int nz = blockIdx.z*blockDim.z + threadIdx.z;
	Cell  *c0 = cells[0];

    s.x += nx;
    s.y += ny;
    s.z += nz;

    emeElement(c0,s,E,H1,H2,J,c1,c2,tau,d1,d2);
}

//////////////////////////////////
void Plasma::emeGPUIterate(int3 s,int3 f,
			double *E,double *H1, double *H2,
			double *J,double c1,double c2, double tau,
			int3 d1,int3 d2)
{
	dim3 dimGrid(f.x-s.x+1,1,1),dimBlock(1,f.y-s.y+1,f.z-s.z+1);

//    GPU_eme//<<<dimGrid,dimBlock>>>(d_CellArray,s,
//    		                            E,H1,H2,
//    					    	  		J,c1,c2,tau,
//    					    	  		d1,d2
//    		);
    void* args[] = { (void* )&d_CellArray,
            		           (void *)&s,
            		           (void *)&E,
            		           (void *)&H1,
            		           (void *)&H2,
            		           (void *)&J,
                               (void *)&c1,
                               (void *)&c2,
                               (void *)&tau,
                               (void *)&d1,
                               (void *)&d2,
            		           0};

              cudaError_t cudaStatus = cudaLaunchKernel(
              	            (const void*)GPU_eme, // pointer to kernel func.
              	            dimGrid,                       // grid
              	            dimBlock,                      // block
              	            args,                          // arguments
              	            0,
              	            0
              	            );


}

void GetElectricFieldStartsDirs(
		int3 *start,
		int3 *d1,
		int3 *d2,
		int dir
		)
{
      start->x = (dir == 0)*0 + (dir == 1)*1 + (dir == 2)*1;
      start->y  = (dir == 0)*1 + (dir == 1)*0 + (dir == 2)*1;
      start->z = (dir == 0)*1 + (dir == 1)*1 + (dir == 2)*0;

      d1->x = (dir == 0)*0    + (dir == 1)*0    + (dir == 2)*(-1);
      d1->y = (dir == 0)*(-1) + (dir == 1)*0    + (dir == 2)*0;
      d1->z = (dir == 0)*0    + (dir == 1)*(-1) + (dir == 2)*0;

      d2->x = (dir == 0)*0    + (dir == 1)*(-1) + (dir == 2)*0;
      d2->y = (dir == 0)*0    + (dir == 1)*0    + (dir == 2)*(-1);
      d2->z = (dir == 0)*(-1) + (dir == 1)*0    + (dir == 2)*0;
}

int Plasma::ElectricFieldTrace(
  double *E,double *H1,double *H2,double *J,
  int dir,double c1,double c2,double tau)
  {
      int3 start,d1,d2,finish = make_int3(Nx,Ny,Nz);


      GetElectricFieldStartsDirs(&start,&d1,&d2,dir);

         emeGPUIterate(start,finish,E,H1,H2,
    	        		    	  		J,c1,c2,tau,
    	        		    	  		d1,d2);

    return 0;
  }

int Plasma::checkFields_beforeMagneticStageOne(double *t_Ex,double *t_Ey,double *t_Ez,
		                               double *t_Hx,double *t_Hy,double *t_Hz,
		                               double *t_Qx,double *t_Qy,double *t_Qz,
		                               double *t_check,int nt)
{

	 memory_monitor("beforeComputeField_FirstHalfStep",nt);

//	         t_check[0] = checkControlMatrix("emh1",nt,"qx",t_Qx);
//			 t_check[1] = checkControlMatrix("emh1",nt,"qy",t_Qy);
//			 t_check[2] = checkControlMatrix("emh1",nt,"qz",t_Qz);
//
//			 t_check[3] = checkControlMatrix("emh1",nt,"ex",t_Ex);
//			 t_check[4] = checkControlMatrix("emh1",nt,"ey",t_Ey);
//			 t_check[5] = checkControlMatrix("emh1",nt,"ez",t_Ez);
//
//			 t_check[6] = checkControlMatrix("emh1",nt,"hx",t_Hx);
//			 t_check[7] = checkControlMatrix("emh1",nt,"hy",t_Hy);
//			 t_check[8] = checkControlMatrix("emh1",nt,"hz",t_Hz);
	return 0;
}

int Plasma::checkFields_afterMagneticStageOne(double *t_Hx,double *t_Hy,double *t_Hz,
		                              double *t_Qx,double *t_Qy,double *t_Qz,
		                              double *t_check,int nt)
{
//	         t_check[9] = checkControlMatrix("emh1",nt,"qx",t_Qx);
//			 t_check[10] = checkControlMatrix("emh1",nt,"qy",t_Qy);
//			 t_check[11] = checkControlMatrix("emh1",nt,"qz",t_Qz);
//
//			 t_check[12] = checkControlMatrix("emh1",nt,"hx",t_Hx);
//			 t_check[13] = checkControlMatrix("emh1",nt,"hy",t_Hy);
//			 t_check[14] = checkControlMatrix("emh1",nt,"hz",t_Hz);


			 CPU_field = 1;



			 checkControlPoint(50,nt,0);
			 memory_monitor("afterComputeField_FirstHalfStep",nt);

	return 0;
}

void  Plasma::ComputeField_FirstHalfStep(
		   int nt
		   )
{
	 double t_check[15];
    cudaError_t err;
	err = cudaGetLastError();
    if(err != cudaSuccess) { printf("%s:%d - error %d %s\n",__FILE__,__LINE__,err,cudaGetErrorString(err)); }

     checkFields_beforeMagneticStageOne(d_Ex,d_Ey,d_Ez,
		 		                               d_Hx,d_Hy,d_Hz,
		 		                               d_Qx,d_Qy,d_Qz,
		 		                               t_check,nt);


	err = cudaGetLastError();
    if(err != cudaSuccess) { printf("%s:%d - error %d %s\n",__FILE__,__LINE__,err,cudaGetErrorString(err)); }
	 MagneticStageOne(d_Qx,d_Qy,d_Qz,d_Hx,d_Hy,d_Hz,d_Ex,d_Ey,d_Ez);

	err = cudaGetLastError();
    if(err != cudaSuccess) { printf("%s:%d - error %d %s\n",__FILE__,__LINE__,err,cudaGetErrorString(err)); }
	 checkFields_afterMagneticStageOne(d_Hx,d_Hy,d_Hz,
		 		                           d_Qx,d_Qy,d_Qz,
		 		                           t_check,nt);
	err = cudaGetLastError();
    if(err != cudaSuccess) { printf("%s:%d - error %d %s\n",__FILE__,__LINE__,err,cudaGetErrorString(err)); }
	 AssignCellsToArraysGPU();
	err = cudaGetLastError();
    if(err != cudaSuccess) { printf("%s:%d - error %d %s\n",__FILE__,__LINE__,err,cudaGetErrorString(err)); }

}

void Plasma::ComputeField_SecondHalfStep(int nt)
{

     SetPeriodicCurrents(nt);



     MagneticFieldStageTwo(d_Hx,d_Hy,d_Hz,nt,d_Qx,d_Qy,d_Qz);




     ElectricFieldEvaluate(d_Ex,d_Ey,d_Ez,nt,d_Hx,d_Hy,d_Hz,d_Jx,d_Jy,d_Jz);


}

void Plasma::ElectricFieldComponentEvaluateTrace(
		  double *E,double *H1,double *H2,double *J,
		  int dir,
		  double c1,double c2,double tau,
		  int dir_1,int start1_1,int end1_1,int start2_1,int end2_1,int N_1,
		  int dir_2,int start1_2,int end1_2,int start2_2,int end2_2,int N_2
		  )

{
     ElectricFieldTrace(E,H1,H2,J,dir,c1,c2,tau);

//     PeriodicBoundaries(E, dir_1,start1_1,end1_1,start2_1,end2_1,N_1);
//     PeriodicBoundaries(E, dir_2,start1_2,end1_2,start2_2,end2_2,N_2);
}


void Plasma::ElectricFieldComponentEvaluatePeriodic(
		  double *E,double *H1,double *H2,double *J,
		  int dir,
		  double c1,double c2,double tau,
		  int dir_1,int start1_1,int end1_1,int start2_1,int end2_1,int N_1,
		  int dir_2,int start1_2,int end1_2,int start2_2,int end2_2,int N_2
		  )

{
//     ElectricFieldTrace(E,H1,H2,J,dir,c1,c2,tau);

     if (dir != 0)
     {
    	 PeriodicBoundaries(E, dir_1,start1_1,end1_1,start2_1,end2_1,N_1);
     }

     if (dir != 2)
     {
    	 PeriodicBoundaries(E, dir_2,start1_2,end1_2,start2_2,end2_2,N_2);
     }
}


void Plasma::ElectricFieldEvaluate(double *locEx,double *locEy,double *locEz,
		   int nt,
		   double *locHx,double *locHy,double *locHz,
		   double *loc_npJx,double *loc_npJy,double *loc_npJz)
{
	 CPU_field = 0;
      double3 c1 = getMagneticFieldTimeMeshFactors();

      ElectricFieldComponentEvaluateTrace(
    		  locEx,locHz,locHy,loc_npJx,
    		  0,c1.y,c1.z,tau,
              1,0,Nx,1,Nz,Ny,
              2,0,Nx,0,Ny+1,Nz);

      ElectricFieldComponentEvaluateTrace(
          		  locEy,locHx,locHz,loc_npJy,
          		  1,c1.z,c1.x,tau,
                    0,0,Ny,1,Nz,Nx,
                    2,0,Nx+1,0,Ny,Nz);



      ElectricFieldComponentEvaluateTrace(
         		  locEz,locHy,locHx,loc_npJz,
         		  2,c1.x,c1.y,tau,
                   0,1,Ny,0,Nz,Nx,
                   1,0,Nx+1,0,Nz,Ny);

      checkControlPoint(550,nt,0);


      ElectricFieldComponentEvaluatePeriodic(
     		  locEx,locHz,locHy,loc_npJx,
     		  0,c1.y,c1.z,tau,
               1,0,Nx,1,Nz,Ny,
               2,0,Nx,0,Ny+1,Nz);

      ElectricFieldComponentEvaluatePeriodic(
    		  locEy,locHx,locHz,loc_npJy,
    		  1,c1.z,c1.x,tau,
              0,0,Ny,1,Nz,Nx,
              2,0,Nx+1,0,Ny,Nz);

//      SinglePeriodicBoundary(locEy,1,0,Nx+1,0,Nz+1,Ny);


      ElectricFieldComponentEvaluatePeriodic(
        		  locEz,locHy,locHx,loc_npJz,
        		  2,c1.x,c1.y,tau,
                  0,1,Ny,0,Nz,Nx,
                  1,0,Nx+1,0,Nz,Ny);

         checkControlPoint(600,nt,0);

         memory_monitor("after_ComputeField_SecondHalfStep",nt);
}

double3 Plasma::getMagneticFieldTimeMeshFactors()
{
    Cell c = (*AllCells)[0];
	double hx = c.get_hx(),hy = c.get_hy(),hz = c.get_hz();
	double3 d;
    d.x = tau/(hx);
    d.y = tau/(hy);
    d.z = tau/hz;

	return d;
}

void Plasma::MagneticStageOne(
                  double *Qx,double *Qy,double *Qz,
                  double *Hx,double *Hy,double *Hz,
   	              double *Ex,double *Ey,double *Ez
		           )
{
	double3 c1 = getMagneticFieldTimeMeshFactors();

    MagneticFieldTrace(Qx,Hx,Ey,Ez,Nx+1,Ny,Nz,c1.z,c1.y,0);

    MagneticFieldTrace(Qy,Hy,Ez,Ex,Nx,Ny+1,Nz,c1.x,c1.z,1);

    MagneticFieldTrace(Qz,Hz,Ex,Ey,Nx,Ny,Nz+1,c1.y,c1.x,2);

}

void Plasma::MagneticFieldStageTwo(double *Hx,double *Hy,double *Hz,
		            int nt,
		            double *Qx,double *Qy,double *Qz)
{
    Cell c = (*AllCells)[0];

    SimpleMagneticFieldTrace(c,Qx,Hx,Nx+1,Ny,Nz);
    SimpleMagneticFieldTrace(c,Qy,Hy,Nx,Ny+1,Nz);
    SimpleMagneticFieldTrace(c,Qz,Hz,Nx,Ny,Nz+1);

    checkControlPoint(500,nt,0);
}


int Plasma::PushParticles(int nt)
{
	double mass = -1.0/1836.0,q_mass = -1.0;

	memory_monitor("before_CellOrder_StepAllCells",nt);

    CellOrder_StepAllCells(nt,	mass,q_mass,1);
    puts("cell_order");

	memory_monitor("after_CellOrder_StepAllCells",nt);

	//checkParticleAttributes(nt);

	checkControlPoint(270,nt,1);

	return 0;
}


int Plasma::MagneticFieldTrace(double *Q,double *H,double *E1,double *E2,int i_end,int l_end,int k_end,double c1,double c2,int dir)
	{
	      int3 d1,d2;

	      getMagneticFieldTraceShifts(dir,d1,d2);

   		dim3 dimGrid(i_end+1,l_end+1,k_end+1),dimBlock(1,1,1);

//	    GPU_emh1//<<<dimGrid,dimBlock>>>(d_CellArray,Q,H,E1,E2,c1,c2,
//	    		d1,d2);
	    void* args[] = { (void* )&d_CellArray,
	            		           (void *)&Q,
	            		           (void *)&H,
	            		           (void *)&E1,
	            		           (void *)&E2,
	            		           (void *)&c1,
	                               (void *)&c2,
	            		           (void *)&d1,
	                               (void *)&d2,
	            		           0};
	              cudaError_t cudaStatus = cudaLaunchKernel(
	              	            (const void*)GPU_emh1, // pointer to kernel func.
	              	            dimGrid,                       // grid
	              	            dimBlock,                      // block
	              	            args,                          // arguments
	              	            0,
	              	            0
	              	            );



	      return 0;
	  }



int Plasma::SimpleMagneticFieldTrace(Cell &c,double *Q,double *H,int i_end,int l_end,int k_end)
	{


		   		dim3 dimGrid(i_end+1,l_end+1,k_end+1),dimBlock(1,1,1);

//			    GPU_emh2//<<<dimGrid,dimBlock>>>(d_CellArray,0,0,0,Q,H);
			    int i_s = 0;
			    int l_s = 0;
			    int k_s = 0;

			    void* args[] = { (void* )&d_CellArray,
			    		         (void *)&i_s,
			    		         (void *)&l_s,
			    		         (void *)&k_s,
			    		         (void *)&Q,
			    		         (void *)&H,
			   	            		     0};
			   	              cudaError_t cudaStatus = cudaLaunchKernel(
			   	              	            (const void*)GPU_emh2, // pointer to kernel func.
			   	              	            dimGrid,                       // grid
			   	              	            dimBlock,                      // block
			   	              	            args,                          // arguments
			   	              	            0,
			   	              	            0
			   	              	            );



	      return 0;
	  }

int Plasma::PeriodicBoundaries(double *E,int dir,int start1,int end1,int start2,int end2,int N)
	  {
	      Cell  c = (*AllCells)[0];

//	      if(CPU_field == 0)
//	      {
	    		dim3 dimGrid(end1-start1+1,1,end2-start2+1),dimBlock(1,1,1);

//	    	    GPU_periodic//<<<dimGrid,dimBlock>>>(d_CellArray,start1,start2,E,dir,0,N);
	    	    int zero = 0;
	    	    void* args[] = { (void* )&d_CellArray,
	    	   			         (void *)&start1,
	    	   			    	 (void *)&start2,
	    	   			         (void *)&E,
	    	   			    	 (void *)&dir,
	    	   			    	 (void *)&zero,
	    	   			    	 (void *)&N,
	    	   			   	     0};
	    	     cudaError_t cudaStatus = cudaLaunchKernel(
	    	   	   	            (const void*)GPU_periodic, // pointer to kernel func.
	    	   			   	    dimGrid,                       // grid
	    	   			   	    dimBlock,                      // block
	    	   			   	    args,                          // arguments
	    	   			   	    0,
	    	   			   	    0
	    	   			   	    );

//	    	    GPU_periodic//<<<dimGrid,dimBlock>>>(d_CellArray,start1,start2,E,dir,N+1,1);
	    	    int one = 1;
	    	    int N1 = N+1;

				void *args1[] = { (void* )&d_CellArray,
								 (void *)&start1,
								 (void *)&start2,
								 (void *)&E,
								 (void *)&dir,
								 (void *)&N1,
								 (void *)&one,
								 0};
				 cudaStatus = cudaLaunchKernel(
								(const void*)GPU_periodic, // pointer to kernel func.
								dimGrid,                       // grid
								dimBlock,                      // block
								args1,                          // arguments
								0,
								0
								);
//
//	      }
//	      else
//	      {
//
//	      for(int k = start2;k <= end2;k++)
//	      {
//		  for(int i = start1;i <= end1;i++)
//		  {
//			  periodicElement(&c,i,k,E,dir,0,N);
//		  }
//	      }
//	      for(int k = start2;k <= end2;k++)
//	      {
//	         for(int i = start1;i <= end1;i++)
//	      	 {
//	        	 periodicElement(&c,i,k,E,dir,N+1,1);
//		  }
//	      }
//	      }
	      return 0;
	}



int Plasma::SinglePeriodicBoundary(double *E,int dir,int start1,int end1,int start2,int end2,int N)
{
    Cell  c = (*AllCells)[0];

    if(CPU_field == 0)
    {
    	dim3 dimGrid(end1-start1+1,1,end2-start2+1),dimBlock(1,1,1);

//   	    GPU_periodic//<<<dimGrid,dimBlock>>>(d_CellArray,start1,start2,E,dir,N+1,1);
   	    int N1 = N+1;
   	    int one = 1;
     	void* args[] = { (void* )&d_CellArray,
   	         		           (void *)&start1,
   	         		           (void *)&start2,
   	         		           (void *)&E,
   	         		           (void *)&dir,
   	         		           (void *)&N1,
   	                            (void *)&one,
   	         		           0};
   	           cudaError_t cudaStatus = cudaLaunchKernel(
   	           	            (const void*)GPU_periodic, // pointer to kernel func.
   	           	            dimGrid,                       // grid
   	           	            dimBlock,                      // block
   	           	            args,                          // arguments
   	           	            16000,
   	           	            0
   	           	            );


    }
    else
    {
       for(int k = start2;k <= end2;k++)
       {
	  	  for(int i = start1;i <= end1;i++)
	  	  {
	  		  int3 i0,i1;

	                    int n   = c.getGlobalBoundaryCellNumber(i,k,dir,N+1);
	  		            int n1  = c.getGlobalBoundaryCellNumber(i,k,dir,1);
	  		            E[n]    = E[n1];
	  		            i0= c.getCellTripletNumber(n);
	  		            i1= c.getCellTripletNumber(n1);
	  		            std::cout << "ex1 "<< i0.x+1 << " "<< i0.y+1 << " " << i0.z+1  <<" " <<  i1.x+1 << " " << i1.y+1 << " " << i1.z+1  << " " << E[n]  << " " << E[n1] << std::endl;
	  		   	  }
	        }
    }
    return 0;
}

int SetPeriodicCurrentComponent(GPUCell **cells,double *J,int dir,int Nx,int Ny,int Nz)
	  {
		  dim3 dimGridX(Ny+2,1,Nz+2),dimGridY(Nx+2,1,Nz+2),dimGridZ(Nx+2,1,Ny+2),dimBlock(1,1,1);


//          GPU_CurrentPeriodic//<<<dimGridX,dimBlock>>>(cells,J,dir,0,0,0,Nx+2);
          int dir2 = 0;
          int i_s  = 0;
          int k_s  = 0;
          int N    = Nx+2;
          void* args[] = { (void* )&cells,
        		           (void *)&J,
        		           (void *)&dir,
        		           (void *)&dir2,
        		           (void *)&i_s,
        		           (void *)&k_s,
                           (void *)&N,
        		           0};
          cudaError_t cudaStatus = cudaLaunchKernel(
          	            (const void*)GPU_CurrentPeriodic, // pointer to kernel func.
          	            dimGridX,                       // grid
          	            dimBlock,                      // block
          	            args,                          // arguments
          	            16000,
          	            0
          	            );

//	      GPU_CurrentPeriodic//<<<dimGridY,dimBlock>>>(cells,J,dir,1,0,0,Ny+2);
	               dir2 = 1;
	               N    = Ny+2;
          cudaStatus = cudaLaunchKernel(
	               	            (const void*)GPU_CurrentPeriodic, // pointer to kernel func.
	               	            dimGridY,                       // grid
	               	            dimBlock,                      // block
	               	            args,                          // arguments
	               	            16000,
	               	            0
	               	            );

//	      GPU_CurrentPeriodic//<<<dimGridZ,dimBlock>>>(cells,J,dir,2,0,0,Nz+2);
	      dir2 = 2;
	      N    = Nz+2;
	      cudaStatus = cudaLaunchKernel(
	     	               	            (const void*)GPU_CurrentPeriodic, // pointer to kernel func.
	     	               	            dimGridZ,                       // grid
	     	               	            dimBlock,                      // block
	     	               	            args,                          // arguments
	     	               	            16000,
	     	               	            0
	     	               	            );

		  return 0;
	  }

int Plasma::SetPeriodicCurrentComponent(GPUCell **cells,double *J,int dir,int Nx,int Ny,int Nz)
	  {
		  dim3 dimGridX(Ny+2,1,Nz+2),dimGridY(Nx+2,1,Nz+2),dimGridZ(Nx+2,1,Ny+2),dimBlock(1,1,1);


//          GPU_CurrentPeriodic//<<<dimGridX,dimBlock>>>(cells,J,dir,0,0,0,Nx+2);
         int dir2 = 0;
         int i_s  = 0;
         int k_s  = 0;
         int N    = Nx+2;
         void* args[] = { (void* )&cells,
       		           (void *)&J,
       		           (void *)&dir,
       		           (void *)&dir2,
       		           (void *)&i_s,
       		           (void *)&k_s,
                          (void *)&N,
       		           0};
         cudaError_t cudaStatus = cudaLaunchKernel(
         	            (const void*)GPU_CurrentPeriodic, // pointer to kernel func.
         	            dimGridX,                       // grid
         	            dimBlock,                      // block
         	            args,                          // arguments
         	            16000,
         	            0
         	            );

//	      GPU_CurrentPeriodic//<<<dimGridY,dimBlock>>>(cells,J,dir,1,0,0,Ny+2);
	               dir2 = 1;
	               N    = Ny+2;
         cudaStatus = cudaLaunchKernel(
	               	            (const void*)GPU_CurrentPeriodic, // pointer to kernel func.
	               	            dimGridY,                       // grid
	               	            dimBlock,                      // block
	               	            args,                          // arguments
	               	            16000,
	               	            0
	               	            );

//	      GPU_CurrentPeriodic//<<<dimGridZ,dimBlock>>>(cells,J,dir,2,0,0,Nz+2);
	      dir2 = 2;
	      N    = Nz+2;
	      cudaStatus = cudaLaunchKernel(
	     	               	            (const void*)GPU_CurrentPeriodic, // pointer to kernel func.
	     	               	            dimGridZ,                       // grid
	     	               	            dimBlock,                      // block
	     	               	            args,                          // arguments
	     	               	            16000,
	     	               	            0
	     	               	            );

		  return 0;
	  }

void Plasma::AssignCellsToArraysGPU()
{
	dim3 dimGrid(Nx,Ny,Nz),dimBlockExt(CellExtent,CellExtent,CellExtent);
	cudaError_t err = cudaGetLastError();
    printf("%s:%d - error %d %s\n",__FILE__,__LINE__,err,cudaGetErrorString(err));

	size_t sz;
	err = cudaDeviceGetLimit(&sz,cudaLimitStackSize);
	printf("%s:%d - stack limit %d err = %d\n",__FILE__,__LINE__,((int)sz),err);
	err = cudaDeviceSetLimit(cudaLimitStackSize, 64*1024);
	printf("%s:%d - set stack limit %d \n",__FILE__,__LINE__,err);
	err= cudaDeviceGetLimit(&sz,cudaLimitStackSize);
	printf("%s:%d - stack limit %d err %d\n",__FILE__,__LINE__,((int)sz),err);

//	GPU_SetFieldsCel..ls//<<<dimGrid, dimBlockExt>>>(d_CellArray,d_Ex,d_Ey,d_Ez,d_Hx,d_Hy,d_Hz);
	void* args[] = { (void* )&d_CellArray,&d_Ex,&d_Ey,&d_Ez,&d_Hx,&d_Hy,&d_Hz,0};
	cudaError_t cudaStatus = cudaLaunchKernel(
	                        (const void*)GPU_SetFieldsToCells, // pointer to kernel func.
	                        dimGrid,                       // grid
	                        dimBlockExt,                      // block
	                        args,                          // arguments
	                        16000,
	                        0
	                        );


	cudaDeviceSynchronize();
	err = cudaGetLastError();
    printf("%s:%d - error %d %s\n",__FILE__,__LINE__,err,cudaGetErrorString(err));

}


int Plasma::SetCurrentsInCellsToZero(int nt)
{
	dim3 dimGrid(Nx+2,Ny+2,Nz+2),dimBlockExt(CellExtent,CellExtent,CellExtent);
	char name[100];
	sprintf(name,"before_set_to_zero_%03d.dat",nt);

	write3D_GPUArray(name,d_Jx);

//	GPU_SetAllCurrentsToZero//<<<dimGrid, dimBlockExt,16000>>>(d_CellArray);
	void* args[] = { (void* )&d_CellArray,0};
    cudaError_t cudaStatus = cudaLaunchKernel(
                 (const void*)GPU_SetAllCurrentsToZero, // pointer to kernel func.
                 dimGrid,                       // grid
                 dimBlockExt,                   // block
                 args,                          // arguments
                 16000,
                 0
             );


	return 0;
}


int Plasma::MakeParticleList(int nt,int *stage,int *stage1,int **d_stage,int **d_stage1)
{
	dim3 dimGrid(Nx+2,Ny+2,Nz+2),dimGridOne(1,1,1),dimBlock(512,1,1),
	     dimBlockOne(1,1,1),dimBlockGrow(1,1,1),dimBlockExt(CellExtent,CellExtent,CellExtent);
	dim3 dimGridBulk(Nx,Ny,Nz);
	cudaError_t before_MakeDepartureLists,after_MakeDepartureLists;
//      before_ArrangeFlights;//,after_ArrangeFlights;

    before_MakeDepartureLists = cudaGetLastError();
    printf("before_MakeDepartureLists %d %s blockdim %d %d %d\n",before_MakeDepartureLists,
    cudaGetErrorString(before_MakeDepartureLists),dimGrid.x,dimGrid.y,dimGrid.z);

//    int stage[4000],stage1[4000];//,*d_stage,*d_stage1;
    cudaMalloc(d_stage,sizeof(int)*(Nx+2)*(Ny+2)*(Nz+2));

    cudaMalloc(d_stage1,sizeof(int)*(Nx+2)*(Ny+2)*(Nz+2));

//    GPU_MakeDepartureLists//<<<dimGrid, dimBlockOne>>>(d_CellArray,nt,*d_stage);

    void* args[] = {
    		         (void *)&d_CellArray,
    		         (void *)&nt,
    		         (void *)d_stage,
    		         0};
        cudaError_t cudaStatus = cudaLaunchKernel(
                     (const void*)GPU_MakeDepartureLists, // pointer to kernel func.
                     dimGrid,                       // grid
                     dimBlockOne,                   // block
                     args,                          // arguments
                     16000,
                     0
                 );

    after_MakeDepartureLists = cudaGetLastError();
    if(after_MakeDepartureLists != cudaSuccess)
    {
       printf("after_MakeDepartureLists %d %s\n",after_MakeDepartureLists,cudaGetErrorString(after_MakeDepartureLists));
    }

    cudaDeviceSynchronize();

    int err = cudaGetLastError();

    if(err != cudaSuccess)
        {
           printf("MakeParticleList sync error %d %s\n",err,getErrorString(err));
        }
    err = MemoryCopy(stage,*d_stage,sizeof(int)*(Nx+2)*(Ny+2)*(Nz+2),DEVICE_TO_HOST);

    if(err != cudaSuccess)
    {
       printf("MakeParticleList error %d %s\n",err,getErrorString(err));
       exit(0);
    }

    return (int)err;
}


int Plasma::inter_stage_diagnostic(int *stage,int nt)
{
           if(stage[0] == TOO_MANY_PARTICLES)
            {
               printf("too many particles flying to (%d,%d,%d) from (%d,%d,%d) \n",
                          stage[1],stage[2],stage[3],stage[4],stage[5],stage[6]);
               exit(0);
            }

            ListAllParticles(nt,"aMakeDepartureLists");
#ifdef BALANCING_PRINTS
            before_ArrangeFlights = cudaGetLastError();
            printf("before_ArrangeFlights %d %s\n",before_ArrangeFlights,cudaGetErrorString(before_ArrangeFlights));
#endif

            return 0;
}

int Plasma::reorder_particles(int nt)
{
    int stage[4000],stage1[4000],*d_stage,*d_stage1,err;

    MakeParticleList(nt,stage,stage1,&d_stage,&d_stage1);

    inter_stage_diagnostic(stage,nt);

    err = reallyPassParticlesToAnotherCells(nt,stage1,d_stage1);

    return (int)err;
}



























