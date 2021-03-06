/*
 * gpucell.h
 *
 *  Created on: Oct 19, 2013
 *      Author: snytav
 */

#ifndef GPUCELL_H_
#define GPUCELL_H_

//#include "cuPrintf.cu"
#include "cell.h"
#include "archAPI.h"
#include <stdlib.h>


void dbgPrintGPUParticleAttribute(Cell<Particle> *d_c,int n_particle,int attribute,char *name )
{
    double t;
    Cell<Particle> *h_c;
    int shift = (attribute + n_particle*sizeof(Particle)/sizeof(double));
    cudaError_t err;

    h_c = new Cell<Particle>;

    err = cudaMemcpy(h_c,d_c,sizeof(Cell<Particle>),cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
        {
        	printf("pyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }
    double *vec = h_c->doubParticleArray + shift;

    cudaMemcpy(&t,vec,sizeof(double),cudaMemcpyDeviceToHost);

    printf("%s %10.3e \n",name,t);
}

__global__ void testKernelBefore(double *vec,int n_particle,int attribute)
{
   //  	cuPrintf("vecBefore %15.5e \n",vec[attribute + n_particle*sizeof(Particle)/sizeof(double)]);
}

__global__ void testKernel(double *vec)
{
    // 	cuPrintf("vec %15.5e \n",vec[1]);
}

template <class Particle >
class GPUCell: public Cell<Particle>
{


public:
	  double *d_wrong_current_particle_attributes,*h_wrong_current_particle_attributes;



 #ifdef __CUDACC__
 __host__ __device__
 #endif

    GPUCell(){}

 #ifdef __CUDACC__
 __host__ __device__
 #endif

   ~GPUCell(){}

 #ifdef __CUDACC__
 __host__ __device__
 #endif

    GPUCell(int i1,int l1,int k1,double Lx,double Ly, double Lz,int Nx1, int Ny1, int Nz1,double tau1):
       Cell<Particle>(i1,l1,k1,Lx,Ly,Lz,Nx1,Ny1,Nz1,tau1){}

double compareArrayHostToDevice(double *h, double *d,int size,char *legend)
{
	double h_d[8*CellExtent*CellExtent*CellExtent],t;

//	h_d = (double *)malloc(size);

	MemoryCopy(h_d,d,size,DEVICE_TO_HOST);

	t = compare(h,h_d,size/sizeof(double),legend,TOLERANCE);

	return t;
}

GPUCell<Particle>* copyCellToDevice()
{
	GPUCell<Particle> *h_src,*d_dst;//,*h_ctrl;
	int err1,err2,err3,err4,err5,err6,err7,err8,err9,err10;
	int err11,err12,err13,err14,err15,err16,err17,err18,err19,err20;
	int err21,err22,err23,err24,err25;


	h_src = new GPUCell<Particle>;
	//h_ctrl = new GPUCell<Particle>;

	h_src->number_of_particles = Cell<Particle>::number_of_particles;
	h_src->Nx = Cell<Particle>::Nx;
	h_src->Ny = Cell<Particle>::Ny;
	h_src->Nz = Cell<Particle>::Nz;
	h_src->hx = Cell<Particle>::hx;
	h_src->hy = Cell<Particle>::hy;
	h_src->hz = Cell<Particle>::hz;
	h_src->i  = Cell<Particle>::i;
	h_src->k  = Cell<Particle>::k;
	h_src->l  = Cell<Particle>::l;
	h_src->x0 = Cell<Particle>::x0;
	h_src->y0 = Cell<Particle>::y0;
	h_src->z0 = Cell<Particle>::z0;
	h_src->xm = Cell<Particle>::xm;
	h_src->ym = Cell<Particle>::ym;
	h_src->zm = Cell<Particle>::zm;
	h_src->tau = Cell<Particle>::tau;
	h_src->jmp = Cell<Particle>::jmp;
	h_src->d_ctrlParticles = Cell<Particle>::d_ctrlParticles;

	h_src->busyParticleArray = Cell<Particle>::busyParticleArray;

	//cudaPrintfInit();
	cudaMalloc(&(h_src->doubParticleArray),sizeof(Particle)*MAX_particles_per_cell);
	err1 = cudaGetLastError();



	cudaMemset(h_src->doubParticleArray,0,sizeof(Particle)*MAX_particles_per_cell);
	err2 = cudaGetLastError();

	//testKernelBefore<<<1,1>>>(h_src->doubParticleArray,50,1);
	//cudaThreadSynchronize();


	cudaMemcpy(h_src->doubParticleArray,Cell<Particle>::doubParticleArray,
			   sizeof(Particle)*MAX_particles_per_cell,cudaMemcpyHostToDevice);
	err3 = cudaGetLastError();

//	compareArrayHostToDevice((double *)Cell<Particle>::doubParticleArray,
	//		(double *)h_src->doubParticleArray,sizeof(Particle)*MAX_particles_per_cell,"part");
	//printf("after copy %e\n",this->ParticleArrayRead(50,1));
	//dbgPrintGPUParticleAttribute(d_dst,50,1," IN_COPY0 " );
	//testKernelBefore<<<1,1>>>(h_src->doubParticleArray,50,1);
	//cudaPrintfDisplay(stdout, true);
	//cudaPrintfEnd();

	MemoryAllocate((void**)&(h_src->Jx),sizeof(CellDouble));
	err4 = getLastError();

	MemoryCopy(h_src->Jx,Cell<Particle>::Jx,sizeof(CellDouble),HOST_TO_DEVICE);
	err5 = getLastError();

	//compareArrayHostToDevice((double *)Cell<Particle>::Jx,(double *)h_src->Jx,sizeof(CellDouble),"Jx");

	MemoryAllocate((void**)&(h_src->Jy),sizeof(CellDouble));
	err6 = getLastError();

	MemoryCopy(h_src->Jy,Cell<Particle>::Jy,sizeof(CellDouble),HOST_TO_DEVICE);
	//compareArrayHostToDevice((double *)Cell<Particle,dims>::Jy,(double *)h_src->Jy,sizeof(CellDouble),"Jy");
	err7 = getLastError();


	MemoryAllocate((void**)&(h_src->Jz),sizeof(CellDouble));
	err8 = getLastError();

	MemoryCopy(h_src->Jz,Cell<Particle>::Jz,sizeof(CellDouble),HOST_TO_DEVICE);
	err9 = getLastError();

	//compareArrayHostToDevice((double *)Cell<Particle>::Jz,(double *)h_src->Jz,sizeof(CellDouble),"Jz");

	MemoryAllocate((void**)&(h_src->Ex),sizeof(CellDouble));
	err10 = getLastError();

	MemoryCopy(h_src->Ex,Cell<Particle>::Ex,sizeof(CellDouble),HOST_TO_DEVICE);
	err11 = getLastError();

	//compareArrayHostToDevice((double *)Cell<Particle>::Ex,(double *)h_src->Ex,sizeof(CellDouble),"Ex");

	MemoryAllocate((void**)&(h_src->Ey),sizeof(CellDouble));
	err12 = getLastError();

	MemoryCopy(h_src->Ey,Cell<Particle>::Ey,sizeof(CellDouble),HOST_TO_DEVICE);
	err13 = getLastError();

	//compareArrayHostToDevice((double *)Cell<Particle>::Ey,(double *)h_src->Ey,sizeof(CellDouble),"Ey");

	MemoryAllocate((void**)&(h_src->Ez),sizeof(CellDouble));
	err14 = getLastError();

	MemoryCopy(h_src->Ez,Cell<Particle>::Ez,sizeof(CellDouble),HOST_TO_DEVICE);
	err15 = getLastError();

	//compareArrayHostToDevice((double *)Cell<Particle>::Ez,(double *)h_src->Ez,sizeof(CellDouble),"Ez");

	MemoryAllocate((void**)&(h_src->Hx),sizeof(CellDouble));
	err16 = getLastError();

	MemoryCopy(h_src->Hx,Cell<Particle>::Hx,sizeof(CellDouble),HOST_TO_DEVICE);
	err17 = getLastError();

	//compareArrayHostToDevice((double *)Cell<Particle>::Hx,(double *)h_src->Hx,sizeof(CellDouble),"Hx");

	cudaMalloc(&(h_src->Hy),sizeof(CellDouble));
	err18 = cudaGetLastError();

	MemoryCopy(h_src->Hy,Cell<Particle>::Hy,sizeof(CellDouble),HOST_TO_DEVICE);
	err19 = getLastError();

	//compareArrayHostToDevice((double *)Cell<Particle>::Hy,(double *)h_src->Hy,sizeof(CellDouble),"Hy");

	MemoryAllocate((void**)&(h_src->Hz),sizeof(CellDouble));
	err20 = getLastError();

	MemoryCopy(h_src->Hz,Cell<Particle>::Hz,sizeof(CellDouble),HOST_TO_DEVICE);
	err21 = getLastError();

	//compareArrayHostToDevice((double *)Cell<Particle>::Hz,(double *)h_src->Hz,sizeof(CellDouble),"Hz");

	MemoryAllocate((void**)&(h_src->Rho),sizeof(CellDouble));
	err22 = getLastError();

	MemoryCopy(h_src->Rho,Cell<Particle>::Rho,sizeof(CellDouble),HOST_TO_DEVICE);
	err23 = getLastError();

	//compareArrayHostToDevice((double *)Cell<Particle>::Rho,(double *)h_src->Rho,sizeof(CellDouble),"Rho");

	//memcpy((unsigned char *)dst.Jx,(unsigned char *)src.Jx,sizeof(CellDouble));
	//printf("i %d l %d k %d q_m %15.5e \n",h_src->i,h_src->k,h_src->l,Cell<Particle>::ParticleArrayRead(0,7));

    MemoryAllocate((void**)&d_dst,sizeof(GPUCell<Particle>));
	err24 = getLastError();



    MemoryCopy(d_dst,h_src,sizeof(GPUCell<Particle>),HOST_TO_DEVICE);
	err25 = getLastError();

	if(
			(err1 != 0) ||
			(err2 != 0) ||
			(err3 != 0) ||
			(err4 != 0) ||
			(err5 != 0) ||
			(err6 != 0) ||
			(err7 != 0) ||
			(err8 != 0) ||
			(err9 != 0) ||
			(err10 != 0) ||
			(err11 != 0) ||
			(err12 != 0) ||
			(err13 != 0) ||
			(err14 != 0) ||
			(err15 != 0) ||
			(err16 != 0) ||
			(err17 != 0) ||
			(err18 != 0) ||
			(err19 != 0) ||
			(err20 != 0) ||
			(err21 != 0) ||
			(err22 != 0) ||
			(err23 != 0) ||
			(err24 != 0) ||
			(err25 != 0)
	  )
	{
		//int qq = 0;
	}

 //   cudaMemcpy(h_ctrl,d_dst,sizeof(Cell<Particle>),cudaMemcpyDeviceToHost);
//
  //      dbgPrintGPUParticleAttribute(d_dst,50,1," IN_COPY " );
  //  cudaMemcpy(Cell<Particle>::doubParticleArray,h_ctrl->doubParticleArray,
    //			   sizeof(Particle)*MAX_particles_per_cell,cudaMemcpyDeviceToHost);

   // printf("before copy return  %e\n",this->ParticleArrayRead(50,1));
   // dbgPrintGPUParticleAttribute(d_dst,50,1," IN_COPY " );


     return d_dst;
}

void copyCellFromDevice(GPUCell<Particle>* d_src,GPUCell<Particle>* h_dst,char *where,int nt)
{
	static GPUCell<Particle> *h_copy_of_d_src;
	static int first = 1;
	int code;

//#ifdef WRONG_CURRENTS_CHECK
//	static int first = 1;

	if(first == 1)
	{
	   first = 0;
	   h_copy_of_d_src = new GPUCell<Particle>;
	   h_copy_of_d_src->Init();

	}
//	cudaError_t err_attr = cudaMemcpy(h_wrong_current_particle_attributes,d_wrong_current_particle_attributes,
//			 sizeof(double)*PARTICLE_ATTRIBUTES*MAX_particles_per_cell,cudaMemcpyDeviceToHost);
//#endif

	int err = getLastError();
	if(err != 0)
		{
			 printf(" copyCellFromDevice enter %d %s \n ",err,getErrorString(err));
			 exit(0);
		}

//    is the device array of Cell pointers being really copied to Host?

	ThreadSynchronize();

	err = MemoryCopy(h_copy_of_d_src,d_src,sizeof(GPUCell<Particle>),DEVICE_TO_HOST);
	if(err != 0)
	{
		 printf(" copyCellFromDevice1 %d %s \n ",err,getErrorString(err));
		 exit(0);
	}
	//printf("Cuda error: %d: %s.\n", code,getErrorString((cudaError_t) code));
    if(h_copy_of_d_src->number_of_particles < 0 || h_copy_of_d_src->number_of_particles > MAX_particles_per_cell)
    {
    	int qq = 0;
    }
	//code = cudaMemcpy(h_dst,&h_copy_of_d_src,sizeof(GPUCell<Particle>),cudaMemcpyHostToHost);
#ifdef COPY_CELLS_MEMORY_PRINTS
	printf("step %d %s number of particles %5d %3d %3d %d \n",nt,where,h_copy_of_d_src->i,h_copy_of_d_src->l,h_copy_of_d_src->k,h_copy_of_d_src->number_of_particles);
#endif
//	if(code != cudaSuccess)
//	{
//		 printf(" copyCellFromDevice2 %d \n ",code);
//		 exit(0);
//	}


//	cudaPrintfInit();
//	h_dst->doubParticleArray = (double*)malloc(sizeof(Particle)*MAX_particles_per_cell);

	h_dst->number_of_particles = h_copy_of_d_src->number_of_particles;

	code = MemoryCopy(h_dst->doubParticleArray,h_copy_of_d_src->doubParticleArray,
			   sizeof(Particle)*MAX_particles_per_cell,DEVICE_TO_HOST);
	if(code != 0)
	{
		 printf(" copyCellFromDevice3 %d %s \n ",code,getErrorString(code));
		 exit(0);
	}

	//printf("i %d l %d k %d q_m %15.5e \n",h_dst->i,h_dst->k,h_dst->l,h_dst->ParticleArrayRead(50,1));
//**************************************************************************************************
//	          cudaPrintfInit();
//
//	          testKernelBefore<<<1,1>>>(h_copy_of_d_src->doubParticleArray,50,1);
//	          cudaPrintfDisplay(stdout, true);
//	          cudaPrintfEnd();
//**************************************************************************************************


	//h_dst->Jx = new CellDouble;
	//compareArrayHostToDevice((double *)h_dst->Jx,(double *)(h_copy_of_d_src.Jx),sizeof(CellDouble),"TEST");

	code = MemoryCopy(h_dst->Jx,h_copy_of_d_src->Jx,sizeof(CellDouble),DEVICE_TO_HOST);
	if(code != 0)
	{
		 printf(" copyCellFromDevice4 %d \n ",code);
		 exit(0);
	}



	//h_dst->Jy = new CellDouble;
	code = MemoryCopy(h_dst->Jy,h_copy_of_d_src->Jy,sizeof(CellDouble),DEVICE_TO_HOST);
	if(code != 0)
	{
		 printf(" copyCellFromDevice5 %d \n ",code);
		 exit(0);
	}

	//h_dst->Jz = new CellDouble;
	code = MemoryCopy(h_dst->Jz,h_copy_of_d_src->Jz,sizeof(CellDouble),DEVICE_TO_HOST);
	if(code != 0)
	{
		 printf(" copyCellFromDevice6 %d \n ",code);
		 exit(0);
	}

	//h_dst->Ex = new CellDouble;
	code = MemoryCopy(h_dst->Ex,h_copy_of_d_src->Ex,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Ey = new CellDouble;
	code = MemoryCopy(h_dst->Ey,h_copy_of_d_src->Ey,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Ez = new CellDouble;
	code = MemoryCopy(h_dst->Ez,h_copy_of_d_src->Ez,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Hx = new CellDouble;
	code = MemoryCopy(h_dst->Hx,h_copy_of_d_src->Hx,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Hy = new CellDouble;
	code = MemoryCopy(h_dst->Hy,h_copy_of_d_src->Hy,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Hz = new CellDouble;
	code = MemoryCopy(h_dst->Hz,h_copy_of_d_src->Hz,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Rho = new CellDouble;
	code = MemoryCopy(h_dst->Rho,h_copy_of_d_src->Rho,sizeof(CellDouble),DEVICE_TO_HOST);
	if(code != 0)
	{
		 printf(" copyCellFromDevice10 %d \n ",code);
		 exit(0);
	}


	//memcpy((unsigned char *)dst.Jx,(unsigned char *)src.Jx,sizeof(CellDouble));
	//printf("i %d l %d k %d q_m %15.5e \n",h_dst->i,h_dst->k,h_dst->l,h_dst->ParticleArrayRead(50,1));

    //cudaMalloc(&d_dst,sizeof(Cell<Particle>));


    //cudaMemcpy(h_dst,d_src,sizeof(Cell<Particle>),cudaMemcpyDeviceToHost);
   // cudaMemcpy(h_ctrl,d_dst,sizeof(Cell<Particle>),cudaMemcpyDeviceToHost);

//     return h_dst;
}

GPUCell<Particle>* allocateCopyCellFromDevice()
{
	GPUCell<Particle> *h_dst,*h_copy_of_d_src;
	//static int first = 1;
	int code;


	   h_dst = new GPUCell<Particle>;
	//h_ctrl = new GPUCell<Particle>;

//	cudaPrintfInit();
	h_dst->doubParticleArray = (double*)malloc(sizeof(Particle)*MAX_particles_per_cell);

	h_dst->Jx = new CellDouble;
	h_dst->Jy = new CellDouble;
	h_dst->Jz = new CellDouble;

	h_dst->Ex = new CellDouble;
	h_dst->Ey = new CellDouble;
	h_dst->Ez = new CellDouble;
	h_dst->Hx = new CellDouble;
	h_dst->Hy = new CellDouble;
	h_dst->Hz = new CellDouble;
	h_dst->Rho = new CellDouble;

    return h_dst;
}

void freeCopyCellFromDevice(GPUCell<Particle> *h_dst)
{
//	GPUCell<Particle> *h_dst,*h_copy_of_d_src;
	//static int first = 1;
	int code;


//	   h_dst = new GPUCell<Particle>;
	//h_ctrl = new GPUCell<Particle>;

//	cudaPrintfInit();
	free(h_dst->doubParticleArray);// = (double*)malloc(sizeof(Particle)*MAX_particles_per_cell);

	delete (h_dst->Jx);// = new CellDouble;
	delete (h_dst->Jy);// = new CellDouble;
	delete (h_dst->Jz);// = new CellDouble;

	delete (h_dst->Ex);// = new CellDouble;
	delete (h_dst->Ey);// = new CellDouble;
	delete (h_dst->Ez);// = new CellDouble;
	delete (h_dst->Hx);// = new CellDouble;
	delete (h_dst->Hy);// = new CellDouble;
	delete (h_dst->Hz);// = new CellDouble;
	delete (h_dst->Rho);// = new CellDouble;

	delete h_dst;
//    return h_dst;
}


#ifdef __CUDACC__
__host__
#endif
void printFileCellParticles(FILE *f,GPUCell<Particle> *h_copy_of_d_src)
{
	Particle p;
	int sorts[3] = {0,0,0};

//	cudaError_t code = cudaMemcpy(this,h_copy_of_d_src,sizeof(GPUCell<Particle>),cudaMemcpyDeviceToHost);
//	Cell<Particle>::doubParticleArray = (double*)malloc(sizeof(Particle)*MAX_particles_per_cell);

	//num = &(Cell<Particle>::number_of_particles);
//	cudaError_t code = cudaMemcpy(&num,
//			                      &(h_copy_of_d_src->number_of_particles),
//					              sizeof(int),cudaMemcpyDeviceToHost);

//	cudaError_t code = cudaMemcpy(Cell<Particle>::doubParticleArray,h_copy_of_d_src->doubParticleArray,
//				                  sizeof(Particle)*MAX_particles_per_cell,cudaMemcpyDeviceToHost);
    fprintf(f,"(%3d,%3d,%3d) ========================================================================================== \n",this->i,this->l,this->k);
	for(int i = 0;i < h_copy_of_d_src->number_of_particles;i++)
	{
		h_copy_of_d_src->readParticleFromSurfaceDevice(i,&p);
		fprintf(f,"(%3d,%3d,%3d) i %3d sort %d FN %10d c  pointInCell %d %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e \n",
						this->i,this->l,this->k,i,(int)p.sort,p.fortran_number,Cell<Particle>::isPointInCell(p.GetX()),
				        p.x,p.y,p.z,p.pu ,p.pv,p.pw);

		sorts[(int)p.sort] += 1;
	}
	fprintf(f,"ions %3d electrons %3d beam %3d \n",sorts[0],sorts[1],sorts[2]);
}

double compareToCell(Cell<Particle> & d_src)
{
	//copyCellFromDevice(&d_src);
	//dbgPrintGPUParticleAttribute(&d_src,50,1,"COMPARE_TO_CELL" );
	return Cell<Particle>::compareToCell(d_src);
}

void addParticlesToCellOnDevice(GPUCell<Particle>* d_src,GPUCell<Particle>* h_dst,char *where,int nt)
{
	static GPUCell<Particle> *h_copy_of_d_src;
	static int first = 1;
	int code;


	if(first == 1)
	{
	   first = 0;
	   h_copy_of_d_src = new GPUCell<Particle>;
	   h_copy_of_d_src->Init();

	}

	int err = getLastError();
	if(err != 0)
		{
			 printf(" copyCellFromDevice enter %d %s \n ",err,getErrorString(err));
			 exit(0);
		}


	ThreadSynchronize();

	err = MemoryCopy(h_copy_of_d_src,d_src,sizeof(GPUCell<Particle>),DEVICE_TO_HOST);
	if(err != 0)
	{
		 printf(" copyCellFromDevice1 %d %s \n ",err,getErrorString(err));
		 exit(0);
	}
	//printf("Cuda error: %d: %s.\n", code,getErrorString((cudaError_t) code));
    if(h_copy_of_d_src->number_of_particles < 0 || h_copy_of_d_src->number_of_particles > MAX_particles_per_cell)
    {
    	int qq = 0;
    }
	//code = MemoryCopy(h_dst,&h_copy_of_d_src,sizeof(GPUCell<Particle,dims>),cudaMemcpyHostToHost);
#ifdef COPY_CELLS_MEMORY_PRINTS
	printf("step %d %s number of particles %5d %3d %3d %d \n",nt,where,h_copy_of_d_src->i,h_copy_of_d_src->l,h_copy_of_d_src->k,h_copy_of_d_src->number_of_particles);
#endif



	h_dst->number_of_particles = h_copy_of_d_src->number_of_particles;

	code = MemoryCopy(h_dst->doubParticleArray,h_copy_of_d_src->doubParticleArray,
			   sizeof(Particle)*MAX_particles_per_cell,DEVICE_TO_HOST);
	if(code != 0)
	{
		 printf(" copyCellFromDevice3 %d %s \n ",code,getErrorString(code));
		 exit(0);
	}

	//printf("i %d l %d k %d q_m %15.5e \n",h_dst->i,h_dst->k,h_dst->l,h_dst->ParticleArrayRead(50,1));


	//h_dst->Jx = new CellDouble;
	//compareArrayHostToDevice((double *)h_dst->Jx,(double *)(h_copy_of_d_src.Jx),sizeof(CellDouble),"TEST");

	code = MemoryCopy(h_dst->Jx,h_copy_of_d_src->Jx,sizeof(CellDouble),DEVICE_TO_HOST);
	if(code != 0)
	{
		 printf(" copyCellFromDevice4 %d \n ",code);
		 exit(0);
	}



	//h_dst->Jy = new CellDouble;
	code = MemoryCopy(h_dst->Jy,h_copy_of_d_src->Jy,sizeof(CellDouble),DEVICE_TO_HOST);
	if(code != 0)
	{
		 printf(" copyCellFromDevice5 %d \n ",code);
		 exit(0);
	}

	//h_dst->Jz = new CellDouble;
	code = MemoryCopy(h_dst->Jz,h_copy_of_d_src->Jz,sizeof(CellDouble),DEVICE_TO_HOST);
	if(code != 0)
	{
		 printf(" copyCellFromDevice6 %d \n ",code);
		 exit(0);
	}

	//h_dst->Ex = new CellDouble;
	code = MemoryCopy(h_dst->Ex,h_copy_of_d_src->Ex,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Ey = new CellDouble;
	code = MemoryCopy(h_dst->Ey,h_copy_of_d_src->Ey,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Ez = new CellDouble;
	code = MemoryCopy(h_dst->Ez,h_copy_of_d_src->Ez,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Hx = new CellDouble;
	code = MemoryCopy(h_dst->Hx,h_copy_of_d_src->Hx,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Hy = new CellDouble;
	code = MemoryCopy(h_dst->Hy,h_copy_of_d_src->Hy,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Hz = new CellDouble;
	code = MemoryCopy(h_dst->Hz,h_copy_of_d_src->Hz,sizeof(CellDouble),DEVICE_TO_HOST);
	//h_dst->Rho = new CellDouble;
	code = MemoryCopy(h_dst->Rho,h_copy_of_d_src->Rho,sizeof(CellDouble),DEVICE_TO_HOST);
	if(code != 0)
	{
		 printf(" copyCellFromDevice10 %d \n ",code);
		 exit(0);
	}


	//memcpy((unsigned char *)dst.Jx,(unsigned char *)src.Jx,sizeof(CellDouble));
	//printf("i %d l %d k %d q_m %15.5e \n",h_dst->i,h_dst->k,h_dst->l,h_dst->ParticleArrayRead(50,1));

    //cudaMalloc(&d_dst,sizeof(Cell<Particle,dims>));


    //cudaMemcpy(h_dst,d_src,sizeof(Cell<Particle,dims>),cudaMemcpyDeviceToHost);
   // cudaMemcpy(h_ctrl,d_dst,sizeof(Cell<Particle,dims>),cudaMemcpyDeviceToHost);

}





};




#endif /* GPUCELL_H_ */
