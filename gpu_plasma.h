/*
 * gpu_plasma.h
 *
 *  Created on: Aug 21, 2013
 *      Author: snytav
 */
#include "cuPrintf.cu"

#ifndef GPU_PLASMA_H_
#define GPU_PLASMA_H_

#include<stdlib.h>
#include<stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <string>

//#include <unistd.h>
//#include <stdio.h>
#include <errno.h>

#ifdef __CUDACC__
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>
#endif


#include <time.h>

//#ifdef __OMP__
#include <omp.h>
//#endif

#ifdef __CUDACC__
#include <cuda.h>
#endif

#include "archAPI.h"
#include "rnd.h"
#include "plasma.h"
#include "gpucell.h"
#include "mpi_shortcut.h"

#include "service_functions.h"

#include <sys/resource.h>
#include <stdint.h>

#include <sys/sysinfo.h>
#include <sys/time.h>


#include "init.h"
#include "diagnose.h"

#include<string>
#include <iostream>

#include "particle_target.h"

#include "params.h"

#include "memory_control.h"

#include "kernels.cu"



using namespace std;






#define FORTRAN_ORDER






template <template <class Particle> class Cell >
class GPUPlasma
{
public:
   GPUCell<Particle> **h_CellArray,**d_CellArray;
   GPUCell<Particle> **cp;
   thrust::device_vector<Cell<Particle> > *d_AllCells;
   double *d_Ex,*d_Ey,*d_Ez,*d_Hx,*d_Hy,*d_Hz,*d_Jx,*d_Jy,*d_Jz,*d_Rho,*d_npJx,*d_npJy,*d_npJz;
   double *d_Qx,*d_Qy,*d_Qz;
   double *dbg_x,*dbg_y,*dbg_z,*dbg_px,*dbg_py,*dbg_pz;
   int total_particles;

   int h_controlParticleNumberArray[4000];

   int  jx_wrong_points_number;
   int3 *jx_wrong_points,*d_jx_wrong_points;

//#ifdef ATTRIBUTES_CHECK
   double *ctrlParticles,*d_ctrlParticles,*check_ctrlParticles;
//#endif

   int jmp,size_ctrlParticles;
   double ami,amb,amf;
   int real_number_of_particle[3];
   FILE *f_prec_report;



   int CPU_field;


int InitializeGPU()
{
    InitGPUParticles();
    InitGPUFields(&d_Ex,&d_Ey,&d_Ez,
    	          &d_Hx,&d_Hy,&d_Hz,
    		      &d_Jx,&d_Jy,&d_Jz,
    		      &d_npJx,&d_npJy,&d_npJz,
                  &d_Qx,&d_Qy,&d_Qz
            );

    setPrintfLimit();

    int err = cudaSetDevice(0);

    printf("InitializeGPU error %d \n",err);

    return 0;
}

void Initialize()
{
	InitializeCPU();
	copyCellsWithParticlesToGPU();
	InitializeGPU();
}

void cudaMalloc3D(double **X,double **Y,double**Z,int nx,int ny,int nz)
{
	cudaMalloc(X,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc(Y,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc(Z,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

}

void InitGPUFields(
		double **d_Ex,double **d_Ey,double **d_Ez,
		double **d_Hx,double **d_Hy,double **d_Hz,
		double **d_Jx,double **d_Jy,double **d_Jz,
		double **d_npJx,double **d_npJy,double **d_npJz,
        double **d_Qx,double **d_Qy,double **d_Qz
        )
{
	cudaMalloc3D(d_Ex,d_Ey,d_Ez,Nx,Ny,Nz);
	cudaMalloc3D(d_Hx,d_Hy,d_Hz,Nx,Ny,Nz);
	cudaMalloc3D(d_Jx,d_Jy,d_Jz,Nx,Ny,Nz);
	cudaMalloc3D(d_npJx,d_npJy,d_npJz,Nx,Ny,Nz);
	cudaMalloc3D(d_Qx,d_Qy,d_Qz,Nx,Ny,Nz);



    copyFieldsToGPU();
}

void copyFieldsToGPU()
{
	cudaError_t err;

    err = cudaMemcpy(d_Ex,Ex,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
    	printf("1copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
    	exit(0);
    }
    err = cudaMemcpy(d_Ey,Ey,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
     	printf("2copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
    	exit(0);
    }

    err = cudaMemcpy(d_Ez,Ez,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("3copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }

    err = cudaMemcpy(d_Hx,Hx,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("4copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }
    err = cudaMemcpy(d_Hy,Hy,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("5copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }
    err = cudaMemcpy(d_Hz,Hz,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("6copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }

    err = cudaMemcpy(d_Jx,Jx,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("7copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }
    err = cudaMemcpy(d_Jy,Jy,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("8copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }

    err = cudaMemcpy(d_Jz,Jz,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("9copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }

    err = cudaMemcpy(d_npJx,npJx,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("10copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }

    err = cudaMemcpy(d_npJy,npJy,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("11copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }

    err = cudaMemcpy(d_npJz,npJz,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("12copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }

    err = cudaMemcpy(d_Qx,Qx,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("13copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }

    err = cudaMemcpy(d_Qy,Qy,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("14copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }

    err = cudaMemcpy(d_Qz,Qz,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("15copyFieldsToGPU err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }
}

void InitGPUParticles()
 //   :InitParticles(fname,vp)
{
	int size;
	GPUCell<Particle> *d_c,*h_ctrl;
	GPUCell<Particle> *n;
	GPUCell<Particle> *h_copy,*h_c;
	double t;
	dim3 dimGrid(Nx+2,Ny+2,Nz+2),dimBlockOne(1,1,1);

	 readControlFile(START_STEP_NUMBER);

	size = (*AllCells).size();

	 size_t m_free,m_total;

	h_ctrl = new Cell<Particle>;
	n = new Cell<Particle>;

    h_CellArray = (Cell<Particle> **)malloc(size*sizeof(Cell<Particle> *));
    cudaError_t err = cudaMalloc(&d_CellArray,size*sizeof(Cell<Particle> *));

//    h_controlParticleNumberArray = (int*)malloc(size*sizeof(int));

    printf("%s : size = %d\n", __FILE__, size);
    for(int i = 0;i < size;i++)
    {
        //printf("GPU cell %d begins******************************************************\n",i);
    	GPUCell<Particle> c;
    	c = (*AllCells)[i];

    	h_controlParticleNumberArray[i] = c.number_of_particles;
    	/////////////////////////////////////////
    	*n = c;
#ifdef ATTRIBUTES_CHECK
    	c.SetControlSystem(jmp,d_ctrlParticles);
#endif

    	//t = c.compareToCell(*n);

       // puts("COMPARE------------------------------");
    	//printf("%d: %d\n", i, c.busyParticleArray);
        d_c = c.copyCellToDevice();
        cudaMemGetInfo(&m_free,&m_total);
        double mfree,mtot;
        mfree = m_free;
        mtot  = m_total;
#ifdef COPY_CELL_PRINTS
        printf("cell %10d Device cell array allocated error %d %s memory: free %10.2f total %10.2f\n",i,err,cudaGetErrorString(err),
        		                                                mfree/1024/1024/1024,mtot/1024/1024/1024);
        puts("");

	  dbgPrintGPUParticleAttribute(d_c,50,1," CO2DEV " );
	  puts("COPY----------------------------------");
#endif
//	    h_copy = new GPUCell<Particle>;
//	    cudaError_t err = cudaMemcpy(h_c,d_c,sizeof(GPUCell<Particle>),cudaMemcpyDeviceToHost);
//	    cudaMalloc(&d_c,sizeof(GPUCell<Particle>));

//        d_c incorrect!!!!
//        c.copyCellFromDevice(d_c,h_copy);
//        err = cudaMemcpy(h_copy,d_c,sizeof(GPUCell<Particle>),cudaMemcpyDeviceToHost);


//	    dbgPrintGPUParticleAttribute(d_c,50,1," COPY " );
//
//        h_c = new GPUCell<Particle>;
//        cudaMemcpy(h_c,d_c,sizeof(GPUCell<Particle>),cudaMemcpyDeviceToHost);
//        c.compareArrayHostToDevice((double *)c.Jx,(double *)h_c->Jx,sizeof(CellDouble),"Jx");
//        c.compareArrayHostToDevice((double *)c.Jy,(double *)h_c->Jy,sizeof(CellDouble),"Jy");
//        t = c.compareToCell(*h_copy);

#ifdef PARTICLE_PRINTS

        if(t < 1.0)
        {
        	t = c.compareToCell(*h_copy);
        }
#endif
        ////////////////////////////////////////.
        h_CellArray[i] = d_c;
        cudaMemcpy(h_ctrl,d_c,sizeof(Cell<Particle>),cudaMemcpyDeviceToHost);
#ifdef InitGPUParticles_PRINTS
	    dbgPrintGPUParticleAttribute(d_c,50,1," CPY " );

       cudaPrintfInit();

        testKernel<<<1,1>>>(h_ctrl->d_ctrlParticles,h_ctrl->jmp);
        cudaPrintfDisplay(stdout, true);
        cudaPrintfEnd();

        printf("i %d l %d k n %d %d %e src %e num %d\n",h_ctrl->i,h_ctrl->l,h_ctrl->k,i,
        		c.ParticleArrayRead(0,7),c.number_of_particles
        		);
	printf("GPU cell %d ended ******************************************************\n",i);
#endif
    }

    //cudaError_t err;
    err = cudaMemcpy(d_CellArray,h_CellArray,size*sizeof(Cell<Particle> *),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("bGPU_WriteControlSystem err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }

//	d_AllCells = new thrust::device_vector<Cell<Particle> >(size);

//	*d_AllCells = (*AllCells);
#ifdef ATTRIBUTES_CHECK
    GPU_WriteControlSystem<<<dimGrid, dimBlockOne,16000>>>(d_CellArray);
#endif
	size = 0;

}


void copyCells(char *where,int nt)
{
	static int first = 1;
	size_t m_free,m_total;
	int size = (*AllCells).size();
	//double accum = 0.0;
	struct sysinfo info;
	unsigned long c1,c2;

    if(first == 1)
    {
    	cp = (Cell<Particle> **)malloc(size*sizeof(Cell<Particle> *));
    }

	unsigned long m1,m2,delta,accum;
	memory_monitor("beforeCopyCells",nt);

//	err = cudaMemGetInfo(&m_free,&m_total);
//	freemem=get_meminfo();
//	sysinfo(&info);


	for(int i = 0;i < size;i++)
	{
		if(i == 141)
		{
			int qq = 0;
		}
		cudaError_t err = cudaMemGetInfo(&m_free,&m_total);
		//double freemem=get_meminfo();
		sysinfo(&info);
		m1 = info.freeram;
	 	GPUCell<Particle> c,*d_c,*z0;
	 //	m1 = get_meminfo();
	 	//c = new GPUCell<Particle>;
	 	z0 = h_CellArray[i];
	 	if(first == 1)
	 	{
	       d_c = c.allocateCopyCellFromDevice();
     	   cp[i] = d_c;
	 	}
	 	else
	 	{
	 	   d_c = cp[i];
	 	}
	    c.copyCellFromDevice(z0,d_c,where,nt);
	   // m2 = get_meminfo();
		//sysinfo(&info);
		m2 = info.freeram;

	    delta = m1-m2;
        accum += delta;

	//    printf("cell allocated delta %u size %d free CPU memory %u accum %u \n",delta,sizeof(GPUCell<Particle>),m2,accum);

	}

	if(first == 1)
	{
		first = 0;
	}

	memory_monitor("afterCopyCells",nt);


}

void freeCellCopies(Cell<Particle> **cp)
{
	int size = (*AllCells).size();

	for(int i = 0;i < size;i++)
	{
		GPUCell<Particle> *d_c,c;

		d_c = cp[i];

		c.freeCopyCellFromDevice(d_c);

	}
	free(cp);
}

double compareCells(int nt)
{
	double t = 0.0,t1;
	struct sysinfo info;
//	Cell<Particle> **cp;

	int size = (*AllCells).size();



	//copyCells(cp);
	checkParticleNumbers(cp,-1);
	memory_monitor("compareCells",nt);

	//h_ctrl = new Cell<Particle>;

	for(int i = 0;i < size;i++)
	{
	 	Cell<Particle> c = (*AllCells)[i];
	    t1 = c.compareToCell(*(cp[i]));

	    Particle p;
	    //int j;

	    c.readParticleFromSurfaceDevice(0,&p);

	 //   j = p.fortran_number;

	    if(t1 < 1.0)
	    {
	       	t1 = c.compareToCell(*(cp[i]));
	    }
	    if(isNan(t1))
	    {
	    	t1 = c.compareToCell(*(cp[i]));
	    }

	    t += t1;

	}
	memory_monitor("compareCells2",nt);

	return t/size;
}

double checkGPUArray(double *a,double *d_a,char *name,char *where,int nt)
{
	 static double *t;
	 static int f1 = 1;
	 char fname[1000];
	 double res;
	 FILE *f;
#ifndef CHECK_ARRAY_OUTPUT
   return 0.0;
#endif

	 sprintf(fname,"diff_%s_at_%s_nt%08.dat",name,where,nt);


	 if(f1 == 1)
	 {
		 t = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
		 f1 = 0;
	 }
	 cudaError_t err;
	 err = cudaMemcpy(t,d_a,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyDeviceToHost);
	 if(err != cudaSuccess)
	         {
	          	printf("bCheckArray err %d %s \n",err,cudaGetErrorString(err));
	        	exit(0);
	         }

	 if((f = fopen(fname,"wt")) != NULL)
	 {
		 res = CheckArray(a,t,f);
		 fclose(f);
	 }

	 return res;
}

double checkGPUArray(double *a,double *d_a)
{
	 static double *t;
	 static int f = 1;

	 if(f == 1)
	 {
		 t = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
		 f = 0;
	 }
	 cudaError_t err;
	 err = cudaMemcpy(t,d_a,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyDeviceToHost);
	 if(err != cudaSuccess)
	         {
	          	printf("bCheckArray err %d %s \n",err,cudaGetErrorString(err));
	        	exit(0);
	         }

	 return CheckArray(a,t);

}


double compareFields(){return 1.0;}

double compareCPUtoGPU()
{
//       return (compareFields() +
//
//       compareCells())/2.0;
	return 1.0;
}

void StepAllCells()
{
	dim3 dimGrid(Nx+2,Ny+2,Nz+2),
			dimBlock(MAX_particles_per_cell/2,1,1);

//	cudaPrintfInit();
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
   // GPU_StepAllCells<<<dimGrid, dimBlock,16000>>>(d_CellArray);
    cudaError_t err;
    err = cudaGetLastError();
    printf("Err: %d %s\n", err, cudaGetErrorString(err));
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("Err: %d %s\n", err, cudaGetErrorString(err));
	//cudaPrintfDisplay(stdout, true);

	//cudaPrintfEnd();
//	exit(0);
}


public:
//	GPUPlasma(){}


//	 void Initialize(){
//		 Initialize();
//	 }
//

void virtual emeGPUIterate(int3 s,int3 f,
			double *E,double *H1, double *H2,
			double *J,double c1,double c2, double tau,
			int3 d1,int3 d2)
{
	dim3 dimGrid(f.x-s.x+1,1,1),dimBlock(1,f.y-s.y+1,f.z-s.z+1);

    GPU_eme<<<dimGrid,dimBlock>>>(d_CellArray,s,
    		                            E,H1,H2,
    					    	  		J,c1,c2,tau,
    					    	  		d1,d2
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

int virtual ElectricFieldTrace(
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

int checkFields_beforeMagneticStageOne(double *t_Ex,double *t_Ey,double *t_Ez,
		                               double *t_Hx,double *t_Hy,double *t_Hz,
		                               double *t_Qx,double *t_Qy,double *t_Qz,
		                               double *t_check,int nt)
{

	 memory_monitor("beforeComputeField_FirstHalfStep",nt);

	         t_check[0] = checkControlMatrix("emh1",nt,"qx",t_Qx);
			 t_check[1] = checkControlMatrix("emh1",nt,"qy",t_Qy);
			 t_check[2] = checkControlMatrix("emh1",nt,"qz",t_Qz);

			 t_check[3] = checkControlMatrix("emh1",nt,"ex",t_Ex);
			 t_check[4] = checkControlMatrix("emh1",nt,"ey",t_Ey);
			 t_check[5] = checkControlMatrix("emh1",nt,"ez",t_Ez);

			 t_check[6] = checkControlMatrix("emh1",nt,"hx",t_Hx);
			 t_check[7] = checkControlMatrix("emh1",nt,"hy",t_Hy);
			 t_check[8] = checkControlMatrix("emh1",nt,"hz",t_Hz);
	return 0;
}

int checkFields_afterMagneticStageOne(double *t_Hx,double *t_Hy,double *t_Hz,
		                              double *t_Qx,double *t_Qy,double *t_Qz,
		                              double *t_check,int nt)
{
	         t_check[9] = checkControlMatrix("emh1",nt,"qx",t_Qx);
			 t_check[10] = checkControlMatrix("emh1",nt,"qy",t_Qy);
			 t_check[11] = checkControlMatrix("emh1",nt,"qz",t_Qz);

			 t_check[12] = checkControlMatrix("emh1",nt,"hx",t_Hx);
			 t_check[13] = checkControlMatrix("emh1",nt,"hy",t_Hy);
			 t_check[14] = checkControlMatrix("emh1",nt,"hz",t_Hz);


			 CPU_field = 1;



			 checkControlPoint(50,nt,0);
			 memory_monitor("afterComputeField_FirstHalfStep",nt);

	return 0;
}

void  ComputeField_FirstHalfStep(
//		   double *locEx,double *locEy,double *locEz,
		   int nt//,
//		   double *locHx,double *locHy,double *locHz,
//		   double *loc_npJx,double *loc_npJy,double *loc_npJz,
//		   double *locQx,double *locQy,double *locQz
		   )
{
	 double t_check[15];

     checkFields_beforeMagneticStageOne(d_Ex,d_Ey,d_Ez,
		 		                               d_Hx,d_Hy,d_Hz,
		 		                               d_Qx,d_Qy,d_Qz,
		 		                               t_check,nt);


	 MagneticStageOne(d_Qx,d_Qy,d_Qz,d_Hx,d_Hy,d_Hz,d_Ex,d_Ey,d_Ez);

	 checkFields_afterMagneticStageOne(d_Hx,d_Hy,d_Hz,
		 		                           d_Qx,d_Qy,d_Qz,
		 		                           t_check,nt);
	 AssignCellsToArraysGPU();

}

virtual void ComputeField_SecondHalfStep(int nt)
{

     SetPeriodicCurrents(nt);



     MagneticFieldStageTwo(d_Hx,d_Hy,d_Hz,nt,d_Qx,d_Qy,d_Qz);




     ElectricFieldEvaluate(d_Ex,d_Ey,d_Ez,nt,d_Hx,d_Hy,d_Hz,d_Jx,d_Jy,d_Jz);


}

void ElectricFieldComponentEvaluate(
		  double *E,double *H1,double *H2,double *J,
		  int dir,
		  double c1,double c2,double tau,
		  int dir_1,int start1_1,int end1_1,int start2_1,int end2_1,int N_1,
		  int dir_2,int start1_2,int end1_2,int start2_2,int end2_2,int N_2
		  )

{
     ElectricFieldTrace(E,H1,H2,J,dir,c1,c2,tau);

     PeriodicBoundaries(E, dir_1,start1_1,end1_1,start2_1,end2_1,N_1);
     PeriodicBoundaries(E, dir_2,start1_2,end1_2,start2_2,end2_2,N_2);
}

void ElectricFieldEvaluate(double *locEx,double *locEy,double *locEz,
		   int nt,
		   double *locHx,double *locHy,double *locHz,
		   double *loc_npJx,double *loc_npJy,double *loc_npJz)
{
	 CPU_field = 0;
      double3 c1 = getMagneticFieldTimeMeshFactors();

      ElectricFieldComponentEvaluate(
    		  locEx,locHz,locHy,loc_npJx,
    		  0,c1.y,c1.z,tau,
              1,0,Nx,1,Nz,Ny,
              2,0,Nx,0,Ny+1,Nz);

      ElectricFieldComponentEvaluate(
    		  locEy,locHx,locHz,loc_npJy,
    		  1,c1.z,c1.x,tau,
              0,0,Ny,1,Nz,Nx,
              2,0,Nx+1,0,Ny,Nz);

      SinglePeriodicBoundary(locEy,1,0,Nx+1,0,Nz+1,Ny);

      ElectricFieldComponentEvaluate(
    		  locEz,locHy,locHx,loc_npJz,
    		  2,c1.x,c1.y,tau,
              0,1,Ny,0,Nz,Nx,
              1,0,Nx+1,0,Nz,Ny);

         checkControlPoint(600,nt,0);

         memory_monitor("after_ComputeField_SecondHalfStep",nt);
}

double3 getMagneticFieldTimeMeshFactors()
{
    Cell<Particle> c = (*AllCells)[0];
	double hx = c.get_hx(),hy = c.get_hy(),hz = c.get_hz();
	double3 d;
    d.x = tau/(hx);
    d.y = tau/(hy);
    d.z = tau/hz;

	return d;
}

virtual void MagneticStageOne(
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

virtual void MagneticFieldStageTwo(double *Hx,double *Hy,double *Hz,
		            int nt,
		            double *Qx,double *Qy,double *Qz)
{
    Cell<Particle> c = (*AllCells)[0];

    SimpleMagneticFieldTrace(c,Qx,Hx,Nx+1,Ny,Nz);
    SimpleMagneticFieldTrace(c,Qy,Hy,Nx,Ny+1,Nz);
    SimpleMagneticFieldTrace(c,Qz,Hz,Nx,Ny,Nz+1);

    checkControlPoint(500,nt,0);
}


int PushParticles(int nt)
{
	double mass = -1.0/1836.0,q_mass = -1.0;

	memory_monitor("before_CellOrder_StepAllCells",nt);

    CellOrder_StepAllCells(nt,	mass,q_mass,1);

	memory_monitor("after_CellOrder_StepAllCells",nt);

	//checkParticleAttributes(nt);

	checkControlPoint(270,nt,1);

	return 0;
}


int readStartPoint(int nt)
{
	char fn[100];

	 if(nt == START_STEP_NUMBER)
	 {
		 readControlPoint(NULL,fn,0,nt,0,1,Ex,Ey,Ez,Hx,Hy,Hz,Jx,Jy,Jz,Qx,Qy,Qz,
							   dbg_x,dbg_y,dbg_z,dbg_px,dbg_py,dbg_pz
			 );
		copyFieldsToGPU();
	 }

     checkControlPoint(0,nt,1);

	 return 0;
}



	void Step(int nt)
	 {
		ComputeField_FirstHalfStep(nt);

		PushParticles(nt);

		ComputeField_SecondHalfStep(nt);

		 Diagnose(nt);

	 }
	virtual double getElectricEnergy()
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

		GPU_getCellEnergy<<<dimGrid, dimBlockOne,16000>>>(d_CellArray,d_ee,d_Ex,d_Ey,d_Ez);

        cudaMemcpy(&ee,d_ee,sizeof(double),cudaMemcpyDeviceToHost);

        return ee;

	}
	void Diagnose(int nt)
	{
		double ee;
		static FILE *f;
		static int first = 1;

		if(first == 1)
		{
			f = fopen("energy.dat","wt");
			first = 0;
		}
		else
		{
			f = fopen("energy.dat","at");

		}



        ee = getElectricEnergy();
       // sumMPIenergy(&ee);

        //if(getRank() == 0)
        	fprintf(f,"%10d %25.15e \n",nt,ee);

        fclose(f);
		//puts("GPU-Plasma");

	}
	virtual ~GPUPlasma(){
		//~Plasma<Cell>();
		}

	  int Nx,Ny,Nz;

	  int n_per_cell;

	  int meh;

	  int magf;

	  double ion_q_m,tau;

	  double Lx,Ly,Lz;

	  double ni;

	  double *Qx,*Qy,*Qz,*dbg_Qx,*dbg_Qy,*dbg_Qz;

	  double *Ex,*Ey,*Ez,*Hx,*Hy,*Hz,*Jx,*Jy,*Jz,*Rho,*npJx,*npJy,*npJz;
	  double *dbgEx,*dbgEy,*dbgEz,*dbgHx,*dbgHy,*dbgHz,*dbgJx,*dbgJy,*dbgJz;
	  double *dbgEx0,*dbgEy0,*dbgEz0;
	  double *npEx,*npEy,*npEz;



	  thrust::host_vector<Cell<Particle> > *AllCells;

	  int getBoundaryLimit(int dir){return ((dir == 0)*Nx  + (dir == 1)*Ny + (dir == 2)*Nz + 2);}

	  virtual void Alloc()
	  {

		  AllCells = new thrust::host_vector<Cell<Particle> >;

	     Ex  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Ey  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Ez  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Hx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Hy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Hz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Jx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Jy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Jz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Rho = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	     npJx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     npJy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     npJz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	     npEx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     npEy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     npEz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	     Qx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Qy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Qz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	#ifdef DEBUG_PLASMA

	     dbgEx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgEy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgEz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgEx0  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgEy0  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgEz0  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	     dbgHx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgHy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgHz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgJx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgJy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgJz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	     dbg_Qx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbg_Qy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbg_Qz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	#endif
	  }

	  virtual void InitFields()
	  {
	     for(int i = 0;i < (Nx+2)*(Ny+2)*(Nz+2);i++)
	     {
	         Ex[i] = 0.0;
	         Ey[i] = 0.0;
	         Ez[i] = 0.0;
	         Hx[i] = 0.0;
	         Hy[i] = 0.0;
	         Hz[i] = 0.0;

	         dbgEx[i] = 0.0;
	         dbgEy[i] = 0.0;
	         dbgEz[i] = 0.0;
	         dbgHx[i] = 0.0;
	         dbgHy[i] = 0.0;
	         dbgHz[i] = 0.0;
	     }
	  }

	  virtual void InitCells()
	  {
	     for(int i = 0;i < Nx+2;i++)
	     {
	         for(int l = 0;l < Ny+2;l++)
		 {
		     for(int k = 0;k < Nz+2;k++)
		     {
	                 Cell<Particle> * c = new Cell<Particle>(i,l,k,Lx,Ly,Lz,Nx,Ny,Nz,tau);
	                 c->Init();
			         (*AllCells).push_back(*c);
#ifdef INIT_CELLS_DEBUG_PRINT
	                printf("%5d %5d %5d size %d \n",i,l,k,(*AllCells).size());
#endif
		     }

		 }

	     }
	  }

	  virtual void InitCurrents()
	  {
	     for(int i = 0;i < (Nx+2)*(Ny+2)*(Nz+2);i++)
	     {
	         Jx[i]  = 0.0;
	         Jy[i]  = 0.0;
	         Jz[i]  = 0.0;
	         Rho[i] = 0.0;

	         dbgJx[i]  = 0.0;
	         dbgJy[i]  = 0.0;
	         dbgJz[i]  = 0.0;

	     }
	  }

	  void InitCurrents(string fnjx,string fnjy,string fnjz,
	                    string dbg_fnjx,string dbg_fnjy,string dbg_fnjz,
	                    string np_fnjx,string np_fnjy,string np_fnjz,
			            int dbg)
	  {

	     read3Darray(np_fnjx, npJx);
	     read3Darray(np_fnjy, npJy);
	     read3Darray(np_fnjz, npJz);

	     if(dbg == 0)
	     {
	        read3Darray(fnjx, Jx);
	        read3Darray(fnjy, Jy);
	        read3Darray(fnjz, Jz);
	     }
	#ifdef DEBUG_PLASMA
	     read3Darray(dbg_fnjx, dbgJx);
	     read3Darray(dbg_fnjy, dbgJy);
	     read3Darray(dbg_fnjz, dbgJz);

	#endif
	   }

	  void InitFields(char *fnex,char *fney,char *fnez,
			          char *fnhx,char *fnhy,char *fnhz,
			          char *dbg_fnex,char *dbg_fney,char *dbg_fnez,
			          char *dbg_0fnex,char *dbg_0fney,char *dbg_0fnez,
			          char *np_ex,char *np_ey,char *np_ez,
			          char *dbg_fnhx,char *dbg_fnhy,char *dbg_fnhz)
	  {
	     InitFields();

	      //double t271 = Hy[27];

	     read3Darray(fnex, Ex);
	     read3Darray(fney, Ey);
	     read3Darray(fnez, Ez);
	     read3Darray(fnhx, Hx);
	     read3Darray(fnhy, Hy);
	     read3Darray(fnhz, Hz);

	#ifdef DEBUG_PLASMA
	     read3Darray(dbg_fnex, dbgEx);
	     read3Darray(dbg_fney, dbgEy);
	     read3Darray(dbg_fnez, dbgEz);

	     read3Darray(dbg_0fnex, dbgEx0);
	     read3Darray(dbg_0fney, dbgEy0);
	     read3Darray(dbg_0fnez, dbgEz0);

	     read3Darray(dbg_fnhx, dbgHx);
	     read3Darray(dbg_fnhy, dbgHy);
	     read3Darray(dbg_fnhz, dbgHz);

	     read3DarrayLog(np_ex, npEx,50,8);
	     read3DarrayLog(np_ey, npEy,50,8);
	     read3DarrayLog(np_ez, npEz,50,8);
	#endif

	  //   double t27 = Hy[27];

	  }


	  virtual void InitParticles(thrust::host_vector<Particle> & vp)
	  {
	     InitIonParticles(n_per_cell,ion_q_m,vp);
	  }

	  virtual void InitParticles(char *fname,thrust::host_vector<Particle>& vp)
	  {
	     FILE *f;
	     char str[1000];
	     double x,y,z,px,py,pz,q_m,m;
	     int n = 0;

	     if((f = fopen(fname,"rt")) == NULL) return;

	     while(fgets(str,1000,f) != NULL)
	     {
	          x   = atof(str);
	          y   = atof(str + 25);
	          z   = atof(str + 50);
	          px  = atof(str + 75);
	          py  = atof(str + 100);
	          pz  = atof(str + 125);
	          m   = fabs(atof(str + 150));
	          q_m = atof(str + 175);
	#undef GPU_PARTICLE
		  Particle *p = new Particle(x,y,z,px,py,pz,m,q_m);
//		      if(n == 829)
//		      {
//		    	  int qq = 0;
//		    	  qq = 1;
//		      }
		  p->fortran_number = ++n;
		  vp.push_back(*p);
	#define GPU_PARTICLE

	     }


         dbg_x = (double *)malloc(sizeof(double)*vp.size());
         dbg_y = (double *)malloc(sizeof(double)*vp.size());
         dbg_z = (double *)malloc(sizeof(double)*vp.size());
         dbg_px = (double *)malloc(sizeof(double)*vp.size());
         dbg_py = (double *)malloc(sizeof(double)*vp.size());
         dbg_pz = (double *)malloc(sizeof(double)*vp.size());

         total_particles = vp.size();

	     magf = 1;
	  }


	  void debugPrintParticleCharacteristicArray(double *p_ch,int np,int nt,char *name,int sort)
	  {
		   char fname[200];
		   FILE *f;

#ifndef PRINT_PARTICLE_INITIALS
		   return;

#else
		   sprintf(fname,"particle_init_%s_%05d_sort%02d.dat",name,nt,sort);

		   if((f = fopen(fname,"wt")) == NULL) return;

		   for (int i = 0;i < np;i++)
		   {
			   fprintf(f,"%10d %10d %25.16e \n",i,i+1,p_ch[i]);
		   }
		   fclose(f);
#endif
	  }

      virtual int readBinaryParticleArraysOneSort(
    		  FILE *f,
    		  double **dbg_x,
    		  double **dbg_y,
    		  double **dbg_z,
    		  double **dbg_px,
    		  double **dbg_py,
    		  double **dbg_pz,
    		  double *qq_m,
    		  double *mm,
    		  int nt,
    		  int sort
    		  )
      {
		//     char str[1000];
		     double /*x,y,z,px,py,pz,*/q_m,/* *buf,*/tp,m;
		     int t;
		     Cell<Particle> c0 = (*AllCells)[0];
		     int total_particles;
		     int err;

		     if((err = ferror(f)) != 0)
		     {
		     	 return err ;
		     }

		     fread(&t,sizeof(int),1,f);
		     if((err = ferror(f)) != 0)
		    	 {
		    	 	 return err ;
		    	 }
		     fread(&tp,sizeof(double),1,f);
		     if((err = ferror(f)) != 0)
		    	 {
		    	 	 return err ;
		    	 }

		     total_particles = (int)tp;
		     fread(&q_m,sizeof(double),1,f);
		     if((err = ferror(f)) != 0)
		    	 {
		    	 	 return err ;
		    	 }

		     fread(&m,sizeof(double),1,f);
		     if((err = ferror(f)) != 0)
		    	 {
		    	 	 return err ;
		    	 }

		    // m = fabs(m);
		     fread(&t,sizeof(int),1,f);
		     if((err = ferror(f)) != 0)
		    	 {
		    	 	 return err ;
		    	 }

	         *dbg_x = (double *)malloc(sizeof(double)*total_particles);
	         debugPrintParticleCharacteristicArray(*dbg_x,total_particles,nt,"x",sort);

	         *dbg_y = (double *)malloc(sizeof(double)*total_particles);
	         debugPrintParticleCharacteristicArray(*dbg_y,total_particles,nt,"y",sort);

	         *dbg_z = (double *)malloc(sizeof(double)*total_particles);
	         debugPrintParticleCharacteristicArray(*dbg_z,total_particles,nt,"z",sort);

	         *dbg_px = (double *)malloc(sizeof(double)*total_particles);
	         debugPrintParticleCharacteristicArray(*dbg_px,total_particles,nt,"px",sort);

	         *dbg_py = (double *)malloc(sizeof(double)*total_particles);
	         debugPrintParticleCharacteristicArray(*dbg_py,total_particles,nt,"py",sort);

	         *dbg_pz = (double *)malloc(sizeof(double)*total_particles);
	         debugPrintParticleCharacteristicArray(*dbg_pz,total_particles,nt,"pz",sort);

		 	readFortranBinaryArray(f,*dbg_x);
		 	readFortranBinaryArray(f,*dbg_y);
		 	readFortranBinaryArray(f,*dbg_z);
		 	readFortranBinaryArray(f,*dbg_px);
		 	readFortranBinaryArray(f,*dbg_py);
		 	readFortranBinaryArray(f,*dbg_pz);

		 	//printf("particle 79943 %25.15e \n",(*dbg_x)[79943]);

		 	*qq_m = q_m;
		 	*mm   = m;

		 	if((err = ferror(f)) != 0)
            {
	   	 	    return err ;
			}

		 	return total_particles;
      }

      void printPICstatitstics(double m,double q_m, int total_particles)
      {
    	  int pn_min,pn_ave,pn_max,pn_sum,err;

              pn_min = 1000000000;
              pn_max = 0;
              pn_ave = 0;
 		     for(int n = 0;n < (*AllCells).size();n++)
 		     {
 		    	 Cell<Particle> & c = (*AllCells)[n];

 		    	 pn_ave += c.number_of_particles;
 		    	 if(pn_min > c.number_of_particles) pn_min = c.number_of_particles;
 		    	 if(pn_max < c.number_of_particles) pn_max = c.number_of_particles;

 		     }

 		     pn_sum = pn_ave;
 		     pn_ave /= (*AllCells).size();

 		     printf("SORT m %15.5e q_m %15.5e %10d (sum %10d) particles in %8d cells: MIN %10d MAX %10d average %10d \n",
 		    		 m,            q_m,       total_particles,pn_sum,
 		    		 (*AllCells).size(),
 		    		 pn_min,pn_max,pn_ave);


      }


      int addParticleListToCells(std::vector<Particle>& vp)
      {
    	  Cell<Particle> c0 = (*AllCells)[0];
    	  int n;

    	  for(int i = 0; i < vp.size();i++)
    	  {
    	      Particle p = vp[i]; // = new Particle(x,y,z,px,py,pz,m,q_m);

    	  	  double3 d;
    	      d.x = p.x;
    	      d.y = p.y;
    	      d.z = p.z;

    	      n = c0.getPointCell(d);

    	      Cell<Particle> & c = (*AllCells)[n];


    	      if(c.Insert(p) == true)
    	      {
#ifdef PARTICLE_PRINTS1000
    	  		             if((i+1)%1000 == 0 )
    	  		             {
    	  		        	     printf("particle %d (%e,%e,%e) is number %d in cell (%d,%d,%d)\n",
    	  		        	    		 i+1,
    	  				    		x,y,z,c.number_of_particles,c.i,c.l,c.k);
    	  		             }
#endif
    	  			      }
   		      }// END total_particles LOOP

      }

      int convertParticleArraysToSTLvector(
    		  double *dbg_x,
    		  double *dbg_y,
			  double *dbg_z,
			  double *dbg_px,
			  double *dbg_py,
			  double *dbg_pz,
			  double q_m,
			  double m,
			  int total_particles,
			  particle_sorts sort,
    		  std::vector<Particle> & vp
    		  )
      {
    	  double x,y,z,px,py,pz;

    	  for(int i = 0; i < total_particles;i++)
    	  {
			  x   = dbg_x[i];
			  y   = dbg_y[i];
			  z   = dbg_z[i];
			  px   = dbg_px[i];
			  py   = dbg_py[i];
			  pz   = dbg_pz[i];


			  Particle p(x,y,z,px,py,pz,m,q_m);

			  p.fortran_number = i+1;
			  p.sort = sort;

			  vp.push_back(p);

    	  }
      }

      int getParticlesOneSortFromFile(
    		                          FILE *f,
                                      particle_sorts sort,
                                      int nt,
                                      std::vector<Particle> & vp,
                                      double *q_m,
                                      double *m
                                      )
      {
 		     double x,y,z,px,py,pz;

 		     int err;

 		     if((err = ferror(f)) != 0) return 1;

 		     total_particles = readBinaryParticleArraysOneSort(f,&dbg_x,&dbg_y,&dbg_z,
 		    		                                             &dbg_px,&dbg_py,&dbg_pz,q_m,m,nt,
 		    		                                             sort);

 		     real_number_of_particle[(int)sort] = total_particles;

 		     if((err = ferror(f)) != 0) return 1;
 		     convertParticleArraysToSTLvector(dbg_x,dbg_y,dbg_z,dbg_px,dbg_py,dbg_pz,*q_m,*m,
 		    			  total_particles,sort,vp);

      }

	  virtual void readBinaryParticlesOneSort(FILE *f,std::vector<Particle> & vp,
			                                  particle_sorts sort,int nt)

	  {
		    double q_m,m;
		    int err;
		    getParticlesOneSortFromFile(f,sort,nt,vp,&q_m,&m);

		    err = ferror(f);
		    free(dbg_x);
			free(dbg_y);
			free(dbg_z);
		    free(dbg_px);
			free(dbg_py);
			free(dbg_pz);
			err = ferror(f);

			struct sysinfo info;
            sysinfo(&info);
			printf("before1  %d free %u \n",nt,info.freeram/1024/1024);
			err = ferror(f);

			printPICstatitstics(m,q_m,total_particles);
	  }



	  FILE *readPreliminary3Darrays(string fn,int nt)
	  {
		     double *buf;
		     FILE *f;

		     buf = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

		     if((f = fopen(fn.c_str(),"rb")) == NULL) return NULL;
		     struct sysinfo info;

		     sysinfo(&info);
		     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);
		     readFortranBinaryArray(f,buf);
		     sysinfo(&info);
		     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);
		     readFortranBinaryArray(f,buf);
		     sysinfo(&info);
		     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);

		     readFortranBinaryArray(f,buf);
		     sysinfo(&info);
		     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);

		     readFortranBinaryArray(f,buf);
		     sysinfo(&info);
		     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);

		     readFortranBinaryArray(f,buf);
		     readFortranBinaryArray(f,buf);

		     readFortranBinaryArray(f,buf);
		     readFortranBinaryArray(f,buf);
		     readFortranBinaryArray(f,buf);

		     readFortranBinaryArray(f,buf);
		     readFortranBinaryArray(f,buf);
		     readFortranBinaryArray(f,buf);
		     int err;
	//--------------------------------------------
		     err = ferror(f);

		     if(err != 0) return NULL;

		     return f;
	  }

	  int addAllParticleListsToCells(std::vector<Particle> & ion_vp,
			                         std::vector<Particle> & el_vp,
			                         std::vector<Particle> & beam_vp)
	  {
			 addParticleListToCells(ion_vp);
			 addParticleListToCells(el_vp);
			 addParticleListToCells(beam_vp);

			 return 0;
	  }

	  int readBinaryParticlesAllSorts(FILE *f,int nt,
			                          std::vector<Particle> & ion_vp,
                                      std::vector<Particle> & el_vp,
                                      std::vector<Particle> & beam_vp)
	  {
		  readBinaryParticlesOneSort(f,ion_vp,ION,nt);
          readBinaryParticlesOneSort(f,el_vp,PLASMA_ELECTRON,nt);
    	  readBinaryParticlesOneSort(f,beam_vp,BEAM_ELECTRON,nt);

    	  return 0;
	  }

	  int readParticles(FILE *f,int nt)
	  {
		 std::vector<Particle> ion_vp,el_vp,beam_vp;

		 readBinaryParticlesAllSorts(f,nt,ion_vp,el_vp,beam_vp);


		 addAllParticleListsToCells(ion_vp,el_vp,beam_vp);

		 return 0;
	  }

	  string getBinaryFileName(int nt)
	  {
		  char part_name[100];
		  string s;

		  sprintf(part_name,"mumu000%08d.dat",nt);

		  s = part_name;

		  return s;
	  }

	  virtual void InitBinaryParticles(int nt)
	  {
	     FILE *f;
	     string part_name = getBinaryFileName(nt);

		 if((f = readPreliminary3Darrays(part_name,nt)) == NULL) return;

		 std::vector<Particle> ion_vp,el_vp,beam_vp;

		 readBinaryParticlesAllSorts(f,nt,ion_vp,el_vp,beam_vp);


		 addAllParticleListsToCells(ion_vp,el_vp,beam_vp);

	     fclose(f);

	     magf = 1;
	  }



	  virtual void InitElectronParticles(){}
	  virtual void InitIonParticles(int n_per_cell1,double q_m,thrust::host_vector<Particle> &vecp)
	  {
	     int total_ions = Nx*Ny*Nz*n_per_cell;
	     Particle *p;
	     //double ami = ni /((double)n_per_cell);
	     double x,y,z;

	     for(int j = 0;j < total_ions;j++)
	     {
		z = Lz * rnd_uniform();
		y = meh * Ly + Ly * rnd_uniform();
		x = Lx * rnd_uniform();

		p = new Particle(x,y,z,0.0,0.0,0.0,ni,q_m);

	#ifdef DEBUG_PLASMA
//		printf("particle %d \n",j);
	#endif

		vecp.push_back(*p);
	     }
	  }

	  virtual void InitBeamParticles(int n_per_cell1){}
	  void Distribute(thrust::host_vector<Particle> &vecp)
	  {
	     Cell<Particle> c0 = (*AllCells)[0],c111;
	     int    n;//,i;
	     int  vec_size = vecp.size();

	     for(int j = 0;j < vecp.size();j++)
	     {
		 Particle p = vecp[j];
		 double3 d;
		 d.x = p.x;
		 d.y = p.y;
		 d.z = p.z;

		 n = c0.getPointCell(d);

		 Cell<Particle> & c = (*AllCells)[n];;
	//	 c.SetZero();
	//	 c = (*AllCells)[n];
//		 if((vec_size-vecp.size()) == 136)
//		 {
//			 i = 0;
//		 }

//		 if(c.i == 0 && c.k == 0 && c.l == 0)
//		 {
//			 int z67 = 0;
//		 }

		 if(c.Insert(p) == true)
		 {
	#ifdef PARTICLE_PRINTS1000
         if((vec_size-vecp.size())%1000 == 0 )	printf("particle %d (%e,%e,%e) is number %d in cell (%d,%d,%d)\n",vec_size-vecp.size(),
		    		p.x,p.y,p.z,c.number_of_particles,c.i,c.l,c.k);
         if((vec_size-vecp.size()) == 10000) exit(0);
	#endif
		    vecp.erase(vecp.begin()+j);
		    j--;
		 }
	     }
	     int pn_min = 1000000,pn_max = 0,pn_ave = 0;
	     for(int n = 0;n < (*AllCells).size();n++)
	     {
	    	 Cell<Particle> & c = (*AllCells)[n];

	    	 pn_ave += c.number_of_particles;
	    	 if(pn_min > c.number_of_particles) pn_min = c.number_of_particles;
	    	 if(pn_max < c.number_of_particles) pn_max = c.number_of_particles;

	     }
	     pn_ave /= (*AllCells).size();

	     printf("%10d particles in %8d cells: MIN %5d MAX %5d average %5d \n",vec_size,(*AllCells).size(),
	    		                                                              pn_min,pn_max,pn_ave);
	  }



//	void virtual emeIterate(int i_s,int i_f,int l_s,int l_f,int k_s,int k_f,
//			double *E,double *H1, double *H2,
//			double *J,double c1,double c2, double tau,
//			int dx1,int dy1,int dz1,int dx2,int dy2,int dz2)
//	{
//		Cell<Particle>  c0 = (*AllCells)[0],*c;
//
//		c = &c0;
//
//		for(int i = i_s;i <= i_f;i++)
//		{
//			  for(int l = l_s;l <= i_f;l++)
//			  {
//			      for(int k = k_s;k <= k_f;k++)
//			      {
//			    	  emeElement(c,i,l,k,E,H1,H2,
//			    	  		J,c1,c2,tau,
//			    	  		dx1,dy1,dz1,dx2,dy2,dz2);
//			      }
//			  }
//		}
//	}

	//  int virtual ElectricFieldTrace(Cell<Particle> &c,char *lname,int nt,
	//  double *E,double *H1,double *H2,double *J,double *dbg_E0,double *dbg_E,double *dbg_H1,double *dbg_H2,double *dbg_J,int dir,double c1,double c2,double tau)
	//  {
	//      int i_start,l_start,k_start,dx1,dy1,dz1,dx2,dy2,dz2;
	//      int check_E_local;
	//      double t0,t,*dbg_E_aper,*Jloc,*ldH1,*ldH2,*ldE;
	//
	//#ifdef DEBUG_PLASMA_EFIELDS
	//      char logname[100];
	//
	//      sprintf(logname,"%s%03d.dat",lname,nt);
	//
	//      dbg_E_aper = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	//      Jloc = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	//      ldH1 = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	//      ldH2 = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	//      ldE  = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	//
	//
	//      char dbg_fnex[100];
	//
	//
	//
	//     read3DarrayLog(logname, ldE,50,0);
	//     read3DarrayLog(logname, ldH1,50,1);
	//     read3DarrayLog(logname, ldH2,50,3);
	//     read3DarrayLog(logname, Jloc,50,5);
	//     read3DarrayLog(logname, dbg_E_aper,50,8);
	//
	//
	//
	//
	////
	////     sprintf(dbg_fnex,"dnex%06d.dat",5);
	////     read3Darray(dbg_fnex, dbgEx);
	////     CheckArray(E,dbgEx);
	//
	//     CheckArray(E,ldE);
	//     puts("Ex prev");
	//     //exit(0);
	//     CheckArray(Jloc,dbg_J);
	//     CheckArray(J,dbg_J);
	//     CheckArray(J,Jloc);
	//     puts("Jx");
	//     //exit(0);
	//     CheckArray(H1,ldH1);
	//     puts("H1");
	//     //exit(0);
	//     CheckArray(H2,ldH2);
	//     puts("H2");
	//     //exit(0);
	//#endif
	//
	//      i_start = (dir == 0)*0 + (dir == 1)*1 + (dir == 2)*1;
	//      l_start = (dir == 0)*1 + (dir == 1)*0 + (dir == 2)*1;
	//      k_start = (dir == 0)*1 + (dir == 1)*1 + (dir == 2)*0;
	//
	//      dx1 = (dir == 0)*0    + (dir == 1)*0    + (dir == 2)*(-1);
	//      dy1 = (dir == 0)*(-1) + (dir == 1)*0    + (dir == 2)*0;
	//      dz1 = (dir == 0)*0    + (dir == 1)*(-1) + (dir == 2)*0;
	//
	//      dx2 = (dir == 0)*0    + (dir == 1)*(-1) + (dir == 2)*0;
	//      dy2 = (dir == 0)*0    + (dir == 1)*0    + (dir == 2)*(-1);
	//      dz2 = (dir == 0)*(-1) + (dir == 1)*0    + (dir == 2)*0;
	//
	//      emeIterate(c,i_start,Nx,l_start,Ny,k_start,Nz,
	//    		                E,H1,H2,
	//      		    	  		J,c1,c2,tau,
	//      		    	  		dx1,dy1,dz1,dx2,dy2,dz2);
	//
	////      for(int i = i_start;i <= Nx;i++)
	////      {
	////	  for(int l = l_start;l <= Ny;l++)
	////	  {
	////	      for(int k = k_start;k <= Nz;k++)
	////	      {
	////		  int n  = c.getGlobalCellNumber(i,l,k);
	////		  int n1 = c.getGlobalCellNumber(i+dx1,l+dy1,k+dz1);
	////		  int n2 = c.getGlobalCellNumber(i+dx2,l+dy2,k+dz2);
	////
	////#ifdef DEBUG_PLASMA
	////		  double h1_n  = H1[n];
	////		  double h1_n1 = H1[n1];
	////		  double h2_n  = H2[n];
	////		  double h2_n2 = H2[n2];
	////		  double E_n =   E[n];
	////		  double J_n   = J[n];
	////		  double term1 = c1*(H1[n] - H1[n1]);
	////		  double term2 = c2*(H2[n] - H2[n2]);
	////#endif
	////		  E[n] += c1*(H1[n] - H1[n1]) - c2*(H2[n] - H2[n2]) - tau*J[n];
	////#ifdef DEBUG_PLASMA
	////		  t = fabs(E[n]-dbg_E[n]);
	////
	////		  std::cout << i << " " << l << " " << k << " " << t << std::endl;
	////		  if((t > TOLERANCE) && ((i == 10 ) && (l == 10) && (k == 0)))
	////		  {
	////		     printf("WRONG i %3d l %3d k %3d %15.5e dbg %15.5e\n",i,l,k,t,dbg_E[n]);
	////		  }
	////
	//
	////#endif
	////
	////	      }
	////	  }
	////      }
	//#ifdef DEBUG_PLASMA
	//      int check_E_final;
	//#ifdef DEBUG_PLASMA_EFIELDS
	//      check_E_final  = CheckArray(E,dbg_E_aper);
	//      puts("Ex inside");
	//#endif
	//      check_E_final  = CheckArray(E,dbg_E);
	//#endif
	//
	////    exit(0);
	//
	//    return 0;
	//  }

	int getMagneticFieldTraceShifts(int dir,int3 &d1,int3 &d2)
	{
	      d1.x = (dir == 0)*0 + (dir == 1)*1 + (dir == 2)*0;
	      d1.y = (dir == 0)*0 + (dir == 1)*0 + (dir == 2)*1;
	      d1.z = (dir == 0)*1 + (dir == 1)*0 + (dir == 2)*0;

	      d2.x = (dir == 0)*0 + (dir == 1)*0 + (dir == 2)*1;
	      d2.y = (dir == 0)*1 + (dir == 1)*0 + (dir == 2)*0;
	      d2.z = (dir == 0)*0 + (dir == 1)*1 + (dir == 2)*0;

	      return 0;
	}

	int MagneticFieldTrace(double *Q,double *H,double *E1,double *E2,int i_end,int l_end,int k_end,double c1,double c2,int dir)
	{
	      int3 d1,d2;

	      getMagneticFieldTraceShifts(dir,d1,d2);

   		dim3 dimGrid(i_end+1,l_end+1,k_end+1),dimBlock(1,1,1);

	    GPU_emh1<<<dimGrid,dimBlock>>>(d_CellArray,Q,H,E1,E2,c1,c2,
	    		d1,d2);

	      return 0;
	  }

	int SimpleMagneticFieldTrace(Cell<Particle> &c,double *Q,double *H,int i_end,int l_end,int k_end)
	{


		   		dim3 dimGrid(i_end+1,l_end+1,k_end+1),dimBlock(1,1,1);

			    GPU_emh2<<<dimGrid,dimBlock>>>(d_CellArray,0,0,0,Q,H);


	      return 0;
	  }


	  int PeriodicBoundaries(double *E,int dir,int start1,int end1,int start2,int end2,int N)
	  {
	      Cell<Particle>  c = (*AllCells)[0];

	      if(CPU_field == 0)
	      {
	    		dim3 dimGrid(end1-start1+1,1,end2-start2+1),dimBlock(1,1,1);

	    	    GPU_periodic<<<dimGrid,dimBlock>>>(d_CellArray,start1,start2,E,dir,0,N);
	    	    GPU_periodic<<<dimGrid,dimBlock>>>(d_CellArray,start1,start2,E,dir,N+1,1);

	      }
	      else
	      {

	      for(int k = start2;k <= end2;k++)
	      {
		  for(int i = start1;i <= end1;i++)
		  {
			  periodicElement(&c,i,k,E,dir,0,N);
//		      //std::cout << "ex2 "<< i+1 << " "<< N+2 << " " << k+1  <<" " <<  i+1 << " " << " 2 " << " " << k+1  << std::endl;
//			  int3 i0,i1;
//
//	                  int n   = c.getGlobalBoundaryCellNumber(i,k,dir,0);
//			  int n1  = c.getGlobalBoundaryCellNumber(i,k,dir,N);
//			  E[n]    = E[n1];
//			  i0= c.getCellTripletNumber(n);
//			  i1= c.getCellTripletNumber(n1);
			   //std::cout << "ex1 "<< i0.x+1 << " "<< i0.y+1 << " " << i0.z+1  <<" " <<  i1.x+1 << " " << i1.y+1 << " " << i1.z+1  << " " << E[n]  << " " << E[n1] << std::endl;
		  }
	      }
	      for(int k = start2;k <= end2;k++)
	      {
	         for(int i = start1;i <= end1;i++)
	      	 {
	        	 periodicElement(&c,i,k,E,dir,N+1,1);
//	          int n       = c.getGlobalBoundaryCellNumber(i,k,dir,N+1);
//			  int n1      = c.getGlobalBoundaryCellNumber(i,k,dir,1);
//			  E[n]    = E[n1];
//			  int3 i0= c.getCellTripletNumber(n);
//			  int3 i1= c.getCellTripletNumber(n1);
			  //std::cout << "ex2 "<< i0.x+1 << " "<< i0.y+1 << " " << i0.z+1  <<" " <<  i1.x+1 << " " << i1.y+1 << " " << i1.z+1  << " " << E[n]  << " " << E[n1] << std::endl;
		  }
//		      int qq = 0;
	      }
	      }
	      return 0;
	}

int SinglePeriodicBoundary(double *E,int dir,int start1,int end1,int start2,int end2,int N)
{
    Cell<Particle>  c = (*AllCells)[0];

    if(CPU_field == 0)
    {
    	dim3 dimGrid(end1-start1+1,1,end2-start2+1),dimBlock(1,1,1);

   	    GPU_periodic<<<dimGrid,dimBlock>>>(d_CellArray,start1,start2,E,dir,N+1,1);

    }
    else
    {
       for(int k = start2;k <= end2;k++)
       {
	  	  for(int i = start1;i <= end1;i++)
	  	  {

	  	      //std::cout << "ex2 "<< i+1 << " "<< N+2 << " " << k+1  <<" " <<  i+1 << " " << " 2 " << " " << k+1  << std::endl;
	  		  int3 i0,i1;

	                    int n   = c.getGlobalBoundaryCellNumber(i,k,dir,N+1);
	  		            int n1  = c.getGlobalBoundaryCellNumber(i,k,dir,1);
	  		            E[n]    = E[n1];
	  		            i0= c.getCellTripletNumber(n);
	  		            i1= c.getCellTripletNumber(n1);
	  		            std::cout << "ex1 "<< i0.x+1 << " "<< i0.y+1 << " " << i0.z+1  <<" " <<  i1.x+1 << " " << i1.y+1 << " " << i1.z+1  << " " << E[n]  << " " << E[n1] << std::endl;
	  		   	  }
	  	      //int qq = 0;
	        }
    }
    return 0;
}



      void getWrongCurrentCellList(int num,int nt)
      {
    	   FILE *f;
    	   char fn_copy[100];
    	   Cell<Particle> c = (*AllCells)[0];
    	   int wrong_flag[(Nx+2)*(Ny+2)*(Nz+2)],*d_wrong_flag;
    	   double_pointer wrong_attributes[(Nx+2)*(Ny+2)*(Nz+2)],*d_wrong_attributes;

    	   readControlPoint(&f,fn_copy,num,nt,1,0,dbgEx,dbgEy,dbgEz,dbgHx,dbgHy,dbgHz,dbgJx,dbgJy,dbgJz,dbg_Qx,dbg_Qy,dbg_Qz,
    	                     dbg_x,dbg_y,dbg_z,dbg_px,dbg_py,dbg_pz);

    	   for(int n = 0;n < (Nx + 2)*(Ny + 2)*(Nz + 2);n++)
    	       	   {
    		           wrong_flag[n] = 0;
    	       	   }
    	   cudaMalloc(&d_wrong_flag,sizeof(int)*(Nx + 2)*(Ny + 2)*(Nz + 2));
    	   cudaMalloc(&d_wrong_attributes,sizeof(double_pointer)*(Nx+2)*(Ny+2)*(Nz+2));

    	   static double *t;
    	   static int first = 1;

    	   if(first == 1)
    	   {
    	  	 t = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
    	  	 first = 0;
    	   }
    	   cudaError_t err;
    	   err = cudaMemcpy(t,d_Jx,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyDeviceToHost);
    	   if(err != cudaSuccess)
    	   {
    	     printf("getWrongCurrentCellList err %d %s \n",err,cudaGetErrorString(err));
    	  	 exit(0);
    	   }

    	   double diff = 0.0;
    	   jx_wrong_points_number = 0;
    	  // puts("begin array checking=============================");
    	   for(int n = 0;n < (Nx + 2)*(Ny + 2)*(Nz + 2);n++)
    	   {
   	           if(fabs(dbgJx[n] - t[n]) > TOLERANCE)
    	   	   {
      		       int3 i = c.getCellTripletNumber(n);
                   jx_wrong_points_number++;
                   wrong_flag[n] = 1;

//#ifdef WRONG_CURRENTS_CHECK
//                   double *d_w;
//                   cudaError_t err_attr = cudaMalloc(
//                	        &d_w,
//                   			sizeof(double)*PARTICLE_ATTRIBUTES*MAX_particles_per_cell);
//                   printf("wrong current attributes alloc %d %s \n",err_attr,cudaGetErrorString(err_attr));
//                   wrong_attributes[n] = d_w;
//#endif
        	   }
    	   }
    	   jx_wrong_points = (int3 *)malloc(jx_wrong_points_number*sizeof(int3));

    	   int num_cell = 0;
    	   for(int n = 0;n < (Nx + 2)*(Ny + 2)*(Nz + 2);n++)
    	   {
    	       if(fabs(dbgJx[n] - t[n]) > TOLERANCE)
    	       {
    	          int3 i = c.getCellTripletNumber(n);
    	          jx_wrong_points[num_cell++] = i;
    	       }
    	   }
    	   cudaMalloc(&(d_jx_wrong_points),jx_wrong_points_number*sizeof(int3));

    	   cudaMemcpy(d_wrong_flag,wrong_flag,sizeof(int)*(Nx + 2)*(Ny + 2)*(Nz + 2),cudaMemcpyHostToDevice);

    	   cudaMemcpy(d_wrong_attributes,wrong_attributes,
    			                              sizeof(double_pointer)*(Nx + 2)*(Ny + 2)*(Nz + 2),cudaMemcpyHostToDevice);

           copy_pointers<<<(Nx + 2)*(Ny + 2)*(Nz + 2),1>>>(d_CellArray,d_wrong_flag,d_wrong_attributes);

      }

      void WrongCurrentCell_AttributeMalloc(int num,int nt)
      {

      }

	  double checkFirstHalfstepFields(int nt)
	  {
//		  double /*t = 0.0,*/*dbg,t_ex,t_ey,t_ez,t_hx,t_hy,t_hz;

//		  dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

		  //read magnetic field from "nt+1" exlg file - to consider emh2
//		  readDebugArray("exlg",dbg,nt+1,1);
//		  t_hz = CheckArray(Hz,dbg);
//		  readDebugArray("exlg",dbg,nt+1,3);
//		  t_hy = CheckArray(Hy,dbg);
//		  readDebugArray("eylg",dbg,nt+1,1);
//		  t_hx = CheckArray(Hx,dbg);
//		  //electric from "nt+1" exlg file to take periodicity into account
//		  readDebugArray("exlg",dbg,nt+1,0);
//		  t_ex = CheckArray(Ex,dbg);
//		  readDebugArray("eylg",dbg,nt+1,0);
//		  t_ey = CheckArray(Ey,dbg);
//		  readDebugArray("ezlg",dbg,nt+1,0);
//		  t_ez = CheckArray(Ez,dbg);

	      //printf("First half-step fields %.5f \n",(t_ex+t_ey+t_ez+t_hx+t_hy+t_hz)/6.0);
		  return 1.0;//(t_ex+t_ey+t_ez+t_hx+t_hy+t_hz)/6.0;
	  }

	  double checkFirstHalfstep_emh2_GPUMagneticFields(int nt)
	  	  {
	  		  double /*t = 0.0,*/*dbg/*,t_ex,t_ey,t_ez*/,t_hx,t_hy,t_hz;

	  		  dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

	  		  //read magnetic field from "nt+1" exlg file - to consider emh2
//	  		  readDebugArray("exlg",dbg,nt+1,1);
//	  		  t_hz = checkGPUArray(dbg,d_Hz);
//	  		  readDebugArray("exlg",dbg,nt+1,3);
//	  		  t_hy = checkGPUArray(dbg,d_Hy);
//	  		  readDebugArray("eylg",dbg,nt+1,1);
//	  		  t_hx = checkGPUArray(dbg,d_Hx);

	  	      //printf("First half-step EMH-2 fields %.5f \n",(t_hx+t_hy+t_hz)/3.0);
	  		  return 1.0;
	  	  }


	  double checkFirstHalfstep_emh1_GPUMagneticFields(int nt)
	  {
		  double /*t = 0.0,*/*dbg,t_ex,t_ey,t_ez,t_hx,t_hy,t_hz;

		  dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

//		  readDebugArray("hzlg",dbg,nt,0);
//		  t_ez = checkGPUArray(dbg,d_Qz);
//		  readDebugArray("hylg",dbg,nt,0);
//		  t_ey = checkGPUArray(dbg,d_Qy);
//		  readDebugArray("hxlg",dbg,nt,0);
//		  t_ex = checkGPUArray(dbg,d_Qx);
//		  //read magnetic field from "nt+1" exlg file - to consider emh2
//		  readDebugArray("hzlg",dbg,nt,6);
//		  t_hz = checkGPUArray(dbg,d_Hz);
//		  readDebugArray("hylg",dbg,nt,6);
//		  t_hy = checkGPUArray(dbg,d_Hy);
//		  readDebugArray("hxlg",dbg,nt,6);
//		  t_hx = checkGPUArray(dbg,d_Hx);



	      //printf("First half-step MAGNETIC GPU fields %.5f \n",(t_ex+t_ey+t_ez+t_hx+t_hy+t_hz)/6.0);
		  return 1.0;//(t_ex+t_ey+t_ez+t_hx+t_hy+t_hz)/6.0;
	  }

	  double checkFirstHalfstep_emh1_MagneticFields(int nt,double *Qx,double *Qy,double *Qz,
			                                               double *Hx,double *Hy,double *Hz)
	  {
		  double *dbg,t_ex,t_ey,t_ez,t_hx,t_hy,t_hz;

		  //dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

//		  readDebugArray("hzlg",dbg,nt,0);
//		  t_ez = CheckArray(dbg,Qz);
//		  readDebugArray("hylg",dbg,nt,0);
//		  t_ey = CheckArray(dbg,Qy);
//		  readDebugArray("hxlg",dbg,nt,0);
//		  t_ex = CheckArray(dbg,Qx);
//		  //read magnetic field from "nt+1" exlg file - to consider emh2
//		  readDebugArray("hzlg",dbg,nt,6);
//		  t_hz = CheckArray(dbg,Hz);
//		  readDebugArray("hylg",dbg,nt,6);
//		  t_hy = CheckArray(dbg,Hy);
//		  readDebugArray("hxlg",dbg,nt,6);
//		  t_hx = CheckArray(dbg,Hx);



	      //printf("First half-step MAGNETIC GPU fields %.5f \n",(t_ex+t_ey+t_ez+t_hx+t_hy+t_hz)/6.0);
		  return 1.0;//(t_ex+t_ey+t_ez+t_hx+t_hy+t_hz)/6.0;
	  }


	  double checkFirstHalfstepElectricFields(int nt)
	  {
		  double *dbg,t_ex,t_ey,t_ez;//,t_hx,t_hy,t_hz;

		//  dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));


		  //electric from "nt+1" exlg file to take periodicity into account
//		  readDebugArray("exlg",dbg,nt+1,0);
//		  t_ex = CheckArray(Ex,dbg);
//		  readDebugArray("eylg",dbg,nt+1,0);
//		  t_ey = CheckArray(Ey,dbg);
//		  readDebugArray("ezlg",dbg,nt+1,0);
//		  t_ez = CheckArray(Ez,dbg);

	      //printf("First half-step fields %.5f \n",(t_ex+t_ey+t_ez+t_hx+t_hy+t_hz)/3.0);
		  return 1.0;//(t_ex+t_ey+t_ez)/3.0;
	  }

	  double checkFirstHalfstepGPUElectricFields(int nt)
	  {
//		  double *dbg,t_ex,t_ey,t_ez;//,t_hx,t_hy,t_hz;

//		  dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));


		  //electric from "nt+1" exlg file to take periodicity into account
//		  readDebugArray("exlg",dbg,nt+1,0);
//		  t_ex = checkGPUArray(dbg,d_Ex);
//		  readDebugArray("eylg",dbg,nt+1,0);
//		  t_ey = checkGPUArray(dbg,d_Ey);
//		  readDebugArray("ezlg",dbg,nt+1,0);
//		  t_ez = checkGPUArray(dbg,d_Ez);

//	      printf("First half-step GPU fields %.5f \n",(t_ex+t_ey+t_ez)/3.0);
		  return 1.0;//(t_ex+t_ey+t_ez)/3.0;
	  }

	  double checkSecondHalfstepFields(int nt)
	    {
//	  	  double *dbg,t_ex,t_ey,t_ez;//,t_hx,t_hy,t_hz;

//	  	  dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

	  	  //read magnetic field from "nt+1" exlg file - to consider emh2

	  	  //electric from "nt+1" exlg file to take periodicity into account
//	  	  readDebugArray("exlg",dbg,nt+2,0);
//	  	  t_ex = CheckArray(Ex,dbg);
//	  	  readDebugArray("eylg",dbg,nt+2,0);
//	  	  t_ey = CheckArray(Ey,dbg);
//	  	  readDebugArray("ezlg",dbg,nt+2,0);
//	  	  t_ez = CheckArray(Ez,dbg);

	//        printf("Second half-step fields %.5f \n",(t_ex+t_ey+t_ez)/3.0);
	  	  return 1.0;//(t_ex+t_ey+t_ez)/3.0;
	    }
	  double checkGPUSecondHalfstepFields(int nt)
	  	    {
//	  	  	  double /*t = 0.0,*/*dbg,t_ex,t_ey,t_ez; // ,t_hx,t_hy,t_hz;

//	  	  	  dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

	  	  	  //read magnetic field from "nt+1" exlg file - to consider emh2

	  	  	  //electric from "nt+1" exlg file to take periodicity into account
//	  	  	  readDebugArray("exlg",dbg,nt+1,8);
//	  	  	  t_ex = checkGPUArray(dbg,d_Ex);
//	  	  	  readDebugArray("eylg",dbg,nt+1,8);
//	  	  	  t_ey = checkGPUArray(dbg,d_Ey);
//	  	  	  readDebugArray("ezlg",dbg,nt+1,8);
//	  	  	  t_ez = checkGPUArray(dbg,d_Ez);

	  	        //printf("Second GPU half-step fields %.5f \n",(t_ex+t_ey+t_ez)/3.0);
	  	  	  return 1.0;//(t_ex+t_ey+t_ez)/3.0;
	  	    }


	  double checkCurrents(int nt)
	    {
	  	  double  *dbg,/*t_ex,t_ey,t_ez,*/t_hx,t_hy,t_hz;
	  	  static int first = 1;

	  	  if(first == 1)
	  	  {
	  	     dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	  	     first = 0;
	  	  }

	  	  //read magnetic field from "nt+1" exlg file - to consider emh2
//	  	  readDebugArray("exlg",dbg,nt+1,5);
//	  	  t_hz = CheckArray(npJx,dbg);
//	  	  readDebugArray("eylg",dbg,nt+1,5);
//	  	  t_hy = CheckArray(npJy,dbg);
//	  	  readDebugArray("ezlg",dbg,nt+1,5);
//	  	  t_hx = CheckArray(npJz,dbg);

//	      printf("half-step currents %.5f \n",(t_hx+t_hy+t_hz)/3.0);

	  	  return 1.0;//(t_hx+t_hy+t_hz)/3.0;
	    }

	  int SetPeriodicCurrentComponent(Cell<Particle>  **cells,double *J,int dir,int Nx,int Ny,int Nz)
	  {
		  dim3 dimGridX(Ny+2,1,Nz+2),dimGridY(Nx+2,1,Nz+2),dimGridZ(Nx+2,1,Ny+2),dimBlock(1,1,1);

          GPU_CurrentPeriodic<<<dimGridX,dimBlock>>>(cells,J,dir,0,0,0,Nx+2);
	      GPU_CurrentPeriodic<<<dimGridY,dimBlock>>>(cells,J,dir,1,0,0,Ny+2);
	      GPU_CurrentPeriodic<<<dimGridZ,dimBlock>>>(cells,J,dir,2,0,0,Nz+2);

		  return 0;
	  }

	  void SetPeriodicCurrents(int nt)
	  {

		  memory_monitor("before275",nt);

		  checkControlPoint(275,nt,0);

//		  dim3 dimGridX(Ny+2,1,Nz+2),dimGridY(Nx+2,1,Nz+2),dimGridZ(Nx+2,1,Ny+2),dimBlock(1,1,1);

		  SetPeriodicCurrentComponent(d_CellArray,d_Jx,0,Nx,Ny,Nz);
		  SetPeriodicCurrentComponent(d_CellArray,d_Jy,1,Nx,Ny,Nz);
		  SetPeriodicCurrentComponent(d_CellArray,d_Jz,2,Nx,Ny,Nz);

//         GPU_CurrentPeriodic<<<dimGridX,dimBlock>>>(d_CellArray,d_Jx,0,0,0,0,Nx+2);
//	     GPU_CurrentPeriodic<<<dimGridY,dimBlock>>>(d_CellArray,d_Jx,0,1,0,0,Ny+2);
//	     GPU_CurrentPeriodic<<<dimGridZ,dimBlock>>>(d_CellArray,d_Jx,0,2,0,0,Nz+2);

//	     GPU_CurrentPeriodic<<<dimGridX,dimBlock>>>(d_CellArray,d_Jy,1,0,0,0,Nx+2);
//	     GPU_CurrentPeriodic<<<dimGridY,dimBlock>>>(d_CellArray,d_Jy,1,1,0,0,Ny+2);
//	     GPU_CurrentPeriodic<<<dimGridZ,dimBlock>>>(d_CellArray,d_Jy,1,2,0,0,Nz+2);
//
//	     GPU_CurrentPeriodic<<<dimGridX,dimBlock>>>(d_CellArray,d_Jz,2,0,0,0,Nx+2);
//	     GPU_CurrentPeriodic<<<dimGridY,dimBlock>>>(d_CellArray,d_Jz,2,1,0,0,Ny+2);
//	     GPU_CurrentPeriodic<<<dimGridZ,dimBlock>>>(d_CellArray,d_Jz,2,2,0,0,Nz+2);

	     checkControlPoint(400,nt,0);

	   }

	  void InitQdebug(std::string fnjx,std::string fnjy,std::string fnjz)
	  {


	     read3Darray(fnjx, dbg_Qx);
	     read3Darray(fnjy, dbg_Qy);
	     read3Darray(fnjz, dbg_Qz);
	  }

	  int LoadMeshData(int nt)
	  {
//		 thrust::host_vector<Particle> vp,bin_vp;
//		 char part_name[100];
		 std::string jxfile,jyfile,jzfile,d_jxfile,d_jyfile,d_jzfile,np_jxfile,np_jyfile,np_jzfile,
		 qxfile,qyfile,qzfile;

		 readDebugArray("hxlg",Hx,nt,5);
		 readDebugArray("hylg",Hy,nt,5);
		 readDebugArray("hzlg",Hz,nt,5);


		 get_load_data_file_names(jxfile,jyfile,jzfile,d_jxfile,d_jyfile,d_jzfile,np_jxfile,np_jyfile,np_jzfile,
				 qxfile,qyfile,qzfile,nt);

		 InitQdebug(qxfile,qyfile,qzfile);

		 InitCurrents(jxfile,jyfile,jzfile,d_jxfile,d_jyfile,d_jzfile,
				 np_jxfile,np_jyfile,np_jzfile,0);

		 return 0;
	  }

	  int LoadParticleData(int nt,
			               std::vector<Particle> & ion_vp,
			               std::vector<Particle> & el_vp,
			               std::vector<Particle> & beam_vp)
	  {
		 if(nt > 1)
		 {
			 ClearAllParticles();
		 }

		 FILE *f;

		 string part_name = getBinaryFileName(nt);

		 if((f = readPreliminary3Darrays(part_name,nt)) == NULL) return 1;

//		 std::vector<Particle> ion_vp,el_vp,beam_vp;

		 readBinaryParticlesAllSorts(f,nt,ion_vp,el_vp,beam_vp);

		 fclose(f);




	     magf = 1;

		 return 0;
	  }

//	  int LoadData(i)

	  void LoadTestData(int nt,
			            int part_nt,
			            std::vector<Particle> & ion_vp,
			            std::vector<Particle> & el_vp,
			            std::vector<Particle> & beam_vp)
	  {
	     LoadMeshData(nt);

	     LoadParticleData(nt,ion_vp,el_vp,beam_vp);


	  }
	  virtual void readBinaryParticlesOneSort(FILE *f,thrust::host_vector<Particle>& vp,
			                                  particle_sorts sort,int nt)

	  {

//		     char str[1000];
		     double x,y,z,px,py,pz,q_m,m;
		     int n = 0;//,t;
		     Cell<Particle> c0 = (*AllCells)[0];
		     int pn_min,pn_ave,pn_max,pn_sum,err;


		     if((err = ferror(f)) != 0) return;

		     total_particles = readBinaryParticleArraysOneSort(f,&dbg_x,&dbg_y,&dbg_z,
		    		                                             &dbg_px,&dbg_py,&dbg_pz,&q_m,&m,nt,
		    		                                             sort);

		     real_number_of_particle[(int)sort] = total_particles;

		    err = ferror(f);
		    for(int i = 0; i < total_particles;i++)
		     {

		    	  if((err = ferror(f)) != 0)
		    	  {
		    		 // int qq = 0;
		    	  }

		         // if(i%getSize() != 0) continue;
		      //    printf("rank %d part %d \n",getRank(),i);

		    	  x   = dbg_x[i];
//		    	  if(i  == 269 && sort == ION)
//		    	  {
//		    		  int qq = 0;
//		    		  qq = 0;
//		    		 // printf("number %5d x %25.15e \n",i,x);
//		    	  }


		          y   = dbg_y[i];
		          z   = dbg_z[i];
  		          px   = dbg_px[i];
		          py   = dbg_py[i];
		          pz   = dbg_pz[i];

//		          if(i+1 == 138)
//		          {
//		        	  int qq = 0;
//		          }

		         // if(sort == PLASMA_ELECTRON) continue;

			      Particle p;// = new Particle(x,y,z,px,py,pz,m,q_m);
			      p.x   = x;
			      p.y   = y;
			      p.z   = z;
			      p.pu  = px;
			      p.pv  = py;
			      p.pw  = pz;
			      p.m   = m;
			      p.q_m = q_m;

		    	  if((err = ferror(f)) != 0)
		    	  {
		    		  //int qq = 0;
		    	  }

//			      if(n == 829)
//			      {
////			    	  int qq = 0;
////			    	  qq = 1;
//			      }
			      p.fortran_number = i+1;
			      p.sort = sort;
//////////////////////////////////////////////////
			      double3 d;
			      d.x = x;
			      d.y = y;
			      d.z = z;

			      n = c0.getPointCell(d);

			      Cell<Particle> & c = (*AllCells)[n];
		    	  if((err = ferror(f)) != 0)
		    	  {
		    		  int qq = 0;
		    	  }

		    	  if(i == 3189003 && sort == PLASMA_ELECTRON)
		    	  {
		    		  int qq = 0;
		    	  }


   			      if(c.Insert(p) == true)
			      {
#ifdef PARTICLE_PRINTS1000
		             if((i+1)%1000 == 0 )
		             {
		        	     printf("particle %d (%e,%e,%e) is number %d in cell (%d,%d,%d)\n",
		        	    		 i+1,
				    		x,y,z,c.number_of_particles,c.i,c.l,c.k);
		             }
		             //if((i+1) == 10000) exit(0);
#endif
			      }
   			   if((err = ferror(f)) != 0)
   			  		    	  {
   			  		    		 // int qq = 0;
   			  		    	  }

//		#define GPU_PARTICLE ///////////////////////

		     }// END total_particles LOOP
	    	  if((err = ferror(f)) != 0)
	    	  {
	    		  //int qq = 0;
	    	  }

		    err = ferror(f);
		    free(dbg_x);
			free(dbg_y);
			free(dbg_z);
		    free(dbg_px);
			free(dbg_py);
			free(dbg_pz);
			err = ferror(f);
			 if((err = ferror(f)) != 0)
					    	  {
					    		//  int qq = 0;
					    	  }

             pn_min = 1000000000;
             pn_max = 0;
             pn_ave = 0;
		     for(int n = 0;n < (*AllCells).size();n++)
		     {
		    	 Cell<Particle> & c = (*AllCells)[n];

		    	 pn_ave += c.number_of_particles;
		    	 if(pn_min > c.number_of_particles) pn_min = c.number_of_particles;
		    	 if(pn_max < c.number_of_particles) pn_max = c.number_of_particles;

		     }
		     if((err = ferror(f)) != 0)
		    		    	  {
		    		    		  //int qq = 0;
		    		    	  }
		     err = ferror(f);
		     pn_sum = pn_ave;
		     pn_ave /= (*AllCells).size();

		     printf("SORT m %15.5e q_m %15.5e %10d (sum %10d) particles in %8d cells: MIN %10d MAX %10d average %10d \n",
		    		 m,            q_m,       total_particles,pn_sum,
		    		 (*AllCells).size(),
		    		 pn_min,pn_max,pn_ave);
		     if((err = ferror(f)) != 0)
		    		    	  {
		    		    		 // int qq = 0;
		    		    	  }

		     err = ferror(f);
			//exit(0);
	  }


	  virtual void InitBinaryParticles(char *fn,thrust::host_vector<Particle>& vp,int nt)
	  {
	     FILE *f;
//	     char str[1000];
	     double /*x,y,z,px,py,pz,q_m*/*buf;//,tp,m;
//	     int n = 0,t;

	     buf = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

	     if((f = fopen(fn,"rb")) == NULL) return;
	     struct sysinfo info;

	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);
	     readFortranBinaryArray(f,buf);
	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);
	     readFortranBinaryArray(f,buf);
	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);

	     readFortranBinaryArray(f,buf);
	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);

	     readFortranBinaryArray(f,buf);
	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);

	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);

	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);

	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);
	     int err;
//--------------------------------------------
	     err = ferror(f);
	     readBinaryParticlesOneSort(f,vp,ION,nt);
	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);
	     err = ferror(f);

	     readBinaryParticlesOneSort(f,vp,PLASMA_ELECTRON,nt);
	     err = ferror(f);
	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);
         err = ferror(f);
	     readBinaryParticlesOneSort(f,vp,BEAM_ELECTRON,nt);
//--------------------------------------------

	     fclose(f);

	     magf = 1;
	  }


void readParticles(char *pfile,char *nextpfile)
{
	thrust::host_vector<Particle> vp;

	if(!strncmp(pfile,"mumu",4))
	{
		InitBinaryParticles(pfile,vp,4);
	}
	else
	{
       InitParticles(pfile,vp);
	}
#ifdef DEBUG_PLASMA
	if(!strncmp(pfile,"mumu",4))
	{
	    InitBinaryParticlesNext(nextpfile, vp);
	}
	else
	{
	    InitParticlesNext(nextpfile, vp);
	}
#endif
    Distribute(vp);

    AssignArraysToCells();
}

void AssignCellsToArraysGPU()
{
	dim3 dimGrid(Nx,Ny,Nz),dimBlockExt(CellExtent,CellExtent,CellExtent);

	GPU_SetFieldsToCells<<<dimGrid, dimBlockExt>>>(d_CellArray,d_Ex,d_Ey,d_Ez,d_Hx,d_Hy,d_Hz);

}


	  void AssignCellsToArrays()
	{

	     //double g;
	     for(int n = 0;n < (*AllCells).size();n++)
	     {
//	         if(n == 972)
//		 {
//		    double g = 0;
//		 }
	         Cell<Particle>  c = (*AllCells)[n];
		 c.writeAllToArrays(Jx,Jy,Jz,Rho,0);
//		 if(n == 54 || n == 108)
//		 {
//		    CheckArray(Jx,dbgJx);
//		 }
	     }
	     CheckArray(Jx, dbgJx);
	     SetPeriodicCurrents(0);
	     CheckArray(Jx, dbgJx);
	    // g = 0.0;
	}
	  void AssignArraysToCells()
	  {
	     for(int n = 0;n < (*AllCells).size();n++)
	     {

	         Cell<Particle> c = (*AllCells)[n];
//	         if(c.i == 1 &&  c.l  == 1 && c.k == 1)
//	         {
////	         	   int j = 0;
//	         }
		     c.readFieldsFromArrays(Ex,Ey,Ez,Hx,Hy,Hz);
	     }
	  }

	  void ParticleLog()
	{
	#ifndef DEBUG_PLASMA
	     return;
	#endif

	     FILE *f;
	     char  fname[100];
	     int   num = 0;

	     sprintf(fname,"particles.dat");

	     if((f = fopen(fname,"wt")) == NULL) return;

	     for(int n = 0;n < (*AllCells).size();n++)
	     {
	         Cell<Particle>  c = (*AllCells)[n];
	#ifdef GPU_PARTICLE
	   	 thrust::host_vector<Particle>  pvec_device;// = c.GetParticles();
	   	 thrust::host_vector<Particle> pvec = pvec_device;
	#else
		 thrust::host_vector<Particle>  pvec = c.GetParticles();
	#endif

		 for(int i = 0;i < pvec.size();i++)
		 {
		      Particle p = pvec[i];

		      p.Print(f,num++);
		 }

	     }

	     fclose(f);
	  }


	  void write3Darray(char *name,double *d)
	  {
	    char fname[100];
	    GPUCell<Particle> c = (*AllCells)[0];
	    FILE *f;

	    sprintf(fname,"%s_fiel3d.dat",name);

	    if((f = fopen(fname,"wt")) == NULL) return;

	    for(int i = 1;i < Nx+1;i++)
	    {
	        for(int l = 1;l < Ny+1;l++)
	        {
	            for(int k = 1;k < Nz+1;k++)
		    {
		        int n = c.getGlobalCellNumber(i,l,k);

			fprintf(f,"%15.5e %15.5e %15.5e %25.15e \n",c.getNodeX(i),c.getNodeY(l),c.getNodeZ(k),d[n]);
		    }
		}
	    }

	    fclose(f);
	}

void write3D_GPUArray(char *name,double *d_d)
{
	double *d;

#ifndef WRITE_3D_DEBUG_ARRAYS
	return;
#endif

	d = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

	cudaError_t err = cudaMemcpy(d,d_d,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyDeviceToHost);

	write3Darray(name,d);
}

void readControlPoint(FILE **f1,char *fncpy,int num,int nt,int part_read,int field_assign,
		double *ex,double *ey,double *ez,
		double *hx,double *hy,double *hz,
		double *jx,double *jy,double *jz,
		double *qx,double *qy,double *qz,
		double *x,double *y,double *z,
		double *px,double *py,double *pz
		)
{
	char fn[100],fn_next[100];
	FILE *f;

	sprintf(fn,"mumu%03d%08d.dat",num,nt);
	strcpy(fncpy,fn);
	sprintf(fn_next,"mumu%03d%05d.dat",num,nt+1);
	if((f = fopen(fn,"rb")) == NULL) return;
	if(part_read)
	{
	   *f1 = f;
	}

	readFortranBinaryArray(f,ex);
	readFortranBinaryArray(f,ey);
	readFortranBinaryArray(f,ez);
	readFortranBinaryArray(f,hx);
	readFortranBinaryArray(f,hy);
	readFortranBinaryArray(f,hz);
	readFortranBinaryArray(f,jx);
	readFortranBinaryArray(f,jy);
	readFortranBinaryArray(f,jz);

	readFortranBinaryArray(f,qx);
	readFortranBinaryArray(f,qy);
	readFortranBinaryArray(f,qz);

//	readFortranBinaryArray(f,x);
//	readFortranBinaryArray(f,y);
//	readFortranBinaryArray(f,z);
//	readFortranBinaryArray(f,px);
//	readFortranBinaryArray(f,py);
//	readFortranBinaryArray(f,pz);



//	fclose(f);

	//if(part_read == 1) readParticles(fn,fn_next);

	if(field_assign == 1) AssignArraysToCells();

}

double checkControlMatrix(char *wh,int nt,char *name, double *d_m)
{
	double /*t_ex,t_ey,t_ez,t_hx,t_hy,t_hz,*/t_jx,t_jy,t_jz;
	char fn[100];//,fn_next[100];
	FILE *f;

#ifndef CHECK_CONTROL_MATRIX
	return 0.0;
#endif

	sprintf(fn,"wcmx_%4s_%08d_%2s.dat",wh,nt,name);
	if((f = fopen(fn,"rb")) == NULL) return -1.0;

	readFortranBinaryArray(f,dbgJx);

	t_jx = checkGPUArray(dbgJx,d_m);

    return t_jx;
}


void checkCurrentControlPoint(int j,int nt)
{
	 double /*t_ex,t_ey,t_ez,t_hx,t_hy,t_hz,*/t_jx,t_jy,t_jz;
		char fn[100];//,fn_next[100];
		FILE *f;

		sprintf(fn,"curr%03d%05d.dat",nt,j);
		if((f = fopen(fn,"rb")) == NULL) return;

		readFortranBinaryArray(f,dbgJx);
		readFortranBinaryArray(f,dbgJy);
		readFortranBinaryArray(f,dbgJz);

		int size = (Nx+2)*(Ny+2)*(Nz+2);

	 t_jx = CheckArraySilent(Jx,dbgJx,size);
	 t_jy = CheckArraySilent(Jy,dbgJy,size);
	 t_jz = CheckArraySilent(Jz,dbgJz,size);

     printf("Jx %15.5e Jy %15.5e Jz %15.5e \n",t_jx,t_jy,t_jz);
}

void checkControlPoint(int num,int nt,int check_part)
{
	 double t_ex,t_ey,t_ez,t_hx,t_hy,t_hz,t_jx,t_jy,t_jz;
	 double t_qx,t_qy,t_qz,t_njx,t_njy,t_njz;

	 if((nt != TOTAL_STEPS) && (num == 600))
	 {
#ifndef CONTROL_POINT_CHECK
	     return;
#endif
	 }

	 FILE *f;
	 char fn_copy[100];
	 struct sysinfo info;

	 memory_monitor("checkControlPoint1",nt);

	 if(nt % FORTRAN_NUMBER_OF_SMALL_STEPS != 0) return;

	 memory_monitor("checkControlPoint2",nt);

	 readControlPoint(&f,fn_copy,num,nt,1,0,dbgEx,dbgEy,dbgEz,dbgHx,dbgHy,dbgHz,dbgJx,dbgJy,dbgJz,dbg_Qx,dbg_Qy,dbg_Qz,
                     dbg_x,dbg_y,dbg_z,dbg_px,dbg_py,dbg_pz);

	 memory_monitor("checkControlPoint3",nt);

	 int size = (Nx+2)*(Ny+2)*(Nz+2);


	 t_ex = CheckArraySilent(Ex,dbgEx,size);
	 t_ey = CheckArraySilent(Ey,dbgEy,size);
	 t_ez = CheckArraySilent(Ez,dbgEz,size);
	 t_hx = CheckArraySilent(Hx,dbgHx,size);
	 t_hy = CheckArraySilent(Hy,dbgHy,size);
	 t_hz = CheckArraySilent(Hz,dbgHz,size);
	 t_jx = CheckArraySilent(Jx,dbgJx,size);
	 t_jy = CheckArraySilent(Jy,dbgJy,size);
	 t_jz = CheckArraySilent(Jz,dbgJz,size);

	 memory_monitor("checkControlPoint4",nt);

	 t_ex = CheckGPUArraySilent(dbgEx,d_Ex);
	 t_ey = CheckGPUArraySilent(dbgEy,d_Ey);
	 t_ez = CheckGPUArraySilent(dbgEz,d_Ez);
	 t_hx = CheckGPUArraySilent(dbgHx,d_Hx);
	 t_hy = CheckGPUArraySilent(dbgHy,d_Hy);
	 t_hz = CheckGPUArraySilent(dbgHz,d_Hz);

	 t_qx = CheckGPUArraySilent(dbg_Qx,d_Qx);
	 t_qy = CheckGPUArraySilent(dbg_Qy,d_Qy);
	 t_qz = CheckGPUArraySilent(dbg_Qz,d_Qz);

	 t_jx = CheckGPUArraySilent(dbgJx,d_Jx);
	 t_jy = CheckGPUArraySilent(dbgJy,d_Jy);
	 t_jz = CheckGPUArraySilent(dbgJz,d_Jz);

	 t_njx = CheckGPUArraySilent(dbgJx,d_Jx);
	 t_njy = CheckGPUArraySilent(dbgJy,d_Jy);
	 t_njz = CheckGPUArraySilent(dbgJz,d_Jz);

	 memory_monitor("checkControlPoint5",nt);

	 double t_cmp_jx = checkGPUArray(dbgJx,d_Jx,"Jx","step",nt);
	 double t_cmp_jy = checkGPUArray(dbgJy,d_Jy,"Jy","step",nt);
	 double t_cmp_jz = checkGPUArray(dbgJz,d_Jz,"Jz","step",nt);

#ifdef CONTROL_DIFF_GPU_PRINTS
     printf("GPU: Ex %15.5e Ey %15.5e Ez %15.5e \n",t_ex,t_ey,t_ez);
     printf("GPU: Hx %15.5e Hy %15.5e Ez %15.5e \n",t_hx,t_hy,t_hz);
     printf("GPU: Jx %15.5e Jy %15.5e Jz %15.5e \n",t_jx,t_jy,t_jz);
     printf("GPU compare : Jx %15.5e Jy %15.5e Jz %15.5e \n",t_cmp_jx,t_cmp_jy,t_cmp_jz);
#endif

     memory_monitor("checkControlPoint6",nt);

     double cp = checkControlPointParticles(num,f,fn_copy,nt);

     f_prec_report = fopen("control_points.dat","at");
     fprintf(f_prec_report,"nt %5d num %3d Ex %15.5e Ey %15.5e Ez %15.5e Hx %15.5e Hy %15.5e Hz %15.5e Jx %15.5e Jy %15.5e Jz %15.5e Qx %15.5e Qy %15.5e Qz %15.5e particles %15.5e\n",
    		 nt,num,
    		 t_ex,t_ey,t_ez,
    		 t_hx,t_hy,t_hz,
    		 t_jx,t_jy,t_jz,
    		 t_qx,t_qy,t_qz,
    		 cp
    		 );
     fclose(f_prec_report);

     memory_monitor("checkControlPoint7",nt);

     fclose(f);
}

	int readFortranBinaryArray(FILE *f, double* d)
	{
//	    char str[100];
	    Cell<Particle>  c = (*AllCells)[0];
	    int t,err;//,n;
//	    double t0;


	    //sprintf(fname,"%s_fiel3d.dat",name);
	    fread(&t,sizeof(int),1,f);
	     if((err = ferror(f)) != 0)
	    	 {
	    	 	 return err ;
	    	 }

	    fread(d,1,t,f);
	     if((err = ferror(f)) != 0)
	    	 {
	    	 	 return err ;
	    	 }

//	    t0 = d[269];
//	    t0 = d[270];
	    fread(&t,sizeof(int),1,f);
	     if((err = ferror(f)) != 0)
	    	 {
	    	 	 return err ;
	    	 }



#ifdef READ_DEBUG_PRINTS
	    for(int i = 1; i <= Nx+2;i++)
	    {
	    	for(int l = 1; l <= Ny+2;l++)
	    	{
	    		for(int k = 1;k <= Nz+2;k++)
	    		{
	    			n = c.getFortranCellNumber(i,l,k);
	    			printf("%5d %5d %5d %25.15e \n",i,l,k,d[n]);
	    		}
	    	}
	    }
#endif

	    	    return t;
	}




	int readDebugArray(char* name, double* d,int nt,int col)
	{
		char dfile[100];

		if(!strncmp(name+2,"lg",2))
		{
			sprintf(dfile,"%s%03d.dat",name,nt);
			if(name[0] == 'e')
			{
			   read3DarrayLog(dfile, d,50,col);
			}
			else
			{
				read3DarrayLog(dfile,d,40,col);
			}
		}
		else
		{
			sprintf(dfile,"%s%06d.dat",name,nt);
			read3Darray(dfile,d);
		}
		return 0;
	}


	  void read3DarrayModified(char* name, double* d,double a)
	  {
	      char str[100];
	      Cell<Particle>  c = (*AllCells)[0];
	      FILE *f;

	      //sprintf(fname,"%s_fiel3d.dat",name);

	      if((f = fopen(name,"rt")) == NULL) return;

	      while(fgets(str,100,f) != NULL)
	      {
	            int i = atoi(str) - 1;
	            int l = atoi(str + 10) - 1;
	            int k = atoi(str + 20) - 1;
	  	  double t = atof(str + 30);
	  	  int n = c.getGlobalCellNumber(i,l,k);
	  	   t = a*(i + 100*l + 10000*k);
	  	   d[n] = t;
	      }

	      fclose(f);

	  }



	  void write3DcellArray(char *name,int code)
	  {
	    char fname[100];
	    Cell<Particle> & c = AllCells[0];
	    FILE *f;
#ifndef WRITE_3D_CELL_ARRAY
	    return;
#endif

	    sprintf(fname,"%03d_cells.dat",code,name);

	    if((f = fopen(fname,"wt")) == NULL) return;

	    for(int i = 1;i < Nx+1;i++)
	    {
	        for(int l = 1;l < Ny+1;l++)
	        {
	            for(int k = 1;k < Nz+1;k++)
		    {
		        int n = c.getGlobalCellNumber(i,l,k);
			Cell<Particle> & cc = AllCells[n];

			fprintf(f,"%15.5e %15.5e %15.5e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",c.getNodeX(i),c.getNodeY(l),c.getNodeZ(k),
				cc.getCoreCell(code,1,1,1),
				cc.getCoreCell(code,1,1,2),
				cc.getCoreCell(code,1,2,1),
				cc.getCoreCell(code,1,2,2),
				cc.getCoreCell(code,2,1,1),
				cc.getCoreCell(code,2,1,2),
				cc.getCoreCell(code,2,2,1),
				cc.getCoreCell(code,2,2,2)
			);
		    }
		}
	    }

	    fclose(f);
	}



void copyCellCurrentsToDevice(CellDouble *d_jx,CellDouble *d_jy,CellDouble *d_jz,
		                      CellDouble *h_jx,CellDouble *h_jy,CellDouble *h_jz)
{
	cudaError_t err;

 	err = cudaMemcpy(d_jx,h_jx,sizeof(CellDouble),cudaMemcpyHostToDevice);
 	if(err != cudaSuccess)
 	        {
 	         	printf("1copyCellCurrentsToDevice err %d %s \n",err,cudaGetErrorString(err));
 	       	exit(0);
 	        }
 	err = cudaMemcpy(d_jy,h_jy,sizeof(CellDouble),cudaMemcpyHostToDevice);
 	if(err != cudaSuccess)
 	        {
 	         	printf("2copyCellCurrentsToDevice err %d %s \n",err,cudaGetErrorString(err));
 	       	exit(0);
 	        }
 	err = cudaMemcpy(d_jz,h_jz,sizeof(CellDouble),cudaMemcpyHostToDevice);
 	if(err != cudaSuccess)
 	        {
 	         	printf("3copyCellCurrentsToDevice err %d %s \n",err,cudaGetErrorString(err));
 	       	exit(0);
 	        }

}


double CheckArray	(double* a, double* dbg_a,FILE *f)
	{
	    Cell<Particle> c = (*AllCells)[0];
	    int wrong = 0;
	    double diff = 0.0;



//#ifdef CHECK_ARRAY_DETAIL_PRINTS
	    fprintf(f,"begin array checking=============================\n");
//#endif
	    for(int n = 0;n < (Nx + 2)*(Ny + 2)*(Nz + 2);n++)
	    {
//	        double t  = a[n];
//		    double dt = dbg_a[n];
            diff += pow(a[n] - dbg_a[n],2.0);

	        if(fabs(a[n] - dbg_a[n]) > TOLERANCE)
		    {

		       int3 i = c.getCellTripletNumber(n);
#ifdef CHECK_ARRAY_DETAIL_PRINTS
		       fprintf(f,"n %5d i %3d l %3d k %3d %15.5e dbg %15.5e diff %15.5e wrong %10d \n",
				   n,i.x+1,i.y+1,i.z+1,a[n],dbg_a[n],fabs(a[n] - dbg_a[n]),wrong++);
#endif
     		}
	    }
#ifdef CHECK_ARRAY_DETAIL_PRINTS
	    fprintf(f,"  end array checking============================= %.4f less than %15.5e diff %15.5e \n",
	    		(1.0-((double)wrong/((Nx + 2)*(Ny + 2)*(Nz + 2)))),TOLERANCE,
	    		pow(diff/((Nx + 2)*(Ny + 2)*(Nz + 2)),0.5)
	    	  );
#endif

	    return (1.0-((double)wrong/((Nx + 2)*(Ny + 2)*(Nz + 2))));
	}

double CheckArray	(double* a, double* dbg_a)
	{
	    Cell<Particle> c = (*AllCells)[0];
	    int wrong = 0;
	    double diff = 0.0;
#ifdef CHECK_ARRAY_DETAIL_PRINTS
	    puts("begin array checking2=============================");
#endif
	    for(int n = 0;n < (Nx + 2)*(Ny + 2)*(Nz + 2);n++)
	    {
//	        double t  = a[n];
//		    double dt = dbg_a[n];
            diff += pow(a[n] - dbg_a[n],2.0);

	        if(fabs(a[n] - dbg_a[n]) > TOLERANCE)
		    {

		       int3 i = c.getCellTripletNumber(n);
#ifdef CHECK_ARRAY_DETAIL_PRINTS
		       printf("n %5d i %3d l %3d k %3d %15.5e dbg %15.5e diff %15.5e wrong %10d \n",
				   n,i.x+1,i.y+1,i.z+1,a[n],dbg_a[n],fabs(a[n] - dbg_a[n]),wrong++);
#endif
     		}
	    }
#ifdef CHECK_ARRAY_DETAIL_PRINTS
	    printf("  end array checking============================= %.4f less than %15.5e diff %15.5e \n",
	    		(1.0-((double)wrong/((Nx + 2)*(Ny + 2)*(Nz + 2)))),TOLERANCE,
	    		pow(diff/((Nx + 2)*(Ny + 2)*(Nz + 2)),0.5)
	    	  );
#endif

	    return (1.0-((double)wrong/((Nx + 2)*(Ny + 2)*(Nz + 2))));
	}




double CheckGPUArraySilent	(double* a, double* d_a)
	{
	    static double *t;
	    static int f = 1;
	    cudaError_t err;


	    if(f == 1)
	    {
	    	 t = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	    	 f = 0;
	    }
	    cudaMemcpy(t,d_a,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyDeviceToHost);
	    err = cudaGetLastError();
	    if(err != cudaSuccess)
	            {
	             	printf("CheckArraySilent err %d %s \n",err,cudaGetErrorString(err));
	            	exit(0);
	            }


	   return CheckArraySilent(a,t,(Nx+2)*(Ny+2)*(Nz+2));
	}




	int CheckValue(double *a, double *dbg_a, int n)
	{
	    Cell<Particle>  c = (*AllCells)[0];
//	    double t  = a[n];
//	    double dt = dbg_a[n];

	    if(fabs(a[n] - dbg_a[n]) > TOLERANCE)
	    {

	       int3 i = c.getCellTripletNumber(n);
#ifdef CHECK_VALUE_DEBUG_PRINTS
	       printf("value n %5d i %3d l %3d k %3d %15.5e dbg %1.5e diff %15.5e \n",n,i.x,i.y,i.z,a[n],dbg_a[n],fabs(a[n] - dbg_a[n]));
#endif

	       return 0;

	    }

	    return 1;
	}


	void read3DarrayLog(char* name, double* d, int offset, int col)
	{
	    char str[1000];
	    Cell<Particle> c = (*AllCells)[0];
	    FILE *f;

	    //sprintf(fname,"%s_fiel3d.dat",name);

	    if((f = fopen(name,"rt")) == NULL) return;

	    while(fgets(str,1000,f) != NULL)
	    {
	          //str += offset;

	          int i = atoi(str + offset)      - 1;
	          int l = atoi(str + offset + 5)  - 1;
	          int k = atoi(str + offset + 10) - 1;
//	          if(i == 1 && l == 1 && k == 1 && col == 0)
//	          {
//	        	  int qq = 0;
//	          }
		  double t = atof(str + offset + 15 + col*25);
		  int n = c.getGlobalCellNumber(i,l,k);
		  d[n] = t;
#ifdef READ_ARRAY_LOG_PRINTS
		  printf("%d %d %5d %5d %15.5e \n",i,l,k,n,t);
#endif
	    }

	    fclose(f);

	}

	void read3Darray(char* name, double* d)
	{
	  char str[100];
	  Cell<Particle>  c = (*AllCells)[0];
	  FILE *f;

	  //sprintf(fname,"%s_fiel3d.dat",name);

	  if((f = fopen(name,"rt")) == NULL) return;

	  while(fgets(str,100,f) != NULL)
	  {
	        int i = atoi(str);
	        int l = atoi(str + 10);
	        int k = atoi(str + 20);
		  double t = atof(str + 30);
		  //int n = c.getGlobalCellNumber(i,l,k);
		  int i1,l1,k1,n = c.getFortranCellNumber(i,l,k);
		  c.getFortranCellTriplet(n,&i1,&l1,&k1);
		  d[n] = t;
	  }

	  fclose(f);

	}


	void read3Darray(string name, double* d)
	{
	  char str[100];

	  strcpy(str,name.c_str());

	  read3Darray(str,d);

	}

	void InitParticlesNext(char* fname, thrust::host_vector< Particle >& vp)
	{
	    FILE *f;
	     char str[1000];
	     //double3 x;
	     int n = 0;

	     if((f = fopen(fname,"rt")) == NULL) return;

	     while(fgets(str,1000,f) != NULL)
	     {
//	          x.x   = atof(str);
//	          x.y   = atof(str + 25);
//	          x.z   = atof(str + 50);

	          //px  = atof(str + 75);
	          //py  = atof(str + 100);
	          //pz  = atof(str + 125);
	          //m   = atof(str + 150);
	          //q_m = atof(str + 175);

		  Particle & p = vp[n++];
		  //p.SetXnext(x);
	     }
	}

	void InitBinaryParticlesNext(char *fn, thrust::host_vector< Particle >& vp)
	{
	     FILE *f;
//	     char str[1000];
//	     double3 x;
	     double *buf,q_m,m,tp;
	     int n = 0,t;

	     buf = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

	     if((f = fopen(fn,"rb")) == NULL) return;
	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);

	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);

	     readFortranBinaryArray(f,buf);
	     	     readFortranBinaryArray(f,buf);
	     	     readFortranBinaryArray(f,buf);

	     	     fread(&t,sizeof(int),1,f);
	     	     fread(&tp,sizeof(double),1,f);
	     	     total_particles = (int)tp;
	     	     fread(&q_m,sizeof(double),1,f);
	     	     fread(&m,sizeof(double),1,f);
	     	     fread(&t,sizeof(int),1,f);
	              dbg_x = (double *)malloc(sizeof(double)*total_particles);
	              dbg_y = (double *)malloc(sizeof(double)*total_particles);
	              dbg_z = (double *)malloc(sizeof(double)*total_particles);
	              dbg_px = (double *)malloc(sizeof(double)*total_particles);
	              dbg_py = (double *)malloc(sizeof(double)*total_particles);
	              dbg_pz = (double *)malloc(sizeof(double)*total_particles);

        dbg_x = (double *)malloc(sizeof(double)*total_particles);
        dbg_y = (double *)malloc(sizeof(double)*total_particles);
        dbg_z = (double *)malloc(sizeof(double)*total_particles);

	 	readFortranBinaryArray(f,dbg_x);
	 	readFortranBinaryArray(f,dbg_y);
	 	readFortranBinaryArray(f,dbg_z);

	     for(int i = 0;i< total_particles;i++)
	     {
//	          x.x   = atof(str);
//	          x.y   = atof(str + 25);
//	          x.z   = atof(str + 50);

	          //px  = atof(str + 75);
	          //py  = atof(str + 100);
	          //pz  = atof(str + 125);
	          //m   = atof(str + 150);
	          //q_m = atof(str + 175);

		  Particle & p = vp[n++];
		//  p.SetXnext(x);
	     }
	}

	int PeriodicCurrentBoundaries(double* E, int dirE,int dir, int start1, int end1, int start2, int end2)
	{
	      Cell<Particle>  c = (*AllCells)[0];

	      int N = getBoundaryLimit(dir);

	      for(int k = start2;k <= end2;k++)
	      {
	    	  for(int i = start1;i <= end1;i++)
		  {
		      int n1    = c.getGlobalBoundaryCellNumber(i,k,dir,1);
		      int n_Nm1 = c.getGlobalBoundaryCellNumber(i,k,dir,N-1);
	#ifdef DEBUG_PLASMA
		      int3 n1_3 = c.getCellTripletNumber(n1);
		      int3 n_Nm1_3 = c.getCellTripletNumber(n_Nm1);
	#endif
		      if(dir != dirE)
		      {
		         E[n1] += E[n_Nm1];


		      }
		      if(dir != 1 || dirE != 1)
		      {
		         E[n_Nm1] =  E[n1];
		      }
		      int n_Nm2 = c.getGlobalBoundaryCellNumber(i,k,dir,N-2);
		      int n0    = c.getGlobalBoundaryCellNumber(i,k,dir,0);
	#ifdef DEBUG_PLASMA
		      int3 n0_3 = c.getCellTripletNumber(n0);
		      int3 n_Nm2_3 = c.getCellTripletNumber(n_Nm2);
	#endif

		      E[n0] += E[n_Nm2];

		      E[n_Nm2] = E[n0];


		      //   E[n0] = E[n_Nm2];
		      //   E[n_Nm1] = E[n1];


		     // }
		  }
	      }
	      return 0;
	}

	void ClearAllParticles(void )
	{
	    for(int n = 0;n < (*AllCells).size();n++)
	    {
	        Cell<Particle> c = (*AllCells)[n];

		c.ClearParticles();

	    }
	}





	public:

	  GPUPlasma(int nx,int ny,int nz,double lx,double ly,double lz,double ni1,int n_per_cell1,double q_m,double TAU)
	   {
	     Nx = nx;
	     Ny = ny;
	     Nz = nz;

	     Lx = lx;
	     Ly = ly;
	     Lz = lz;

	     ni = ni1;

	     n_per_cell = n_per_cell1;
	     ion_q_m    = q_m;
	     tau        = TAU;
	   }

	   int initControlPointFile()
	   {
		   f_prec_report = fopen("control_points.dat","wt");
		   fclose(f_prec_report);

		   return 0;
	   }


	   int initMeshArrays()
	   {
		   initControlPointFile();

		   Alloc();

		   Cell<Particle> c000;

		   InitCells();
		   c000 = (*AllCells)[0];

		   InitFields();
		   c000 = (*AllCells)[0];
		   InitCurrents();

		   return 0;
	   }

	   int copyCellsWithParticlesToGPU()
	   {
		   Cell<Particle> c000 = (*AllCells)[0];
		   magf = 1;

		   int size = (Nx+2)*(Ny+2)*(Nz+2);

		   cp = (Cell<Particle> **)malloc(size*sizeof(Cell<Particle> *));

		   for(int i = 0;i < size;i++)
		   {
		     	Cell<Particle> c,*d_c;
		   	   	// 	z0 = h_CellArray[i];
		   	    d_c = c.allocateCopyCellFromDevice();

		   	    cp[i] = d_c;
		   }
	   }


	   virtual void InitializeCPU()
	   {
		  std::vector<Particle> ion_vp,el_vp,beam_vp;

	      initMeshArrays();

	      LoadTestData(START_STEP_NUMBER,START_STEP_NUMBER, ion_vp,el_vp,beam_vp);

	      addAllParticleListsToCells(ion_vp,el_vp,beam_vp);

	      AssignArraysToCells();


	   }

	   void Free();

	   virtual void SetInitialConditions(){}

	   virtual void ParticleSort(){}

	   //void ApplyToAllParticles(cell_work_function cwf);

	//   virtual void ComputeField(int nt)
	//{
	//   double t27 = Hy[27];
	//
	//#ifdef UNITY_ELECTRIC_FIELD
	//     for(int i = 0;i < (Nx+2)*(Ny+2)*(Nz+2);i++)
	//     {
	//         Ex[i] = 1.0;
	//         Ey[i] = 1.0;
	//         Ez[i] = 1.0;
	//     }
	//#else
	//        double t271 = Hy[27];
	//
	//     CheckArray(Hx,dbgHx);
	//     CheckArray(Hy,dbgHy);
	//     CheckArray(Hz,dbgHz);
	//     CheckArray(Ex,dbgEx);
	//     CheckArray(Ey,dbgEy);
	//     CheckArray(Ez,dbgEz);
	//
	//     emh1(Qx,Qy,Qz,Hx,Hy,Hz,nt,Ex,Ey,Ez);
	//
	//     eme(Ex,Ey,Ez,nt,Hx,Hy,Hz,npJx,npJy,npJz);
	//     CheckArray(Ex,dbgEx);
	//     CheckArray(Ey,dbgEy);
	//     CheckArray(Ez,dbgEz);
	//
	//     emh2(Hx,Hy,Hz,nt+1,Qx,Qy,Qz);
	//
	//
	//     eme(Ex,Ey,Ez,nt+1,Hx,Hy,Hz,npJx,npJy,npJz);
	//
	//     CheckArray(Hx,dbgHx);
	//     CheckArray(Hy,dbgHy);
	//     CheckArray(Hz,dbgHz);
	//     puts("H");
	//     //exit(0);
	//
	////for(int i = 0;i < (Nx+2)*(Ny+2)*(Nz+2);i++)
	////     {
	////         Ex[i] = 0.0;
	////         Ey[i] = 0.0;
	////         Ez[i] = 0.0;
	////     }
	////    LoadTestData(3);
	//     eme(Ex,Ey,Ez,nt,Hx,Hy,Hz,npJx,npJy,npJz);
	//     CheckArray(Ex,dbgEx);
	//     CheckArray(Ey,dbgEy);
	//     CheckArray(Ez,dbgEz);
	//#endif
	//}

	//    void  ComputeField_FirstHalfStep(
	//		   double *locEx,double *locEy,double *locEz,
	//		   double nt,
	//		   double *locHx,double *locHy,double *locHz,
	//		   double *loc_npJx,double *loc_npJy,double *loc_npJz)
	//{
	//	 double *locQx,*locQy,*locQz;
	//	 static int first = 0;
	//
	//	 if(first == 0)
	//	 {
	//		 locQx = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	//		 locQy = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	//		 locQz = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	//
	//		 first = 1;
	//	 }
	//     emh1(locQx,locQy,locQz,locHx,locHy,locHz,nt,locEx,locEy,locEz);
	//     CheckArray(dbg_Qx,locQx);
	//     eme(locEx,locEy,locEz,nt,locHx,locHy,locHz,loc_npJx,loc_npJy,loc_npJz);
	//     CheckArray(dbg_Qx,locQx);
	//     emh2(locHx,locHy,locHz,nt,locQx,locQy,locQz);
	//}



//	void StepAllCells()
//	{
//		int cell_sum = 0;
//
//		for(int n = 0;n < (*AllCells).size();n++)
//		{
//			int sum = 0,f;
//
//		    Cell<Particle> c = (*AllCells)[n];
//
//	        for(int i = 0; i < c.number_of_particles;i++)
//	        {
//		        f = c.Move(i);
//		        sum += f;
//	//	        c.writeAllToArrays(Jx,Jy,Jz,Rho);
//	        }
//	        if(sum != c.number_of_particles)
//	                {
//	                	int qq = 0;
//	                }
//	        cell_sum += sum == c.number_of_particles;
//
//		}
//		printf("passed %10d cells of %10d total \n",cell_sum,(*AllCells).size());
//
//		AssignCellsToArrays();
//	}

	   void ListAllParticles(int nt,char *where)
	   	{
//	   		int cell_sum = 0;
//	   		int part_number = 0;
//	   		double t_hx,t_hy,t_hz,*dbg;
	   		FILE *f;
	   		char str[200];
	   		//Cell<Particle> **cp;

#ifndef LIST_ALL_PARTICLES
	   		return;
#endif


	   		sprintf(str,"List%05d_%s.dat\0",nt,where);

	   		if((f = fopen(str,"wt")) == NULL) return;


	   		int size = (*AllCells).size();

//	   		cp = (Cell<Particle> **)malloc(size*sizeof(Cell<Particle> *));

	   		copyCells(where,nt);

	   			//h_ctrl = new Cell<Particle>;

	   		for(int i = 0;i < size;i++)
	   		{
	   		 	Cell<Particle> c = (*AllCells)[i];

   			    Particle p;
	 //   			int j;

	   			c.readParticleFromSurfaceDevice(0,&p);
	   	        //h_c.copyCellFromDevice(&d_c);
	   	        c.printFileCellParticles(f,cp[i]);
	   		}
	   		fclose(f);
	   	}

	void FortranOrder_StepAllCells(int nt)
	{
		int cell_sum = 0;
		int part_number = 0;
	//	double t_hx,t_hy,t_hz,*dbg;

		memset(Jx,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
		memset(Jy,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
		memset(Jz,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

		for(int n = 0;n < (*AllCells).size();n++)
		{
			Cell<Particle> c = (*AllCells)[n];
	        part_number += c.number_of_particles;
		}

		for(int j = 1;j <= part_number;j++)
		{
		    for(int n = 0;n < (*AllCells).size();n++)
		    {
			    int f;

		        Cell<Particle> c = (*AllCells)[n];

	            for(int i = 0; i < c.number_of_particles;i++)
	            {
	            	if(c.getFortranParticleNumber(i) != j) continue;
	            	int num = c.getFortranParticleNumber(i);
//	            	if(num == 1000 )
//	            	{
//	            		int qq = 0;
//	            	}
	            	c.SetAllCurrentsToZero();
		            f = c.Move(i);
		        //    sum += f;
		            c.writeAllToArrays(Jx,Jy,Jz,Rho,c.getFortranParticleNumber(i));
		            printf("particle number %10d \n",num);


	            }
		    }

	        //if(sum != c.number_of_particles)
		}
		printf("passed %10d cells of %10d total \n",cell_sum,(*AllCells).size());

		checkNonPeriodicCurrents(nt);

	    SetPeriodicCurrents(nt);
		CheckArray(Jx,dbgJx);
		CheckArray(Jy,dbgJy);
		CheckArray(Jz,dbgJz);

		//AssignCellsToArrays();
	}

	double TryCheckCurrent(int nt,double *npJx)
	{
		double *dbg,t_hx;//,t_hy,t_hz;



	  	dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

	  	// read magnetic field from "nt+1" exlg file - to consider emh2


//	  	readDebugArray("npjx",dbg,nt,5);
//	    t_hx = CheckArray(npJx,dbg);
//	  	readDebugArray("npjx",dbg,nt+1,5);
//	    t_hx = CheckArray(npJx,dbg);
//	    readDebugArray("exlg",dbg,nt,5);
//	    t_hx = CheckArray(npJx,dbg);
//	    readDebugArray("exlg",dbg,nt+1,5);
//	    t_hx = CheckArray(npJx,dbg);

	    return 1.0;//t_hx;
	}

	double checkNonPeriodicCurrents(int nt)
	{
//		double *dbg,t_hx,t_hy,t_hz;

		printf("CHECKING Non-periodic currents !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");

	  //	dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	  	TryCheckCurrent(nt,npJx);

	  	// read magnetic field from "nt+1" exlg file - to consider emh2
//	  	readDebugArray("npjx",dbg,nt,5);
//	    t_hx = CheckArray(npJx,dbg);
//	  	readDebugArray("npjy",dbg,nt,5);
//	  	t_hy = CheckArray(npJy,dbg);
//	  	readDebugArray("npjz",dbg,nt,5);
//	  	t_hz = CheckArray(npJz,dbg);
//
//	  	printf("Non-periodic currents %.5f \n",(t_hx+t_hy+t_hz)/3.0);

		return 1.0;//(t_hx+t_hy+t_hz)/3.0;
	}

	double checkPeriodicCurrents(int nt)
	{
		double *dbg,t_hx,t_hy,t_hz;

		printf("CHECKING periodic currents !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");

	  	dbg = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

	  	TryCheckCurrent(nt,Jx);

	  	// read magnetic field from "nt+1" exlg file - to consider emh2
//	  	readDebugArray("npjx",dbg,nt,5);
//	    t_hx = CheckArray(Jx,dbg);
//	    readDebugArray("exlg",dbg,nt+1,5);
//	    t_hx = CheckArray(Jx,dbg);
//	  	readDebugArray("npjy",dbg,nt,5);
//	  	t_hy = CheckArray(Jy,dbg);
//	  	readDebugArray("npjz",dbg,nt,5);
//	  	t_hz = CheckArray(Jz,dbg);
//
//	  	printf("Non-periodic currents %.5f \n",(t_hx+t_hy+t_hz)/3.0);

		return 1.0;//(t_hx+t_hy+t_hz)/3.0;
	}

	void printCurrentTensor(CellDouble dbg_cell_Jx,CellDouble dbg_cell_Jy,CellDouble dbg_cell_Jz,CurrentTensor t1)
	{
		double t_b,t_a,t;
		puts("Jx");
		t_b = dbg_cell_Jx.M[t1.Jx.i11][t1.Jx.i12][t1.Jx.i13];
		t   = t1.Jx.t[0];
		t_a = t_b + t;
		printf("before %15.5e t1 %15.5e after %15.5e \n",t_b,t,t_a);

		t_b = dbg_cell_Jx.M[t1.Jx.i21][t1.Jx.i22][t1.Jx.i23];
		t   = t1.Jx.t[1];
		t_a = t_b + t;
		printf("before %15.5e t1 %15.5e after %15.5e \n",t_b,t,t_a);

		t_b = dbg_cell_Jx.M[t1.Jx.i31][t1.Jx.i32][t1.Jx.i33];
		t   = t1.Jx.t[2];
		t_a = t_b + t;
		printf("before %15.5e t1 %15.5e after %15.5e \n",t_b,t,t_a);

		t_b = dbg_cell_Jx.M[t1.Jx.i41][t1.Jx.i42][t1.Jx.i43];
		t   = t1.Jx.t[3];
		t_a = t_b + t;
		printf("before %15.5e t1 %15.5e after %15.5e \n",t_b,t,t_a);

puts("Jy");
		printf("before %15.5e t1 %15.5e after %15.5ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½\n",
        dbg_cell_Jy.M[t1.Jy.i11][t1.Jy.i12][t1.Jy.i13],t1.Jy.t[0],dbg_cell_Jy.M[t1.Jy.i11][t1.Jy.i12][t1.Jy.i13]);
		printf("before %15.5e t1 %15.5e after %15.5ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½\n",
        dbg_cell_Jy.M[t1.Jy.i21][t1.Jy.i22][t1.Jy.i23],t1.Jy.t[1],dbg_cell_Jy.M[t1.Jy.i21][t1.Jy.i22][t1.Jy.i23]);
		printf("before %15.5e t1 %15.5e after %15.5ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½\n",
        dbg_cell_Jy.M[t1.Jy.i31][t1.Jy.i32][t1.Jy.i33],t1.Jy.t[2],dbg_cell_Jy.M[t1.Jy.i31][t1.Jy.i32][t1.Jy.i33]);
		printf("before %15.5e t1 %15.5e after %15.5ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½\n",
        dbg_cell_Jy.M[t1.Jy.i41][t1.Jy.i42][t1.Jy.i43],t1.Jy.t[3],dbg_cell_Jy.M[t1.Jy.i41][t1.Jy.i42][t1.Jy.i43]);
        puts("Jz");
		printf("before %15.5e t1 %15.5e after %15.5ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½\n",
        dbg_cell_Jz.M[t1.Jz.i11][t1.Jz.i12][t1.Jz.i13],t1.Jz.t[0],dbg_cell_Jz.M[t1.Jz.i11][t1.Jz.i12][t1.Jz.i13]);
		printf("before %15.5e t1 %15.5e after %15.5ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½\n",
        dbg_cell_Jz.M[t1.Jz.i21][t1.Jz.i22][t1.Jz.i23],t1.Jz.t[1],dbg_cell_Jz.M[t1.Jz.i21][t1.Jz.i22][t1.Jz.i23]);
		printf("before %15.5e t1 %15.5e after %15.5ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½\n",
        dbg_cell_Jz.M[t1.Jz.i31][t1.Jz.i32][t1.Jz.i33],t1.Jz.t[2],dbg_cell_Jz.M[t1.Jz.i31][t1.Jz.i32][t1.Jz.i33]);
		printf("before %15.5e t1 %15.5e after %15.5ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½\n",
        dbg_cell_Jz.M[t1.Jz.i41][t1.Jz.i42][t1.Jz.i43],t1.Jz.t[3],dbg_cell_Jz.M[t1.Jz.i41][t1.Jz.i42][t1.Jz.i43]);
	}

int SetCurrentArraysToZero()
{
	memset(Jx,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
    memset(Jy,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	memset(Jz,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(d_Jx,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(d_Jy,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(d_Jz,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	return 0;
}

int SetCurrentsInCellsToZero(int nt)
{
	dim3 dimGrid(Nx+2,Ny+2,Nz+2),dimBlockExt(CellExtent,CellExtent,CellExtent);
	char name[100];
	sprintf(name,"before_set_to_zero_%03d.dat",nt);

	write3D_GPUArray(name,d_Jx);

	GPU_SetAllCurrentsToZero<<<dimGrid, dimBlockExt,16000>>>(d_CellArray);

	return 0;
}

int StepAllCells_fore_diagnostic(int nt)
{
	char name[100];

	memory_monitor("CellOrder_StepAllCells3",nt);

	sprintf(name,"before_step_%03d.dat",nt);
	write3D_GPUArray(name,d_Jx);
	//			printCellCurrents(270,nt,"jx","step");
	ListAllParticles(nt,"bStepAllCells");

	return 0;
}

int StepAllCells(int nt,double mass,double q_mass)
{
	   dim3 dimGrid(Nx+2,Ny+2,Nz+2),dimBlock(512,1,1);
	   cudaDeviceSynchronize();

	   GPU_StepAllCells<<<dimGrid, dimBlock,16000>>>(d_CellArray,0,d_Jx,
	            		     		                 mass,q_mass);

	   cudaDeviceSynchronize();

	   return 0;
}

int StepAllCells_post_diagnostic(int nt)
{
	  memory_monitor("CellOrder_StepAllCells4",nt);

      ListAllParticles(nt,"aStepAllCells");
      cudaError_t err2 = cudaGetLastError();
	  char err_s[200];
      strcpy(err_s,cudaGetErrorString(err2));

      return (int)err2;
}


int WriteCurrentsFromCellsToArrays(int nt)
{
	char name[100];
	dim3 dimGrid(Nx+2,Ny+2,Nz+2);

	sprintf(name,"before_write_currents_%03d.dat",nt);
	write3D_GPUArray(name,d_Jx);

    dim3 dimExt(CellExtent,CellExtent,CellExtent);
    GPU_WriteAllCurrents<<<dimGrid, dimExt,16000>>>(d_CellArray,0,d_Jx,d_Jy,d_Jz,d_Rho);

    memory_monitor("CellOrder_StepAllCells5",nt);

    sprintf(name,"after_write_currents_%03d.dat",nt);
	write3D_GPUArray(name,d_Jx);

	memory_monitor("CellOrder_StepAllCells6",nt);

	return 0;
}

int MakeParticleList(int nt,int *stage,int *stage1,int **d_stage,int **d_stage1)
{
	dim3 dimGrid(Nx+2,Ny+2,Nz+2),dimGridOne(1,1,1),dimBlock(512,1,1),
	     dimBlockOne(1,1,1),dimBlockGrow(1,1,1),dimBlockExt(CellExtent,CellExtent,CellExtent);
	dim3 dimGridBulk(Nx,Ny,Nz);
	cudaError_t before_MakeDepartureLists,after_MakeDepartureLists,
      before_ArrangeFlights,after_ArrangeFlights;

#ifdef BALANCING_PRINTS
    before_MakeDepartureLists = cudaGetLastError();
    printf("before_MakeDepartureLists %d %s blockdim %d %d %d\n",before_MakeDepartureLists,
    cudaGetErrorString(before_MakeDepartureLists),dimGrid.x,dimGrid.y,dimGrid.z);
#endif

//    int stage[4000],stage1[4000];//,*d_stage,*d_stage1;
    cudaMalloc(d_stage,sizeof(int)*(Nx+2)*(Ny+2)*(Nz+2));

    cudaMalloc(d_stage1,sizeof(int)*(Nx+2)*(Ny+2)*(Nz+2));

    GPU_MakeDepartureLists<<<dimGrid, dimBlockOne>>>(d_CellArray,nt,*d_stage);

    after_MakeDepartureLists = cudaGetLastError();
    if(after_MakeDepartureLists != cudaSuccess)
    {
       printf("after_MakeDepartureLists %d %s\n",after_MakeDepartureLists,cudaGetErrorString(after_MakeDepartureLists));
    }

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess)
        {
           printf("MakeParticleList sync error %d %s\n",err,cudaGetErrorString(err));
        }
    err = cudaMemcpy(stage,*d_stage,sizeof(int)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyDeviceToHost);

    if(err != cudaSuccess)
    {
       printf("MakeParticleList error %d %s\n",err,cudaGetErrorString(err));
       exit(0);
    }

    return (int)err;
}

int inter_stage_diagnostic(int *stage,int nt)
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

int reallyPassParticlesToAnotherCells(int nt,int *stage1,int *d_stage1)
{
    int err,after_ArrangeFlights;
    dim3 dimGridBulk(Nx,Ny,Nz),dimBlockOne(1,1,1);
	cudaMemset(d_stage1,0,sizeof(int)*(Nx+2)*(Ny+2)*(Nz+2));


	    GPU_ArrangeFlights<<<dimGridBulk, dimBlockOne>>>(d_CellArray,nt,d_stage1);
	    after_ArrangeFlights = cudaGetLastError();

#ifdef BALANCING_PRINTS
    printf("after_ArrangeFlights %d %s\n",after_ArrangeFlights,cudaGetErrorString(after_ArrangeFlights));
            cudaDeviceSynchronize();
#endif

	err = cudaMemcpy(stage1,d_stage1,sizeof(int)*(Nx+2)*(Ny+2)*(Nz+2),cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
	{
	   puts("copy error");
	   exit(0);
	}
	ListAllParticles(nt,"aArrangeFlights");


	memory_monitor("CellOrder_StepAllCells7",nt);
	return (int)err;

	return 0;
}

int reorder_particles(int nt)
{
    int stage[4000],stage1[4000],*d_stage,*d_stage1,err;

    MakeParticleList(nt,stage,stage1,&d_stage,&d_stage1);

    inter_stage_diagnostic(stage,nt);

    err = reallyPassParticlesToAnotherCells(nt,stage1,d_stage1);

    return (int)err;
}

int Push(int nt,double mass,double q_mass)
{
	StepAllCells_fore_diagnostic(nt);

	StepAllCells(nt,mass,q_mass);

	return StepAllCells_post_diagnostic(nt);
}

int SetCurrentsToZero(int nt)
{
	SetCurrentArraysToZero();

	return SetCurrentsInCellsToZero(nt);
}


	void CellOrder_StepAllCells(int nt,double mass,double q_mass,int first)
	{
		SetCurrentsToZero(nt);

		Push(nt,mass,q_mass);

        WriteCurrentsFromCellsToArrays(nt);

        reorder_particles(nt);
	}

	void SetCurrentsFromCellsToArrays(int nt)
	{
	    memset(Jx,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	    memset(Jy,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	    memset(Jz,0,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

	    for(int n = 0;n < (*AllCells).size();n++)
	    {
//		    int sum = 0,f;

	        Cell<Particle> c = (*AllCells)[n];

	        c.writeAllToArrays(Jx,Jy,Jz,Rho,c.getFortranParticleNumber(0));

	    }

	    //if(sum != c.number_of_particles)
	    //	}

	    memcpy(npJx,Jx,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	   	memcpy(npJy,Jy,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	   	memcpy(npJz,Jz,sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));
	   	checkNonPeriodicCurrents(nt);
	    SetPeriodicCurrents(nt);
	    CheckArray(Jx,dbgJx);
	    CheckArray(Jy,dbgJy);
	    CheckArray(Jz,dbgJz);

	}

	//   virtual void Diagnose(){ puts("Plasma");}

	   void virtual Step()
	{
	    // double t = dbgJx[0];
	  //   int j;
	     Particle p;

	     //ComputeField();
		   Cell<Particle> & c000 = (*AllCells)[0];
	    // AssignArraysToCells();

	     StepAllCells();

	     puts("particles moved!!!");
	     //return;
	     exit(0);
	/*
	     c000 = (*AllCells)[0];
	     AssignCellsToArrays();

	     for(int n = 0;n < (*AllCells).size();n++)
	     {
	         Cell<Particle>  c = (*AllCells)[n];
		     thrust::host_vector<Particle> vecp = c.getFlyList();
		     if(vecp.size() > 0)
		     {
		        int q = 0;
		        for(j = 0;j < vecp.size();j++)
		        {
		        	p = vecp[j];
		        	printf("cell %5d particle %5d %10.3e \n",n,j,p.x);
		        }
		     }
		     Distribute(vecp);
	     }
	     c000 = (*AllCells)[0];
	   //  LoadTestData(2);

	     ComputeField();

	#ifdef DEBUG_PLASMA
	     CheckArray(Jx, dbgJx);
	#endif

	     ParticleLog();*/
	}

double checkControlPointParticlesOneSort(int check_point_num,FILE *f,GPUCell<Particle> **copy_cells,int nt,int sort)
{

    double t = 0.0;
    int size = 1;
#ifdef CPU_DEBUG_RUN
    double q_m,m;
    struct sysinfo info;

    memory_monitor("checkControlPointParticlesOneSort",nt);

  //  double x,y,z,px,pz,q_m,*buf,tp,m;
    //double dbg_x,dbg_y,dbg_z,dbg_px,dbg_py,dbg_pz;

    Cell<Particle> c0 = (*AllCells)[0];
    //int pn_min/*,pn_ave,pn_max*/;
    if(check_point_num == 50)
    {
    	int qq = 0;
    }
    total_particles = readBinaryParticleArraysOneSort(f,&dbg_x,&dbg_y,&dbg_z,
   		                                             &dbg_px,&dbg_py,&dbg_pz,&q_m,&m,nt,sort);
    memory_monitor("checkControlPointParticlesOneSort2",nt);

    size = (*AllCells).size();

   	for(int i = 0;i < size;i++)
   	{
   	 	GPUCell<Particle> c = *(copy_cells[i]);

#ifdef checkControlPointParticles_PRINT
             printf("cell %d particles %20d \n",i,c.number_of_particles);
#endif

   	 	t += c.checkCellParticles(check_point_num,dbg_x,dbg_y,dbg_z,dbg_px,dbg_py,dbg_pz,q_m,m);
//   	 	if(t < 1.0)
//   	 	{
//   	 	   t += c.checkCellParticles(check_point_num,dbg_x,dbg_y,dbg_z,dbg_px,dbg_py,dbg_pz,q_m,m);
//   	 	}
   	}
   	memory_monitor("checkControlPointParticlesOneSort3",nt);

   	free(dbg_x);
   	free(dbg_y);
   	free(dbg_z);

   	free(dbg_px);
   	free(dbg_py);
   	free(dbg_pz);
   	memory_monitor("checkControlPointParticlesOneSort4",nt);
#endif
	return t/size;
}

double checkControlPointParticles(int check_point_num,FILE *f,char *fname,int nt)
{
	double te = 0.0,ti = 0.0,tb = 0.0;
	struct sysinfo info;
#ifdef CPU_DEBUG_RUN
 //   Cell<Particle> **cp;

	int size = (*AllCells).size();
//  разобраться, как делается девайсрвый массив указателей на ячейки и сделать его копию здесь
//	cp = (GPUCell<Particle> **)malloc(size*sizeof(GPUCell<Particle> *));

//	copyCells(h_CellArray);

	char where[100];
	sprintf(where, "checkpoint%03d",check_point_num);
	copyCells(where,nt);

	//checkParticleNumbers(cp,check_point_num);

#ifdef FREE_RAM_MONITOR
	sysinfo(&info);
#ifdef checkControlPointParticles_PRINTS
	printf("checkControlPointParticles %u \n",info.freeram/1024/1024);
#endif
#endif


//	if(check_point_num == 100)
//		{
//				int qq = 0;
//			//	tb  = checkControlPointParticlesOneSort(check_point_num,f,cp);
//		}
	GPUCell<Particle> c = *(cp[141]);
#ifdef checkControlPointParticles_PRINTS
	printf("checkControlPointParticlesOneSort cell 141 particles %20d \n",c.number_of_particles);
#endif

#ifdef FREE_RAM_MONITOR
	sysinfo(&info);
#ifdef checkControlPointParticles_PRINTS
	printf("checkControlPointParticles0.9 %u \n",info.freeram/1024/1024);
#endif
#endif

	ti  = checkControlPointParticlesOneSort(check_point_num,f,cp,nt,ION);
//	printf("IONS\n");
#ifdef FREE_RAM_MONITOR
	sysinfo(&info);
#ifdef checkControlPointParticles_PRINTS
	printf("checkControlPointParticles1 %u \n",info.freeram/1024/1024);
#endif
#endif

	te  = checkControlPointParticlesOneSort(check_point_num,f,cp,nt,PLASMA_ELECTRON);
//	printf("ELECTRONS\n");

#ifdef FREE_RAM_MONITOR
	sysinfo(&info);
#ifdef checkControlPointParticles_PRINTS
    printf("checkControlPointParticles1.5 %u \n",info.freeram/1024/1024);
#endif
#endif

	tb  = checkControlPointParticlesOneSort(check_point_num,f,cp,nt,BEAM_ELECTRON);
//	printf("BEAM\n");

#ifdef FREE_RAM_MONITOR
	sysinfo(&info);
#ifdef checkControlPointParticles_PRINTS
	printf("checkControlPointParticles2 %u \n",info.freeram/1024/1024);
#endif
#endif


//	if(tb < 1.0 && check_point_num == 100)
//	{
//		int qq = 0;
//		tb  = checkControlPointParticlesOneSort(check_point_num,f,cp);
//	}

//    printf("checkpoint %s electrons %e ions %e beam %e \n",fname,te,ti,tb);

#endif

//    freeCellCopies(cp);
    memory_monitor("after_free",nt);
	return (te+ti+tb)/3.0;
}

int readControlFile(int nt)
{


#ifndef ATTRIBUTES_CHECK
	return 0;
#else
	FILE *f;
	char fname[100];
	static int first = 1;
	int size;//,jmp1;

	sprintf(fname,"ctrl%05d",nt);

	if((f = fopen(fname,"rb")) == NULL)
		{
		  puts("no ini-file");
		  exit(0);
		}

	fread(&size,sizeof(int),1,f);
	fread(&ami,sizeof(double),1,f);
	fread(&amf,sizeof(double),1,f);
	fread(&amb,sizeof(double),1,f);
	fread(&size,sizeof(int),1,f);

	fread(&size,sizeof(int),1,f);

	if(first == 1)
	{
		first = 0;
        ctrlParticles = (double *)malloc(size);
#ifdef ATTRIBUTES_CHECK
        memset(ctrlParticles,0,size);
        cudaMalloc(&d_ctrlParticles,size);
        cudaMemset(d_ctrlParticles,0,size);
        size_ctrlParticles = size;
#endif
	}
	fread(ctrlParticles,1,size,f);


	jmp = size/sizeof(double)/PARTICLE_ATTRIBUTES/3;

	//double x,y,z;
	//int pos;

	//pos = ParticleAttributePositionFortran(jmp,1,ION,1);
	//x = ctrlParticles[pos];
	//printf("x %s \n",FortranExpWrite(x));

	//pos = ParticleAttributePositionFortran(jmp,1,ION,2);

	//y = ctrlParticles[pos];
	//pos = ParticleAttributePositionFortran(jmp,1,ION,3);

	//z = ctrlParticles[pos];

	return 0;
#endif
}

int checkDoublePrecisionIdentity(double a,double b)
{
	char as[50],bs[50];
	int point_pos,i;

	if(fabs(a-b) > PARTICLE_TOLERANCE) return 0;

	sprintf(as,"%25.15e",a);
	sprintf(bs,"%25.15e",b);

	for(i = 0;i < strlen(as);i++)
	{
		if(as[i] == bs[i] && as[i] == '.')
		{
			point_pos = i;
		}

		if(as[i] != bs[i]) break;
	}


	return (i - point_pos);
}






int checkParticleNumbers(GPUCell<Particle> ** h_cp,int num)
{
	int *d_numbers,*h_numbers,size;

	if(num >= 270) return -1;

	size = (*AllCells).size();

	h_numbers = (int *)malloc(size*sizeof(int));
	cudaMalloc(&d_numbers,size*sizeof(int));

	GPU_GetCellNumbers<<<(Nx+2)*(Ny+2)*(Nz+2),1>>>(d_CellArray,d_numbers);

	cudaError_t err = cudaMemcpy(h_numbers,d_numbers,size*sizeof(int),cudaMemcpyDeviceToHost);


	for(int i = 0;i < (*AllCells).size();i++)
	{
		GPUCell<Particle> c = *(h_cp[i]);

	    if(h_numbers[i] != h_controlParticleNumberArray[i])
	    {
	    	printf("checkpoint %d: cell %d incorrect number of particles in DEVICE array %15d (must be %15d)\n",
	    			num,i,
	    			h_numbers[i],h_controlParticleNumberArray[i]
	    			);
	    	exit(0);
	    }
	}


	for(int i = 0;i < (*AllCells).size();i++)
	{
		GPUCell<Particle> c = *(h_cp[i]);

	    if(c.number_of_particles != h_controlParticleNumberArray[i])
	    {
	    	printf("checkpoint %d: cell %d incorrect number of particles in HOST copy array %15d (must be %15d)\n",
	    			num,i,
	    			c.number_of_particles,h_controlParticleNumberArray[i]
	    			);
	    	exit(0);
	    }
	}

    return 0;
}

void printCellCurrents(int num,int nt,char *name,char *where)
{
	int size = (*AllCells).size();
	GPUCell<Particle> ** cp = (GPUCell<Particle> **)malloc(size*sizeof(GPUCell<Particle> *));
	FILE *f;
	char fname[100];
	CellDouble *m;

	sprintf(fname,"%s_at_%s_cells_%03d.dat",name,where,nt);

	if((f = fopen(fname,"wt")) == NULL) return;

	//	copyCells(h_CellArray);
		copyCells(where,nt);

		for(int i = 0;i <size;i++)
		{
			GPUCell<Particle> c = *(cp[i]);

			for(int i1  = 0;i1 < CellExtent;i1++)
			{
				for(int k1  = 0;k1 < CellExtent;k1++)
				{
					for(int l1  = 0;l1 < CellExtent;l1++)
					{
						if(!strcmp(name,"jx"))
						{
							m = c.Jx;
						}
						else
						{
							if(!strcmp(name,"jy"))
							{
								m = c.Jy;
							}
							else
							{
								m = c.Jz;
							}
						}
						fprintf(f,"%10d %5d %5d %5d %25.15e \n",i,i1,k1,l1,m->M[i1][k1][l1]);
					}
				}
			}
		}
        fclose(f);
}

int memory_monitor(char *legend,int nt)
{
	static int first = 1;
	static FILE *f;

#ifndef FREE_RAM_MONITOR
	return 1;
#endif

	if(first == 1)
	{
		first = 0;
		f = fopen("memory_monitor.log","wt");
	}

	size_t m_free,m_total;
	struct sysinfo info;


	cudaError_t err = cudaMemGetInfo(&m_free,&m_total);

	sysinfo(&info);
	fprintf(f,"step %10d %50s GPU memory total %10d free %10d free CPU memory %10u \n",nt,legend,m_total/1024/1024,m_free/1024/1024,info.freeram/1024/1024);

}

int memory_status_print(int nt)
{
	size_t m_free,m_total;
	struct sysinfo info;


	cudaMemGetInfo(&m_free,&m_total);
	sysinfo(&info);

	#ifdef MEMORY_PRINTS
    printf("before Step  %10d CPU memory free %10u GPU memory total %10d free %10d\n",
 		   nt,info.freeram/1024/1024,m_total/1024/1024,m_free/1024/1024);
#endif

	return 0;
}


int Compute()
{
	   double t = compareCPUtoGPU();
	   printf("----------------------------------------------------------- plasma check before move %.5f\n",t);
	   size_t m_free,m_total;

	   cudaMemGetInfo(&m_free,&m_total);

	   struct sysinfo info;


	   for(int nt = START_STEP_NUMBER;nt <= TOTAL_STEPS;nt++)
	   {
		   memory_status_print(nt);

	       Step(nt);

	       memory_status_print(nt);
	   }


	   t = compareCPUtoGPU();
	   printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ plasma check after move %.5f\n",t);

}


};







#endif /* GPU_PLASMA_H_ */
