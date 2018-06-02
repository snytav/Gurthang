//#include "archAPI.h"
#include "rnd.h"
//#include "plasma.h"
#include "cell.h"


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
__global__ void  GPU_WriteAllCurrents(Cell<Particle>  **cells,int n0,
		      double *jx,double *jy,double *jz,double *rho)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
	Cell<Particle>  *c,*c0 = cells[0],nc;
	__shared__ extern CellDouble fd[9];

	c = cells[ c0->getGlobalCellNumber(nx,ny,nz)];

	 nc = *c;

	             int i1,l1,k1;
	        	 i1 = threadIdx.x;
	        	 l1 = threadIdx.y;
	        	 k1 = threadIdx.z;
    	         int n = nc.getFortranCellNumber(nc.i+i1-1,nc.l+l1-1,nc.k+k1-1);

    	         if (n < 0 ) n = -n;
        		 double t,t_x,t_y;
		         t_x = nc.Jx->M[i1][l1][k1];
		         int3 i3 = nc.getCellTripletNumber(n);


		         cuda_atomicAdd(&(jx[n]),t_x);
		         t_y= nc.Jy->M[i1][l1][k1];
		         cuda_atomicAdd(&(jy[n]),t_y);
		         t = nc.Jz->M[i1][l1][k1];
		         cuda_atomicAdd(&(jz[n]),t);

}
