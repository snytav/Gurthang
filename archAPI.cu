/*
 * archAPI.cxx
 *
 *  Created on: Apr 10, 2018
 *      Author: snytav
 */
#include<stdlib.h>
#include<string.h>


#include "archAPI.h"



#ifdef __CUDACC__
int SetDevice(int n){return cudaSetDevice(n);}
#else
int SetDevice(int n){return 0;}
#endif

#ifdef __CUDACC__
__device__
void AsyncCopy(double *dst,double *src,int n,int size)
{
	int j;
	j = n;
	if(j < size)
	{
	   dst[j] = src[j];
	}

}
#else
void AsyncCopy(double *dst,double *src,int n,int size){ memcpy(dst,src,n,size);}
#endif


#ifdef __CUDACC__
 int MemoryCopy(void* dst,void *src,size_t size,int dir)
{
//	int err = 0;


	cudaMemcpyKind cuda_dir;

	if(dir == HOST_TO_DEVICE) cuda_dir = cudaMemcpyHostToDevice;
	if(dir == HOST_TO_HOST) cuda_dir = cudaMemcpyHostToHost;
	if(dir == DEVICE_TO_HOST) cuda_dir = cudaMemcpyDeviceToHost;
	if(dir == DEVICE_TO_DEVICE) cuda_dir = cudaMemcpyDeviceToDevice;



	return ((int)cudaMemcpy(dst,src,size,cuda_dir));
}
#else
 int MemoryCopy(void* dst,void *src,size_t size,int dir);
#endif


#ifdef __CUDACC__
 int MemoryAllocate(void** dst,size_t size)
{
	cudaMalloc(dst,size);
    return 0;
}
#else
 int MemoryAllocate(void** dst,size_t size);
#endif

#ifndef __CUDACC__
int GetDeviceMemory(size_t *m_free,size_t *m_total)
{
	*m_free = 0;
	*m_total = 0;
	return 0;
}
#endif

#ifndef __CUDACC__
int MemorySet(void *s, int c, size_t n)
{
	memset(s,c,n);
    return 0;
}
#endif


#ifndef __CUDACC__
int DeviceSynchronize()
{
    return 0;
}

#ifdef __CUDACC__
 int __host__ ThreadSynchronize()
{
	 return cudaThreadSynchronize();
}
#else
 int ThreadSynchronize()
 {
 	 return 0;
 }
#endif

 int getLastError()
{
	return 0;
}
#else
int getLastError()
{
	return (int)cudaGetLastError();
}
#endif

#ifdef __CUDACC__
 __device__ void BlockThreadSynchronize()
 {
     __syncthreads();
 }
#else
 void BlockThreadSynchronize(){}
#endif

#ifdef __CUDACC__
__device__ double MultiThreadAdd(double *address, double val)
{
    double assumed,old=*address;
    do {
        assumed=old;
        old= __longlong_as_double(atomicCAS((unsigned long long int*)address,
                    __double_as_longlong(assumed),
                    __double_as_longlong(val+assumed)));
    }while (assumed!=old);

    *address += val;

    old = *address;

    return old;
}
#else
double MultiThreadAdd(double *address, double val){
#pragma omp critical
	*address += val;

}
#endif

#ifdef __CUDACC__
 const char *getErrorString(int err)
{
	return cudaGetErrorString((cudaError_t)err);
}
#else
const char *getErrorString(int err){return "";}

#endif


#ifdef __CUDACC__
 int GetDeviceMemory(size_t *m_free,size_t *m_total)
{
	return cudaMemGetInfo(m_free,m_total);
}
#else
 int GetDeviceMemory(size_t *m_free,size_t *m_total){*m_free = -1; *m_total = -1;}
#endif


#ifdef __CUDACC__
int MemorySet(void *s, int c, size_t n)
{
	return (int)cudaMemset(s,c,n);

//    return 0;
}
#else
int MemorySet(void *s, int c, size_t n)
{
	return memset(s,c,n);
}
#endif

int get_num_args(void **args)
{
	int i;
	for(i = 0;args[i] != NULL;i++);

	return i;
}

#ifndef __CUDACC__
dim3 threadIdx,blockIdx;
#endif

typedef void (*func_1)(void);

void call_with_args(const void *func, void **args)
{
	int num = get_num_args(args);

	switch(num)
	{
		case 0:  func_1 f = (func_1)func;
			     f();
				 break;

		case 1:  func(*(args[0]));
		         break;

		case 2:  func(*(args[0]),*(args[1]));
		         break;

		case 3:  func(*(args[0]),*(args[1]),*(args[2]));
		         break;

		case 4:  func(*(args[0]),*(args[1]),*(args[2]),*(args[3]));
		         break;

		case 5:  func(*(args[0]),*(args[1]),*(args[2]),*(args[3]),*(args[4]));
		         break;

		case 6:  func(*(args[0]),*(args[1]),*(args[2]),*(args[3]),*(args[4]),*(args[5]));
		         break;

		case 7:  func(*(args[0]),*(args[1]),*(args[2]),*(args[3]),*(args[4]),*(args[5]),*(args[6]));
		         break;


	}

}

int cudaLaunchKernel_onCPU(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
{

	for(int i = 0;i < gridDim.x;i++)
	{
		for(int l = 0;l < gridDim.y;l++)
		{
			for(int k = 0;k < gridDim.z;k++)
			{
				blockIdx.x = i;
				blockIdx.y = l;
				blockIdx.z = k;

				for(int i1 = 0;i1 < blockDim.x;i1++)
				{
					for(int l1 = 0;l1 < blockDim.y;l1++)
					{
						for(int k1 = 0;k1 < blockDim.z;k1++)
						{
							threadIdx.x = i1;
							threadIdx.x = l1;
							threadIdx.x = k1;

							call_with_args(func, args);
						}
					}
				}
			}
		}
	}

}

