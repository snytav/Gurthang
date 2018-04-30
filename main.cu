#include "gpu_plasma.h"
#include <stdlib.h>
#include "mpi_shortcut.h"
//TODO: gpu cell in the global array at copy from there appears to be not initialized

int main(int argc,char*argv[])
{
   GPUPlasma<GPUCell> *plasma;

   plasma = new GPUPlasma<GPUCell>(100,4,4,1.1424,0.05,0.05,1.0,2000,1.0,0.001);

   plasma->Initialize();

   double t = plasma->compareCPUtoGPU();
   printf("----------------------------------------------------------- plasma check before move %.5f\n",t);
   size_t m_free,m_total;

   cudaMemGetInfo(&m_free,&m_total);

   struct sysinfo info;


   for(int nt = START_STEP_NUMBER;nt <= TOTAL_STEPS;nt++)
   {
	   cudaMemGetInfo(&m_free,&m_total);
	   sysinfo(&info);
#ifdef MEMORY_PRINTS
       printf("before Step  %10d CPU memory free %10u GPU memory total %10d free %10d\n",
    		   nt,info.freeram/1024/1024,m_total/1024/1024,m_free/1024/1024);
#endif

       plasma->Step(nt);

       cudaMemGetInfo(&m_free,&m_total);
       sysinfo(&info);
#ifdef MEMORY_PRINTS
       printf("after  Step  %10d CPU memory free %10u GPU memory total %10d free %10d\n",
    		   nt,info.freeram/1024/1024/1024,m_total/1024/1024/1024,m_free/1024/1024/1024);
#endif
   }


   t = plasma->compareCPUtoGPU();
   printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ plasma check after move %.5f\n",t);

   delete plasma;
   

   return 0;
}
