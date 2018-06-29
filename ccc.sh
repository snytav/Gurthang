     memory_monitor("checkControlPointParticlesOneSort",nt);
               ^
     memory_monitor("checkControlPointParticlesOneSort2",nt);
               ^
     memory_monitor("checkControlPointParticlesOneSort3",nt);
               ^
     memory_monitor("checkControlPointParticlesOneSort4",nt);
               ^
gpu_plasma.h: In instantiation of ‘void GPUPlasma<Cell>::ListAllParticles(int, char*) [with Cell = GPUCell]’:
       sprintf(str,"List%05d_%s.dat\0",nt,where);
        ^
g++ -g -c -o rnd.o rnd.cpp  
     } cag05b_ = { 1., -1. };
       ^
     } cag05a_ = { 1, 255, 25555 };
       ^
mpicxx -g -c -o mpi_shortcut.o mpi_shortcut.cxx  
nvcc -g -c -o service_functions.o service_functions.cu -lineinfo --maxrregcount=128 -g -I/usr/local/cuda/include/	
     printf(d_hyfile);
                ^
nvcc -g -c -o compare.o compare.cu -lineinfo --maxrregcount=128 -g -I/usr/local/cuda/include/
nvcc -g -c -o maxwell.o maxwell.cu -lineinfo --maxrregcount=128 -g -I/usr/local/cuda/include/	
nvcc -g -c -o load_data.o load_data.cu -lineinfo --maxrregcount=128 -g -I/usr/local/cuda/include/	
mpicxx -g -o all main.o rnd.o mpi_shortcut.o service_functions.o compare.o maxwell.o load_data.o   -g -L/usr/local/cuda/lib64 -lcuda -lcudart  
