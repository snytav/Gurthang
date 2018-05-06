#export PATH=/opt/cuda_80/bin:/home/snytav/backUp/WRFV35/external/io_netcdf:$PATH
#export PATH=/usr/local/cuda/bin:/home/snytav/backUp/WRFV35/external/io_netcdf:$PATH
echo SETTING FOR CUDA SERVER AT NSU : CUDA 8.0
export LD_LIBRARY_PATH=/opt/cuda_80/lib64/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/ifs/opt/2013/intel/impi/4.1.3.049/intel64/lib:$LD_LIBRARY_PATH

rm *.o all
g++ -m64 -c rnd.cpp
g++ -c service_functions.cxx
mpicxx -c mpi_shortcut.c -m64 >&mpi_err.c
#nvcc -O2 -lineinfo -c main.cu --ptxas-options=-v --keep -I/usr/local/cuda/include >&c_out
nvcc -O2  -lineinfo -c main.cu  -g --ptxas-options=-v -fmad=false  -m64 -I/opt/cuda_80/include >&c_out
mpicxx -o all main.o rnd.o mpi_shortcut.o service_functions.o -lcuda -lcudart -lm -lmpich -lpthread -m64 -L/opt/cuda_80/lib64/ -I/opt/cuda_80/include/ -I/ifs/opt/2013/intel/impi/4.1.3.049/intel64/include  -L/ifs/opt/2013/intel/impi/4.1.3.049/intel64/lib >&link_out

ls -l all
date
