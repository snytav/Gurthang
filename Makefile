

all:    service_functions.o rnd.o mpi_shortcut.o main.o


main.o:
	nvcc -O2  -lineinfo -c main.cu  -g --ptxas-options=-v -fmad=false  -m64 -I/opt/cuda_80/include

mpi_shortcut.o:
	mpicxx -c mpi_shortcut.c -m64

service_functions.o:
	g++ -c service_functions.cxx
rnd.o:
	g++ -m64 -c rnd.cpp

clean:
	rm *.o
