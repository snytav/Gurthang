LD=mpicxx
CUDACC=nvcc
CXX=mpicxx
CPP=g++
CUDA=/usr/local/cuda
CUDALIB=$(CUDA)/lib64

LDFLAGS= -lm -L$(CUDALIB)
#CUDAFLAGS= --maxrregcount=128  -arch=sm_35 --ptxas-options=-v -I/usr/local/cuda-7.5/include/
CUDAFLAGS=  -lineinfo --maxrregcount=128 -g -I$(CUDA)/include/
CUDALIBS=  -g -L$(CUDALIB) -lcuda -lcudart #-lthrust 
MPIFLAGS=
CFLAGS=

OBJ = main.o rnd.o mpi_shortcut.o service_functions.o compare.o 
#plasma.o
            
main.o: main.cu $(DEPS)
	$(CUDACC) -g -c -o $@ $< $(CUDAFLAGS) 
	
compare.o: compare.cu $(DEPS)
	$(CUDACC) -g -c -o $@ $< $(CUDAFLAGS)
	
#kernels.o: kernels.cu $(DEPS)
#	$(CUDACC) -g -c -o $@ $< $(CUDAFLAGS) 		 	
                    
#plasma.o: plasma.cu $(DEPS)
#	$(CUDACC) -g -c -o $@ $< $(CUDAFLAGS)                     
                    
%.o: %.cxx $(DEPS)
	$(CXX) -g -c -o $@ $< $(MPIFLAGS)

%.o: %.cpp $(DEPS)
	$(CPP) -g -c -o $@ $< $(CBFLAGS)
                            
all: $(OBJ)
	$(LD) -g -o $@ $^ $(CFLAGS) $(DBFLAGS) $(CUDALIBS)

clean:
	rm *.o all    
