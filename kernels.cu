





template <template <class Particle> class Cell >
global_for_CUDA
void printParticle(Cell<Particle>  **cells,int num,int sort)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
//	int i,l,k;
	Cell<Particle>  *c,*c0 = cells[0],nc;
	//double t;
	__shared__ extern CellDouble fd[9];
	//double *src,*dst;
	//int pqr2;
	//CurrentTensor t1,t2;
	Particle p;

	c = cells[ c0->getGlobalCellNumber(nx,ny,nz)];
//	c = cells[ n ];

	nc = *c;
    if(nc.number_of_particles < threadIdx.x) return;

	nc.readParticleFromSurfaceDevice(threadIdx.x,&p);

		if(p.fortran_number == num && (int)p.sort == sort)
		{
//#ifdef PARTICLE_CELL_DEBUG_PRINTS
			printf("particle-print %5d thread %3d cell (%d,%d,%d) sort %d  %25.15e,%25.15e,%25.15e \n",p.fortran_number,threadIdx.x,c->i,c->l,c->k,(int)p.sort,p.x,p.y,p.z);
//#endif
		}
}














template <template <class Particle> class Cell >
global_for_CUDA void GPU_WriteControlSystem(Cell<Particle>  **cells)
{
	unsigned int nx = blockIdx.x;
	unsigned int ny = blockIdx.y;
	unsigned int nz = blockIdx.z;
//	int i,l,k;
	Cell<Particle>  *c,*c0 = cells[0],nc;
	//double t;
	__shared__ extern CellDouble fd[9];
	//double *src; //,*dst;
//	int pqr2;
	//CurrentTensor t1,t2;

	c = cells[ c0->getGlobalCellNumber(nx,ny,nz)];
//	c = cells[ n ];

	 nc = *c;

	 nc.SetControlSystemToParticles();

}

//TODO : 1. 3 separate kernels :
//            A. form 3x3x3 array with number how many to fly and departure list with start positions in 3x3x3 array
//            B. func to get 3x3x3 indexes from a pair of cell numbers, to and from function
//            C. 2nd kernel to write arrival 3x3x3 matrices
///           D. 3rd kernel to form arrival positions in the particle list
//            E. 4th to write arriving particles




























