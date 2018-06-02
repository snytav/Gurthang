/*
 * kernels.h
 *
 *  Created on: Jun 2, 2018
 *      Author: snytav
 */

#ifndef KERNELS_H_
#define KERNELS_H_


template <template <class Particle> class Cell >
__global__ void GPU_WriteAllCurrents(Cell<Particle>  **cells,int n0,
		      double *jx,double *jy,double *jz,double *rho);


#endif /* KERNELS_H_ */
