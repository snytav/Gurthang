/*
 * tensor.h
 *
 *  Created on: Jul 10, 2018
 *      Author: snytav
 */

#ifndef TENSOR_H_
#define TENSOR_H_

#ifndef __CUDACC__
#define __host__
#define __device__
#define __forceinline__
#endif


class CurrentTensorComponent {
public:
	char i11, i12, i13,
	 i21, i22, i23,
	 i31, i32, i33,
	 i41, i42, i43;
	double t[4];


	__host__ __device__	CurrentTensorComponent & operator=(CurrentTensorComponent b)
	{
		CurrentTensorComponent a;

		a.i11 = b.i11;
		a.i12 = b.i12;
		a.i13 = b.i13;

		a.i21 = b.i21;
		a.i22 = b.i22;
		a.i23 = b.i23;

		a.i31 = b.i31;
		a.i32 = b.i32;
		a.i33 = b.i33;

		a.i41 = b.i41;
		a.i42 = b.i42;
		a.i43 = b.i43;

		a.t[0] = b.t[0];
		a.t[1] = b.t[1];
		a.t[2] = b.t[2];
		a.t[3] = b.t[3];

		return a;
	}

};

class CurrentTensor {
public:
	CurrentTensorComponent Jx,Jy,Jz;


	__host__ __device__	CurrentTensor & operator=(CurrentTensor b)
	{
		CurrentTensor a;

		a.Jx = b.Jx;
		a.Jy = b.Jy;
		a.Jz = b.Jz;

		return a;
	}
};

class DoubleCurrentTensor {
public:
	CurrentTensor t1,t2;

__host__ __device__	DoubleCurrentTensor & operator=(DoubleCurrentTensor b)
	{
		DoubleCurrentTensor a;

		a.t1 = b.t1;
		a.t2 = b.t2;

		return a;
	}
};



#endif /* TENSOR_H_ */
