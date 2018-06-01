#ifndef PARTICLE_H
#define PARTICLE_H

#include "types.h"

#include <stdio.h>
#include <string>
#include <string.h>
#include <iostream>
#include <math.h>
#include "run_control.h"
#include "control.h"

#define GPU_PARTICLE
#define TOLERANCE 1e-15
#define SIZE_TOLERANCE 1e-10

typedef struct CurrentTensorComponent {
	int i11, i12, i13,
	 i21, i22, i23,
	 i31, i32, i33,
	 i41, i42, i43;
	double t[4];
} CurrentTensorComponent;

typedef struct CurrentTensor {
	CurrentTensorComponent Jx,Jy,Jz;
} CurrentTensor;

typedef char gpu_string[200];

//char *FortranExpWrite(double t)
//{
//      char str[100],res[100];
//      char *prev,*next,*dot;
//      gpu_string res_str;
//
//      sprintf(str,"%11.4E",t);
//
//      dot = strstr(str,".");
//
//      prev = dot - 1;
//      next = dot + 1;
//
//      sprintf(res,"0.%c%s",*prev,next);
//
//      strcpy(res_str,res);
//
//      return (char *)res_str;
//}

double compare(double *a,double *b,int num,char *legend,double tol)
{
     double t = 0.0;

     for(int i = 0; i < num ;i++)
     {
         if(fabs(a[i] - b[i]) < tol)
         {
            t += 1.0;
#ifdef COMPARE_PRINTS
            printf(" i %5d a %e b %e diff %e\n",i,a[i],b[i],fabs(a[i] - b[i]));
#endif

         }
         else
         {
#ifdef COMPARE_PRINTS
        	printf("WRONG i %5d a %e b %e diff %e\n",i,a[i],b[i],fabs(a[i] - b[i]));
#endif
         }
     }

     if(num > 0) t /= num;
#ifdef COMPARE_PRINTS
     printf("%30s %.5f\n",legend,t);
#endif
     return t;
}

int comd(double a,double b)
{
	return (fabs(a - b) < TOLERANCE);
}

class Particle
{
//	int jmp;
//	double *d_ctrlParticles;
public:  
  
   double x,y,z,pu,pv,pw,m,q_m;
   particle_sorts sort;
//   CurrentTensor t1,t2;

  // void SetControlSystem(int j,double *c){jmp = j;d_ctrlParticles = c;}
   
#ifdef DEBUG_PLASMA
 //  double3 next_x;
//   double ex,ey,ez,hx,hy,hz;
   int fortran_number;

#endif   

__host__ __device__ Particle(){}

__host__ __device__ __forceinline__
Particle(double x1,double y1, double z1,double u1,double v1,double w1,double m1,double q_m1): x(x1), y(y1), z(z1), pu(u1), pv(v1), pw(w1), m(m1), q_m(q_m1) {}

__host__ __device__
  ~Particle(){}

 __device__
void ElectricMove(double3 E,double tau, double q_m,double *tau1,double *pu,double *pv,double *pw,double *ps)
{

	*tau1=q_m*tau*0.5;

		*pu += *tau1*E.x;
		*pv += *tau1*E.y;
		*pw += *tau1*E.z;
		*ps = (*tau1) * rsqrt(((*pu) * (*pu) + (*pv) * (*pv) + (*pw) * (*pw)) * 1. + 1.0);

}

__device__ void MagneticMove(double3 H,double ps,double *pu1,double *pv1,double *pw1)
{
	double bx,by,bz,s1,s2,s3,s4,s5,s6,s,su,sv,sw;

	bx = ps * H.x;
	by = ps * H.y;
	bz = ps * H.z;
	su = pu + pv * bz - pw * by;
	sv = pv + pw * bx - pu * bz;
	sw = pw + pu * by - pv * bx;

	s1 = bx * bx;
	s2 = by * by;
	s3 = bz * bz;
	s4 = bx * by;
	s5 = by * bz;
	s6 = bz * bx;
	s = s1 + 1. + s2 + s3;

	*pu1 = ((s1 + 1.) * su + (s4 + bz) * sv + (s6 - by) * sw) / s;
	*pv1 = ((s4 - bz) * su + (s2 + 1.) * sv + (s5 + bx) * sw) / s;
	*pw1 = ((s6 + by) * su + (s5 - bx) * sv + (s3 + 1.) * sw) / s;
}

__device__
void ElectricMoveStageTwo(double tau1,double tau,double3 E,
//		 double pu,double pv,double pw,
		 double pu1,double pv1,double pw1,double *u,double *v,double *w,double *x,double *y,double *z)
{
	double sx,sy,sz,pu,pv,pw,ps,x1,y1,z1;

	sx = tau1*E.x;
	sy = tau1*E.y;
	sz = tau1*E.z;

	pu = pu1 + sx;
	pv = pv1 + sy;
	pw = pw1 + sz;
	ps = pu * pu + pv * pv + pw * pw;
	ps = pow(((pu * pu + pv * pv + pw * pw) + 1.0),-0.5);

	*u = ps * pu;
	*v = ps * pv;
	*w = ps * pw;
	x1 = *x + tau * (*u);
	y1 = *y + tau * (*v);
	z1 = *z + tau * (*w);

	*x = x1;
	*y = y1;
	*z = z1;


}

__device__ double3 mult(double t,double3 t3)
{
	t3.x *= t;
	t3.y *= t;
	t3.z *= t;

	return t3;
}

 __device__ __forceinline__
void Move(double3 E,double3 H,double tau)
{
    double bx,by,bz,tau1,u,v,w,ps,su,sv,sw,s1,s2,s3,s4,s5,s6,s;
	double sx,sy,sz,x1,y1,z1,pu1,pv1,pw1;
	double3 sx3;


	ElectricMove(E,tau,q_m,&tau1,&pu,&pv,&pw,&ps);

	MagneticMove(H,ps,&pu1,&pv1,&pw1);

//	ElectricMoveStageTwo(tau1,tau,E,pu1,pv1,pw1,&u,&v,&w,&x,&y,&z);

	sx3 = mult(tau1,E);
//	sx = sx3.x;
//	sy = sx3.y;
//	sz = sx3.z;

	pu = pu1 + sx3.x;
	pv = pv1 + sx3.y;
	pw = pw1 + sx3.z;
//	ps = pu * pu + pv * pv + pw * pw;
	ps = pow(((pu * pu + pv * pv + pw * pw) + 1.0),-0.5);

	u = ps * pu;
	v = ps * pv;
	w = ps * pw;
	x1 = x + tau * u;
	y1 = y + tau * v;
	z1 = z + tau * w;

	x = x1;
	y = y1;
	z = z1;

}
   
__host__ __device__
void Collide(double sect){}

__host__ __device__ __forceinline__
   double3 GetX(){double3 d3x; d3x.x = x; d3x.y = y; d3x.z = z; return d3x;}

__host__ __device__ __forceinline__
   double3 GetV(){double3 d3x; d3x.x = x; d3x.y = y; d3x.z = z; return d3x;}

__host__ __device__ __forceinline__
   void    SetX(double3 x1){x = x1.x;y = x1.y;z = x1.z;}

__host__ __device__ __forceinline__
   double  GetMass(){return m;}

__host__ __device__ __forceinline__
   double  GetQ2M(){return q_m;}
   

#ifdef DEBUG_PLASMA
//next
//!!!!!!!!!!!!!!!!!
//__host__ __device__ __forceinline__
//   double3 GetXnext(){return next_x;}
//
//__host__ __device__ __forceinline__
//  void    SetXnext(double3 x1){next_x.x = x1.x;next_x.y = x1.y;next_x.z = x1.z;}

//__host__ __device__ __forceinline__
//   int checkParticle(){return (
//                               (fabs(x -next_x.x) < TOLERANCE) &&
//                               (fabs(y -next_x.y) < TOLERANCE) &&
//                               (fabs(z -next_x.z) < TOLERANCE)
//
//			      );
//
//                      }


#endif


void Print(FILE* f, int num)
{
//     char num_str[20];
//     sprintf(num_str,"num %05d",num);
//
//     gpu_string print_str;
//     strcpy(print_str,num_str);
//
//     strcat(print_str,strcat(" x ",FortranExpWrite(x)));
//     strcat(print_str,strcat(" y ",FortranExpWrite(y)));
//     strcat(print_str,strcat(" z ",FortranExpWrite(z)));
//     strcat(print_str,strcat(" px ",FortranExpWrite(pu)));
//     strcat(print_str,strcat(" py ",FortranExpWrite(pv)));
//     strcat(print_str,strcat(" pz ",FortranExpWrite(pw)));
//     strcat(print_str,strcat(" mass ",FortranExpWrite(m)));
//     strcat(print_str,strcat(" q/m ",FortranExpWrite(q_m)));
//
//     fprintf(f,"%s\n",print_str);
   }

__host__ __device__
Particle & operator=(Particle const & src)
{
	x = src.x;
	y = src.y;
	z = src.z;
	pu = src.pu;
	pv = src.pv;
	pw  = src.pw;
	m   = src.m;
	q_m = src.q_m;
	sort = src.sort;


#ifdef DEBUG_PLASMA
//	next_x = src.next_x;
//	ex     = src.ex;
//	ey     = src.ey;
//	ez     = src.ez;
//
//	hx     = src.hx;
//	hy     = src.hy;
//	hz     = src.hz;
	fortran_number = src.fortran_number;
#endif
	return (*this);
}

};

#endif
