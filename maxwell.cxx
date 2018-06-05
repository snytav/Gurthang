#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include "rnd.h"

#include "read_particles.h"

#include "run_control.h"

#include <vector>

#include "particle.h"


#include "f2c.h"
#include <stdio.h>
// #include "run_control.h"

/* Common Block Declarations */

ParticleArraysGroup initial;
ParticleFloatArraysGroup diagnostics;

struct cag05b_1_ {
    doublereal store1, store2;
};
struct cag05b_2_ {
    doublereal normal, gamma;
};

#define cag05b_1 (*(struct cag05b_1_ *) &cag05b_)
#define cag05b_2 (*(struct cag05b_2_ *) &cag05b_)

struct cag05a_1_ {
    integer ix, iy, iz;
};

#define cag05a_1 (*(struct cag05a_1_ *) &cag05a_)

/* Initialized data */

struct cag05b {
    doublereal e_1[2];
    } cag05b_ = { 1., -1. };

struct cag05a {
    integer e_1[3];
    } cag05a_ = { 1, 255, 25555 };


/* Table of constant values */

static integer c__9 = 9;
static integer c__1 = 1;
static integer c__3 = 3;
static integer c__5 = 5;
static integer c__0 = 0;
static integer c__2 = 2;
static doublereal c_b172 = 1.;
static integer c__12 = 12;
static doublereal c_b417 = 1.1424;
static doublereal c_b419 = .5712;
static doublereal c_b429 = 1.5;
static integer c__20000 = 20000;
static doublereal c_b581 = 0.;
static doublereal c_b582 = .11200000000000002;
static doublereal c_b587 = .14;
static doublereal c_b589 = .8;
static doublereal c_b614 = .001;

/* ------------------------------------------------------ */
/* ������� �.�., */
doublereal g05dde_(doublereal *a, doublereal *b,int dbg_print)
{
    /* Initialized data */

    static doublereal one = 1.;
    static doublereal half = .5;
    static doublereal d__[41] = { 0.,.674489750196082,1.150349380376008,
	    1.534120544352546,1.862731867421652,2.153874694061456,
	    2.417559016236505,2.66006746861746,2.885634912426757,
	    3.097269078198785,3.297193345691964,3.487104104114431,
	    3.668329285121323,3.841930685501911,4.008772594168585,
	    4.169569323349106,4.324919040826046,4.475328424654204,
	    4.621231001499247,4.763001034267814,4.900964207963193,
	    5.035405969463927,5.166578119728753,5.294704084854598,
	    5.419983174916868,5.54259405780294,5.662697617459439,
	    5.780439324478935,5.895951216739571,6.009353565530745,
	    6.120756285971941,6.230260137989044,6.33795775455379,
	    6.443934526538564,6.548269367831731,6.651035379893011,
	    6.752300431407015,6.852127665896068,6.95057594791675,
	    7.047700256664409,7.14355203435219 };

    /* System generated locals */
    doublereal ret_val;

    /* Local variables */
    static integer n;
    static doublereal t, u, v, w, x;
    extern doublereal wrapg05cae_(doublereal *,int);

    u = cag05b_1.store1;
    for (n = 1; n <= 39; ++n) {
	if (u > half) {
	    goto L40;
	}
	u += u;
/* L20: */
    }
    n = 40;
L40:
    t = d__[n - 1];
    u = wrapg05cae_(&x,dbg_print);
L60:
    w = (d__[n] - t) * u;
    v = w * (w * half + t);
L80:
    u = wrapg05cae_(&x,dbg_print);
    if (v <= u) {
	goto L100;
    }
    v = wrapg05cae_(&x,dbg_print);
    if (u > v) {
	goto L80;
    }
    u = (v - u) / (one - u);
    goto L60;
L100:
    u = (u - v) / (one - v);
    if (u > half) {
	goto L120;
    }
    cag05b_1.store1 = u + u;
    ret_val = *a + *b * (w + t);
    return ret_val;
L120:
    cag05b_1.store1 = u + u - one;
    ret_val = *a - *b * (w + t);
    return ret_val;
} /* g05dde_ */

/* ------------------------------------------------------------------ */
doublereal g05cae_(doublereal *x,int dbg_print)
{
    /* System generated locals */
    doublereal ret_val;

    /* Local variables */
    static doublereal ai;
    static integer ii;
    static doublereal ax, ay, az;

    cag05a_1.ix = (cag05a_1.ix - cag05a_1.ix / 177 * 177) * 171 - (
	    cag05a_1.ix / 177 << 1);
    cag05a_1.iy = (cag05a_1.iy - cag05a_1.iy / 176 * 176) * 172 - (
	    cag05a_1.iy / 176 << 1);
    cag05a_1.iz = (cag05a_1.iz - cag05a_1.iz / 178 * 178) * 170 - (
	    cag05a_1.iz / 178 << 1);
    if (cag05a_1.ix < 0) {
	cag05a_1.ix += 30269;
    }
    if (cag05a_1.iy < 0) {
	cag05a_1.iy += 30307;
    }
    if (cag05a_1.iz < 0) {
	cag05a_1.iz += 30323;
    }
    ax = (doublereal) cag05a_1.ix;
    ay = (doublereal) cag05a_1.iy;
    az = (doublereal) cag05a_1.iz;
    ai = ax / 30269. + ay / 30307. + az / 30323.;
    ii = (integer) ai;
    ret_val = ai - ii;
    return ret_val;
} /* g05cae_ */

doublereal wrapg05cae_(doublereal *x,int dbg_print)
{
    static int n = 0;
    double t  = g05cae_(x,dbg_print);
// #ifdef DEBUG_PLASMA
    n++;
    if(dbg_print == 1)
    {
     printf("%10d %25.15e \n",n,t);
    }
// #endif

    return t;
}

doublereal wrapg05dde_(doublereal *a, doublereal *b,int dbg_print)
{
    return g05dde_(a,b,dbg_print);
}

double rnd_uniform(int dbg_print)
{
    doublereal x;

    return (double)wrapg05cae_(&x,dbg_print);
}

double rnd_gaussian(double a,double b,int dbg_print)
{
    return (double)wrapg05dde_(&a,&b,dbg_print);
}


int in_range(double z0,double z,double z1)
{
	return ((z > z0) && (z < z1)) || ((fabs(z - z0) < 1e-13) && (fabs(z - z1) < 1e-13));
}

int InitUniformMaxwellianParticles(int beamf,int jmb,
				   double tex0,double tey0,double tez0,
				   double beam_lx, double beam_ly,double beam_lz,int *jmb_real,
                   double lx,double ly,double lz, int meh,double Tb,double rimp,double rbd,
				   double *xi,double *yi, double *zi,double *ui,double *vi, double *wi,
				   double *xb,double *yb, double *zb,double *ub,double *vb, double *wb,
				   double *xf,double *yf, double *zf,double *uf,double *vf, double *wf
				  )
{
    double x,y,z,vb0,d__1,d__2,d__3,vy,vz,termx,gb0;
    double vf01,vf02,pinv1,pinv2,mfrq = 0.0;
//     double *ux,*uy,*uz;
    double *ux,*uy,*uz;
    double beam_y_max,beam_y_min, beam_sh;
    double beam_z_max,beam_z_min, beam_shz;

    beam_sh = (ly - beam_ly)/2;
    beam_y_max = ly - beam_sh;
    beam_y_min = beam_sh;

    beam_shz   = (lz - beam_lz)/2;
    beam_z_max = lz - beam_shz;
    beam_z_min = beam_shz;

    int j;
    
    ux = (double *)malloc(jmb*sizeof(double));
    uy = (double *)malloc(jmb*sizeof(double));
    uz = (double *)malloc(jmb*sizeof(double));
    
    for (j = 1; j <= jmb; j++) 
    {
	z =   lz * rnd_uniform(0);
	y =   meh * ly + ly * rnd_uniform(0);
	x =   lx * rnd_uniform(0);
	
	xi[j - 1] = x;
	yi[j - 1] = y;
	zi[j - 1] = z;
	ui[j - 1] = 0.0;
	vi[j - 1] = 0.0;
	wi[j - 1] = 0.0;
    }
    
//    parasetrandombeam_();
/*     END RANDOM GENERATOR */
/* ****************** BEAM **************************************** */
    *jmb_real = 0;
	for (j = 1; j <= jmb; j++) 
	{
		double y = yi[j- 1];
		double z = zi[j- 1];
		if((xi[j - 1] < beam_lx) &&
				(y < beam_y_max) && (y > beam_y_min) &&
				in_range(beam_z_min,z,beam_z_max)
		  )
		{
	        xb[*jmb_real] = xi[j - 1];
	        yb[*jmb_real] = yi[j - 1];
	        zb[*jmb_real] = zi[j - 1];
	        vb0       = rnd_gaussian(0.0, Tb*rimp,0);
	        ux[*jmb_real] = vb0 + rimp;
	        uy[*jmb_real] = rnd_gaussian(0.0, Tb*rimp,0);
	        uz[*jmb_real] = rnd_gaussian(0.0, Tb*rimp,0);
#ifdef DEBUG_INITIAL_PARTICLE_PRINTS
            	printf("ion %10d %25.15e %25.15e %25.15e \n",j,xi[j - 1],yi[j - 1],zi[j - 1]);
#endif
	        (*jmb_real)++;
		}
	    
	}
	
	for (j = 1; j <= *jmb_real; j++)
	{
	    double uxt,ubt; 
	    d__1 = ux[j - 1];
	    d__2 = uy[j - 1];
	    d__3 = uz[j - 1];
	    
	    vb0 = sqrt(1.0 - d__1 * d__1 - d__2 * d__2 - d__3 * d__3);
// 	    uxt = ux[j - 1];
// 	    ubt = uxt/vb0;
// 	    ub[j-1] = ubt;
	    ub[j - 1] = ux[j - 1] / vb0;
	    vb[j - 1] = uy[j - 1] / vb0;
	    wb[j - 1] = uz[j - 1] / vb0;
#ifdef DEBUG_INITIAL_PARTICLE_PRINTS
	       printf("beam %10d %25.15e  %25.15e    %25.15e %25.15e %25.15e   %25.15e \n",
	    		         j,  xb[j - 1],yb[j - 1],vb0,    ub[j-1],vb[j - 1],wb[j - 1]);
#endif
	}
/*     MAKING THE RANDOM GENERATOR WORK THE SAME */
//    parasetrandomelectrons_();
/*     END RANDOM GENERATOR */

//     vy = rnd_gaussian(0.0,tey0,1);
//     vz = rnd_gaussian(0.0,tez0,1);
//  
//     termx = rnd_gaussian(0.0,tex0,1); 
	j = 1;
    for (j = 1; j <= jmb;j++) 
    {
    	   if((2*j-1) == 24933)
    	   {
    		   int qq = 0;
    	   }
           xf[2*j-1-1] = xi[j-1];
           yf[2*j-1-1] = yi[j-1];
           zf[2*j-1-1] = zi[j-1];

           xf[2*j-1]   = xi[j-1];
           yf[2*j-1]   = yi[j-1];
           zf[2*j-1]   = zi[j-1];
	   
//         FIRST SETTING TRANVERSE
// razbros v skorostyax
           
           vy=rnd_gaussian(0.0,tey0,0);    
           vz=rnd_gaussian(0.0,tez0,0);        

//          INVERSE CURRENT
           
           termx = rnd_gaussian(0.0,tex0,0);
// 	   printf("termx %15.5e vx %15.5e vy %15.5e \n",termx,vy,vz);
	   
           gb0 = pow(1.0+pow(ub[j-1],2)+pow(vb[j-1],2)+pow(wb[j-1],2),-0.5);
	   
           vb0=ub[j-1]*gb0;
      	   if ((beamf == 1) && ((xi[j - 1] < beam_lx) && (yi[j - 1] < beam_y_max) && (yi[j - 1] > beam_y_min)))
     	   {
        	    vf01=-rbd*vb0+termx;
                vf02=-rbd*vb0-termx;
	       }
       	   else
	       {
      	        vf01=+termx;
      	        vf02=-termx;
	       }
       	   
           pinv1= vf01*pow((1.0-pow(vf01,2)-vy*vy-vz*vz),-0.5); 
           pinv2= vf02*pow((1.0-pow(vf02,2)-vy*vy-vz*vz),-0.5);
            
          vf[2*j-2] =  vy*pow((1.0-pow(vf01,2)-vy*vy-vz*vz),-0.5);
// 	  write(37,127) 2*j-1,vy,vf01,vz,                                                                                                                                                                        
//      +    vy/dsqrt((1d0-vf01**2-vy**2-vz**2))
//           printf("%10d vy %15.5e vf01 %15.5e vz %15.5e vf %15.5e \n",2*j-1,vy,vf01,vz,vy*pow((1.0-pow(vf02,2)-vy*vy-vz*vz),-0.5));
          vf[2*j-1] = -vy*pow((1.0-pow(vf02,2)-vy*vy-vz*vz),-0.5);
	  
          wf[2*j-2] =  vz*pow((1.0-pow(vf01,2)-vy*vy-vz*vz),-0.5);
          wf[2*j-1] = -vz*pow((1.0-pow(vf02,2)-vy*vy-vz*vz),-0.5);

	      uf[2*j-2] = pinv1+0.01*sin(mfrq*2.0*M_PI*xf[2*j-2]/lx);
          uf[2*j-1] = pinv2+0.01*sin(mfrq*2.0*M_PI*xf[2*j-1]/lx);    
          
// c my correct end
//           printf("j %10d ub(j) %15.5e uf(j) %15.5e gb0 %15.5e vb0 %15.5e vf0 %15.5e pinv  %15.5e termx %15.5e\n",
// 		  j,     ub[j-1],     uf[j-1],     gb0,       vb0,       vf01,      pinv1,       termx);
	  
#ifdef DEBUG_INITIAL_PARTICLE_PRINTS
	       printf("electron %10d %25.15e %25.15e %25.15e %25.15e \n",2*j-2,yi[j - 1],uf[2*j-2],vf[2*j-2],wf[2*j-2]);
	       printf("electron %10d %25.15e %25.15e %25.15e %25.15e \n",2*j-1,yi[j - 1],uf[2*j-1],vf[2*j-1],wf[2*j-1]);
#endif
	  }

    free(ux);
    free(uy);
    free(uz);

#ifdef DEBUG_INITIAL_PARTICLE_PRINTS
//    exit(0);
#endif

    return 0;
} /* start_ */

int AddBeamParticles(int jmb,
				   double tex0,double tey0,double tez0,
				   double beam_lx, double beam_ly,int *jmb_real,
                   double lx,double ly,double lz, int meh,double Tb,double rimp,double rbd,
				   double *xb,double *yb, double *zb,double *ub,double *vb, double *wb
				  )
{
    double x,y,z,vb0,d__1,d__2,d__3,vy,vz,termx,gb0;
    double vf01,vf02,pinv1,pinv2,mfrq;
//     double *ux,*uy,*uz;
    double *ux,*uy,*uz;
    double beam_y_max,beam_y_min, beam_sh;

    beam_sh = (ly - beam_ly)/2;
    beam_y_max = ly - beam_sh;
    beam_y_min = beam_sh;

    for (int j = 1; j <= jmb; j++)
    {
    	z =   beam_ly * rnd_uniform(0) + beam_y_min;
	    y =   meh * ly + beam_ly * rnd_uniform(0) + beam_y_min;
	    x =   beam_lx * rnd_uniform(0);

	    xb[j - 1] = x;
	    yb[j - 1] = y;
	    zb[j - 1] = z;

	    double vb0,ux,uy,uz;
        vb0       = rnd_gaussian(0.0, Tb*rimp,0);
        ux        = vb0 + rimp;
        uy        = rnd_gaussian(0.0, Tb*rimp,0);
        uz        = rnd_gaussian(0.0, Tb*rimp,0);

        vb0 = sqrt(1.0 - ux*ux - uy * uy - uz * uz);

        ub[j - 1] = ux / vb0;
        vb[j - 1] = uy / vb0;
        wb[j - 1] = uz / vb0;


#ifdef ADD_BEAM_INITIAL_PARTICLE_PRINTS
	       printf("add beam %10d %25.15e  %25.15e    %25.15e %25.15e %25.15e   %25.15e \n",
	    		         j,  xb[j - 1],yb[j - 1],vb0,    ub[j-1],vb[j - 1],wb[j - 1]);
#endif
    }
	return 0;
}

int getMassCharge(ParticleArrays *ions,ParticleArrays *electrons,ParticleArrays *beam_electrons,
		double ni,double rbd,int lp)
{
    //int lp = ((double)N)/(Nx*Ny*Nz);
	electrons->m[0]      = -ni/lp;                 //!!!!!!
	ions->m[0]           =  2.0*ni/lp*(1.0+rbd);
	beam_electrons->m[0] = electrons->m[0]*rbd*2.0;

	electrons->q_m        = -1.0;
	ions->q_m             =  1.0/1836.0;
	beam_electrons->q_m   = -1.0;

}

int AllocateMemoryForArrays(int N,int sorts)
{

	initial[0].total    = N;
    initial[1].total    = 2*N;
    initial[2].total    = N;

    diagnostics[0].total    = N;
    diagnostics[1].total    = 2*N;
    diagnostics[2].total    = N;

    sorts = 3;

    AllocateBinaryParticlesArrays(&(initial[0]),&(initial[1]),&(initial[2]));
    AllocateBinaryParticlesArraysFloat(&(diagnostics[0]),&(diagnostics[1]),&(diagnostics[2]));

}


int getUniformMaxwellianParticles(std::vector<Particle>  & ion_vp,
		                           std::vector<Particle>  & el_vp,
		                           std::vector<Particle>  & beam_vp)
{

    int total = 160000;

    tex0 = 1e-3;
    tey0 = 1e-3;
    tez0 = 1e-3;
    tol  = 1e-15;

    Tb   = 0.0;
    rimp = 0.2;
    rbd  = 1.0e-2;
    ni   = 1.0;
    meh  = 0;
    lp   = 100;

    int AllocateMemoryForArrays(total,3);

	InitUniformMaxwellianParticles(1,N,tex0,tey0,tez0,
					  beam_max.x,beam_max.y,beam_max.z(),&jmb_real,
					  xmax.x,xmax.y,xmax.z(),meh,Tb,rimp,rbd,
						  initial[0].dbg_x,initial[0].dbg_y,initial[0].dbg_z,
						  initial[0].dbg_px,initial[0].dbg_py,initial[0].dbg_pz,
						  initial[2].dbg_x,initial[2].dbg_y,initial[2].dbg_z,
						  initial[2].dbg_px,initial[2].dbg_py,initial[2].dbg_pz,
						  initial[1].dbg_x,initial[1].dbg_y,initial[1].dbg_z,
						  initial[1].dbg_px,initial[1].dbg_py,initial[1].dbg_pz);


	return 0;


}
