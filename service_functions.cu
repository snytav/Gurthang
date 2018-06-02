#include <math.h>
#include <string>
#include <stdio.h>
#include "particle.h"

using namespace std;

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


double CheckArraySilent	(double* a, double* dbg_a,int size)
	{
	   // Cell<Particle> c = (*AllCells)[0];
	    double diff = 0.0;

	    for(int n = 0;n < size;n++)
	    {
            diff += pow(a[n] - dbg_a[n],2.0);

//	        if(fabs(a[n] - dbg_a[n]) > TOLERANCE)
//		    {
//
//		       int3 i = c.getCellTripletNumber(n);
//
//     		}
	    }

	    return pow(diff/(size),0.5);
	}





void get_load_data_file_names(
		string & t_jxfile,
		string & t_jyfile,
		string & t_jzfile,
		string & t_d_jxfile,
		string & t_d_jyfile,
		string & t_d_jzfile,
		string & t_np_jxfile,
		string & t_np_jyfile,
		string & t_np_jzfile,
		string & t_qxfile,
		string & t_qyfile,
		string & t_qzfile,int nt)
{


    char d_exfile[100],d_eyfile[100],d_ezfile[100],d_hxfile[100],d_hyfile[100],d_hzfile[100];
    char d_0exfile[100],d_0eyfile[100],d_0ezfile[100];
    char jxfile[100],jyfile[100],jzfile[100];
    char np_jxfile[100],np_jyfile[100],np_jzfile[100];
    char np_exfile[100],np_eyfile[100],np_ezfile[100];
    char d_jxfile[100],d_jyfile[100],d_jzfile[100];
    char qxfile[100],qyfile[100],qzfile[100];
    char pfile[100],nextpfile[100];
    char part_name[100];

    sprintf(qxfile,"dnqx%06d.dat",nt);
    sprintf(qyfile,"dnqy%06d.dat",nt);
    sprintf(qzfile,"dnqz%06d.dat",nt);



    sprintf(d_exfile,"dnex%06d.dat",2*nt-1);
    sprintf(d_eyfile,"dney%06d.dat",2*nt-1);
    sprintf(d_ezfile,"dnez%06d.dat",2*nt-1);

    sprintf(d_0exfile,"dnex%06d.dat",2*nt-2);
    sprintf(d_0eyfile,"dney%06d.dat",2*nt-2);
    sprintf(d_0ezfile,"dnez%06d.dat",2*nt-2);

    sprintf(d_hxfile,"dnhx%06d.dat",2*nt-1);
    sprintf(d_hyfile,"dnhy%06d.dat",2*nt-1);
    printf(d_hyfile);
    sprintf(d_hzfile,"dnhz%06d.dat",2*nt-1);

    sprintf(jxfile,"dnjx%06d.dat",2*nt);
    sprintf(jyfile,"dnjy%06d.dat",2*nt);
    sprintf(jzfile,"dnjz%06d.dat",2*nt);

    sprintf(d_jxfile,"npjx%06d.dat",2*nt);
    sprintf(d_jyfile,"npjy%06d.dat",2*nt);
    sprintf(d_jzfile,"npjz%06d.dat",2*nt);

    sprintf(np_jxfile,"npjx%06d.dat",2*nt);
    sprintf(np_jyfile,"npjy%06d.dat",2*nt);
    sprintf(np_jzfile,"npjz%06d.dat",2*nt);

    sprintf(np_exfile,"exlg%03d.dat",2*nt);
    sprintf(np_eyfile,"eylg%03d.dat",2*nt);
    sprintf(np_ezfile,"ezlg%03d.dat",2*nt);

    sprintf(pfile,    "part%06d000.dat",nt);
    sprintf(nextpfile,"part%06d000.dat",nt+2);


    t_jxfile =    jxfile;
    t_jyfile =    jyfile;
    t_jzfile =    jzfile;
    t_d_jxfile =  d_jxfile;
    t_d_jyfile =  d_jyfile;
    t_d_jzfile =  d_jzfile;
    t_np_jxfile = np_jxfile;
    t_np_jyfile = np_jyfile;
    t_np_jzfile = np_jzfile;
    t_qxfile =    qxfile;
    t_qyfile =    qyfile;
    t_qzfile =    qzfile;
}
