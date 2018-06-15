/*
 * read_particle.cxx
 *
 *  Created on: Jun 9, 2018
 *      Author: snytav
 */

#include <stdio.h>
#include <stdlib.h>

#include <sys/resource.h>
#include <stdint.h>

#include <sys/sysinfo.h>
#include <sys/time.h>

#include <string>


#include "load_data.h"

std::string getMumuFileName(int nt)
	  {
		  char part_name[100];
		  std::string s;

		  sprintf(part_name,"mumu000%08d.dat",nt);

		  s = part_name;

		  return s;
	  }


int readFortranBinaryArray(FILE *f, double* d)
	{
//	    char str[100];
//	    Cell<Particle>  c = (*AllCells)[0];
	    int t,err;//,n;
//	    double t0;


	    //sprintf(fname,"%s_fiel3d.dat",name);
	    fread(&t,sizeof(int),1,f);
	     if((err = ferror(f)) != 0)
	    	 {
	    	 	 return err ;
	    	 }

	    fread(d,1,t,f);
	     if((err = ferror(f)) != 0)
	    	 {
	    	 	 return err ;
	    	 }

//	    t0 = d[269];
//	    t0 = d[270];
	    fread(&t,sizeof(int),1,f);
	     if((err = ferror(f)) != 0)
	    	 {
	    	 	 return err ;
	    	 }
   	    return t;
	}

FILE *readPreliminary3Darrays(std::string fn,int nt,int nx,int ny,int nz)
{
	     double *buf;
	     FILE *f;

	     buf = (double *)malloc(sizeof(double)*(nx+2)*(ny+2)*(nz+2));

	     if((f = fopen(fn.c_str(),"rb")) == NULL) return NULL;
	     struct sysinfo info;

	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);
	     readFortranBinaryArray(f,buf);
	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);
	     readFortranBinaryArray(f,buf);
	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);

	     readFortranBinaryArray(f,buf);
	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);

	     readFortranBinaryArray(f,buf);
	     sysinfo(&info);
	     printf("before1  %d free %u \n",nt,info.freeram/1024/1024);

	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);

	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);

	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);
	     readFortranBinaryArray(f,buf);
	     int err;
//--------------------------------------------
	     err = ferror(f);

	     if(err != 0) return NULL;

	     return f;
}

void debugPrintParticleCharacteristicArray(double *p_ch,int np,int nt,char *name,int sort)
{
	   char fname[200];
	   FILE *f;

#ifndef PRINT_PARTICLE_INITIALS
	   return;

#else
	   sprintf(fname,"particle_init_%s_%05d_sort%02d.dat",name,nt,sort);

	   if((f = fopen(fname,"wt")) == NULL) return;

	   for (int i = 0;i < np;i++)
	   {
		   fprintf(f,"%10d %10d %25.16e \n",i,i+1,p_ch[i]);
	   }
	   fclose(f);
#endif
}

int readBinaryParticleArraysOneSort(
		  FILE *f,
		  double **dbg_x,
		  double **dbg_y,
		  double **dbg_z,
		  double **dbg_px,
		  double **dbg_py,
		  double **dbg_pz,
		  double *qq_m,
		  double *mm,
		  int nt,
		  int sort
		  )
{
	//     char str[1000];
	     double /*x,y,z,px,py,pz,*/q_m,/* *buf,*/tp,m;
	     int t;
//		     Cell<Particle> c0 = (*AllCells)[0];
	     int total_particles;
	     int err;

	     if((err = ferror(f)) != 0)
	     {
	     	 return err ;
	     }

	     fread(&t,sizeof(int),1,f);
	     if((err = ferror(f)) != 0)
	    	 {
	    	 	 return err ;
	    	 }
	     fread(&tp,sizeof(double),1,f);
	     if((err = ferror(f)) != 0)
	    	 {
	    	 	 return err ;
	    	 }

	     total_particles = (int)tp;
	     fread(&q_m,sizeof(double),1,f);
	     if((err = ferror(f)) != 0)
	    	 {
	    	 	 return err ;
	    	 }

	     fread(&m,sizeof(double),1,f);
	     if((err = ferror(f)) != 0)
	    	 {
	    	 	 return err ;
	    	 }

	    // m = fabs(m);
	     fread(&t,sizeof(int),1,f);
	     if((err = ferror(f)) != 0)
	    	 {
	    	 	 return err ;
	    	 }

       *dbg_x = (double *)malloc(sizeof(double)*total_particles);

       *dbg_y = (double *)malloc(sizeof(double)*total_particles);

       *dbg_z = (double *)malloc(sizeof(double)*total_particles);


       *dbg_px = (double *)malloc(sizeof(double)*total_particles);

       *dbg_py = (double *)malloc(sizeof(double)*total_particles);


       *dbg_pz = (double *)malloc(sizeof(double)*total_particles);


	 	readFortranBinaryArray(f,*dbg_x);
	 	readFortranBinaryArray(f,*dbg_y);
	 	readFortranBinaryArray(f,*dbg_z);
	 	readFortranBinaryArray(f,*dbg_px);
	 	readFortranBinaryArray(f,*dbg_py);
	 	readFortranBinaryArray(f,*dbg_pz);
	 	debugPrintParticleCharacteristicArray(*dbg_x,total_particles,nt,"x",sort);
        debugPrintParticleCharacteristicArray(*dbg_y,total_particles,nt,"y",sort);
        debugPrintParticleCharacteristicArray(*dbg_z,total_particles,nt,"z",sort);
        debugPrintParticleCharacteristicArray(*dbg_px,total_particles,nt,"px",sort);
        debugPrintParticleCharacteristicArray(*dbg_py,total_particles,nt,"py",sort);
        debugPrintParticleCharacteristicArray(*dbg_pz,total_particles,nt,"pz",sort);

	 	//printf("particle 79943 %25.15e \n",(*dbg_x)[79943]);

	 	*qq_m = q_m;
	 	*mm   = m;

	 	if((err = ferror(f)) != 0)
      {
 	 	    return err ;
		}

	 	return total_particles;
}

int getParticlesOneSortFromFile(
		                          FILE *f,
                                particle_sorts sort,
                                int nt,
                                std::vector<Particle> & vp,
                                double *q_m,
                                double *m
                                )
{
	     double x,y,z,px,py,pz;
	     double *dbg_x,*dbg_y,*dbg_z,*dbg_px,*dbg_py,*dbg_pz;
	     int total_particles;


	     int err;

	     if((err = ferror(f)) != 0) return 1;

	     total_particles = readBinaryParticleArraysOneSort(f,&dbg_x,&dbg_y,&dbg_z,
	    		                                             &dbg_px,&dbg_py,&dbg_pz,q_m,m,nt,
	    		                                             sort);

	    // real_number_of_particle[(int)sort] = total_particles;

	     if((err = ferror(f)) != 0) return 1;
	     convertParticleArraysToSTLvector(dbg_x,dbg_y,dbg_z,dbg_px,dbg_py,dbg_pz,*q_m,*m,
	    			  total_particles,sort,vp);

}

std::vector<Particle> readBinaryParticlesOneSortSTL(FILE *f, particle_sorts sort,int nt)

	      	  {
	      		    double q_m,m;
	      		    int err;
	      		    std::vector<Particle> vp;
	      		    getParticlesOneSortFromFile(f,sort,nt,vp,&q_m,&m);

	      		    err = ferror(f);


	      			struct sysinfo info;
	                 sysinfo(&info);
	      			printf("before1  %d free %u \n",nt,info.freeram/1024/1024);
	      			err = ferror(f);
                    return vp;
	      //			printPICstatitstics(m,q_m,total_particles);
	      	  }

int readBinaryParticlesAllSorts(FILE *f,int nt,
			                          std::vector<Particle> & ion_vp,
                                      std::vector<Particle> & el_vp,
                                      std::vector<Particle> & beam_vp)
	  {
		  ion_vp = readBinaryParticlesOneSortSTL(f,ION,nt);
		  el_vp = readBinaryParticlesOneSortSTL(f,PLASMA_ELECTRON,nt);
    	  beam_vp = readBinaryParticlesOneSortSTL(f,BEAM_ELECTRON,nt);

    	  return 0;
	  }

int LoadParticleData(int nt,
		               std::vector<Particle> & ion_vp,
		               std::vector<Particle> & el_vp,
		               std::vector<Particle> & beam_vp, int nx,int ny,int nz)
{

	 FILE *f;

	 std::string part_name = getMumuFileName(nt);

	 if((f = readPreliminary3Darrays(part_name,nt,nx,ny,nz)) == NULL) return 1;

//		 std::vector<Particle> ion_vp,el_vp,beam_vp;

	 readBinaryParticlesAllSorts(f,nt,ion_vp,el_vp,beam_vp);

	 fclose(f);

	 return 0;
}


