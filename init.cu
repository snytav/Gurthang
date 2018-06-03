
int InitializeGPU()
{
    InitGPUParticles();
    InitGPUFields(&d_Ex,&d_Ey,&d_Ez,
    	          &d_Hx,&d_Hy,&d_Hz,
    		      &d_Jx,&d_Jy,&d_Jz,
    		      &d_npJx,&d_npJy,&d_npJz,
                  &d_Qx,&d_Qy,&d_Qz,
                  Ex,Ey,Ez,
				  Hx,Hy,Hz,
				  Jx,Jy,Jz,
				  npJx,npJy,npJz,
				  Qx,Qy,Qz,
				  Nx,Ny,Nz
            );

    setPrintfLimit();

    int err = cudaSetDevice(0);

    printf("InitializeGPU error %d \n",err);

    return 0;
}

int initMeshArrays()
{
	   initControlPointFile();

	   Alloc();

	   Cell<Particle> c000;

	   InitCells();
	   c000 = (*AllCells)[0];

	   InitFields();
	   c000 = (*AllCells)[0];
	   InitCurrents();

	   return 0;
}

int LoadParticleData(int nt,
		               std::vector<Particle> & ion_vp,
		               std::vector<Particle> & el_vp,
		               std::vector<Particle> & beam_vp)
{
	 if(nt > 1)
	 {
		 ClearAllParticles();
	 }

	 FILE *f;

	 string part_name = getBinaryFileName(nt);

	 if((f = readPreliminary3Darrays(part_name,nt)) == NULL) return 1;

//		 std::vector<Particle> ion_vp,el_vp,beam_vp;

	 readBinaryParticlesAllSorts(f,nt,ion_vp,el_vp,beam_vp);

	 fclose(f);




   magf = 1;

	 return 0;
}

//	  int LoadData(i)

void LoadTestData(int nt,
		            int part_nt,
		            std::vector<Particle> & ion_vp,
		            std::vector<Particle> & el_vp,
		            std::vector<Particle> & beam_vp)
{
   LoadMeshData(nt);

   LoadParticleData(nt,ion_vp,el_vp,beam_vp);


}

void AssignArraysToCells()
{
   for(int n = 0;n < (*AllCells).size();n++)
   {

       Cell<Particle> c = (*AllCells)[n];
//	         if(c.i == 1 &&  c.l  == 1 && c.k == 1)
//	         {
////	         	   int j = 0;
//	         }
	     c.readFieldsFromArrays(Ex,Ey,Ez,Hx,Hy,Hz);
   }
}



virtual void InitializeCPU()
{
   std::vector<Particle> ion_vp,el_vp,beam_vp;

   initMeshArrays();

   LoadTestData(START_STEP_NUMBER,START_STEP_NUMBER, ion_vp,el_vp,beam_vp);

   addAllParticleListsToCells(ion_vp,el_vp,beam_vp);

   AssignArraysToCells();


}

void Initialize()
{
	InitializeCPU();
	copyCellsWithParticlesToGPU();
	InitializeGPU();
}





void InitGPUParticles()
 //   :InitParticles(fname,vp)
{
	int size;
	GPUCell<Particle> *d_c,*h_ctrl;
	GPUCell<Particle> *n;
	GPUCell<Particle> *h_copy,*h_c;
	double t;
	dim3 dimGrid(Nx+2,Ny+2,Nz+2),dimBlockOne(1,1,1);

	 readControlFile(START_STEP_NUMBER);

	size = (*AllCells).size();

	 size_t m_free,m_total;

	h_ctrl = new Cell<Particle>;
	n = new Cell<Particle>;

    h_CellArray = (Cell<Particle> **)malloc(size*sizeof(Cell<Particle> *));
    cudaError_t err = cudaMalloc(&d_CellArray,size*sizeof(Cell<Particle> *));

//    h_controlParticleNumberArray = (int*)malloc(size*sizeof(int));

    printf("%s : size = %d\n", __FILE__, size);
    for(int i = 0;i < size;i++)
    {
        //printf("GPU cell %d begins******************************************************\n",i);
    	GPUCell<Particle> c;
    	c = (*AllCells)[i];

    	h_controlParticleNumberArray[i] = c.number_of_particles;
    	/////////////////////////////////////////
    	*n = c;
#ifdef ATTRIBUTES_CHECK
    	c.SetControlSystem(jmp,d_ctrlParticles);
#endif

    	//t = c.compareToCell(*n);

       // puts("COMPARE------------------------------");
    	//printf("%d: %d\n", i, c.busyParticleArray);
        d_c = c.copyCellToDevice();
        cudaMemGetInfo(&m_free,&m_total);
        double mfree,mtot;
        mfree = m_free;
        mtot  = m_total;
#ifdef COPY_CELL_PRINTS
        printf("cell %10d Device cell array allocated error %d %s memory: free %10.2f total %10.2f\n",i,err,cudaGetErrorString(err),
        		                                                mfree/1024/1024/1024,mtot/1024/1024/1024);
        puts("");

	  dbgPrintGPUParticleAttribute(d_c,50,1," CO2DEV " );
	  puts("COPY----------------------------------");
#endif


#ifdef PARTICLE_PRINTS

        if(t < 1.0)
        {
        	t = c.compareToCell(*h_copy);
        }
#endif
        ////////////////////////////////////////.
        h_CellArray[i] = d_c;
        cudaMemcpy(h_ctrl,d_c,sizeof(Cell<Particle>),cudaMemcpyDeviceToHost);
#ifdef InitGPUParticles_PRINTS
	    dbgPrintGPUParticleAttribute(d_c,50,1," CPY " );

       cudaPrintfInit();

        testKernel<<<1,1>>>(h_ctrl->d_ctrlParticles,h_ctrl->jmp);
        cudaPrintfDisplay(stdout, true);
        cudaPrintfEnd();

        printf("i %d l %d k n %d %d %e src %e num %d\n",h_ctrl->i,h_ctrl->l,h_ctrl->k,i,
        		c.ParticleArrayRead(0,7),c.number_of_particles
        		);
	printf("GPU cell %d ended ******************************************************\n",i);
#endif
    }

    //cudaError_t err;
    err = cudaMemcpy(d_CellArray,h_CellArray,size*sizeof(Cell<Particle> *),cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        {
         	printf("bGPU_WriteControlSystem err %d %s \n",err,cudaGetErrorString(err));
        	exit(0);
        }

//	d_AllCells = new thrust::device_vector<Cell<Particle> >(size);

//	*d_AllCells = (*AllCells);
#ifdef ATTRIBUTES_CHECK
    GPU_WriteControlSystem<<<dimGrid, dimBlockOne,16000>>>(d_CellArray);
#endif
	size = 0;

}


virtual void Alloc()
	  {

		  AllCells = new thrust::host_vector<Cell<Particle> >;

	     Ex  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Ey  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Ez  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Hx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Hy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Hz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Jx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Jy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Jz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Rho = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	     npJx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     npJy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     npJz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	     npEx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     npEy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     npEz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	     Qx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Qy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     Qz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	#ifdef DEBUG_PLASMA

	     dbgEx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgEy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgEz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgEx0  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgEy0  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgEz0  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	     dbgHx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgHy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgHz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgJx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgJy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbgJz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

	     dbg_Qx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbg_Qy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	     dbg_Qz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
	#endif
	  }

	  virtual void InitFields()
	  {
	     for(int i = 0;i < (Nx+2)*(Ny+2)*(Nz+2);i++)
	     {
	         Ex[i] = 0.0;
	         Ey[i] = 0.0;
	         Ez[i] = 0.0;
	         Hx[i] = 0.0;
	         Hy[i] = 0.0;
	         Hz[i] = 0.0;

	         dbgEx[i] = 0.0;
	         dbgEy[i] = 0.0;
	         dbgEz[i] = 0.0;
	         dbgHx[i] = 0.0;
	         dbgHy[i] = 0.0;
	         dbgHz[i] = 0.0;
	     }
	  }

	  virtual void InitCells()
	  {
	     for(int i = 0;i < Nx+2;i++)
	     {
	         for(int l = 0;l < Ny+2;l++)
		 {
		     for(int k = 0;k < Nz+2;k++)
		     {
	                 Cell<Particle> * c = new Cell<Particle>(i,l,k,Lx,Ly,Lz,Nx,Ny,Nz,tau);
	                 c->Init();
			         (*AllCells).push_back(*c);
#ifdef INIT_CELLS_DEBUG_PRINT
	                printf("%5d %5d %5d size %d \n",i,l,k,(*AllCells).size());
#endif
		     }

		 }

	     }
	  }

	  virtual void InitCurrents()
	  {
	     for(int i = 0;i < (Nx+2)*(Ny+2)*(Nz+2);i++)
	     {
	         Jx[i]  = 0.0;
	         Jy[i]  = 0.0;
	         Jz[i]  = 0.0;
	         Rho[i] = 0.0;

	         dbgJx[i]  = 0.0;
	         dbgJy[i]  = 0.0;
	         dbgJz[i]  = 0.0;

	     }
	  }

	  void InitCurrents(string fnjx,string fnjy,string fnjz,
	                    string dbg_fnjx,string dbg_fnjy,string dbg_fnjz,
	                    string np_fnjx,string np_fnjy,string np_fnjz,
			            int dbg)
	  {

	     read3Darray(np_fnjx, npJx);
	     read3Darray(np_fnjy, npJy);
	     read3Darray(np_fnjz, npJz);

	     if(dbg == 0)
	     {
	        read3Darray(fnjx, Jx);
	        read3Darray(fnjy, Jy);
	        read3Darray(fnjz, Jz);
	     }
	#ifdef DEBUG_PLASMA
	     read3Darray(dbg_fnjx, dbgJx);
	     read3Darray(dbg_fnjy, dbgJy);
	     read3Darray(dbg_fnjz, dbgJz);

	#endif
	   }

	  void InitFields(char *fnex,char *fney,char *fnez,
			          char *fnhx,char *fnhy,char *fnhz,
			          char *dbg_fnex,char *dbg_fney,char *dbg_fnez,
			          char *dbg_0fnex,char *dbg_0fney,char *dbg_0fnez,
			          char *np_ex,char *np_ey,char *np_ez,
			          char *dbg_fnhx,char *dbg_fnhy,char *dbg_fnhz)
	  {
	     InitFields();

	      //double t271 = Hy[27];

	     read3Darray(fnex, Ex);
	     read3Darray(fney, Ey);
	     read3Darray(fnez, Ez);
	     read3Darray(fnhx, Hx);
	     read3Darray(fnhy, Hy);
	     read3Darray(fnhz, Hz);

	#ifdef DEBUG_PLASMA
	     read3Darray(dbg_fnex, dbgEx);
	     read3Darray(dbg_fney, dbgEy);
	     read3Darray(dbg_fnez, dbgEz);

	     read3Darray(dbg_0fnex, dbgEx0);
	     read3Darray(dbg_0fney, dbgEy0);
	     read3Darray(dbg_0fnez, dbgEz0);

	     read3Darray(dbg_fnhx, dbgHx);
	     read3Darray(dbg_fnhy, dbgHy);
	     read3Darray(dbg_fnhz, dbgHz);

	     read3DarrayLog(np_ex, npEx,50,8);
	     read3DarrayLog(np_ey, npEy,50,8);
	     read3DarrayLog(np_ez, npEz,50,8);
	#endif

	  //   double t27 = Hy[27];

	  }


	  virtual void InitParticles(thrust::host_vector<Particle> & vp)
	  {
	     InitIonParticles(n_per_cell,ion_q_m,vp);
	  }

	  virtual void InitParticles(char *fname,thrust::host_vector<Particle>& vp)
	  {
	     FILE *f;
	     char str[1000];
	     double x,y,z,px,py,pz,q_m,m;
	     int n = 0;

	     if((f = fopen(fname,"rt")) == NULL) return;

	     while(fgets(str,1000,f) != NULL)
	     {
	          x   = atof(str);
	          y   = atof(str + 25);
	          z   = atof(str + 50);
	          px  = atof(str + 75);
	          py  = atof(str + 100);
	          pz  = atof(str + 125);
	          m   = fabs(atof(str + 150));
	          q_m = atof(str + 175);
	#undef GPU_PARTICLE
		  Particle *p = new Particle(x,y,z,px,py,pz,m,q_m);
//		      if(n == 829)
//		      {
//		    	  int qq = 0;
//		    	  qq = 1;
//		      }
		  p->fortran_number = ++n;
		  vp.push_back(*p);
	#define GPU_PARTICLE

	     }


         dbg_x = (double *)malloc(sizeof(double)*vp.size());
         dbg_y = (double *)malloc(sizeof(double)*vp.size());
         dbg_z = (double *)malloc(sizeof(double)*vp.size());
         dbg_px = (double *)malloc(sizeof(double)*vp.size());
         dbg_py = (double *)malloc(sizeof(double)*vp.size());
         dbg_pz = (double *)malloc(sizeof(double)*vp.size());

         total_particles = vp.size();

	     magf = 1;
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

      virtual int readBinaryParticleArraysOneSort(
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
		     Cell<Particle> c0 = (*AllCells)[0];
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
	         debugPrintParticleCharacteristicArray(*dbg_x,total_particles,nt,"x",sort);

	         *dbg_y = (double *)malloc(sizeof(double)*total_particles);
	         debugPrintParticleCharacteristicArray(*dbg_y,total_particles,nt,"y",sort);

	         *dbg_z = (double *)malloc(sizeof(double)*total_particles);
	         debugPrintParticleCharacteristicArray(*dbg_z,total_particles,nt,"z",sort);

	         *dbg_px = (double *)malloc(sizeof(double)*total_particles);
	         debugPrintParticleCharacteristicArray(*dbg_px,total_particles,nt,"px",sort);

	         *dbg_py = (double *)malloc(sizeof(double)*total_particles);
	         debugPrintParticleCharacteristicArray(*dbg_py,total_particles,nt,"py",sort);

	         *dbg_pz = (double *)malloc(sizeof(double)*total_particles);
	         debugPrintParticleCharacteristicArray(*dbg_pz,total_particles,nt,"pz",sort);

		 	readFortranBinaryArray(f,*dbg_x);
		 	readFortranBinaryArray(f,*dbg_y);
		 	readFortranBinaryArray(f,*dbg_z);
		 	readFortranBinaryArray(f,*dbg_px);
		 	readFortranBinaryArray(f,*dbg_py);
		 	readFortranBinaryArray(f,*dbg_pz);

		 	//printf("particle 79943 %25.15e \n",(*dbg_x)[79943]);

		 	*qq_m = q_m;
		 	*mm   = m;

		 	if((err = ferror(f)) != 0)
            {
	   	 	    return err ;
			}

		 	return total_particles;
      }

      void printPICstatitstics(double m,double q_m, int total_particles)
      {
    	  int pn_min,pn_ave,pn_max,pn_sum,err;

              pn_min = 1000000000;
              pn_max = 0;
              pn_ave = 0;
 		     for(int n = 0;n < (*AllCells).size();n++)
 		     {
 		    	 Cell<Particle> & c = (*AllCells)[n];

 		    	 pn_ave += c.number_of_particles;
 		    	 if(pn_min > c.number_of_particles) pn_min = c.number_of_particles;
 		    	 if(pn_max < c.number_of_particles) pn_max = c.number_of_particles;

 		     }

 		     pn_sum = pn_ave;
 		     pn_ave /= (*AllCells).size();

 		     printf("SORT m %15.5e q_m %15.5e %10d (sum %10d) particles in %8d cells: MIN %10d MAX %10d average %10d \n",
 		    		 m,            q_m,       total_particles,pn_sum,
 		    		 (*AllCells).size(),
 		    		 pn_min,pn_max,pn_ave);


      }


      int addParticleListToCells(std::vector<Particle>& vp)
      {
    	  Cell<Particle> c0 = (*AllCells)[0];
    	  int n;

    	  for(int i = 0; i < vp.size();i++)
    	  {
    	      Particle p = vp[i]; // = new Particle(x,y,z,px,py,pz,m,q_m);

    	  	  double3 d;
    	      d.x = p.x;
    	      d.y = p.y;
    	      d.z = p.z;

    	      n = c0.getPointCell(d);

    	      Cell<Particle> & c = (*AllCells)[n];


    	      if(c.Insert(p) == true)
    	      {
#ifdef PARTICLE_PRINTS1000
    	  		             if((i+1)%1000 == 0 )
    	  		             {
    	  		        	     printf("particle %d (%e,%e,%e) is number %d in cell (%d,%d,%d)\n",
    	  		        	    		 i+1,
    	  				    		x,y,z,c.number_of_particles,c.i,c.l,c.k);
    	  		             }
#endif
    	  			      }
   		      }// END total_particles LOOP

      }

      int convertParticleArraysToSTLvector(
    		  double *dbg_x,
    		  double *dbg_y,
			  double *dbg_z,
			  double *dbg_px,
			  double *dbg_py,
			  double *dbg_pz,
			  double q_m,
			  double m,
			  int total_particles,
			  particle_sorts sort,
    		  std::vector<Particle> & vp
    		  )
      {
    	  double x,y,z,px,py,pz;

    	  for(int i = 0; i < total_particles;i++)
    	  {
			  x   = dbg_x[i];
			  y   = dbg_y[i];
			  z   = dbg_z[i];
			  px   = dbg_px[i];
			  py   = dbg_py[i];
			  pz   = dbg_pz[i];


			  Particle p(x,y,z,px,py,pz,m,q_m);

			  p.fortran_number = i+1;
			  p.sort = sort;

			  vp.push_back(p);

    	  }
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

 		     int err;

 		     if((err = ferror(f)) != 0) return 1;

 		     total_particles = readBinaryParticleArraysOneSort(f,&dbg_x,&dbg_y,&dbg_z,
 		    		                                             &dbg_px,&dbg_py,&dbg_pz,q_m,m,nt,
 		    		                                             sort);

 		     real_number_of_particle[(int)sort] = total_particles;

 		     if((err = ferror(f)) != 0) return 1;
 		     convertParticleArraysToSTLvector(dbg_x,dbg_y,dbg_z,dbg_px,dbg_py,dbg_pz,*q_m,*m,
 		    			  total_particles,sort,vp);

      }

	  virtual void readBinaryParticlesOneSort(FILE *f,std::vector<Particle> & vp,
			                                  particle_sorts sort,int nt)

	  {
		    double q_m,m;
		    int err;
		    getParticlesOneSortFromFile(f,sort,nt,vp,&q_m,&m);

		    err = ferror(f);
		    free(dbg_x);
			free(dbg_y);
			free(dbg_z);
		    free(dbg_px);
			free(dbg_py);
			free(dbg_pz);
			err = ferror(f);

			struct sysinfo info;
            sysinfo(&info);
			printf("before1  %d free %u \n",nt,info.freeram/1024/1024);
			err = ferror(f);

			printPICstatitstics(m,q_m,total_particles);
	  }



	  FILE *readPreliminary3Darrays(string fn,int nt)
	  {
		     double *buf;
		     FILE *f;

		     buf = (double *)malloc(sizeof(double)*(Nx+2)*(Ny+2)*(Nz+2));

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

	  int addAllParticleListsToCells(std::vector<Particle> & ion_vp,
			                         std::vector<Particle> & el_vp,
			                         std::vector<Particle> & beam_vp)
	  {
			 addParticleListToCells(ion_vp);
			 addParticleListToCells(el_vp);
			 addParticleListToCells(beam_vp);

			 return 0;
	  }

	  int readBinaryParticlesAllSorts(FILE *f,int nt,
			                          std::vector<Particle> & ion_vp,
                                      std::vector<Particle> & el_vp,
                                      std::vector<Particle> & beam_vp)
	  {
		  readBinaryParticlesOneSort(f,ion_vp,ION,nt);
          readBinaryParticlesOneSort(f,el_vp,PLASMA_ELECTRON,nt);
    	  readBinaryParticlesOneSort(f,beam_vp,BEAM_ELECTRON,nt);

    	  return 0;
	  }

	  int readParticles(FILE *f,int nt)
	  {
		 std::vector<Particle> ion_vp,el_vp,beam_vp;

		 readBinaryParticlesAllSorts(f,nt,ion_vp,el_vp,beam_vp);


		 addAllParticleListsToCells(ion_vp,el_vp,beam_vp);

		 return 0;
	  }

	  string getBinaryFileName(int nt)
	  {
		  char part_name[100];
		  string s;

		  sprintf(part_name,"mumu000%08d.dat",nt);

		  s = part_name;

		  return s;
	  }

	  virtual void InitBinaryParticles(int nt)
	  {
	     FILE *f;
	     string part_name = getBinaryFileName(nt);

		 if((f = readPreliminary3Darrays(part_name,nt)) == NULL) return;

		 std::vector<Particle> ion_vp,el_vp,beam_vp;

		 readBinaryParticlesAllSorts(f,nt,ion_vp,el_vp,beam_vp);


		 addAllParticleListsToCells(ion_vp,el_vp,beam_vp);

	     fclose(f);

	     magf = 1;
	  }



	  virtual void InitElectronParticles(){}
	  virtual void InitIonParticles(int n_per_cell1,double q_m,thrust::host_vector<Particle> &vecp)
	  {
	     int total_ions = Nx*Ny*Nz*n_per_cell;
	     Particle *p;
	     //double ami = ni /((double)n_per_cell);
	     double x,y,z;

	     for(int j = 0;j < total_ions;j++)
	     {
		z = Lz * rnd_uniform();
		y = meh * Ly + Ly * rnd_uniform();
		x = Lx * rnd_uniform();

		p = new Particle(x,y,z,0.0,0.0,0.0,ni,q_m);

	#ifdef DEBUG_PLASMA
//		printf("particle %d \n",j);
	#endif

		vecp.push_back(*p);
	     }
	  }

	  virtual void InitBeamParticles(int n_per_cell1){}
	  void Distribute(thrust::host_vector<Particle> &vecp)
	  {
	     Cell<Particle> c0 = (*AllCells)[0],c111;
	     int    n;//,i;
	     int  vec_size = vecp.size();

	     for(int j = 0;j < vecp.size();j++)
	     {
		 Particle p = vecp[j];
		 double3 d;
		 d.x = p.x;
		 d.y = p.y;
		 d.z = p.z;

		 n = c0.getPointCell(d);

		 Cell<Particle> & c = (*AllCells)[n];;
	//	 c.SetZero();
	//	 c = (*AllCells)[n];
//		 if((vec_size-vecp.size()) == 136)
//		 {
//			 i = 0;
//		 }

//		 if(c.i == 0 && c.k == 0 && c.l == 0)
//		 {
//			 int z67 = 0;
//		 }

		 if(c.Insert(p) == true)
		 {
	#ifdef PARTICLE_PRINTS1000
         if((vec_size-vecp.size())%1000 == 0 )	printf("particle %d (%e,%e,%e) is number %d in cell (%d,%d,%d)\n",vec_size-vecp.size(),
		    		p.x,p.y,p.z,c.number_of_particles,c.i,c.l,c.k);
         if((vec_size-vecp.size()) == 10000) exit(0);
	#endif
		    vecp.erase(vecp.begin()+j);
		    j--;
		 }
	     }
	     int pn_min = 1000000,pn_max = 0,pn_ave = 0;
	     for(int n = 0;n < (*AllCells).size();n++)
	     {
	    	 Cell<Particle> & c = (*AllCells)[n];

	    	 pn_ave += c.number_of_particles;
	    	 if(pn_min > c.number_of_particles) pn_min = c.number_of_particles;
	    	 if(pn_max < c.number_of_particles) pn_max = c.number_of_particles;

	     }
	     pn_ave /= (*AllCells).size();

	     printf("%10d particles in %8d cells: MIN %5d MAX %5d average %5d \n",vec_size,(*AllCells).size(),
	    		                                                              pn_min,pn_max,pn_ave);
	  }

