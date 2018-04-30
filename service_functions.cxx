#include <math.h>

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
