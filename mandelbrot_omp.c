/*
Parallel implementation of Mandelbrot set using OpenMP
*/

#include<stdio.h>
#include<stdlib.h>
#include"timer.h"
#include<omp.h>

#define xmax 2.0
#define xmin -2.0
#define ymax 2.0
#define ymin -2.0
#define N 1000				//number of divisions on grid
#define itr 10000			//number of iterations for each point
#define NUM_THREADS 5
#define CHUNK 100			//chunk size for dynamic scheduling

//check if z is bounded
int zcheck(double zr, double zi){
  double zmod_sq;
 
	zmod_sq = (zr*zr)+(zi*zi);
	if(zmod_sq<=4){
	  return 1;
	}
	else{
	  return 0;
	}
}

//main function
int main(){
	int i,j,k,ifcontinue;
	double zr, zi, zr2;
	double dx,dy;
	double *x=NULL,*y=NULL;
	
	int **c=NULL;				
	//c stores 1 if point belongs to mandelbrot set else stores 0
	
	int tid;

	FILE *fmbrot_omp = fopen("fmbrot_omp.dat","w+");

	//allocate memory for x[] y[] c[][]
	x = (double*)malloc((N+1)*sizeof(double));		
	y = (double*)malloc((N+1)*sizeof(double));
	c = (int**)malloc((N+1)*sizeof(int*));
	for(i=0;i<N+1;i++){
	  c[i] = (int*)malloc((N+1)*sizeof(int));
	}

	//calculate increment
	dx = (xmax-xmin)/N;
	dy = (ymax-ymin)/N;

	//initialize arrays
	for(i=0;i<N+1;i++){
	  x[i] = xmin + (dx*i);
	  y[i] = ymin + (dy*i);
	  for(j=0;j<N+1;j++){
	    c[i][j] = 0;
	  }
	}

	timespec before, after;
	get_time(&before);

	//create parallel threads
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel shared(x,y,c,dx,dy) private(i,j,k,zr,zi,zr2,ifcontinue)
	{
  		tid = omp_get_thread_num(); 
		printf("%d\n",tid);
	
		//calculations for mandelbrot
		#pragma omp for schedule(dynamic, CHUNK) 
		for(j=0;j<=N;j++){
			for(i=0;i<=N;i++){
				k=0;
				zr=0.0;
				zi=0.0;
				do{
					ifcontinue = zcheck(zr,zi);				
					zr2 = zr;
					zr = (zr*zr) - (zi*zi) + x[i];
					zi = (2*zr2*zi) + y[j];
					k++;  
					if(k==itr){
					  c[i][j]=1;
					}
				}while(k<itr && ifcontinue==1);
			}
		}
    }
	get_time(&after);
	
	timespec time_diff;
	diff(&before,&after,&time_diff);
	double time_s=time_diff.tv_sec+(double)(time_diff.tv_nsec)/1.0e9;
	printf("time = %.09lf\n", time_s);

	//print points belonging to set
	for (i=0;i<=N;i++){
	  for (j=0;j<=N;j++){
		if(c[i][j]==1){
		  fprintf(fmbrot_omp,"%.03lf\t%.03lf\n",x[i],y[j]);
		}
	  }
	}
	fclose(fmbrot_omp);
		
	return 0;
}

