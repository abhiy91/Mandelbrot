/*
A simple serial implementation of the mandelbrot set on the comple plane
*/

#include<stdio.h>

#define xmax 2.0
#define xmin -2.0
#define ymax 2.0
#define ymin -2.0
#define N 50				//number of divisions for the grid
#define itr 50				//number of iterations for each point

int zcheck(double,double);

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
	
	double zr, zi;			//(zr,zi) point on the complex plane
	double dx,dy;

	FILE *fmbrot = fopen("fmbrot.dat","w+");
	FILE *fflag = fopen("fflag.dat","w+");

	//increments in the real and imaginary directions	
	dx = (xmax-xmin)/N;
	dy = (ymax-ymin)/N;
		
	//calculations for mandelbrot
	for(i=0;i<=N;i++){
		for(j=0;j<=N;j++){
			k=0;
			zr=0.0;
			zi=0.0;

			do{
				ifcontinue = zcheck(zr,zi);
				zr = (zr*zr) - (zi*zi) + (xmax-(dx*(i)));
				zi = (2*zr*zi) + (ymax-(dy*(j)));
				k++;  
				if(k==itr){
				  fprintf(fmbrot, "%.06lf\t%.06lf\t%d\n",xmax-(dx*(i)),ymax-(dy*(j)),k);
				}		
			}while(k<itr && ifcontinue==1);
		}
	}

	fclose(fmbrot);
	fclose(fflag);	
		
	return 0;
}

