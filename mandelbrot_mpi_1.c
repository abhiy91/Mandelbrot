/*
Parallel implementation of Mandelbrot set using MPI
*/

#include<stdio.h>
#include<stdlib.h>
#include <mpi.h>

#define N 1000
#define ITR 10000
#define RE_POSMAX 2.0
#define RE_POSMIN -2.0
#define IM_POSMAX 2.0
#define IM_POSMIN -2.0
#define RE_RANGE (RE_POSMAX-RE_POSMIN)
#define IM_RANGE (IM_POSMAX-IM_POSMIN)
#define ROOT 0

//function to test if point belongs to mandelbrot set
int m_pixel_calc(double c_re, double c_im){
	int count = 0;
	double temp, zmod_sq;
	double z_re = 0.0, z_im = 0.0;

	do
	{
		temp = z_re * z_re - z_im * z_im + c_re;
		z_im = (2*z_re*z_im) + c_im;
		z_re = temp;

		zmod_sq = (z_re * z_re) + (z_im * z_im);

		count++;
	}while((zmod_sq < 4.0) && (count < ITR));

	if(count == ITR){
		return 1;
	}else{
		return 0;
	}
}

//main function
int main(int argc, char *argv[]){
	int i, j, k;
	int nprocs,myid,ierr;
	int rem, div;
	int p_rows, p_start, p_end;
	int **m_grid_myid, **m_grid_root;
	int *data_myid, *data_root;
	double c_re, c_im; 
	MPI_Status *status;
	FILE *fp,*fp2;
	double t_start, t_end, t;

	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	//start timer on root process
	if(myid==ROOT){
		t_start = MPI_Wtime();
	}

	//allocate memory on root processor to gather results
	if(myid==ROOT){
		m_grid_root = (int**)malloc(N*sizeof(int *));
		data_root = (int*)malloc(N*N*sizeof(int));

		for(i=0; i<N; ++i){
			m_grid_root[i] = &data_root[i*N];
		}
	}

	//arrays used for displacement and sizes in gatherv
	int displ[nprocs]; 
	int rcounts[nprocs];

	//calculate the no.of rows, start point and end point for each process
	p_rows = N/nprocs;
	p_start = p_rows*myid;

	//allocates extra rows if rem!=0
	rem = (N%nprocs);
	div = (N/nprocs);

	if(myid<rem){
		p_rows= div + 1;
	}else{
		p_rows = div;
	}

	if(myid==ROOT){
		p_start = 0;
	}else if((myid-1)<rem){
		p_start = ((myid * div) + myid);
	}else{
		p_start = ((myid * div) + rem);
	}

	p_end = p_start + p_rows - 1;

	//allocate local memory on each process to store its part of the result
	m_grid_myid = (int**)malloc(p_rows*sizeof(int*));
	data_myid = (int*)malloc(N*p_rows*sizeof(int));

	for(i=0; i<p_rows; ++i){
		m_grid_myid[i] = &data_myid[i*N];
	}

	//calculate mandelbrot set
	for(i=p_start; i<=p_end; ++i){
		for(j=0; j<N; ++j){
			c_re = (RE_POSMIN) + (((RE_RANGE)/(N))*j);
			c_im = (IM_POSMAX) - (((IM_RANGE)/(N))*i);
			m_grid_myid[i-p_start][j] = m_pixel_calc(c_re,c_im);
		}
	}

	//populate rcounts and displ
	for(i=0;i<nprocs;++i){    
		if(i<rem){
		rcounts[i]=N*(div + 1);
		}else{
			rcounts[i]=N*div;
		}

		if(i==0){
			displ[i] = 0;
		}else if((i-1)<rem){
			displ[i] = N*((i * div) + i);
		}else{
			displ[i] = N*((i * div) + rem);
		}
	}

	//gather points to root process
	ierr=MPI_Gatherv(data_myid, p_rows*N, MPI_INT, data_root, rcounts, displ, MPI_INT, 0, MPI_COMM_WORLD);

	//free memory on local process
	free(data_myid);
	free(m_grid_myid);

	//end timer
	if(myid==ROOT){
		t_end = MPI_Wtime();
		printf("time = %.09lf\n", (t_end-t_start));
	}

	//print points to file
	if(myid == 0){
		fp = fopen("dots.dat", "w");
		fp2 = fopen("points.dat", "w");

		for(i=0; i<N; ++i){
			for(j=0; j<N; ++j){
				fprintf(fp, "%d ", m_grid_root[i][j]);
				if(m_grid_root[i][j]==1){
					fprintf(fp2, "%.03lf\t%.03lf\n", (RE_POSMIN) + (((RE_RANGE)/(N))*j),(IM_POSMAX) - (((IM_RANGE)/(N))*i));
				} 
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
		fclose(fp2);
	} 


	//free memory on root process
	if(myid==0){
	free(m_grid_root[0]);
	free(m_grid_root);
	}

	ierr=MPI_Finalize();
	return 0;
}
