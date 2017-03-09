/*
MPI implementation of Mandelbrot set by creating 2 groups of processors and using inter-communicators and intra-communicators
*/

#include<stdio.h>
#include<stdlib.h>
#include <mpi.h>

#define N 1000
#define ITR 10000
#define RE_POSMAX 2.0
#define RE_POSCEN 0.0
#define RE_POSMIN -2.0
#define IM_POSMAX 2.0
#define IM_POSCEN 0.0
#define IM_POSMIN -2.0
#define ROOT 0

//function to test if point belongs to mandelbrot set
int m_pixel_calc(double c_re, double c_im){
	int count = 0;
	double temp, zmod_sq;
	double z_re = 0.0, z_im = 0.0;

	do{
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

//function for mandelbrot computations
void prep(int* myid, int* nprocs, MPI_Comm* comm, int quad, int** data_root, int*** m_grid_root){
	int i,j;
	int ierr;
	int rem, div;
	int p_rows, p_start, p_end;
	int **m_grid_myid; 
	int *data_myid; 
	double c_re, c_im; 
	int re_range, im_range, im_start;
	int Ni = ((0.5)*N);
	FILE *fp,*fp2;

	//arrays used for displacement and sizes in gatherv
	int displ[*nprocs]; 
	int rcounts[*nprocs];

	//calculate the no.of rows, start point and end point for each process
	p_rows = (Ni)/(*nprocs);
	p_start = (p_rows)*(*myid);

	rem = ((Ni)%(*nprocs));
	div = ((Ni)/(*nprocs));

	if(*myid<rem){
		p_rows= div + 1;
	}else{
		p_rows = div;
	}

	if(*myid==ROOT){
		p_start = 0;
	}else if((*myid-1)<rem){
		p_start = ((*myid * div) + *myid);
	}else{
		p_start = ((*myid * div) + rem);
	}

	p_end = p_start + p_rows - 1;

	//allocate local memory on each process to store its part of the result
	m_grid_myid = (int**)malloc((p_rows)*sizeof(int*));
	data_myid = (int*)malloc(N* (p_rows) * sizeof(int));

	for(i=0; i<(p_rows); ++i){
		m_grid_myid[i] = &data_myid[i*N];
	}

	//check which half of the workload it is and calculate bounds
	if(quad==0){
		re_range = (RE_POSMAX) - (RE_POSMIN);
		im_range = (IM_POSMAX) - (IM_POSCEN);
		im_start = (IM_POSMAX);
	}else if(quad==1){
		re_range = (RE_POSMAX) - (RE_POSMIN);
		im_range = (IM_POSCEN) - (IM_POSMIN);
		im_start = (IM_POSCEN);
	}

	//calculate mandelbrot set
	for(i=p_start; i<=p_end; ++i){
		for(j=0; j<N; ++j){
			c_re = (RE_POSMIN) + (((double)re_range/(double)N)*j);
			c_im = (im_start) - ((((double)im_range)/((double)Ni))*i);
			m_grid_myid[i-p_start][j] = m_pixel_calc(c_re,c_im);
		}
	}

	//populate rcounts and displ
	for(i=0;i<*nprocs;++i){    
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
	ierr=MPI_Gatherv(data_myid, p_rows*N, MPI_INT, (*data_root), rcounts, displ, MPI_INT, 0, *comm);

	//free memory on local process
	free(data_myid);
	free(m_grid_myid);

	//print points to file
	if(*myid == 0){
		fp = fopen("dots.dat", "a+");
		fp2 = fopen("points.dat", "a+");

		for(i=0; i<Ni; ++i){
			for(j=0; j<N; ++j){
				fprintf(fp, "%d ", (*m_grid_root)[i][j]);
				if((*m_grid_root)[i][j]==1){
					fprintf(fp2, "%.03lf\t%.03lf\n", (double)(RE_POSMIN)+(((re_range)/(N))*j),(double)(im_start)-(((im_range)/(Ni))*i));
				} 
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
		fclose(fp2);
	}
} 

int main(int argc, char *argv[]){
	int i, j, k;
	int nprocs,myid,ierr;
	int rem, div;
	int p_rows, p_start, p_end;
	int **m_grid_myid, **m_grid_root;
	int *data_myid, *data_root;
	double c_re, c_im; 
	MPI_Status *status;
	MPI_Request *request;
	FILE *fp,*fp2;
	double t_start, t_end, t;
	int world_rank, world_size;
	int nprocs1, nprocs2;
	int **g_grid_root; 
	int *g_data;

	fp = fopen("dots.dat","w");
	fp2 = fopen("points2.dat","w");

	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	//start timer
	if(world_rank==ROOT){
		t_start = MPI_Wtime();
	}

	//create world group
	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	//divide the number of processors in half
	nprocs1 = (world_size)/(2);
	nprocs2 = (world_size)-nprocs1;

	//assign these to 2 group arrays
	int group1_rank[nprocs1];
	int group2_rank[nprocs2];

	for(i=0;i<nprocs1;++i){
		group1_rank[i]=i;
		if(i<nprocs2){
			group2_rank[i]=nprocs1+i;
		}
	}

	//construct groups
	MPI_Group group1;
	MPI_Group_incl(world_group, nprocs1, group1_rank, &group1);

	MPI_Group group2;
	MPI_Group_incl(world_group, nprocs2, group2_rank, & group2);

	//create new communicators
	MPI_Comm group1_comm;
	MPI_Comm_create_group(MPI_COMM_WORLD, group1, 1, &group1_comm);

	MPI_Comm group2_comm;
	MPI_Comm_create_group(MPI_COMM_WORLD, group2, 2, &group2_comm);

	//get process information for the new communicators
	int g1_rank=-1, g1_size=-1;

	if(MPI_COMM_NULL != group1_comm){
	MPI_Comm_rank(group1_comm, &g1_rank);
	MPI_Comm_size(group1_comm, &g1_size);
	}

	int g2_rank=-1, g2_size=-1;

	if(MPI_COMM_NULL != group2_comm){
	MPI_Comm_rank(group2_comm, &g2_rank);
	MPI_Comm_size(group2_comm, &g2_size);
	}

	int Ni = 0.5*N;

	//the grid is split into 2 halves above and below the real axis
	//each group is assigned 1 half of the work

	//mandelbrot calculations for group 1
	if(MPI_COMM_NULL != group1_comm){
		if(g1_rank==ROOT){
			g_grid_root = (int**)malloc(Ni*sizeof(int*));     
			g_data = (int*)malloc(N*Ni*sizeof(int));
			for(i=0;i<Ni;i++){
				g_grid_root[i] = &g_data[i*N];
			}
		}
		//do computations for group 1
		prep(&g1_rank, &g1_size, &group1_comm, 1, &g_data, &g_grid_root);
	}

	//mandelbrot calculations for group 2
	if(MPI_COMM_NULL != group2_comm){
		if(g2_rank==ROOT){
			g_grid_root = (int**)malloc(Ni*sizeof(int*));     
			g_data = (int*)malloc(N*Ni*sizeof(int));
			for(i=0;i<Ni;i++){
				g_grid_root[i] = &g_data[i*N];
			}
		}
		//do computations for group 2
		prep(&g2_rank, &g2_size, &group2_comm, 0, &g_data, &g_grid_root);
	}

	//end timer
	if(myid==ROOT){
		t_end = MPI_Wtime();
		printf("time = %.09lf\n", (t_end-t_start));
	}

	//free memory and communicators for group 1
	if(MPI_COMM_NULL != group1_comm){
		if(g1_rank==ROOT){
			free(g_grid_root);
			free(g_data);
		}
		MPI_Group_free(&group1);
		MPI_Comm_free(&group1_comm);
	}

	//free memory and communicators on group 2	
	if(MPI_COMM_NULL != group2_comm){
		if(g2_rank==ROOT){
			free(g_grid_root);
			free(g_data);
		}
		MPI_Group_free(&group2);
		MPI_Comm_free(&group2_comm);
	}

	//free group for world
	if(MPI_COMM_NULL != MPI_COMM_WORLD){
		MPI_Group_free(&world_group);
	}

	ierr=MPI_Finalize();
	return 0;
}
