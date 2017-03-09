/*
MPI implementation of Mandelbrot set using master-slave dynamic load allocation
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
#define CHUNK 1000


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

int main(int argc, char *argv[]){
	int i, j, k;
	int nprocs,myid,ierr;
	int rem, div;
	int p_rows, p_start, p_end;
	int **m_grid_myid, **m_grid_root;
	int *data_myid, *data_root, *data_root_temp;
	double c_re, c_im; 
	MPI_Status status;
	FILE *fp,*fp2;
	double t_start, t_end, t;
	int sent, recvd, end_flag, end, sender;

	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	//it requires 2 or more processors
	if(nprocs<2){
		printf("it requires 2 or more processors\n");
	}

	//start timer on root process
	if(myid==ROOT){
		t_start = MPI_Wtime();
	}

	div = (N*N)/(CHUNK);
	rem = (N*N)%(CHUNK);

	if(myid==ROOT){
		//allocate memory on root(master) processor to gather points
		m_grid_root = (int**)malloc(N*sizeof(int *));
		data_root = (int*)malloc(N*N*sizeof(int));

		for(i=0; i<N; ++i){
			m_grid_root[i] = &data_root[i*N];
		}

		//allocate memory for temporary receive buffer on master
		data_root_temp = (int*)malloc(CHUNK * sizeof(int));

		//initialize counters
		//initial points on workers
		sent = CHUNK * (nprocs-1);

		//none recvd till now
		recvd = 0;

		//stopping criteria
		if(rem==0){
			end = div;
		}else{
			end = div +1;
		}

		//end flag
		end_flag = -1;

		//start receiving messages from servers
		while(recvd<end){
			MPI_Recv(data_root_temp, CHUNK, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			sender = status.MPI_SOURCE;
			k = status.MPI_TAG;

			//update recvd
			++recvd;

			//use tag as the indicator in m_grid_root
			for(i=k; i<(k+CHUNK); ++i){
				m_grid_root[i/N][i%N] = data_root_temp[i-k];
			}

			//send chunk to idle worker
			if(sent<(N*N)){
				MPI_Send(&sent, 1, MPI_INT, sender, 0, MPI_COMM_WORLD);

				//update sent
				sent += CHUNK;
			}else{
				//send end signal
				MPI_Send(&end_flag, 1, MPI_INT, sender, 0, MPI_COMM_WORLD);
			}
		}
		free(data_root_temp);
	}

	//workers
	if(myid!=ROOT){

		//allocate memory for local storage of chunk
		data_myid = (int*)malloc(CHUNK*sizeof(int));

		//initial k on each processor
		k = (CHUNK)*(myid);

		while((k != -1) && (k < (N*N))){
			for(i=k; i<(k+(CHUNK)); ++i){
				c_re = ((double)RE_POSMIN) + ((double)RE_RANGE/N)*(i%N);
				c_im = ((double)IM_POSMAX) - ((double)IM_RANGE/N)*(i/N);
				data_myid[i-k] = m_pixel_calc(c_re, c_im);  	
			}  
										 
			//send chunk back to root
			MPI_Send(data_myid, CHUNK, MPI_INT, 0, k, MPI_COMM_WORLD);
			//receive next chunk from root
			MPI_Recv(&k, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}   
		free(data_myid);
	} 


	//end timer
	if(myid==ROOT){
		t_end = MPI_Wtime();
		printf("time = %.09lf\n", (t_end-t_start));
	}

	//print points to file
	if(myid == ROOT){
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
	if(myid==ROOT){
	free(data_root);
	free(m_grid_root);
	}

	ierr=MPI_Finalize();
	return 0;
}
