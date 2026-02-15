#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv)
{
	
        //init the data
	MPI_Init(&argc, &argv);

	int rank, P;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);//getting_rank
	MPI_Comm_size(MPI_COMM_WORLD, &P);//get_total_count of process

	if(argc != 6) {//error message
		if(rank == 0)
			printf("Usage: %s M D1 D2 T seed\n", argv[0]);
		MPI_Finalize();
		return 0;
	}
        //parsing arguments passed to program
	int M = atoi(argv[1]);
	int D1 = atoi(argv[2]);
	int D2 = atoi(argv[3]);
	int T = atoi(argv[4]);
	int seed = atoi(argv[5]);
        //checking existence of forward and backward neighbors
	int hasD1 = (rank + D1 <= P - 1);
	int hasD2 = (rank + D2 <= P - 1);
	int preD1 = (rank - D1 >= 0);
	int preD2 = (rank - D2 >= 0);
        //memory allocation
	double *sendD1 = hasD1 ? malloc(M * sizeof(double)) : NULL;
	double *sendD2 = hasD2 ? malloc(M * sizeof(double)) : NULL;

	double *recvD1 = preD1 ? malloc(M * sizeof(double)) : NULL;
	double *recvD2 = preD2 ? malloc(M * sizeof(double)) : NULL;

	srand(seed);
	//init data only for ranks that have can send d1 ahead
	if(hasD1)
		for(int i = 0; i < M; i++) {
			double v = (double)rand() * (rank + 1) / 10000.0;
			sendD1[i] = v;
			if(hasD2) sendD2[i] = v;
		}

	//MPI_Barrier(MPI_COMM_WORLD); //this could decrease time but since it is not allowed we have commented it out
	double time_taken = MPI_Wtime();//start timing
        
	for(int iter = 0; iter < T; iter++)
	{
        /*Sending data to receivers*/

        /*D1 communication via odd-even ordering to prevent deadlock*/
		if((rank / D1) % 2) {
			if(preD1)
				MPI_Recv(recvD1, M, MPI_DOUBLE, rank-D1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(hasD1)
				MPI_Send(sendD1, M, MPI_DOUBLE, rank+D1, 1, MPI_COMM_WORLD);
		}
		else {
			if(hasD1)
				MPI_Send(sendD1, M, MPI_DOUBLE, rank+D1, 1, MPI_COMM_WORLD);
			if(preD1)
				MPI_Recv(recvD1, M, MPI_DOUBLE, rank-D1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
        //D2 communication via odd-even ordering to prevent deadlock
		if((rank / D2) % 2) {
			if(preD2)
				MPI_Recv(recvD2, M, MPI_DOUBLE, rank-D2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(hasD2)
				MPI_Send(sendD2, M, MPI_DOUBLE, rank+D2, 2, MPI_COMM_WORLD);
		}
		else {
			if(hasD2)
				MPI_Send(sendD2, M, MPI_DOUBLE, rank+D2, 2, MPI_COMM_WORLD);
			if(preD2)
				MPI_Recv(recvD2, M, MPI_DOUBLE, rank-D2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		//Calculation by receiver
		if(preD1) {
			for(int i = 0; i < M; i++)
				recvD1[i] = recvD1[i] * recvD1[i];

			if(preD2)
				for(int i = 0; i < M; i++)
					recvD2[i] = log(recvD2[i]);
		}
		//D1 return
		if((rank / D1) % 2) {
			if(preD1)
				MPI_Send(recvD1, M, MPI_DOUBLE, rank-D1, 3, MPI_COMM_WORLD);
			if(hasD1)
				MPI_Recv(sendD1, M, MPI_DOUBLE, rank+D1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		else {
			if(hasD1)
				MPI_Recv(sendD1, M, MPI_DOUBLE, rank+D1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(preD1)
				MPI_Send(recvD1, M, MPI_DOUBLE, rank-D1, 3, MPI_COMM_WORLD);
		}
		//D2 return
		if((rank / D2) % 2) {
			if(preD2)
				MPI_Send(recvD2, M, MPI_DOUBLE, rank-D2, 4, MPI_COMM_WORLD);
			if(hasD2)
				MPI_Recv(sendD2, M, MPI_DOUBLE, rank+D2, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		else {
			if(hasD2)
				MPI_Recv(sendD2, M, MPI_DOUBLE, rank+D2, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(preD2)
				MPI_Send(recvD2, M, MPI_DOUBLE, rank-D2, 4, MPI_COMM_WORLD);
		}
    		//updating data
		if(hasD1)
			for(int i = 0; i < M; i++) {
				sendD1[i] = (unsigned long long)sendD1[i] % (unsigned long long)100000;
				if(hasD2)
					sendD2[i] = sendD2[i] * 100000.0;
			}
	}

	double pair[2] = {-INFINITY, -INFINITY};
	//computing local maximum and sending data by valid senders	
	if(hasD1) {
		for(int i = 0; i < M; i++)
			if(sendD1[i] > pair[0]) pair[0] = sendD1[i];

		if(hasD2)
			for(int i = 0; i < M; i++)
				if(sendD2[i] > pair[1]) pair[1] = sendD2[i];

		if(rank != 0)
			MPI_Send(pair, 2, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
	}


	double globalMaxD1 = pair[0];
	double globalMaxD2 = pair[1];
        //recieving data from valid senders
	if(rank == 0) {
		for(int r = 1; r < P - D1; r++) {
			double pair[2];
			MPI_Recv(pair, 2, MPI_DOUBLE, r, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(pair[0] > globalMaxD1) globalMaxD1 = pair[0];
			if(pair[1] > globalMaxD2) globalMaxD2 = pair[1];
		}
		time_taken = MPI_Wtime() - time_taken;
	}
	//Final output
	if(rank == 0)
		printf("%lf %lf %lf\n", globalMaxD1, globalMaxD2, time_taken);
	//freeing up memory
	free(sendD1);
	free(sendD2);
	free(recvD1);
	free(recvD2);

	MPI_Finalize();
	return 0;
}
