//
//  main.c
//  guassianElim
//
//  Created by Sten Golds on 11/5/16.
//  Copyright © 2016 Sten Golds. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>

void gaussianFunc(double ar[], double b[], int n);
void Row_solve(double A[], double b[], double x[], int n, int thread_count);

int thread_count;
int size;
double *x, *A, *b;

int main(int argc, char * argv[]) {
    // insert code here...
	if(argc != 3){
		printf("Missing arguement\n");
		exit(0);
	}
	FILE *file = fopen(argv[1], "r");
	if(file == NULL){
		printf("Error in opening file\n");
		exit(0);
	}
	thread_count = (int) strtol(argv[2], NULL, 10);
	int m, n, i, temp;
	char line[512];
	fscanf(file, " %d", &m);
	fscanf(file, " %d", &n);
	size = m * n;
	A = malloc(size * sizeof(double));
	x = malloc(m * sizeof(double));
	b = malloc(m * sizeof(double));
	for(i = 0; i < size; i++){
		fscanf(file, " %d", &temp);
		A[i] = temp;
	}
	for(i = 0; i < n; i++){
		fscanf(file, " %d", &temp);
		b[i] = temp;
	}

   	fclose(file); 
    return 0;
}

void guassianFunc(double ar[], double b[], double x[], int n) {
    int i, j, q, maxRow;
    double maxCoeff, temp, coeffToMult;
    for(i = 0; i < n; i++) {
        
//May not need the code from here to
        maxCoeff = abs(ar[i*n + i]);
        maxRow = i;
        
        for(j = i + 1; j < n; j++) {
            if(abs(ar[j*n + i]) > maxCoeff) {
                maxCoeff = abs(ar[j*n + i]);
                maxRow = j;
            }
        }
        
        
        for(j = i; j < n; j++) {
            temp = ar[maxRow*n + j];
            ar[maxRow*n + j] = ar[i*n + j];
            ar[i*n + j] = temp;
        }
        
        temp = b[maxRow];
        b[maxRow] = b[i];
        b[i] = maxRow;
//here, as is just swaps rows for numerical stability, I feel it would be hard to
//parallelize
        
        for(j = i + 1; j < n; j++) {
            coeffToMult = (float)ar[j*n + i]/ar[i*n + i] * -1;
            for(q = i; q <= n; q++) {
                if(q == i) {
                    ar[j*n + q] = 0;
                } else {
                    ar[j*n + q] += (coeffToMult * ar[i*n + q]);
                }
            }
        }
        
        Row_solve(ar, b, x, n, thread_count);
        
    }
}

/*--------------------------------------------------------------------
 * Function:  Row_solve
 * Purpose:   Solve a triangular system using the row-oriented algorithm
 * In args:   A, b, n, thread_count
 * Out arg:   x
 *
 * Notes:
 */
void Row_solve(double A[], double b[], double x[], int n, int thread_count) {
    int i, j;
    double tmp;
    
#pragma  omp parallel num_threads(thread_count) \
default(none) private(i, j) shared(A, b, x, n, tmp)
    for (i = n-1; i >= 0; i--) {
#pragma omp single
        tmp = b[i];
        
#pragma omp for reduction(+: tmp)
        for (j = i+1; j < n; j++)
            tmp += -A[i*n+j]*x[j];
        
#pragma omp single
        x[i] = tmp/A[i*n+i];
    }
}  /* Row_solve */
