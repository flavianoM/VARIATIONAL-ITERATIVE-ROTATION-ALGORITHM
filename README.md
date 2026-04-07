This code optimizes the variational parameters (alpha, beta, gamma) for the Variational Iterative Rotation Algorithm applied to the Sherrington-Kirkpatrick spin-glass model and validate the optimized parameters on a large instance (N=16000), reporting both continuous and thresholded energy densities.

To compile the code with OpenMP using gcc: 
gcc -O3 -fopenmp viral.c -o viral -lm

To run the simulation:
./viral
