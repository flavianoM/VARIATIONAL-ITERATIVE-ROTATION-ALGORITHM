# VIRAL — training on the SK model with backpropagation

This code trains the variational parameters {alpha, beta_n, gamma_n} 
of the Variational Iterative Rotation Algorithm (VIRAL) on the 
Sherrington-Kirkpatrick spin-glass model by exact backpropagation 
through the spin dynamics.

## Compile (requires OpenMP)

    gcc -O3 -march=native -fopenmp -ffast-math viral_backprop.c -o viral_backprop -lm

## Run

    ./viral_backprop

## Output

- `best_energy_N*_NUM_INSTANCES*_PMAX*.txt`: best continuous and 
  thresholded energy density at each depth p.
- `best_param_N*_NUM_INSTANCES*_PMAX*_Layer*.txt`: the optimized 
  schedule (beta_n, gamma_n) at each depth; the optimal alpha and 
  the final schedule are also printed to stdout.

## Parameters

System size, number of disorder instances, maximal depth, and 
optimizer settings are set by the #define's at the top of the file 
(defaults: N = 1024, 10 instances, P_MAX = 100). The results in the 
paper were obtained with N = 2^14, 24 instances, and depths up to 
p = 130.
