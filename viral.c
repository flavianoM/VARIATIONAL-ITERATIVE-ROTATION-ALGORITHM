#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N_OPT 256           // System size for optimization
#define N_LARGE 16000        // System size for final validation
#define P 30                // Circuit depth
#define NUM_INSTANCES 50    // Number of instances for averaging
#define GRID_POINTS 50      // Grid search resolution
#define MAX_ITER 50         // Max optimization sweeps
#define SEARCH_RADIUS 0.15   // Adaptive search window radius
#define CYCLES 1            // Number of optimization cycles

#define PI 3.14159265358979323846

// Spin Structure
typedef struct {
    double x;
    double y;
    double z;
} Spin;

// Global pointer that stores all the coupling matrices of all optimization instances
double *J_opt_instances;

// Random Number Generator 
double rand_normal(double mean, double stddev) {
    static int haveSpare = 0;
    static double rand1, rand2;
    if (haveSpare) {
        haveSpare = 0;
        return mean + stddev * sqrt(rand1) * sin(rand2);
    }
    haveSpare = 1;
    rand1 = rand() / ((double) RAND_MAX);
    if (rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * 2 * PI;
    return mean + stddev * sqrt(rand1) * cos(rand2);
}

static inline double sign_val(double x) {
    return (x >= 0.0) ? 1.0 : -1.0;
}

// Initialization of all the coupling matrices
void init_opt_instances() {
    J_opt_instances = (double *)malloc(NUM_INSTANCES * N_OPT * N_OPT * sizeof(double));
    double stddev = sqrt(1.0 / N_OPT);

    // Parallelize instance generation
    #pragma omp parallel
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num();

        #pragma omp for
        for (int k = 0; k < NUM_INSTANCES; k++) {
            for (int i = 0; i < N_OPT; i++) {
                J_opt_instances[k*N_OPT*N_OPT + i*N_OPT + i] = 0.0;
                for (int j = i + 1; j < N_OPT; j++) {
                    
                    double u1 = rand_r(&seed) / ((double) RAND_MAX);
                    double u2 = rand_r(&seed) / ((double) RAND_MAX);
                    if (u1 < 1e-100) u1 = 1e-100;
                    double val = stddev * sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);

                    J_opt_instances[k*N_OPT*N_OPT + i*N_OPT + j] = val;
                    J_opt_instances[k*N_OPT*N_OPT + j*N_OPT + i] = val;
                }
            }
        }
    }
}

// Z-rotation function
static inline void rotate_z(Spin *s, double theta) {
    double c = cos(theta);
    double sn = sin(theta);
    double nx = s->x * c - s->y * sn;
    double ny = s->x * sn + s->y * c;
    s->x = nx;
    s->y = ny;
}

// X-rotation function
static inline void rotate_x(Spin *s, double theta) {
    double c = cos(theta);
    double sn = sin(theta);
    double ny = s->y * c - s->z * sn;
    double nz = s->y * sn + s->z * c;
    s->y = ny;
    s->z = nz;
}

// Run a variational classical circuit of depth P
double run_simulation_generic(int n, double *J_matrix, double alpha, double beta[P], double gamma[P]) {
    Spin *spins = (Spin *)malloc(n * sizeof(Spin));
    double *fields = (double *)malloc(n * sizeof(double));

    // Precompute initial state
    double cos_alpha = cos(alpha);
    double sin_alpha = sin(alpha);

    // Initialize |s_0>
    for (int i = 0; i < n; i++) {
        spins[i].x = cos_alpha;
        spins[i].y = 0.0;
        spins[i].z = sin_alpha;
    }

    // Evolve through layers
    for (int l = 0; l < P; l++) {

        // Compute local fields
        for (int i = 0; i < n; i++) {
            double h = 0.0;
            for (int j = 0; j < n; j++) {
                h += J_matrix[i*n + j] * spins[j].z;
            }
            fields[i] = h;
        }

        // Apply rotations
        double gamma_l = gamma[l];
        double beta_l = beta[l];
        for (int i = 0; i < n; i++) {
            rotate_z(&spins[i], gamma_l * fields[i]);
            rotate_x(&spins[i], beta_l);
        }
    }

    // Compute Energy 
    double energy = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            energy += J_matrix[i*n + j] * spins[i].z * spins[j].z;
        }
    }

    free(fields);
    free(spins);
    return energy / n;
}

// Computate Average Energy 
double average_energy_opt(double alpha, double beta[P], double gamma[P]) {
    double total_E = 0.0;

    // Run all disorder instances in parallel, then sum up their energies 
    #pragma omp parallel for reduction(+:total_E) schedule(dynamic)
    for (int k = 0; k < NUM_INSTANCES; k++) {
        double *J_curr = &J_opt_instances[k * N_OPT * N_OPT];
        total_E += run_simulation_generic(N_OPT, J_curr, alpha, beta, gamma);
    }

    return total_E / NUM_INSTANCES;
}

void evaluate_large_instance(double alpha, double beta[P], double gamma[P]) {
    printf("\n Validating on Large Instance (N=%d) \n", N_LARGE);
    double *J_large = (double *)malloc((size_t)N_LARGE * N_LARGE * sizeof(double));

    double stddev = sqrt(1.0 / N_LARGE);
    for (int i = 0; i < N_LARGE; i++) {
        J_large[i*N_LARGE + i] = 0.0;
        for (int j = i + 1; j < N_LARGE; j++) {
            double val = rand_normal(0.0, stddev);
            J_large[i*N_LARGE + j] = val;
            J_large[j*N_LARGE + i] = val;
        }
    }

    double energy = run_simulation_generic(N_LARGE, J_large, alpha, beta, gamma);
    printf("Final Continuous Energy Density: %f\n", energy);
    printf("(Compare to Parisi value ~ -0.763)\n");
    free(J_large);
}

void evaluate_large_instance_thresholded(double alpha, double beta[P], double gamma[P]) {
    printf("\n Validating on Large Instance (N=%d) [Thresholded] \n", N_LARGE);

    double *J_large = (double *)malloc((size_t)N_LARGE * N_LARGE * sizeof(double));
    Spin *spins = (Spin *)malloc(N_LARGE * sizeof(Spin));
    double *fields = (double *)malloc(N_LARGE * sizeof(double));

    if (!J_large || !spins || !fields) {
        printf("Memory allocation failed.\n");
        return;
    }

    double stddev = sqrt(1.0 / N_LARGE);
    for (int i = 0; i < N_LARGE; i++) {
        J_large[i*N_LARGE + i] = 0.0;
        for (int j = i + 1; j < N_LARGE; j++) {
            double val = rand_normal(0.0, stddev);
            J_large[i*N_LARGE + j] = val;
            J_large[j*N_LARGE + i] = val;
        }
    }

    // Precompute
    double cos_alpha = cos(alpha);
    double sin_alpha = sin(alpha);

    // Initialize
    for (int i = 0; i < N_LARGE; i++) {
        spins[i].x = cos_alpha;
        spins[i].y = 0.0;
        spins[i].z = sin_alpha;
    }

    // Evolve
    for (int l = 0; l < P; l++) {
        for (int i = 0; i < N_LARGE; i++) {
            double h = 0.0;
            for (int j = 0; j < N_LARGE; j++) {
                h += J_large[i*N_LARGE + j] * spins[j].z;
            }
            fields[i] = h;
        }
        double gamma_l = gamma[l];
        double beta_l = beta[l];
        for (int i = 0; i < N_LARGE; i++) {
            rotate_z(&spins[i], gamma_l * fields[i]);
            rotate_x(&spins[i], beta_l);
        }
    }

    // Compute Thresholded Energy
    double energy_thresholded = 0.0;
    for (int i = 0; i < N_LARGE; i++) {
        for (int j = i + 1; j < N_LARGE; j++) {
            double s_i_rounded = sign_val(spins[i].z);
            double s_j_rounded = sign_val(spins[j].z);
            energy_thresholded += J_large[i*N_LARGE + j] * s_i_rounded * s_j_rounded;
        }
    }

    printf("Final Thresholded (Ising) Energy Density: %f\n", energy_thresholded / N_LARGE);

    free(fields);
    free(spins);
    free(J_large);
}

// Main 
int main() {
    // omp_set_num_threads(8); // Set number of threads (use all available by default)

    printf("OpenMP enabled with %d threads\n", omp_get_max_threads());

    srand(time(NULL));

    double beta[P], gamma[P];
    double alpha = PI / 4.0;
    double current_energy;

    for (int k = 0; k < P; k++) {
        beta[k] = (PI / 2.0) * (1.0 - (double)k / (double)(P));
        gamma[k] = -1.2;
    }

    for (int cycle = 0; cycle < CYCLES; cycle++) {
        printf("\n CYCLE %d of %d \n", cycle + 1, CYCLES);

        if (cycle > 0) free(J_opt_instances);
        printf("Initializing %d instances of size N=%d\n", NUM_INSTANCES, N_OPT);
        init_opt_instances();

        current_energy = average_energy_opt(alpha, beta, gamma);
        printf("Avg Energy: %f\n", current_energy);

        double t_start = omp_get_wtime();

        for (int iter = 0; iter < MAX_ITER; iter++) {
            double iter_start_e = current_energy;
            double iter_t_start = omp_get_wtime();
            printf("Iteration %d ", iter+1);
            fflush(stdout);

            for (int k = 0; k < P; k++) {

                {
                    double best_b = beta[k];
                    double min_e = current_energy;

                    double search_min = beta[k] - SEARCH_RADIUS;
                    double search_max = beta[k] + SEARCH_RADIUS;
                    double step = (search_max - search_min) / GRID_POINTS;

                    for (int g = 0; g <= GRID_POINTS; g++) {
                        double val = search_min + g * step;
                        double old_val = beta[k];
                        beta[k] = val;

                        if (fabs(val - old_val) > 1e-5) {
                            double e = average_energy_opt(alpha, beta, gamma);
                            if (e < min_e) {
                                min_e = e;
                                best_b = val;
                            }
                        }
                        beta[k] = old_val;
                    }
                    beta[k] = best_b;
                    current_energy = min_e;
                }

                {
                    double best_g = gamma[k];
                    double min_e = current_energy;

                    double search_min = gamma[k] - SEARCH_RADIUS;
                    double search_max = gamma[k] + SEARCH_RADIUS;
                    double step = (search_max - search_min) / GRID_POINTS;

                    for (int g = 0; g <= GRID_POINTS; g++) {
                        double val = search_min + g * step;
                        double old_val = gamma[k];
                        gamma[k] = val;

                        if (fabs(val - old_val) > 1e-5) {
                            double e = average_energy_opt(alpha, beta, gamma);
                            if (e < min_e) {
                                min_e = e;
                                best_g = val;
                            }
                        }
                        gamma[k] = old_val;
                    }
                    gamma[k] = best_g;
                    current_energy = min_e;
                }
            }

            {
                double best_a = alpha;
                double min_e = current_energy;

                double search_min = alpha - SEARCH_RADIUS;
                double search_max = alpha + SEARCH_RADIUS;
                double step = (search_max - search_min) / GRID_POINTS;

                for(int g = 0; g <= GRID_POINTS; g++) {
                    double val = search_min + g * step;
                    double old_val = alpha;
                    alpha = val;

                    if (fabs(val - old_val) > 1e-6) {
                        double e = average_energy_opt(alpha, beta, gamma);
                        if (e < min_e) {
                            min_e = e;
                            best_a = val;
                        }
                    }
                    alpha = old_val;
                }
                alpha = best_a;
                current_energy = min_e;
            }

            double iter_time = omp_get_wtime() - iter_t_start;
            printf("Energy: %f (%.2f sec)\n", current_energy, iter_time);

            if (fabs(current_energy - iter_start_e) < 1e-5) {
                printf("Optimization Converged \n");
                break;
            }
        }

        double t_opt = omp_get_wtime() - t_start;
        printf("\nCycle %d took %.2f seconds\n", cycle + 1, t_opt);
    }

    // Output final parameters and evaluate on large instance
    printf("\n Results (Layer, Beta/PI, Gamma) \n");
    printf("Optimal Initial Angle (Alpha): %f\n", alpha);
    for (int k = 0; k < P; k++) {
        printf("param %d %f %f\n", k+1, beta[k]/PI, gamma[k]);
    }

    evaluate_large_instance(alpha, beta, gamma);
    evaluate_large_instance_thresholded(alpha, beta, gamma);

    free(J_opt_instances);
    return 0;
}
