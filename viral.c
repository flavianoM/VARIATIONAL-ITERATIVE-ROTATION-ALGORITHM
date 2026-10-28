/*
* Optimize {alpha, gamma_n, beta_n} of the VIRAL with backpropagation
*
* Traininng is performed sequenctially for p = P_STEP, 2*P_STEP,..., P_MAX.
*
* Parallelized over disorder instances using OpenMP (you must have openMP).
*
* Compile:  gcc -O3 -march=native -fopenmp -ffast-math viral.c -o viral -lm
*
* To Run:      ./viral
*
* Flaviano Morone 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>

// Parameters 
#define N_OPT           1024
#define P_MAX           100
#define P_STEP          2
#define NUM_INSTANCES   10

// Convergence parameters
#define MIN_ITER  50
#define MAX_ITER  500
#define CONV_TOL  1e-5
#define CONV_WIN  50      /* Move on if energy doesn't improve over 50 iterations */

// Learning rate and some parameters for ADAM
#define LR_MAX   3e-3
#define LR_MIN   1e-5
#define B1       0.9
#define B2       0.999
#define ADAM_EPS 1e-8

#define T0     100 // Needed because my Learning Rate decays up to T0 iterations then stays constant
#define GRAD_CLIP 5.0 // To clip gradient just in case

#define PI 3.14159265358979323846

// Spin state vector
typedef struct { double x, y, z; } Vec3;

// Rotation Rz
static inline Vec3 rz(Vec3 s, double theta) {
    double c = cos(theta), sn = sin(theta);
    return (Vec3){ c*s.x - sn*s.y,  sn*s.x + c*s.y,  s.z };
}
// Rotation Rx
static inline Vec3 rx(Vec3 s, double theta) {
    double c = cos(theta), sn = sin(theta);
    return (Vec3){ s.x,  c*s.y - sn*s.z,  sn*s.y + c*s.z };
}

/*******************************************/
/*******  My random number generator *******/
/*******************************************/
typedef struct { uint64_t s[4]; } Rng;

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}
static inline uint64_t rng_next(Rng *r) {
    uint64_t *s = r->s;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}
static inline double rng_uniform(Rng *r) {
    return (double)(rng_next(r) >> 11) * 0x1.0p-53 + 0x1.0p-64;
}
static Rng rng_seed(uint64_t seed) {
    Rng r;
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        r.s[i] = z ^ (z >> 31);
    }
    return r;
}

// Struct for the intermediate states and for backpropagation of gradients
typedef struct {
    Vec3   *s;    /* (P_MAX+1)*N  spin states across layers        */
    Vec3   *t;    /* P_MAX*N      state after z-rot, before x-kick */
    double *h;    /* P_MAX*N      local z-fields                   */
    Vec3   *delta;/* N            backward gradient dE/ds          */
    Vec3   *mu;   /* N            delta backpropagated through Rx      */
    double *w;    /* N            store scalar  mu.y*t.x - mu.x*t.y      */
    double *Jw;   /* N            store Jw  */
    double *sz;   /* N            z-components of the spins */
} Workspace;

static Workspace ws_alloc(int N) {
    Workspace ws;
    ws.s     = malloc((size_t)(P_MAX+1)*N*sizeof(Vec3));
    ws.t     = malloc((size_t) P_MAX   *N*sizeof(Vec3));
    ws.h     = malloc((size_t) P_MAX   *N*sizeof(double));
    ws.delta = malloc((size_t)          N*sizeof(Vec3));
    ws.mu    = malloc((size_t)          N*sizeof(Vec3));
    ws.w     = malloc((size_t)          N*sizeof(double));
    ws.Jw    = malloc((size_t)          N*sizeof(double));
    ws.sz    = malloc((size_t)          N*sizeof(double));
    return ws;
}
static void ws_free(Workspace *ws) {
    free(ws->s); free(ws->t); free(ws->h);
    free(ws->delta); free(ws->mu); free(ws->w); free(ws->Jw); free(ws->sz);
}

// K instancers Jij stored in a single array of dimention K × N × N
static float *J_all;   /* dimension K × N × N */

//Init instances with zero mean and variance 1/N
static void init_instances(int N, int K) {
    J_all = malloc((size_t)K*N*N*sizeof(float));
    double std = sqrt(1.0/N);
    uint64_t base_seed = (uint64_t)time(NULL);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < K; k++) {
        Rng rng = rng_seed(base_seed ^ ((uint64_t)k * 0x9e3779b97f4a7c15ULL));
        float *J = J_all + (size_t)k*N*N;
        for (int i = 0; i < N; i++) {
            J[i*N+i] = 0.0f;
            for (int j = i+1; j < N; j++) {
                double u1 = rng_uniform(&rng);
                double u2 = rng_uniform(&rng);
                float v = (float)(std*sqrt(-2.0*log(u1))*cos(2.0*PI*u2));
                J[i*N+j] = v;  J[j*N+i] = v;
            }
        }
    }
}


/**********************************************************************************/
/*******  Forward and backward pass for one Jij instance **************************/
/* returns E/N for that instance and accumulate gradients for alpha, beta, gamma */
/**********************************************************************************/
static double forward_backward(
        int N, int p, const float *J,
        double alpha, const double *beta, const double *gamma,
        double *grad_alpha, double *grad_beta, double *grad_gamma,
        Workspace *ws, double *e_thresh_out)
{
    Vec3   *s = ws->s,  *t  = ws->t;
    double *h = ws->h;
    Vec3   *delta = ws->delta, *mu = ws->mu;
    double *w = ws->w,  *Jw = ws->Jw,  *sz = ws->sz;

                    //FORWARD PASS

    // store s[n], t[n], h[n] at every layer            
    double ca = cos(alpha), sa = sin(alpha);
    for (int i = 0; i < N; i++)
        s[i] = (Vec3){ ca, 0.0, sa };

    for (int n = 0; n < p; n++) {
        Vec3   *s_in  = s + (size_t) n   *N;
        Vec3   *t_n   = t + (size_t) n   *N;
        Vec3   *s_out = s + (size_t)(n+1)*N;
        double *h_n   = h + (size_t) n   *N;

        // Compute local field hz on layer n 
        for (int i = 0; i < N; i++) sz[i] = s_in[i].z;
        for (int i = 0; i < N; i++) {
            const float *Ji = J + i*N;
            double hh = 0.0;
            for (int j = 0; j < N; j++) hh += Ji[j]*sz[j];
            h_n[i] = hh;
        }
        // Apply Z-rotation and X-kick 
        for (int i = 0; i < N; i++) {
            t_n[i]   = rz(s_in[i], gamma[n]*h_n[i]);
            s_out[i] = rx(t_n[i],  beta[n]);
        }
    }
    // Compute energy E/N and initialize delta[i]
    // Remeber delta[i] = dE/ds_i(p)
    Vec3 *s_p = s + (size_t)p*N;
    for (int i = 0; i < N; i++) sz[i] = s_p[i].z;
    double energy = 0.0;
    for (int i = 0; i < N; i++) {
        const float *Ji = J + i*N;
        double hf = 0.0;
        for (int j = 0; j < N; j++) hf += Ji[j]*sz[j];
        energy   += hf * sz[i];
        delta[i]  = (Vec3){ 0.0, 0.0, hf/N };
    }
    energy /= 2.0*N;

    //Compute thresholded energy 
    double e_thr = 0.0;
    for (int i = 0; i < N; i++) {
        const float *Ji = J + i*N;
        double si = s_p[i].z >= 0.0 ? 1.0 : -1.0;
        for (int j = i+1; j < N; j++) {
            double sj = s_p[j].z >= 0.0 ? 1.0 : -1.0;
            e_thr += Ji[j] * si * sj;
        }
    }
    *e_thresh_out += e_thr / N;

                //BACKWARD PASS

    // From layer p to layer 1                  
    for (int n = p-1; n >= 0; n--) {
        Vec3   *s_n = s + (size_t)(n+1)*N;  /* output of layer n */
        Vec3   *t_n = t + (size_t) n   *N;  /* after z-rot       */
        double *h_n = h + (size_t) n   *N;  /* local fields      */

        
        // Gradient of the energy with respect to beta_n
        double gb = 0.0;
        for (int i = 0; i < N; i++)
            gb += delta[i].y*(-s_n[i].z) + delta[i].z*s_n[i].y;
        grad_beta[n] += gb;

        // Backpropagate the gradient through Rx
        for (int i = 0; i < N; i++)
            mu[i] = rx(delta[i], -beta[n]);

        for (int i = 0; i < N; i++)
            w[i] = mu[i].y*t_n[i].x - mu[i].x*t_n[i].y;

        // Gradient of the energy with respect to gamma_n
        double gg = 0.0;
        for (int i = 0; i < N; i++) gg += h_n[i]*w[i];
        grad_gamma[n] += gg;

        // Backpropagate the gradient through Rz
        for (int i = 0; i < N; i++) {
            const float *Ji = J + i*N;
            double jw = 0.0;
            for (int j = 0; j < N; j++) jw += Ji[j]*w[j];
            Jw[i] = jw;
        }

        for (int i = 0; i < N; i++) {
            Vec3 d = rz(mu[i], -gamma[n]*h_n[i]);
            d.z   += gamma[n]*Jw[i];
            delta[i] = d;
        }
    }

    // Gradient of the energy with respect to alpha
    double ga = 0.0;
    for (int i = 0; i < N; i++)
        ga += delta[i].x*(-sa) + delta[i].z*ca;
    *grad_alpha += ga;

    return energy;
}

// Simple Adam optimizer implementation
typedef struct { double *m, *v; int t, np; } Adam;

static Adam adam_alloc(int np) {
    return (Adam){ calloc(np, sizeof(double)),
                   calloc(np, sizeof(double)), 0, np };
}
static void adam_reset(Adam *a) {
    memset(a->m, 0, a->np*sizeof(double));
    memset(a->v, 0, a->np*sizeof(double));
    a->t = 0;
}
static void adam_step(Adam *a, double *params,
                      const double *grads, double lr) {
    a->t++;
    double lr_t = lr * sqrt(1.0 - pow(B2,a->t))
                     / (1.0 - pow(B1,a->t));
    for (int i = 0; i < a->np; i++) {
        a->m[i] = B1*a->m[i] + (1.0-B1)*grads[i];
        a->v[i] = B2*a->v[i] + (1.0-B2)*grads[i]*grads[i];
        params[i] -= lr_t * a->m[i]
                          / (sqrt(a->v[i]) + ADAM_EPS);
    }
}

// Use the old schdule at depth p_old to start the deeper schedule p_new 
static void stretch(double *v, int p_old, int p_new) {
    double *tmp = malloc(p_new*sizeof(double));
    for (int n = 0; n < p_new; n++) {
        double t  = (double)n*(p_old-1)/(double)(p_new-1);
        int    lo = (int)t;
        int    hi = (lo+1 < p_old) ? lo+1 : p_old-1;
        double f  = t - lo;
        tmp[n] = (1.0-f)*v[lo] + f*v[hi];
    }
    memcpy(v, tmp, p_new*sizeof(double));
    free(tmp);
}


/*******************************************/
/*******  MAIN program starts here *********/
/*******************************************/

int main(void) {
    printf("You have %d threads. I'll use all of them.\n\n", omp_get_max_threads());

    int N = N_OPT, K = NUM_INSTANCES;
    init_instances(N, K);

    // Variational parameters for VIRAL
    // alpha is a single global parameter (initial angle) initialized to pi/4
    // beta and gamma are arrays of size P_MAX
    double alpha  = PI/4.0;
    double *beta  = malloc(P_MAX*sizeof(double));
    double *gamma = malloc(P_MAX*sizeof(double));

    // Gradients 
    double *grads  = malloc((1+2*P_MAX)*sizeof(double));

    //Initialize beta as a linear ramp from pi/2 to 0 and gamma to -1.2
    for (int n = 0; n < P_STEP; n++) {
        beta[n]  = (P_STEP > 1)
                   ? (PI/2.0)*(1.0 - (double)n/(P_STEP-1))
                   : PI/2.0;
        gamma[n] = -1.2;
    }

    // Arrange all param into a single array [alpha,beta,gamma] for Adam
    int     np     = 1 + 2*P_MAX;
    double *params = malloc(np*sizeof(double));
    params[0] = alpha;
    memcpy(params+1,      beta,  P_MAX*sizeof(double));
    memcpy(params+1+P_MAX,gamma, P_MAX*sizeof(double));

    Adam opt = adam_alloc(np);


    //Each thread works on a different instance and sum the gradients 
    int nthreads = omp_get_max_threads();
    Workspace *ws_arr = malloc(nthreads*sizeof(Workspace));
    for (int t = 0; t < nthreads; t++) ws_arr[t] = ws_alloc(N);

    // Each openMP thread accumulate gradients for alpha, beta and gamma
    double *tgrads = malloc((size_t)nthreads*np*sizeof(double));

    // File to save best energy and best parameters (overwritten if new best energy found)
    char ename[256];
    sprintf(ename, "best_energy_N%d_NUM_INSTANCES%d_PMAX%d.txt",
            N, K, P_MAX);
    FILE *fe = fopen(ename, "w");
    fprintf(fe, "# p   best_E/N   best_Et/N   best_alpha\n");
    fclose(fe);


    // Train first p=2, then p=4, then p=6, up to PMAX
    for (int p = P_STEP; p <= P_MAX; p += P_STEP) {

        printf("****************\n");
        printf(" Depth  p = %d\n", p);
        printf("****************\n");
    

        // Reuse the best param at previous p as the starting point for the next p
        alpha = params[0];
        memcpy(beta,  params+1,       P_MAX*sizeof(double));
        memcpy(gamma, params+1+P_MAX, P_MAX*sizeof(double));

        // Since at the new depth p there are 2 more layers 
        // insert p points on the old schedule p-2 equally spaced 
        if (p > P_STEP) {
            stretch(beta,  p-P_STEP, p);
            stretch(gamma, p-P_STEP, p);
           
            params[0] = alpha;
            memcpy(params+1,       beta,  P_MAX*sizeof(double));
            memcpy(params+1+P_MAX, gamma, P_MAX*sizeof(double));
        }

        
        // Initialize Adam
        int t_cur = 0;
        double best_E = 1e30, best_Et = 1e30;
        double *best_params = malloc(np*sizeof(double));
        memcpy(best_params, params, np*sizeof(double));
        double best_E_window = 1e30;  
        int converged = 0;

        // Train for MAX_ITER iterations or until convergence
        for (int iter = 0; iter < MAX_ITER && !converged; iter++) {

            //Initialize gradients and energies
            memset(tgrads, 0, (size_t)nthreads*np*sizeof(double));
            double total_E = 0.0, total_Et = 0.0;

            #pragma omp parallel reduction(+:total_E,total_Et)
            {
                int     tid = omp_get_thread_num();
                double *tg  = tgrads + (size_t)tid*np;
                
                #pragma omp for schedule(dynamic)
                for (int k = 0; k < K; k++) {
                    const float *Jk = J_all + (size_t)k*N*N;
                    double et_k = 0.0;
                    total_E += forward_backward(
                                    N, p, Jk,
                                    params[0],
                                    params+1,
                                    params+1+P_MAX,
                                    &tg[0],
                                    &tg[1],
                                    &tg[1+P_MAX],
                                    &ws_arr[tid], &et_k);
                    total_Et += et_k;
                }
            }

            total_Et /= K;

            // Average gradients over K instances of Jij
            memset(grads, 0, np*sizeof(double));
            for (int t = 0; t < nthreads; t++) {
                double *tg = tgrads + (size_t)t*np;
                for (int i = 0; i < np; i++) grads[i] += tg[i];
            }
            total_E /= K;
            for (int i = 0; i < np; i++) grads[i] /= K;

            // Gradient clipping just in case
            double gnorm = 0.0;
            for (int i = 0; i < np; i++) gnorm += grads[i]*grads[i];
            gnorm = sqrt(gnorm);
            if (gnorm > GRAD_CLIP) {
                double scale = GRAD_CLIP/gnorm;
                for (int i = 0; i < np; i++) grads[i] *= scale;
            }

            // Learning rate decays as a cosine from LR_MAX to LR_MIN
            // At iteration T0 the learning rate reaches LR_MIN and stays there
            double lr_max_p = LR_MAX * fmin(1.0, 40.0/p);
            double frac = fmin(1.0, (double)t_cur / T0);
            double lr = LR_MIN + 0.5*(lr_max_p-LR_MIN)*(1.0 + cos(PI*frac));
            t_cur++;

            if (total_E < best_E) {
                best_E = total_E;
                memcpy(best_params, params, np*sizeof(double));
            }
            if (total_Et < best_Et) best_Et = total_Et;

            adam_step(&opt, params, grads, lr);

            // Check if converges
            if (iter >= MIN_ITER && iter % CONV_WIN == 0) {
                double improvement = best_E_window - best_E;
                if (improvement < CONV_TOL) {
                    printf("p=%3d  converged at iter %d\n\n",
                           p, iter);
                    fflush(stdout);
                    converged = 1;
                }
                best_E_window = best_E;
            }

            if (iter % 10 == 0) {
                printf("p=%3d  iter %4d  E/N = %.6f  E_t/N = %.6f\n",
                       p, iter, total_E, total_Et);
                fflush(stdout);
            }
        }

        // Writes on file the best energy and best alpha at this depth 
        fe = fopen(ename, "a");
        fprintf(fe, "%4d   %.8f   %.8f   %.10f\n", p, best_E, best_Et, best_params[0]);
        fclose(fe);

        
         // Writes on a file the best energy and best parameters (alpha,beta,gamma) for this depth 
        char pname[256];
        sprintf(pname, "best_param_N%d_NUM_INSTANCES%d_PMAX%d_Layer%d.txt",
                N, K, P_MAX, p);
        FILE *fp = fopen(pname, "w");
        fprintf(fp, "# alpha  (best E/N = %.8f, best Et/N = %.8f)\n%.10f\n",
                best_E, best_Et, best_params[0]);
        fprintf(fp, "# layer  beta  gamma\n");
        for (int n = 0; n < p; n++)
            fprintf(fp, "%4d  %.10f  %.10f\n",
                    n+1, best_params[1+n], best_params[1+P_MAX+n]);
        fclose(fp);

        // Use best params as starting point for next depth
        memcpy(params, best_params, np*sizeof(double));
        free(best_params);
    }

    // Print final optimal schedule at screen 
    printf("Optimal alpha = %.6f\n\n", params[0]);
    printf("  layer   beta/pi    gamma\n");
    for (int k = 0; k < P_MAX; k++)
        printf("  %3d    %7.4f    %7.4f\n",
               k+1, params[1+k]/PI, params[1+P_MAX+k]);

    // Free all memory 
    for (int t = 0; t < nthreads; t++) ws_free(&ws_arr[t]);
    free(ws_arr); free(tgrads);
    free(opt.m); free(opt.v);
    free(params); free(beta); free(gamma); free(grads);
    free(J_all);
    return 0;
}
