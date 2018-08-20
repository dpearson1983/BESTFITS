#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include "../include/tpods.h"
#include "../include/shells.h"
#include "../include/file_io.h"
#include "../include/transformers.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

double get_V_ijk(double k_i, double k_j, double k_k, double delta_k) {
//     return (k_i*k_j*k_k*delta_k*delta_k*delta_k)/(8.0*PI*PI*PI*PI);
    return 8.0*PI*PI*(k_i*k_j*k_k*delta_k*delta_k*delta_k);
}

double get_V_f(vec3<double> L) {
    return (8.0*PI*PI*PI)/(L.x*L.y*L.z);
}

void get_shell(fftw_complex *shell, fftw_complex *dk, 
               std::vector<double> &kx, std::vector<double> &ky, std::vector<double> &kz, 
               double k_shell, double delta_k, vec3<int> N) {
    #pragma omp parallel for
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k <= N.z/2; ++k) {
                double k_mag = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k]);
                size_t index = k + (N.z/2 + 1)*(j + N.y*i);
                shell[index][0] = 0.0;
                shell[index][1] = 0.0;
                
                if (k_mag >= k_shell - 0.5*delta_k && k_mag < k_shell + 0.5*delta_k) {
                    shell[index][0] = dk[index][0];
                    shell[index][1] = dk[index][1];
                }
            }
        }
    }
}

void get_shells(std::vector<std::vector<double>> &shells, std::vector<double> &dk, std::vector<double> &kx,
                std::vector<double> &ky, std::vector<double> &kz, double k_min, double k_max, double delta_k, 
                vec3<int> N, std::string wisdomFile) {
    std::vector<double> shell(N.x*N.y*2*(N.z/2 + 1));
    generate_wisdom_bipc2r(shell, N, wisdomFile, omp_get_max_threads());
    int N_shells = int((k_max - k_min)/delta_k);
    for (int i =0; i < N_shells; ++i) {
        double k_shell = k_min + (i + 0.5)*delta_k;
        get_shell((fftw_complex *)shell.data(), (fftw_complex *)dk.data(), kx, ky, kz, k_shell, delta_k,
                  N);
        bip_c2r(shell, N, wisdomFile, omp_get_max_threads());
        shells.push_back(shell);
    }
}

void zero_shell(std::vector<double> &shell) {
#pragma omp parallel for
    for (size_t i = 0; i < shell.size(); ++i)
        shell[i] = 0.0;
}

double shell_prod(std::vector<double> &r_1, std::vector<double> &r_2, std::vector<double> &r_3,
                  vec3<int> N) {
    std::vector<double> result(omp_get_max_threads());
    for (int i = 0; i < omp_get_max_threads(); ++i)
        result[i] = 0.0;
    #pragma omp parallel for
    for (size_t i = 0; i < N.x; ++i) {
        for (size_t j = 0; j < N.y; ++j) {
            for (size_t k = 0; k < N.z; ++k) {
                int tid = omp_get_thread_num();
                size_t index = k + 2*(N.z/2 + 1)*(j + N.y*i);
                result[tid] += (r_1[index]*r_2[index]*r_3[index]);
            }
        }
    }
    
    for (int i = 1; i < omp_get_max_threads(); ++i)
        result[0] += result[i];
    
    return result[0];
}

