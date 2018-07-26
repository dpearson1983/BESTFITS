#ifndef _SHELLS_H_
#define _SHELLS_H_

#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include "tpods.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

double get_V_ijk(double k_i, double k_j, double k_k, double delta_k) {
    return (k_i*k_j*k_k*delta_k*delta_k*delta_k)/(8.0*PI*PI*PI*PI);
}

double get_V_f(vec3<double> L) {
    return (8.0*PI*PI*PI)/(L.x*L.y*L.z);
}

void get_shell(fftw_complex *shell, fftw_complex *dk, 
               std::vector<double> &kx, std::vector<double> &ky, std::vector<double> &kz, 
               double k_shell, double delta_k, vec3<int> N) {
    if (shell.size() == dk.size()) {
        #pragma omp parallel for
        for (int i = 0; i < N.x; ++i) {
            for (int j = 0; j < N.y; ++j) {
                for (int k = 0; k <= N.z/2; ++k) {
                    double k_mag = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k]);
                    size_t index = k + (N.z/2 + 1)*(j + N.y*i);
                    
                    if (k_mag >= k_shell - 0.5*delta_k && k_mag < k_shell + 0.5*delta_k) {
                        shell[index][0] = dk[index][0];
                        shell[index][1] = dk[index][1];
                    } else {
                        shell[index][0] = 0.0;
                        shell[index][1] = 0.0;
                    }
                }
            }
        }
    } else {
        std::stringstream err_msg;
        err_msg << "Array size mismatch.\n";
        throw std::runtime_error(err_msg.str());
    }
}

double get_bispectrum(std::vector<double> &r_1, std::vector<double> &r_2, std::vector<double> &r_3) {
    if (r_1.size() == r_2.size() && r_2.size() == r_3.size()) {
        std::vector<double> result(omp_get_max_threads());
        #pragma omp parallel for
        for (size_t i = 0; i < r_1.size(); ++i) {
            int tid = omp_get_thread_num();
            result[tid] += r_1[i]*r_2[i]*r_3[i];
        }
        
        for (int i = 1; i < omp_get_max_threads(); ++i)
            result[0] += result[i];
        
        return result[0];
    } else {
        std::stringstream err_msg;
        err_msg << "Array size mismatch.\n";
        throw std::runtime_error(err_msg.str());
    }
}

#endif
