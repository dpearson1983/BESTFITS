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
#include "../include/bispec.h"

void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<std::vector<double>> &shells, double delta_k, double k_min, double k_max) {
    vec3<double> kt;
//     int N_shells = int((k_max - k_min)/delta_k);
    double V_f = get_V_f(L);
    double N_tot = N.x*N.y*N.z;
    double alpha3 = alpha*alpha*alpha;
    
    for (int i = 0; i < ks.size(); ++i) {
        kt.x = ks[i];
        for (int j = i; j < ks.size(); ++j) {
            kt.y = ks[j];
            for (int k = j; k < ks.size(); ++k) {
                kt.z = ks[k];
                if (ks[k] <= ks[i] + ks[j]) {
                    double V_ijk = get_V_ijk(ks[i], ks[j], ks[k], delta_k);
                    
                    double B_est = shell_prod(shells[i], shells[j], shells[k], N)/N_tot;
                    B_est /= gal_bk_nbw.z;
                    B_est *= V_f*V_f;
                    B_est /= V_ijk;
                    double SN = (P[i] + P[j] + P[k])*gal_bk_nbw.y + gal_bk_nbw.x - alpha3*ran_bk_nbw.x;
                    SN /= gal_bk_nbw.z;
                    B_est -= SN;
                    B.push_back(B_est);
                    k_trip.push_back(kt);
                }
            }
        }
    }
}

void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<double> &delta, std::vector<double> &kx, std::vector<double> &ky, 
                    std::vector<double> &kz, double delta_k, std::string wisdomFile) {
    std::vector<double> shell_1(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> shell_2(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> shell_3(N.x*N.y*2*(N.z/2 + 1));
    generate_wisdom_bipc2r(shell_1, N, wisdomFile, omp_get_max_threads());
    generate_wisdom_bipc2r(shell_2, N, wisdomFile, omp_get_max_threads());
    generate_wisdom_bipc2r(shell_3, N, wisdomFile, omp_get_max_threads());
    double N_tot = N.x*N.y*N.z;
    double V_f = get_V_f(L);
    vec3<double> kt;
    double alpha3 = alpha*alpha*alpha;
    for (int i = 0; i < ks.size(); ++i) {
        get_shell((fftw_complex *) shell_1.data(), (fftw_complex *) delta.data(), kx, ky, kz, ks[i], 
                  delta_k, N);
        bip_c2r(shell_1, N, wisdomFile, omp_get_max_threads());
        kt.x = ks[i];
        for (int j = i; j < ks.size(); ++j) {
            get_shell((fftw_complex *) shell_2.data(), (fftw_complex *) delta.data(), kx, ky, kz, ks[j], 
                      delta_k, N);
            bip_c2r(shell_2, N, wisdomFile, omp_get_max_threads());
            kt.y = ks[j];
            for (int k = j; k < ks.size(); ++k) {
                if (ks[k] <= ks[i] + ks[j]) {
                    double V_ijk = get_V_ijk(ks[i], ks[j], ks[k], delta_k);
                    get_shell((fftw_complex *) shell_3.data(), (fftw_complex *) delta.data(), kx, ky, 
                              kz, ks[k], delta_k, N);
                    bip_c2r(shell_3, N, wisdomFile, omp_get_max_threads());
                    kt.z = ks[k];
                    
                    double B_est = shell_prod(shell_1, shell_2, shell_3, N)/N_tot;
                    B_est /= gal_bk_nbw.z;
                    B_est *= V_f*V_f;
                    B_est /= V_ijk;
                    double SN = (P[i] + P[j] + P[k])*gal_bk_nbw.y + gal_bk_nbw.x - alpha3*ran_bk_nbw.x;
                    SN /= gal_bk_nbw.z;
                    B_est -= SN;
                    B.push_back(B_est);
                    k_trip.push_back(kt);
                }
            }
        }
    }
}
