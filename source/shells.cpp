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
}

double shell_prod(std::vector<double> &r_1, std::vector<double> &r_2, std::vector<double> &r_3,
                  vec3<int> N) {
    std::vector<double> result(omp_get_max_threads());
    #pragma omp parallel for
    for (size_t i = 0; i < N.x; ++i) {
        for (size_t j = 0; j < N.y; ++j) {
            for (size_t k = 0; k < N.z; ++k) {
                int tid = omp_get_thread_num();
                size_t  index = k + 2*(N.z/2 + 1)*(j + N.y*i);
                result[tid] += r_1[index]*r_2[index]*r_3[index];
            }
        }
    }
    
    for (int i = 1; i < omp_get_max_threads(); ++i)
        result[0] += result[i];
    
    return result[0];
}

double shell_prod(std::string k1File, std::string k2File, std::string k3File, vec3<int> N) {
    double result = 0.0;
    size_t N_tot = N.x*N.y*N.z;
    if (k1File != k2File && k1File != k3File && k2File != k3File) {
        std::ifstream k1in(k1File, std::ios::in|std::ios::binary);
        std::ifstream k2in(k2File, std::ios::in|std::ios::binary);
        std::ifstream k3in(k3File, std::ios::in|std::ios::binary);
        
        for (size_t i = 0; i < N_tot; ++i) {
            double dr1, dr2, dr3;
            k1in.read((char *) &dr1, sizeof(double));
            k2in.read((char *) &dr2, sizeof(double));
            k3in.read((char *) &dr3, sizeof(double));
            result += dr1*dr2*dr3;
        }
        k1in.close();
        k2in.close();
        k3in.close();
    } else if (k1File == k2File && k1File != k3File) {
        std::ifstream k12in(k1File, std::ios::in|std::ios::binary);
        std::ifstream k3in(k3File, std::ios::in|std::ios::binary);
        
        for (size_t i = 0; i < N_tot; ++i) {
            double dr12, dr3;
            k12in.read((char *) &dr12, sizeof(double));
            k3in.read((char *) &dr3, sizeof(double));
            result += dr12*dr12*dr3;
        }
        k12in.close();
        k3in.close();
    } else if (k1File == k3File && k1File != k2File) {
        std::ifstream k13in(k1File, std::ios::in|std::ios::binary);
        std::ifstream k2in(k2File, std::ios::in|std::ios::binary);
        
        for (size_t i = 0; i < N_tot; ++i) {
            double dr13, dr2;
            k13in.read((char *) &dr13, sizeof(double));
            k2in.read((char *) &dr2, sizeof(double));
            result += dr13*dr2*dr13;
        }
        k13in.close();
        k2in.close();
    } else if (k2File == k3File && k1File != k2File) {
        std::ifstream k23in(k2File, std::ios::in|std::ios::binary);
        std::ifstream k1in(k1File, std::ios::in|std::ios::binary);
        
        for (size_t i = 0; i < N_tot; ++i) {
            double dr1, dr23;
            k1in.read((char *) &dr1, sizeof(double));
            k23in.read((char *) &dr23, sizeof(double));
            result += dr1*dr23*dr23;
        }
        k23in.close();
        k1in.close();
    } else if (k1File == k2File && k1File == k3File) {
        std::ifstream kin(k1File, std::ios::in|std::ios::binary);
        
        for (size_t i = 0; i < N_tot; ++i) {
            double dr;
            kin.read((char *) &dr, sizeof(double));
            result += dr*dr*dr;
        }
    }
        
    return result;
}

void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, double alpha, std::vector<double> &B, 
                    std::vector<vec3<double>> &k_trip, std::string shellBase, std::string shellExt) {
    vec3<double> kt;
    for (int i = 0; i < ks.size(); ++i) {
        std::string k1File = filename(shellBase, 2, i, shellExt);
        kt.x = ks[i];
        for (int j = i; j < ks.size(); ++j) {
            std::string k2File = filename(shellBase, 2, j, shellExt);
            kt.y = ks[j];
            for (int k = j; k < ks.size(); ++k) {
                std::string k3File = filename(shellBase, 2, k, shellExt);
                kt.z = ks[k];
                double B_est = shell_prod(k1File, k2File, k3File, N);
                B_est -= (P[i] + P[j] + P[k])*gal_bk_nbw.y;
                B_est -= gal_bk_nbw.x - alpha*alpha*ran_bk_nbw.x;
                B_est /= gal_bk_nbw.z;
                B.push_back(B_est);
                k_trip.push_back(kt);
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
    double V_f = get_V_f(L);
    std::cout << V_f << std::endl;
    vec3<double> kt;
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
                    std::cout << ks[i] << ", " << ks[j] << ", " << ks[k] << ", ";
                    double V_ijk = get_V_ijk(ks[i], ks[j], ks[k], delta_k);
                    get_shell((fftw_complex *) shell_3.data(), (fftw_complex *) delta.data(), kx, ky, 
                              kz, ks[k], delta_k, N);
                    bip_c2r(shell_3, N, wisdomFile, omp_get_max_threads());
                    kt.z = ks[k];
                    
                    double B_est = shell_prod(shell_1, shell_2, shell_3, N);
                    std::cout << B_est << ", ";
                    B_est /= gal_bk_nbw.z;
                    std::cout << B_est << ", ";
                    double SN = ((P[i] + P[j] + P[k])*gal_bk_nbw.y + gal_bk_nbw.x - alpha*alpha*ran_bk_nbw.x)/gal_bk_nbw.z;
                    B_est *= 1.0/V_ijk;
                    std::cout << B_est << ", ";
                    B_est -= SN;
                    std::cout << B_est << std::endl;
                    B.push_back(B_est);
                    k_trip.push_back(kt);
                }
            }
        }
    }
}

void normalize_delta(std::vector<double> &delta, vec3<int> N) {
    double N_tot = N.x*N.y*N.z;
    #pragma omp parallel for
    for (size_t i = 0; i < delta.size(); ++i)
        delta[i] /= N_tot;
}
