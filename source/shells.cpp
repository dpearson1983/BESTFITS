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
    int N_shells = (k_max - k_min)/delta_k;
    for (int i =0; i < N_shells; ++i) {
        double k_shell = k_min + (i + 0.5)*delta_k;
        get_shell((fftw_complex *)shell.data(), (fftw_complex *)dk.data(), kx, ky, kz, k_shll, delta_k, N);
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

std::vector<size_t> get_num_triangles() {
    std::vector<size_t> num_triangles;
    std::ifstream fin("numTriangles.dat");
    
    while (!fin.eof()) {
        double k1, k2, k3, num_theo, diff;
        size_t num_act;
        fin >> k1 >> k2 >> k3 >> num_act >> num_theo >> diff;
        if (!fin.eof()) {
            num_triangles.push_back(num_act);
        }
    }
    fin.close();
    
    return num_triangles;
}

void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<std::vector<double>> &shells, double delta_k, double k_min, double k_max) {
    vec3<double> kt;
    int N_shells = (k_max - k_min)/delta_k;
    double V_f = get_V_f(L);
    
    for (int i = 0; i < N_shells; ++i) {
        kt.x = ks[i];
        for (int j = i; i < N_shells; ++j) {
            kt.y = ks[j];
            for (int k = j; j < N_shells; ++k) {
                kt.z = ks[k];
                if (kt.z <= kt.x + kt.y) {
                    double V_ijk = get_V_ijk(ks[i], ks[j], ks[k], delta_k);
                    double B_est = shell_prod(shells[i], shells[j], shells[k], N)/(N.x*N.y*N.z);
                    B_est /= gal_bk_nbw.z;
                    B_est *= V_f*V_f;
                    B_est /= V_ijk;
                    double SN = (P[i] + P[j] + P[k])*gal_bk_nbw.y + gal_bk_nbw.x - alpha*alpha*ran_bk_nbw.x;
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
    fftw_init_threads();
    fftw_import_wisdom_from_filename(wisdomFile.c_str());
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_plan trans_shell_1 = fftw_plan_dft_c2r_3d(N.x, N.y, N.z, (fftw_complex *)shell_1.data(),
                                                   shell_1.data(), FFTW_MEASURE);
    fftw_plan trans_shell_2 = fftw_plan_dft_c2r_3d(N.x, N.y, N.z, (fftw_complex *)shell_2.data(),
                                                   shell_2.data(), FFTW_MEASURE);
    fftw_plan trans_shell_3 = fftw_plan_dft_c2r_3d(N.x, N.y, N.z, (fftw_complex *)shell_3.data(),
                                                   shell_3.data(), FFTW_MEASURE);
    fftw_export_wisdom_to_filename(wisdomFile.c_str());
    double N_tot = N.x*N.y*N.z;
    double V_f = get_V_f(L);
    std::cout << V_f << std::endl;
    std::cout << gal_bk_nbw.z << ", " << alpha*ran_bk_nbw.z << std::endl;
    vec3<double> kt;
    zero_shell(shell_1);
    zero_shell(shell_2);
    zero_shell(shell_3);
    std::ofstream fout("shotnoise.dat");
    fout.precision(15);
    double avg_fft_time = 0.0;
    int num_ffts = 0;
    double avg_shell_time = 0.0;
    int num_shells = 0;
    double start_fft, start_shell;
    std::vector<size_t> num_triangles = get_num_triangles();
    int triangle_index = 0;
    for (int i = 0; i < ks.size(); ++i) {
        start_shell = omp_get_wtime();
        get_shell((fftw_complex *) shell_1.data(), (fftw_complex *) delta.data(), kx, ky, kz, ks[i], 
                  delta_k, N);
        avg_shell_time += omp_get_wtime() - start_shell;
        num_shells++;
        start_fft = omp_get_wtime();
        fftw_execute(trans_shell_1);
        avg_fft_time += omp_get_wtime() - start_fft;
        num_ffts++;
        kt.x = ks[i];
        for (int j = i; j < ks.size(); ++j) {
            start_shell = omp_get_wtime();
            get_shell((fftw_complex *) shell_2.data(), (fftw_complex *) delta.data(), kx, ky, kz, ks[j], 
                      delta_k, N);
            avg_shell_time += omp_get_wtime() - start_shell;
            num_shells++;
            start_fft = omp_get_wtime();
            fftw_execute(trans_shell_2);
            avg_fft_time += omp_get_wtime() - start_fft;
            num_ffts++;
            kt.y = ks[j];
            for (int k = j; k < ks.size(); ++k) {
                if (ks[k] <= ks[i] + ks[j]) {
                    std::cout << ks[i] << ", " << ks[j] << ", " << ks[k] << ", ";
                    double V_ijk = get_V_ijk(ks[i], ks[j], ks[k], delta_k);
                    std::cout << V_ijk << ", ";
                    start_shell = omp_get_wtime();
                    get_shell((fftw_complex *) shell_3.data(), (fftw_complex *) delta.data(), kx, ky, 
                              kz, ks[k], delta_k, N);
                    avg_shell_time += omp_get_wtime() - start_shell;
                    num_shells++;
                    start_fft = omp_get_wtime();
                    fftw_execute(trans_shell_3);
                    avg_fft_time += omp_get_wtime() - start_fft;
                    num_ffts++;
                    kt.z = ks[k];
                    
                    double B_est = shell_prod(shell_1, shell_2, shell_3, N)/N_tot;
                    std::cout << B_est << ", ";
                    B_est /= gal_bk_nbw.z;
                    std::cout << B_est << ", ";
                    double SN = ((P[i] + P[j] + P[k])*gal_bk_nbw.y + gal_bk_nbw.x - alpha*alpha*ran_bk_nbw.x)/gal_bk_nbw.z;
//                     B_est *= V_f*V_f;
//                     B_est /= V_ijk;
                    B_est /= num_triangles[triangle_index];
                    triangle_index++;
                    std::cout << B_est << ", ";
                    B_est -= SN;
                    std::cout << B_est << ", " << SN << std::endl;
                    B.push_back(B_est);
                    k_trip.push_back(kt);
                    fout << ks[i] << " " << ks[j] << " " << ks[k] << " " << SN << "\n";
                }
            }
        }
    }
    fout.close();
    fftw_destroy_plan(trans_shell_1);
    fftw_destroy_plan(trans_shell_2);
    fftw_destroy_plan(trans_shell_3);
    fftw_cleanup_threads();
    std::cout << "Average time of " << num_ffts << " FFTs: " << avg_fft_time/num_ffts << " s\n";
    std::cout << "Average time of " << num_shells << " Shells: " << avg_shell_time/num_shells;
    std::cout << std::endl;
}

void normalize_delta(std::vector<double> &delta, vec3<int> N, double gal_bk_nbw_z) {
    double N_tot = N.x*N.y*N.z;
    #pragma omp parallel for
    for (size_t i = 0; i < delta.size(); ++i)
//         delta[i] /= N_tot*pow(gal_bk_nbw_z,1.0/2.0);
        delta[i] /= N_tot;
}

