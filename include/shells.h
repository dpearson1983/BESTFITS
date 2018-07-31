#ifndef _SHELLS_H_
#define _SHELLS_H_

#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include "tpods.h"

double get_V_ijk(double k_i, double k_j, double k_k, double delta_k);

double get_V_f(vec3<double> L);

void get_shell(fftw_complex *shell, fftw_complex *dk, 
               std::vector<double> &kx, std::vector<double> &ky, std::vector<double> &kz, 
               double k_shell, double delta_k, vec3<int> N);

double shell_prod(std::vector<double> &r_1, std::vector<double> &r_2, std::vector<double> &r_3, 
                  vec3<int> N);

void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<double> &delta, std::vector<double> &kx, std::vector<double> &ky, 
                    std::vector<double> &kz, double delta_k, std::string wisdomFile);

void normalize_delta(std::vector<double> &delta, vec3<int> N);

#endif
