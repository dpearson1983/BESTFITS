#ifndef _SHELLS_H_
#define _SHELLS_H_

#include <vector>
#include <string>
#include <fftw3.h>
#include "tpods.h"

double get_V_ijk(double k_i, double k_j, double k_k, double delta_k);

double get_V_f(vec3<double> L);

std::vector<vec3<double>> get_shell_vecs(const vec3<int> N, const vec3<double> L, double k_shell, double Delta_k);

void get_shell(fftw_complex *shell, fftw_complex *dk, 
               std::vector<double> &kx, std::vector<double> &ky, std::vector<double> &kz, 
               double k_shell, double delta_k, vec3<int> N);

void get_shells(std::vector<std::vector<double>> &shells, std::vector<double> &dk, std::vector<double> &kx,
                std::vector<double> &ky, std::vector<double> &kz, double k_min, double k_max, double delta_k, 
                vec3<int> N, std::string wisdomFile);

double shell_prod(std::vector<double> &r_1, std::vector<double> &r_2, std::vector<double> &r_3, 
                  vec3<int> N);

#endif
