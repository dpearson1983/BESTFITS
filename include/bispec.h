#ifndef _BISPEC_H_
#define _BISPEC_H_

#include <vector>
#include <string>
#include "tpods.h"

double get_bispectrum_shot_noise(int k1, int k2, int k3, fftw_complex *A_0, fftw_complex *A_2, fftw_complex *Fw_0, 
                                 fftw_complex *Fw_2, std::vector<std::vector<vec3<double>>> shells, vec3<int> N,
                                 vec3<double> L, vec3<double> gal_bk_nbw, vec3<double> ran_bk_nbw, int l, 
                                 double k_min, double Delta_k);

void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<std::vector<double>> &shells, double delta_k, double k_min, double k_max);

void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<double> &delta, std::vector<double> &kx, std::vector<double> &ky, 
                    std::vector<double> &kz, double delta_k, std::string wisdomFile);

#endif
