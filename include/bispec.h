#ifndef _BISPEC_H_
#define _BISPEC_H_

#include <vector>
#include <string>
#include "tpods.h"

double get_bispectrum_shot_noise(double P_1, double P_2, double P_3, vec3<double> gal_bk_nbw, 
                                 vec3<double> ran_bk_nbw, double alpha);

double get_bispectrum_shot_noise(int k1, int k2, int k3, fftw_complex *A_0, fftw_complex *A_2, fftw_complex *Fw_0, 
                                 fftw_complex *Fw_2, std::vector<std::vector<vec3<double>>> &shells, vec3<int> N,
                                 vec3<double> L, vec3<double> gal_bk_nbw, vec3<double> ran_bk_nbw, int l, 
                                 double k_min, double Delta_k, double alpha);

void generate_triangle_file(const vec3<int> N, const vec3<double> L, double k_min, double k_max, double Delta_k, 
                            std::vector<std::vector<vec3<double>>> shells);

void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<std::vector<double>> &shells, double delta_k, double k_min, double k_max,
                    bool exactTriangles);

void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<std::vector<double>> &shells, double delta_k, double k_min, double k_max, 
                    std::vector<double> &SN, bool exactTriangles);

void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<double> &delta, std::vector<double> &kx, std::vector<double> &ky, 
                    std::vector<double> &kz, double k_min, double k_max, double delta_k, std::string wisdomFile, 
                    bool exactTriangles);

void get_bispectrum_quad(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<std::vector<double>> &A0_shells, std::vector<std::vector<double>> &A2_shells,
                    double delta_k, double k_min, double k_max, std::vector<double> &SN, bool exactTriangles);

void get_bispectrum_quad(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<double> &A_0, std::vector<double> &A_2, std::vector<double> &kx, 
                    std::vector<double> &ky, std::vector<double> &kz, double k_min, double k_max, double delta_k, 
                    std::string wisdomFile, std::vector<double> &SN, bool exactTriangles);

#endif
