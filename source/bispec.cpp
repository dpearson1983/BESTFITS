#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
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

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

vec4<size_t> get_index(vec3<double> k, vec3<double> k_f, vec3<int> N) {
    vec4<size_t> index = {0, 0, 0, 0};
    if (k.x < 0) {
        index.x = int(k.x/k_f.x) + N.x;
    } else {
        index.x = int(k.x/k_f.x);
    }
    
    if (k.y < 0) {
        index.y = int(k.y/k_f.y) + N.y;
    } else {
        index.y = int(k.y/k_f.y);
    }
    
    if (k.z < 0) {
        index.z = int(k.z/k_f.z) + N.z;
        vec3<double> k_neg = {-k.x, -k.y, -k.z};
        vec4<size_t> index_neg = get_index(k_neg, k_f, N);
        index.w = index_neg.w;
    } else {
        index.z = int(k.z/k_f.z);
        index.w = index.z + (N.z/2 + 1)*(index.y + N.y*index.x);
    }
    
    return index;
}

double get_mag(vec3<double> k) {
    return sqrt(k.x*k.x + k.y*k.y + k.z*k.z);
}    

int get_bin(vec3<double> k_1, vec3<double> k_2, vec3<double> k_3, double delta_k, double k_min, double k_max) {
    int N_k = (k_max - k_min)/delta_k;
    std::vector<double> ks;
    ks.push_back(get_mag(k_1));
    ks.push_back(get_mag(k_2));
    ks.push_back(get_mag(k_3));
    std::sort(ks.begin(), ks.end());
    
    int k1_i = (ks[0] - k_min)/delta_k;
    int k2_i = (ks[1] - k_min)/delta_k;
    int k3_i = (ks[2] - k_min)/delta_k;
    
    int bispecBin = 0;
    for (int i = 0; i < N_k; ++i) {
        double k1 = k_min + (i + 0.5)*delta_k;
        for (int j = i; j < N_k; ++j) {
            double k2 = k_min + (j + 0.5)*delta_k;
            for (int k = j; k < N_k; ++k) {
                double k3 = k_min + (k + 0.5)*delta_k;
                if (k3 <= k1 + k2) {
                    if (k1_i == i && k2_i == j && k3_i == k) {
                        return bispecBin;
                    }
                    bispecBin++;
                }
            }
        }
    }
    
    std::cout << "k1_i = " << k1_i << std::endl;
    std::cout << "k2_i = " << k2_i << std::endl;
    std::cout << "k3_i = " << k3_i << std::endl;
    std::cout << ks[0] << " " << ks[1] << " " << ks[2] << std::endl;
    
    return -1;
}

double get_power(size_t index, fftw_complex *A, fftw_complex *B) {
    return A[index][0]*B[index][0] + A[index][1]*B[index][1];
}
    
void process_shells(std::vector<std::vector<double>> &covariance, std::vector<std::vector<size_t>> &N_tri,
                    std::vector<vec3<double>> &k_1, std::vector<vec3<double>> &k_2, fftw_complex *A_0,
                    fftw_complex *A_2, vec3<double> k_f, vec3<int> N, double delta_k, double k_min,
                    double k_max, int N_bin) {
    for (int i = 0; i < k_1.size(); ++i) {
        int k1_bin = (get_mag(k_1[i]) - k_min)/delta_k;
        double k1_bin_mag = k_min + (k1_bin + 0.5)*delta_k;
        for (int j = 0; j < k_2.size(); ++j) {
            int k2_bin = (get_mag(k_2[i]) - k_min)/delta_k;
            double k2_bin_mag = k_min + (k2_bin + 0.5)*delta_k;
            vec3<double> k_3 = {-k_1[i].x - k_2[j].x, -k_1[i].y - k_2[j].y, -k_1[i].z - k_2[j].z};
            vec4<size_t> k1_index = get_index(k_1[i], k_f, N);
            vec4<size_t> k2_index = get_index(k_2[j], k_f, N);
            vec4<size_t> k3_index = get_index(k_3, k_f, N);
            int k3_bin = (get_mag(k_3) - k_min)/delta_k;
            double k3_bin_mag = k_min + (k3_bin + 0.5)*delta_k;
            std::vector<double> ks = {k1_bin_mag, k2_bin_mag, k3_bin_mag};
            std::sort(ks.begin(), ks.end());
            if (ks[2] <= ks[0] + ks[1] && get_mag(k_3) < k_max && get_mag(k_3) >= k_min) {
                int monopole_bin = get_bin(k_1[i], k_2[j], k_3, delta_k, k_min, k_max);
                int quadrupole_bin = monopole_bin + N_bin;
                if (monopole_bin >= 0) {
                    covariance[monopole_bin][monopole_bin] += get_power(k1_index.w, A_0, A_0)*get_power(k2_index.w, A_0, A_0)*get_power(k3_index.w, A_0, A_0);
                    covariance[quadrupole_bin][quadrupole_bin] += get_power(k1_index.w, A_0, A_2)*get_power(k2_index.w, A_0, A_2)*get_power(k3_index.w, A_0, A_0);
                    covariance[monopole_bin][quadrupole_bin] += get_power(k1_index.w, A_0, A_2)*get_power(k2_index.w, A_0, A_0)*get_power(k3_index.w, A_0, A_0);
                    covariance[quadrupole_bin][monopole_bin] += get_power(k1_index.w, A_0, A_2)*get_power(k2_index.w, A_0, A_0)*get_power(k3_index.w, A_0, A_0);
                    N_tri[monopole_bin][monopole_bin]++;
                    N_tri[quadrupole_bin][quadrupole_bin]++;
                    N_tri[monopole_bin][quadrupole_bin]++;
                    N_tri[quadrupole_bin][monopole_bin]++;
                } else {
                    std::stringstream errMsg;
                    errMsg << "Bispectrum bin number not found" << std::endl;
                    errMsg << "(" << get_mag(k_1[i]) << ", " << get_mag(k_2[j]) << ", " << get_mag(k_3) << ")" << std::endl;
                    throw std::runtime_error(errMsg.str());
                }
            }
        }
    }
}
            

void get_covariance(std::vector<std::vector<double>> &covariance, std::vector<std::vector<size_t>> &N_tri,
                    std::vector<std::vector<vec3<double>>> &shells, std::vector<double> &A_0, std::vector<double> &A_2, vec3<int> N, vec3<double> L, double delta_k, double k_min, double k_max, int N_bin) {
    
    vec3<double> k_f = {2.0*PI/L.x, 2.0*PI/L.y, 2.0*PI/L.z};
    
    for (int i = 0; i < shells.size(); ++i) {
        for (int j = i; j < shells.size(); ++j) {
            process_shells(covariance, N_tri, shells[i], shells[j], (fftw_complex *)A_0.data(), 
                           (fftw_complex *)A_2.data(), k_f, N, delta_k, k_min, k_max, N_bin);
            
        }
    }
    
    for (size_t i = 0; i < covariance.size(); ++i) {
        for (size_t j = 0; j < covariance[i].size(); ++j) {
            if (N_tri[i][j] > 0) {
                covariance[i][j] /= N_tri[i][j];
            }
        }
    }
}

size_t get_actual_triangles(const vec3<int> N, const vec3<double> L, std::vector<vec3<double>> &k1,
                            std::vector<vec3<double>> &k2, double k3, double Delta_k) {
    std::vector<size_t> triangles(omp_get_max_threads());
    
    #pragma omp parallel for
    for (size_t i = 0; i < k1.size(); ++i) {
        for (size_t j = 0; j < k2.size(); ++j) {
            vec3<double> k3_vec = {-k1[i].x - k2[j].x, -k1[i].y - k2[j].y, -k1[i].z - k2[j].z};
            double k3_mag = sqrt(k3_vec.x*k3_vec.x + k3_vec.y*k3_vec.y + k3_vec.z*k3_vec.z);
            if (k3_mag >= k3 - 0.5*Delta_k && k3_mag < k3 + 0.5*Delta_k) {
                triangles[omp_get_thread_num()]++;
            }
        }
    }
    
    for (int i = 1; i < omp_get_max_threads(); ++i)
        triangles[0] += triangles[i];
    
    return triangles[0];
}

void generate_triangle_file(const vec3<int> N, const vec3<double> L, double k_min, double k_max, double Delta_k, 
                            std::vector<std::vector<vec3<double>>> shells) {
    int N_kbins = (k_max - k_min)/Delta_k;
    std::vector<size_t> N_tri;
    for (int i = 0; i < N_kbins; ++i) {
        double k1 = k_min + (i + 0.5)*Delta_k;
        for (int j = i; j < N_kbins; ++j) {
            double k2 = k_min + (j + 0.5)*Delta_k;
            for (int k = j; k < N_kbins; ++k) {
                double k3 = k_min + (k + 0.5)*Delta_k;
                if (k3 <= k1 + k2) {
                    size_t Nt = get_actual_triangles(N, L, shells[i], shells[j], k3, Delta_k);
                    N_tri.push_back(Nt);
                }
            }
        }
    }
    writeTriangleFile(N_tri, L, k_min, k_max);
}

double get_bispectrum_shot_noise(double P_1, double P_2, double P_3, vec3<double> gal_bk_nbw, 
                                 vec3<double> ran_bk_nbw, double alpha) {
    double SN = (P_1 + P_2 + P_3)*gal_bk_nbw.y + gal_bk_nbw.x - alpha*alpha*alpha*ran_bk_nbw.x;
    SN /= gal_bk_nbw.z;
    return SN;
}

double complex_product(vec4<size_t> i, vec4<size_t> j, vec3<int> N, fftw_complex *A, fftw_complex *B) {
    fftw_complex a, b;
    if (i.z > N.z/2) {
        a[0] = A[i.w][0];
        a[1] = -A[i.w][1];
    } else {
        a[0] = A[i.w][0];
        a[1] = A[i.w][1];
    }
    
    if (j.z > N.z/2) {
        b[0] = B[j.w][0];
        b[1] = -B[j.w][1];
    } else {
        b[0] = B[j.w][0];
        b[1] = B[j.w][1];
    }
    
    return a[0]*b[0] + a[1]*b[1];
}

double get_mu(double a, double b, double c) {
    return (a*a + b*b - c*c)/(2.0*a*b);
}

double L_2(double mu) {
    return (3.0*mu*mu - 1.0)/2.0;
}
    
double get_bispectrum_shot_noise(int k1, int k2, int k3, fftw_complex *A_0, fftw_complex *A_2, fftw_complex *Fw_0, 
                                 fftw_complex *Fw_2, std::vector<std::vector<vec3<double>>> &shells, vec3<int> N,
                                 vec3<double> L, vec3<double> gal_bk_nbw, vec3<double> ran_bk_nbw, int l, 
                                 double k_min, double Delta_k, double alpha) {
    vec3<double> k_f = {(2.0*PI)/L.x, (2.0*PI)/L.y, (2.0*PI)/L.z};
    double V_f = 1.0;
    double SN1 = 0.0, SN2 = 0.0, SN3 = 0.0;
    double k_1 = k_min + (k1 + 0.5)*Delta_k;
    double k_2 = k_min + (k2 + 0.5)*Delta_k;
    double k_3 = k_min + (k3 + 0.5)*Delta_k;
    double N_1 = 0, N_2 = 0, N_3 = 0;
    for (size_t i = 0; i < shells[k1].size(); ++i) {
        vec3<double> k1_minus = {-shells[k1][i].x, -shells[k1][i].y, -shells[k1][i].z};
        vec4<size_t> index_plus = get_index(shells[k1][i], k_f, N);
        vec4<size_t> index_minus = get_index(k1_minus, k_f, N);
        if (l == 0) {
            SN1 += complex_product(index_plus, index_minus, N, A_0, Fw_0);
            N_1++;
        } else if (l == 2) {
            SN1 += complex_product(index_plus, index_minus, N, A_2, Fw_0);
            N_1++;
        }
    }
    SN1 *= (2*l + 1)/N_1;
    
    for (size_t i = 0; i < shells[k2].size(); ++i) {
        vec3<double> k2_minus = {-shells[k2][i].x, -shells[k2][i].y, -shells[k2][i].z};
        vec4<size_t> index_plus = get_index(shells[k2][i], k_f, N);
        vec4<size_t> index_minus = get_index(k2_minus, k_f, N);
        if (l == 0) {
            SN2 += complex_product(index_plus, index_minus, N, A_0, Fw_0);
            N_2++;
        } else if (l == 2) {
            SN2 += complex_product(index_plus, index_minus, N, A_0, Fw_2);
            N_2++;
        }
    }
    if (l == 0) {
        SN2 /= N_2;
    } else if (l == 2) {
        SN2 *= (2*l + 1)*L_2(get_mu(k_1, k_2, k_3))/N_2;
    }
    
    for (size_t i = 0; i < shells[k3].size(); ++i) {
        vec3<double> k3_minus = {-shells[k3][i].x, -shells[k3][i].y, -shells[k3][i].z};
        vec4<size_t> index_plus = get_index(shells[k3][i], k_f, N);
        vec4<size_t> index_minus = get_index(k3_minus, k_f, N);
        if (l == 0) {
            SN3 += complex_product(index_plus, index_minus, N, A_0, Fw_0);
            N_3++;
        } else if (l == 2) {
            SN3 += complex_product(index_plus, index_minus, N, A_0, Fw_2);
            N_3++;
        }
    }
    if (l == 0) {
        SN3 /= N_3;
    } else if (l == 2) {
        SN3 *= (2*l + 1)*L_2(get_mu(k_1, k_3, k_2))/N_3;
    }
    
    double shotNoise = 0.0;
    if (l == 0) {
        shotNoise = (SN1 + SN2 + SN3 - 2.0*(gal_bk_nbw.x - alpha*alpha*alpha*ran_bk_nbw.x))/gal_bk_nbw.z;
    } else if (l == 2) {
        shotNoise = (SN1 + SN2 + SN3)/gal_bk_nbw.z;
    }
    
    return shotNoise;
}

void get_N_tri(std::string file, std::vector<size_t> &N_tri) {
    std::ifstream fin(file);
    while(!fin.eof()) {
        size_t N;
        fin >> N;
        if (!fin.eof()) {
            N_tri.push_back(N);
        }
    }
    fin.close();
}

// Normal mode for calculating the bispectrum. The shell Fourier transforms are calculated ahead of time and
// stored in memory. This function the simply carries out the product, the summing, the normalization and the 
// shot noise subtraction.
void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec4<double>> &k_trip, 
                    std::vector<std::vector<double>> &shells, double delta_k, double k_min, double k_max, 
                    bool exactTriangles) {
    vec4<double> kt;
    kt.w = 0;
    double V_f = get_V_f(L);
    double N_tot = N.x*N.y*N.z;
    double alpha3 = alpha*alpha*alpha;
    std::vector<size_t> N_tri;
    if (exactTriangles) {
        std::string triangleFile = triangleFilename(L, k_min, k_max);
        get_N_tri(triangleFile, N_tri);
    }
    
    int bispecBin = 0;
    for (int i = 0; i < ks.size(); ++i) {
        kt.x = ks[i];
        for (int j = i; j < ks.size(); ++j) {
            kt.y = ks[j];
            for (int k = j; k < ks.size(); ++k) {
                kt.z = ks[k];
                if (ks[k] <= ks[i] + ks[j]) {                    
                    double B_est = shell_prod(shells[i], shells[j], shells[k], N)/N_tot;
                    B_est /= gal_bk_nbw.z;
                    if (exactTriangles) {
                        B_est /= N_tri[bispecBin];
                    } else {
                        double V_ijk = get_V_ijk(ks[i], ks[j], ks[k], delta_k);
                        B_est *= V_f*V_f;
                        B_est /= V_ijk;
                    }
                    double SN = (P[i] + P[j] + P[k])*gal_bk_nbw.y + gal_bk_nbw.x - alpha3*ran_bk_nbw.x;
                    SN /= gal_bk_nbw.z;
                    B_est -= SN;
                    B.push_back(B_est);
                    k_trip.push_back(kt);
                    bispecBin++;
                }
            }
        }
    }
}

void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec4<double>> &k_trip, 
                    std::vector<std::vector<double>> &shells, double delta_k, double k_min, double k_max, std::vector<double> &SN, bool exactTriangles) {
    vec4<double> kt;
    kt.w = 0;
    double V_f = get_V_f(L);
    double N_tot = N.x*N.y*N.z;
    double alpha3 = alpha*alpha*alpha;
    std::vector<size_t> N_tri;
    if (exactTriangles) {
        std::string triangleFile = triangleFilename(L, k_min, k_max);
        get_N_tri(triangleFile, N_tri);
    }
    
    int bispecBin = 0;
    for (int i = 0; i < ks.size(); ++i) {
        kt.x = ks[i];
        for (int j = i; j < ks.size(); ++j) {
            kt.y = ks[j];
            for (int k = j; k < ks.size(); ++k) {
                kt.z = ks[k];
                if (ks[k] <= ks[i] + ks[j]) {
                    double B_est = shell_prod(shells[i], shells[j], shells[k], N)/N_tot;
                    B_est /= gal_bk_nbw.z;
                    if (exactTriangles) {
                        B_est /= N_tri[bispecBin];
                    } else {
                        double V_ijk = get_V_ijk(ks[i], ks[j], ks[k], delta_k);
                        B_est *= V_f*V_f;
                        B_est /= V_ijk;
                    }
                    B_est -= SN[bispecBin];
                    B.push_back(B_est);
                    k_trip.push_back(kt);
                    bispecBin++;
                }
            }
        }
    }
}

// Low memory mode implementation. Here the shell Fourier transforms are performed on the fly instead of storing
// them all in memory to reduce memory usage by N_shells/3, as we only need to have storage for the three
// Fourier transforms currently in use by the code. The memory savings comes at the cost of speed, however, so
// only use low memory mode when absolutely necessary.
void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec4<double>> &k_trip, 
                    std::vector<double> &delta, std::vector<double> &kx, std::vector<double> &ky, 
                    std::vector<double> &kz, double k_min, double k_max, double delta_k, std::string wisdomFile, 
                    bool exactTriangles) {
    std::vector<double> shell_1(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> shell_2(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> shell_3(N.x*N.y*2*(N.z/2 + 1));
    generate_wisdom_bipc2r(shell_1, N, wisdomFile, omp_get_max_threads());
    generate_wisdom_bipc2r(shell_2, N, wisdomFile, omp_get_max_threads());
    generate_wisdom_bipc2r(shell_3, N, wisdomFile, omp_get_max_threads());
    double N_tot = N.x*N.y*N.z;
    double V_f = get_V_f(L);
    vec4<double> kt;
    kt.w = 0;
    double alpha3 = alpha*alpha*alpha;
    std::vector<size_t> N_tri;
    if (exactTriangles) {
        std::string triangleFile = triangleFilename(L, k_min, k_max);
        get_N_tri(triangleFile, N_tri);
    }
    int bispecBin = 0;
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
                    get_shell((fftw_complex *) shell_3.data(), (fftw_complex *) delta.data(), kx, ky, 
                              kz, ks[k], delta_k, N);
                    bip_c2r(shell_3, N, wisdomFile, omp_get_max_threads());
                    kt.z = ks[k];
                    
                    double B_est = shell_prod(shell_1, shell_2, shell_3, N)/N_tot;
                    B_est /= gal_bk_nbw.z;
                    if (exactTriangles) {
                        B_est /= N_tri[bispecBin];
                    } else {
                        double V_ijk = get_V_ijk(ks[i], ks[j], ks[k], delta_k);
                        B_est *= V_f*V_f;
                        B_est /= V_ijk;
                    }
                    double SN = (P[i] + P[j] + P[k])*gal_bk_nbw.y + gal_bk_nbw.x - alpha3*ran_bk_nbw.x;
                    SN /= gal_bk_nbw.z;
                    B_est -= SN;
                    B.push_back(B_est);
                    k_trip.push_back(kt);
                    bispecBin++;
                }
            }
        }
    }
}

void get_bispectrum_quad(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec4<double>> &k_trip, 
                    std::vector<std::vector<double>> &A0_shells, std::vector<std::vector<double>> &A2_shells,
                    double delta_k, double k_min, double k_max, std::vector<double> &SN, bool exactTriangles) {
    vec4<double> kt;
    kt.w = 2;
    double V_f = get_V_f(L);
    double N_tot = N.x*N.y*N.z;
    double alpha3 = alpha*alpha*alpha;
    std::vector<size_t> N_tri;
    if (exactTriangles) {
        std::string triangleFile = triangleFilename(L, k_min, k_max);
        get_N_tri(triangleFile, N_tri);
    }
    int bispecBin = 0;
    
    for (int i = 0; i < ks.size(); ++i) {
        kt.x = ks[i];
        for (int j = i; j < ks.size(); ++j) {
            kt.y = ks[j];
            for (int k = j; k < ks.size(); ++k) {
                kt.z = ks[k];
                if (ks[k] <= ks[i] + ks[j]) {
                    double B_est = shell_prod(A2_shells[i], A0_shells[j], A0_shells[k], N)/N_tot;
                    B_est /= gal_bk_nbw.z;
                    if (exactTriangles) {
                        B_est /= N_tri[bispecBin];
                    } else {
                        double V_ijk = get_V_ijk(ks[i], ks[j], ks[k], delta_k);
                        B_est *= V_f*V_f;
                        B_est /= V_ijk;
                    }
                    B_est -= SN[bispecBin];
                    B.push_back(B_est);
                    k_trip.push_back(kt);
                    bispecBin++;
                }
            }
        }
    }
}

// Low memory mode implementation. Here the shell Fourier transforms are performed on the fly instead of storing
// them all in memory to reduce memory usage by N_shells/3, as we only need to have storage for the three
// Fourier transforms currently in use by the code. The memory savings comes at the cost of speed, however, so
// only use low memory mode when absolutely necessary.
void get_bispectrum_quad(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec4<double>> &k_trip, 
                    std::vector<double> &A_0, std::vector<double> &A_2, std::vector<double> &kx, 
                    std::vector<double> &ky, std::vector<double> &kz, double k_min, double k_max, double delta_k, 
                    std::string wisdomFile, std::vector<double> &SN, bool exactTriangles) {
    std::vector<double> shell_1(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> shell_2(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> shell_3(N.x*N.y*2*(N.z/2 + 1));
    generate_wisdom_bipc2r(shell_1, N, wisdomFile, omp_get_max_threads());
    generate_wisdom_bipc2r(shell_2, N, wisdomFile, omp_get_max_threads());
    generate_wisdom_bipc2r(shell_3, N, wisdomFile, omp_get_max_threads());
    double N_tot = N.x*N.y*N.z;
    double V_f = get_V_f(L);
    vec4<double> kt;
    kt.w = 2;
    std::vector<size_t> N_tri;
    if (exactTriangles) {
        std::string triangleFile = triangleFilename(L, k_min, k_max);
        get_N_tri(triangleFile, N_tri);
    }
    int bispecBin = 0;
    for (int i = 0; i < ks.size(); ++i) {
        get_shell((fftw_complex *) shell_1.data(), (fftw_complex *) A_2.data(), kx, ky, kz, ks[i], 
                  delta_k, N);
        bip_c2r(shell_1, N, wisdomFile, omp_get_max_threads());
        kt.x = ks[i];
        for (int j = i; j < ks.size(); ++j) {
            get_shell((fftw_complex *) shell_2.data(), (fftw_complex *) A_0.data(), kx, ky, kz, ks[j], 
                      delta_k, N);
            bip_c2r(shell_2, N, wisdomFile, omp_get_max_threads());
            kt.y = ks[j];
            for (int k = j; k < ks.size(); ++k) {
                if (ks[k] <= ks[i] + ks[j]) {
                    get_shell((fftw_complex *) shell_3.data(), (fftw_complex *) A_0.data(), kx, ky, 
                              kz, ks[k], delta_k, N);
                    bip_c2r(shell_3, N, wisdomFile, omp_get_max_threads());
                    kt.z = ks[k];
                    
                    double B_est = shell_prod(shell_1, shell_2, shell_3, N)/N_tot;
                    B_est /= gal_bk_nbw.z;
                    if (exactTriangles) {
                        B_est /= N_tri[bispecBin];
                    } else {
                        double V_ijk = get_V_ijk(ks[i], ks[j], ks[k], delta_k);
                        B_est *= V_f*V_f;
                        B_est /= V_ijk;
                    }
                    B_est -= SN[bispecBin];
                    B.push_back(B_est);
                    k_trip.push_back(kt);
                    bispecBin++;
                }
            }
        }
    }
}

int getNumBispecBins(double k_min, double k_max, double binWidth, std::vector<vec4<double>> &ks) {
    int totBins = 0;
    int N = (k_max - k_min)/binWidth;
    
    for (int i = 0; i < N; ++i) {
        double k_1 = k_min + (i + 0.5)*binWidth;
        for (int j = i; j < N; ++j) {
            double k_2 = k_min + (j + 0.5)*binWidth;
            for (int k = j; k < N; ++k) {
                double k_3 = k_min + (k + 0.5)*binWidth;
                if (k_3 <= k_1 + k_2 && k_3 <= k_max) {
                    totBins++;
                    vec4<double> kt = {k_1, k_2, k_3, 0};
                    ks.push_back(kt);
                }
            }
        }
    }
    
    return totBins;
}

