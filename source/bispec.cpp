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

double real_part(vec4<size_t> i, vec4<size_t> j, vec3<int> N, fftw_complex *A, fftw_complex *B) {
    fftw_complex a, b;
    if (i.z > N.z/2) {
        a[0] = A[i.w][0];
        a[1] = -A[i.w][1];
    } else {
        a[0] = A[i.w][0];
        a[1] = A[i.w][1];
    }
    
    if (j.z > N.z/2) {
        b[0] = B[i.w][0];
        b[1] = -B[i.w][1];
    } else {
        b[0] = B[i.w][0];
        b[1] = B[i.w][1];
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
    double SN1 = 0.0, SN2 = 0.0, SN3 = 0.0;
    double k_1 = k_min + (k1 + 0.5)*Delta_k;
    double k_2 = k_min + (k2 + 0.5)*Delta_k;
    double k_3 = k_min + (k3 + 0.5)*Delta_k;
    int N_1 = 0, N_2 = 0, N_3 = 0;
    for (size_t i = 0; i < shells[k1].size(); ++i) {
        vec3<double> k1_minus = {-shells[k1][i].x, -shells[k1][i].y, -shells[k1][i].z};
        vec4<size_t> index_plus = get_index(shells[k1][i], k_f, N);
        vec4<size_t> index_minus = get_index(k1_minus, k_f, N);
        if (l == 0) {
            SN1 += real_part(index_plus, index_minus, N, A_0, Fw_0);
            N_1++;
        } else if (l == 2) {
            SN1 += real_part(index_plus, index_minus, N, A_2, Fw_0);
            N_1++;
        }
    }
    SN1 *= (2*l + 1)/N_1;
    
    for (size_t i = 0; i < shells[k2].size(); ++i) {
        vec3<double> k2_minus = {-shells[k2][i].x, -shells[k2][i].y, -shells[k2][i].z};
        vec4<size_t> index_plus = get_index(shells[k2][i], k_f, N);
        vec4<size_t> index_minus = get_index(k2_minus, k_f, N);
        if (l == 0) {
            SN2 += real_part(index_plus, index_minus, N, A_0, Fw_0);
            N_2++;
        } else if (l == 2) {
            SN2 += real_part(index_plus, index_minus, N, A_0, Fw_2);
            N_2++;
        }
    }
    SN2 *= (2*l + 1)*L_2(get_mu(k_1, k_2, k_3))/N_2;
    
    for (size_t i = 0; i < shells[k3].size(); ++i) {
        vec3<double> k3_minus = {-shells[k3][i].x, -shells[k3][i].y, -shells[k3][i].z};
        vec4<size_t> index_plus = get_index(shells[k3][i], k_f, N);
        vec4<size_t> index_minus = get_index(k3_minus, k_f, N);
        if (l == 0) {
            SN3 += real_part(index_plus, index_minus, N, A_0, Fw_0);
            N_3++;
        } else if (l == 2) {
            SN3 += real_part(index_plus, index_minus, N, A_0, Fw_2);
            N_3++;
        }
    }
    SN3 *= (2*l + 1)*L_2(get_mu(k_1, k_3, k_2))/N_3;
    
    double shotNoise = 0.0;
    if (l == 0) {
        shotNoise = (SN1 + SN2 + SN3 - 2.0*(gal_bk_nbw.x - alpha*alpha*alpha*ran_bk_nbw.x))/gal_bk_nbw.z;
    } else if (l == 2) {
        shotNoise = SN1 + SN2 + SN3;
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
// stored in memory. This function the simply carries out the pruduct, the summing, the normalization and the 
// shot noise subtraction.
void get_bispectrum(std::vector<double> &ks, std::vector<double> &P, vec3<double> gal_bk_nbw,
                    vec3<double> ran_bk_nbw, vec3<int> N, vec3<double> L, double alpha, 
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<std::vector<double>> &shells, double delta_k, double k_min, double k_max, 
                    bool exactTriangles) {
    vec3<double> kt;
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
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<std::vector<double>> &shells, double delta_k, double k_min, double k_max, std::vector<double> &SN, bool exactTriangles) {
    vec3<double> kt;
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
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
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
    vec3<double> kt;
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
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
                    std::vector<std::vector<double>> &A0_shells, std::vector<std::vector<double>> &A2_shells,
                    double delta_k, double k_min, double k_max, std::vector<double> &SN, bool exactTriangles) {
    vec3<double> kt;
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
                    std::vector<double> &B, std::vector<vec3<double>> &k_trip, 
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
    vec3<double> kt;
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
