#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include "include/file_io.h"
#include "include/cic.h"
#include "include/cosmology.h"
#include "include/transformers.h"
#include "include/tpods.h"
#include "include/shells.h"
#include "include/harppi.h"
#include "include/power.h"
#include "include/line_of_sight.h"
#include "include/bispec.h"

enum mode{
    data,
    mock
};

int main(int argc, char *argv[]) {
    std::cout << "Initializing..." << std::endl;
    parameters p(argv[1]);
    p.print();
    
    mode runMode;
    if (p.gets("runMode") == "data") runMode = data;
    if (p.gets("runMode") == "mock") runMode = mock;
    
    if (runMode == data) {
        // Setup the cosmology class object with values needed to get comoving distances.
        // NOTE: The distances returned will be in h^-1 Mpc, so the value of H_0 is not actually used.
        cosmology cosmo(p.getd("H_0"), p.getd("Omega_M"), p.getd("Omega_L"));
        
        // Storage for values
        vec3<double> gal_pk_nbw = {0.0, 0.0, 0.0};
        vec3<double> gal_bk_nbw = {0.0, 0.0, 0.0};
        vec3<double> ran_pk_nbw = {0.0, 0.0, 0.0};
        vec3<double> ran_bk_nbw = {0.0, 0.0, 0.0};
        
        // Both r_min and L will be set automatically when reading in randoms
        vec3<double> r_min;
        vec3<double> L;
        
        vec3<int> N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
        
        std::cout << "Setting file type variables..." << std::endl;
        FileType dataFileType, ranFileType;
        setFileType(p.gets("dataFileType"), dataFileType);
        setFileType(p.gets("ranFileType"), ranFileType);
        
        std::vector<double> delta(N.x*N.y*N.z);
        std::vector<double> Fw(N.x*N.y*N.z);
        double alpha;
        
        std::cout << "Reading in data and randoms files..." << std::endl;
        // Since the N's can be large values, individual arrays for the FFTs will be quite large. Instead
        // of reusing a fixed number of arrays, by using braced enclosed sections, variables declared
        // within the braces will go out of scope, freeing the associated memory. Here, given how the
        // backend code works, there are two temporary arrays to store the galaxy field and the randoms
        // field.
        {
            std::vector<double> ran(N.x*N.y*N.z);
            std::vector<double> gal(N.x*N.y*N.z);
            std::vector<double> gal2(N.x*N.y*N.z);
            std::vector<double> ran2(N.x*N.y*N.z);
            
            std::cout << "   Getting randoms..." << std::endl;
            readFile(p.gets("randomsFile"), ran, ran2, N, L, r_min, cosmo, ran_pk_nbw, ran_bk_nbw, 
                     p.getd("z_min"), p.getd("z_max"), ranFileType);
            std::cout << "   Getting galaxies..." << std::endl;
            readFile(p.gets("dataFile"), gal, gal2, N, L, r_min, cosmo, gal_pk_nbw, gal_bk_nbw, p.getd("z_min"),
                     p.getd("z_max"), dataFileType);
            
            alpha = gal_pk_nbw.x/ran_pk_nbw.x;
            
            std::cout << "   Computing overdensity..." << std::endl;
            #pragma omp parallel for
            for (size_t i = 0; i < delta.size(); ++i) {
                delta[i] = gal[i] - alpha*ran[i];
                Fw[i] = gal2[i] + alpha*alpha*ran2[i];
            }
        }
        std::cout << "Done!" << std::endl;
        
        std::vector<double> kx = fft_freq(N.x, L.x);
        std::vector<double> ky = fft_freq(N.y, L.y);
        std::vector<double> kz = fft_freq(N.z, L.z);
        
        std::vector<double> A_0(N.x*N.y*2*(N.z/2 + 1));
        std::vector<double> A_2(N.x*N.y*2*(N.z/2 + 1));
        std::vector<double> Fw_0(N.x*N.y*2*(N.z/2 + 1));
        std::vector<double> Fw_2(N.x*N.y*2*(N.z/2 + 1));
        
        std::cout << "Fourier transforming overdensity field..." << std::endl;
        get_A0(delta, A_0, N);
        get_A2(delta, A_2, N, L, r_min);
        get_A0(Fw, Fw_0, N);
        get_A2(Fw, Fw_2, N, L, r_min);
        
        #pragma omp parallel for
        for (size_t i = 0; i < A_0.size(); ++i) {
            A_2[i] *= 1.5;
            A_2[i] -= 0.5*A_0[i];
            Fw_2[i] *= 1.5;
            Fw_2[i] -= 0.5*Fw_0[i];
        }
        
        double average = 0;
        for (size_t i = 0; i < A_2.size(); ++i) {
            if (A_2[i] == 0) {
                std::cout << "A_2[" << i << "] = " << A_2[i] << std::endl;
            }
            average += A_2[i];
        }
        std::cout << average/A_2.size() << std::endl;
        average = 0;
        for (size_t i = 0; i < Fw_0.size(); ++i) {
            if (Fw_0[i] == 0) {
                std::cout << "Fw_0[" << i << "] = " << Fw_0[i] << std::endl;
            }
            average += Fw_0[i];
        }
        std::cout << average/Fw_0.size() << std::endl;
        average = 0;
        for (size_t i = 0; i < Fw_2.size(); ++i) {
            if (Fw_2[i] == 0) {
                std::cout << "Fw_2[" << i << "] = " << Fw_2[i] << std::endl;
            }
            average += Fw_2[i];
        }
        std::cout << average/Fw_2.size() << std::endl;
        
        if (dataFileType != density_field) {
            std::cout << "Applying correction for CIC binning..." << std::endl;
            CICbinningCorrection((fftw_complex *) A_0.data(), N, L, kx, ky, kz);
            CICbinningCorrection((fftw_complex *) A_2.data(), N, L, kx, ky, kz);
            CICbinningCorrection((fftw_complex *) Fw_0.data(), N, L, kx, ky, kz);
            CICbinningCorrection((fftw_complex *) Fw_2.data(), N, L, kx, ky, kz);
        }
        
        // Frequency range 0.04 <= k <= 0.168
        double k_min = p.getd("k_min");
        double k_max = p.getd("k_max");
        double delta_k = p.getd("delta_k");
        int num_k_bins = int((k_max - k_min)/delta_k);
        
        std::vector<double> ks;
        std::vector<std::vector<vec3<double>>> shells;
        for (int i = 0; i < num_k_bins; ++i) {
            double k = k_min + (i + 0.5)*delta_k;
            ks.push_back(k);
            std::vector<vec3<double>> shell = get_shell_vecs(N, L, k, delta_k);
            shells.push_back(shell);
        }
        
        if (p.getb("exactTriangles")) {
            std::string triangleFile = triangleFilename(L, k_min, k_max);
            if (!FileExists(triangleFile)) {
                std::cout << "Computing the exact number of triangles..." << std::endl;
                generate_triangle_file(N, L, k_min, k_max, delta_k, shells);
            }
        }        
        
        std::cout << "Computing the power spectrum..." << std::endl;
        std::vector<double> P(num_k_bins);
        std::vector<int> N_k(num_k_bins);
        std::cout << alpha << std::endl;
        double PkShotNoise = gal_pk_nbw.y + alpha*alpha*ran_pk_nbw.y;
        std::cout << PkShotNoise << std::endl;
        binFrequencies((fftw_complex *)A_0.data(), P, N_k, N, kx, ky, kz, delta_k, k_min, k_max,
                       PkShotNoise);
        normalizePower(P, N_k, gal_pk_nbw.z);
        writePowerSpectrumFile(p.gets("pkFile"), ks, P);
        
        std::vector<double> ks_ext;
        for (int i = 0; i < p.geti("num_pk_bins"); ++i) {
            double k = p.getd("pk_min") + (i + 0.5)*delta_k;
            ks_ext.push_back(k);
        }
        std::vector<double> Pk(p.geti("num_pk_bins"));
        std::vector<int> Nk(p.geti("num_pk_bins"));
        binFrequencies((fftw_complex *)A_0.data(), Pk, Nk, N, kx, ky, kz, delta_k, p.getd("pk_min"), p.getd("pk_max"),
                       PkShotNoise);
        normalizePower(Pk, Nk, gal_pk_nbw.z);
        writePowerSpectrumFile(p.gets("bigPkFile"), ks_ext, Pk);
        
        std::cout << gal_bk_nbw.z << std::endl;
        
        std::cout << "Computing the bispectrum monopole..." << std::endl;
        double start = omp_get_wtime();
        std::vector<double> B;
        std::vector<vec4<double>> k_trip;
        if (p.getb("lowMemoryMode")) {        
            get_bispectrum(ks, P, gal_bk_nbw, ran_bk_nbw, N, L, alpha, B, k_trip, A_0, kx, ky, kz, k_min, k_max,
                           delta_k, p.gets("wisdomFile"), p.getb("exactTriangles"));
        } else {
            std::vector<std::vector<double>> A0_shells;
            get_shells(A0_shells, A_0, kx, ky, kz, k_min, k_max, delta_k, N, p.gets("wisdomFile"));
            get_bispectrum(ks, P, gal_bk_nbw, ran_bk_nbw, N, L, alpha, B, k_trip, A0_shells, delta_k, k_min, k_max, 
                           p.getb("exactTriangles"));
        }
        std::cout << "Time to calculate bispectrum monopole: " << omp_get_wtime() - start << " s" << std::endl;
        
        if (p.getb("calcQuadrupole")) {
            std::cout << "Computing the bispectrum quadrupole..." << std::endl;
            start = omp_get_wtime();
            std::vector<double> SN_0(691);
            std::vector<double> SN_2(691);
            
            double SN_time = omp_get_wtime();
            int bispecBin = 0;
            std::cout << "Calculating quadrupole shot noise..." << std::endl;
            std::ofstream fout("QuadShot.dat");
            fout.precision(15);
            for (int i = 0; i < ks.size(); ++i) {
                for (int j = i; j < ks.size(); ++j) {
                    for (int k = j; k < ks.size(); ++k) {
                        if (ks[k] <= ks[i] + ks[j]) {
                            double shotNoise = get_bispectrum_shot_noise(i, j, k, (fftw_complex *)A_0.data(), 
                                                                         (fftw_complex *)A_2.data(), 
                                                                         (fftw_complex *)Fw_0.data(),
                                                                         (fftw_complex *)Fw_2.data(), shells, N, L, 
                                                                         gal_bk_nbw, ran_bk_nbw, 0, k_min, delta_k,
                                                                         alpha);
                            SN_0[bispecBin] = shotNoise;
                            fout << k_trip[bispecBin].x << " " << k_trip[bispecBin].y << " " << k_trip[bispecBin].z;
                            fout << " " << shotNoise << " " << get_bispectrum_shot_noise(P[i], P[j], P[k], gal_bk_nbw, 
                                                                                         ran_bk_nbw, alpha);
                            shotNoise = get_bispectrum_shot_noise(i, j, k, (fftw_complex *)A_0.data(), 
                                                                  (fftw_complex *)A_2.data(), 
                                                                  (fftw_complex *)Fw_0.data(),
                                                                  (fftw_complex *)Fw_2.data(), shells, N, L, 
                                                                  gal_bk_nbw, ran_bk_nbw, 2, k_min, delta_k,
                                                                  alpha);
                            fout << " " << shotNoise << "\n";
                            bispecBin++;
                        }
                    }
                }
            }
            fout.close();
            std::cout << "Time to calculate quadrupole shot noise: " << omp_get_wtime() - SN_time << " s" << std::endl;
            
            if (p.getb("lowMemoryMode")) {
                get_bispectrum_quad(ks, P, gal_bk_nbw, ran_bk_nbw, N, L, alpha, B, k_trip, A_0, A_2, kx, ky, kz, k_min, 
                                    k_max, delta_k, p.gets("wisdomFile"), SN_2, p.getb("exactTriangles"));
            } else {
                std::vector<std::vector<double>> A0_shells;
                std::vector<std::vector<double>> A2_shells;
                get_shells(A0_shells, A_0, kx, ky, kz, k_min, k_max, delta_k, N, p.gets("wisdomFile"));
                get_shells(A2_shells, A_2, kx, ky, kz, k_min, k_max, delta_k, N, p.gets("wisdomFile"));
                get_bispectrum_quad(ks, P, gal_bk_nbw, ran_bk_nbw, N, L, alpha, B, k_trip, A0_shells, A2_shells,
                                    delta_k, k_min, k_max, SN_2, p.getb("exactTriangles"));
            }
            std::cout << "Time to calculate bispectrum quadrupole: " << omp_get_wtime() - start << " s" << std::endl;
        }
        writeBispectrumFile(p.gets("outFile"), k_trip, B);
        
        //     std::cout << "Getting number of bispectrum bins..." << std::endl;
        //     int bispecBins = getNumBispecBins(k_min, k_max, delta_k, k_trip);
        //     std::cout << "Setting up storage for the covariance..." << std::endl;
        //     std::vector<std::vector<double>> covariance(2*bispecBins, std::vector<double>(2*bispecBins));
        //     std::vector<std::vector<size_t>> N_tri(2*bispecBins, std::vector<size_t>(2*bispecBins));
        //     std::cout << "Getting covariance..." << std::endl;
        //     get_covariance(covariance, N_tri, shells, A_0, A_2, N, L, delta_k, k_min, k_max, bispecBins);
        //     std::cout << "Outputting covariance..." << std::endl;
        //     writeCovarianceFile(p.gets("covarianceFile"), covariance);
    } else {
        // Setup the cosmology class object with values needed to get comoving distances.
        // NOTE: The distances returned will be in h^-1 Mpc, so the value of H_0 is not actually used.
        cosmology cosmo(p.getd("H_0"), p.getd("Omega_M"), p.getd("Omega_L"));
        
        vec3<double> ran_pk_nbw = {0.0, 0.0, 0.0};
        vec3<double> ran_bk_nbw = {0.0, 0.0, 0.0};
        
        // Both r_min and L will be set automatically when reading in randoms
        vec3<double> r_min;
        vec3<double> L;
        
        vec3<int> N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
        
        std::cout << "Setting file type variables..." << std::endl;
        FileType dataFileType, ranFileType;
        setFileType(p.gets("dataFileType"), dataFileType);
        setFileType(p.gets("ranFileType"), ranFileType);
        
        std::vector<double> delta(N.x*N.y*N.z);
        std::vector<double> Fw(N.x*N.y*N.z);
        std::vector<double> ran(N.x*N.y*N.z);
        std::vector<double> ran2(N.x*N.y*N.z);
        double alpha;
        
        std::cout << "Getting randoms..." << std::endl;
        readFile(p.gets("randomsFile"), ran, ran2, N, L, r_min, cosmo, ran_pk_nbw, ran_bk_nbw, p.getd("z_min"), 
                 p.getd("z_max"), ranFileType);
        
        std::vector<double> kx = fft_freq(N.x, L.x);
        std::vector<double> ky = fft_freq(N.y, L.y);
        std::vector<double> kz = fft_freq(N.z, L.z);
        
        std::cout << "Processing mocks..." << std::endl;
        for (int mock = p.geti("start_num"); mock < p.geti("num_mocks") + p.geti("start_num"); ++mock) {
            std::string mockFile = filename(p.gets("mockBase"), p.geti("digits"), mock, p.gets("mockExt"));
            std::string outFile = filename(p.gets("outBase"), p.geti("digits"), mock, p.gets("outExt"));
            
            std::vector<double> gal(N.x*N.y*N.z);
            std::vector<double> gal2(N.x*N.y*N.z);
            
            vec3<double> gal_pk_nbw = {0.0, 0.0, 0.0};
            vec3<double> gal_bk_nbw = {0.0, 0.0, 0.0};
            
            std::cout << "    Getting mock galaxies..." << std::endl;
            readFile(mockFile, gal, gal2, N, L, r_min, cosmo, gal_pk_nbw, gal_pk_nbw, p.getd("z_min"), p.getd("z_max"),
                     dataFileType);
            
            alpha = gal_pk_nbw.x/ran_pk_nbw.x;
            
            for (size_t i = 0; i < delta.size(); ++i) {
                delta[i] = gal[i] - alpha*ran[i];
                Fw[i] = gal2[i] + alpha*alpha*ran2[i];
            }
            
            std::vector<double> A_0(N.x*N.y*2*(N.z/2 + 1));
            std::vector<double> A_2(N.x*N.y*2*(N.z/2 + 1));
            std::vector<double> Fw_0(N.x*N.y*2*(N.z/2 + 1));
            std::vector<double> Fw_2(N.x*N.y*2*(N.z/2 + 1));
            
            std::cout << "Fourier transforming overdensity field..." << std::endl;
            get_A0(delta, A_0, N);
            get_A2(delta, A_2, N, L, r_min);
            get_A0(Fw, Fw_0, N);
            get_A2(Fw, Fw_2, N, L, r_min);
            
            #pragma omp parallel for
            for (size_t i = 0; i < A_0.size(); ++i) {
                A_2[i] *= 1.5;
                A_2[i] -= 0.5*A_0[i];
                Fw_2[i] *= 1.5;
                Fw_2[i] -= 0.5*Fw_0[i];
            }
            
            if (dataFileType != density_field) {
                std::cout << "Applying correction for CIC binning..." << std::endl;
                CICbinningCorrection((fftw_complex *) A_0.data(), N, L, kx, ky, kz);
                CICbinningCorrection((fftw_complex *) A_2.data(), N, L, kx, ky, kz);
                CICbinningCorrection((fftw_complex *) Fw_0.data(), N, L, kx, ky, kz);
                CICbinningCorrection((fftw_complex *) Fw_2.data(), N, L, kx, ky, kz);
            }
            
            // Frequency range 0.04 <= k <= 0.168
            double k_min = p.getd("k_min");
            double k_max = p.getd("k_max");
            double delta_k = p.getd("delta_k");
            int num_k_bins = int((k_max - k_min)/delta_k);
            
            std::vector<double> ks;
            std::vector<std::vector<vec3<double>>> shells;
            for (int i = 0; i < num_k_bins; ++i) {
                double k = k_min + (i + 0.5)*delta_k;
                ks.push_back(k);
                std::vector<vec3<double>> shell = get_shell_vecs(N, L, k, delta_k);
                shells.push_back(shell);
            }
            
            if (p.getb("exactTriangles")) {
                std::string triangleFile = triangleFilename(L, k_min, k_max);
                if (!FileExists(triangleFile)) {
                    std::cout << "Computing the exact number of triangles..." << std::endl;
                    generate_triangle_file(N, L, k_min, k_max, delta_k, shells);
                }
            }        
            
            std::cout << "Computing the power spectrum..." << std::endl;
            std::vector<double> P(num_k_bins);
            std::vector<int> N_k(num_k_bins);
            std::cout << alpha << std::endl;
            double PkShotNoise = gal_pk_nbw.y + alpha*alpha*ran_pk_nbw.y;
            std::cout << PkShotNoise << std::endl;
            binFrequencies((fftw_complex *)A_0.data(), P, N_k, N, kx, ky, kz, delta_k, k_min, k_max,
                           PkShotNoise);
            normalizePower(P, N_k, gal_pk_nbw.z);
            writePowerSpectrumFile(p.gets("pkFile"), ks, P);
            
            std::vector<double> ks_ext;
            for (int i = 0; i < p.geti("num_pk_bins"); ++i) {
                double k = p.getd("pk_min") + (i + 0.5)*delta_k;
                ks_ext.push_back(k);
            }
            std::vector<double> Pk(p.geti("num_pk_bins"));
            std::vector<int> Nk(p.geti("num_pk_bins"));
            binFrequencies((fftw_complex *)A_0.data(), Pk, Nk, N, kx, ky, kz, delta_k, p.getd("pk_min"), p.getd("pk_max"),
                           PkShotNoise);
            normalizePower(Pk, Nk, gal_pk_nbw.z);
            writePowerSpectrumFile(p.gets("bigPkFile"), ks_ext, Pk);
            
            std::cout << gal_bk_nbw.z << std::endl;
            
            std::cout << "Computing the bispectrum monopole..." << std::endl;
            double start = omp_get_wtime();
            std::vector<double> B;
            std::vector<vec4<double>> k_trip;
            if (p.getb("lowMemoryMode")) {        
                get_bispectrum(ks, P, gal_bk_nbw, ran_bk_nbw, N, L, alpha, B, k_trip, A_0, kx, ky, kz, k_min, k_max,
                               delta_k, p.gets("wisdomFile"), p.getb("exactTriangles"));
            } else {
                std::vector<std::vector<double>> A0_shells;
                get_shells(A0_shells, A_0, kx, ky, kz, k_min, k_max, delta_k, N, p.gets("wisdomFile"));
                get_bispectrum(ks, P, gal_bk_nbw, ran_bk_nbw, N, L, alpha, B, k_trip, A0_shells, delta_k, k_min, k_max, 
                               p.getb("exactTriangles"));
            }
            std::cout << "Time to calculate bispectrum monopole: " << omp_get_wtime() - start << " s" << std::endl;
            
            if (p.getb("calcQuadrupole")) {
                std::cout << "Computing the bispectrum quadrupole..." << std::endl;
                start = omp_get_wtime();
//                 std::vector<double> SN_0(691);
                std::vector<double> SN_2(691);
//                 
//                 double SN_time = omp_get_wtime();
//                 int bispecBin = 0;
//                 std::cout << "Calculating quadrupole shot noise..." << std::endl;
//                 std::ofstream fout("QuadShot.dat");
//                 fout.precision(15);
//                 for (int i = 0; i < ks.size(); ++i) {
//                     for (int j = i; j < ks.size(); ++j) {
//                         for (int k = j; k < ks.size(); ++k) {
//                             if (ks[k] <= ks[i] + ks[j]) {
//                                 double shotNoise = get_bispectrum_shot_noise(i, j, k, (fftw_complex *)A_0.data(), 
//                                                                              (fftw_complex *)A_2.data(), 
//                                                                              (fftw_complex *)Fw_0.data(),
//                                                                              (fftw_complex *)Fw_2.data(), shells, N, L, 
//                                                                              gal_bk_nbw, ran_bk_nbw, 0, k_min, delta_k,
//                                                                              alpha);
//                                 SN_0[bispecBin] = shotNoise;
//                                 fout << k_trip[bispecBin].x << " " << k_trip[bispecBin].y << " " << k_trip[bispecBin].z;
//                                 fout << " " << shotNoise << " " << get_bispectrum_shot_noise(P[i], P[j], P[k], gal_bk_nbw, 
//                                                                                              ran_bk_nbw, alpha);
//                                 shotNoise = get_bispectrum_shot_noise(i, j, k, (fftw_complex *)A_0.data(), 
//                                                                       (fftw_complex *)A_2.data(), 
//                                                                       (fftw_complex *)Fw_0.data(),
//                                                                       (fftw_complex *)Fw_2.data(), shells, N, L, 
//                                                                       gal_bk_nbw, ran_bk_nbw, 2, k_min, delta_k,
//                                                                       alpha);
//                                 fout << " " << shotNoise << "\n";
//                                 bispecBin++;
//                             }
//                         }
//                     }
//                 }
//                 fout.close();
//                 std::cout << "Time to calculate quadrupole shot noise: " << omp_get_wtime() - SN_time << " s" << std::endl;
                
                if (p.getb("lowMemoryMode")) {
                    get_bispectrum_quad(ks, P, gal_bk_nbw, ran_bk_nbw, N, L, alpha, B, k_trip, A_0, A_2, kx, ky, kz, k_min, 
                                        k_max, delta_k, p.gets("wisdomFile"), SN_2, p.getb("exactTriangles"));
                } else {
                    std::vector<std::vector<double>> A0_shells;
                    std::vector<std::vector<double>> A2_shells;
                    get_shells(A0_shells, A_0, kx, ky, kz, k_min, k_max, delta_k, N, p.gets("wisdomFile"));
                    get_shells(A2_shells, A_2, kx, ky, kz, k_min, k_max, delta_k, N, p.gets("wisdomFile"));
                    get_bispectrum_quad(ks, P, gal_bk_nbw, ran_bk_nbw, N, L, alpha, B, k_trip, A0_shells, A2_shells,
                                        delta_k, k_min, k_max, SN_2, p.getb("exactTriangles"));
                }
                std::cout << "Time to calculate bispectrum quadrupole: " << omp_get_wtime() - start << " s" << std::endl;
            }
            writeBispectrumFile(outFile, k_trip, B);
            
            //     std::cout << "Getting number of bispectrum bins..." << std::endl;
            //     int bispecBins = getNumBispecBins(k_min, k_max, delta_k, k_trip);
            //     std::cout << "Setting up storage for the covariance..." << std::endl;
            //     std::vector<std::vector<double>> covariance(2*bispecBins, std::vector<double>(2*bispecBins));
            //     std::vector<std::vector<size_t>> N_tri(2*bispecBins, std::vector<size_t>(2*bispecBins));
            //     std::cout << "Getting covariance..." << std::endl;
            //     get_covariance(covariance, N_tri, shells, A_0, A_2, N, L, delta_k, k_min, k_max, bispecBins);
            //     std::cout << "Outputting covariance..." << std::endl;
            //     writeCovarianceFile(p.gets("covarianceFile"), covariance);
        }
    }
    return 0;
}
