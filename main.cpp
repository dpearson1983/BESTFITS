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

int main(int argc, char *argv[]) {
    std::cout << "Initializing..." << std::endl;
    parameters p(argv[1]);
    p.print();
    
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
        
        std::cout << "   Getting randoms..." << std::endl;
        readFile(p.gets("randomsFile"), ran, N, L, r_min, cosmo, ran_pk_nbw, ran_bk_nbw, 
                 p.getd("z_min"), p.getd("z_max"), ranFileType);
        std::cout << "   Getting galaxies..." << std::endl;
        readFile(p.gets("dataFile"), gal, N, L, r_min, cosmo, gal_pk_nbw, gal_bk_nbw, p.getd("z_min"),
                 p.getd("z_max"), dataFileType);
        
        alpha = gal_pk_nbw.x/ran_pk_nbw.x;
        
        std::cout << "   Computing overdensity..." << std::endl;
        #pragma omp parallel for
        for (size_t i = 0; i < delta.size(); ++i) {
            delta[i] = gal[i] - alpha*ran[i];
            Fw[i] = (gal_pk_nbw.y + alpha*alpha*ran_pk_nbw.y)/gal_pk_nbw.z;
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
    
    std::cout << "A_0[0] = " << A_0[0] << std::endl;
    std::cout << "A_2[0] = " << A_2[0] << std::endl;
    
    std::cout << "Fw_0[0] = " << Fw_0[0] << std::endl;
    std::cout << "Fw_2[0] = " << Fw_2[0] << std::endl;
    
    std::cout << "Fw_0[1] = " << Fw_0[1] << std::endl;
    std::cout << "Fw_2[1] = " << Fw_2[1] << std::endl;
    
    std::cout << "Fourier transforming overdensity field..." << std::endl;
    get_A0(delta, A_0, N);
    get_A2(delta, A_2, N, L, r_min);
    get_A0(Fw, Fw_0, N);
    get_A2(Fw, Fw_2, N, L, r_min);
    
    std::cout << "A_0[0] = " << A_0[0] << std::endl;
    std::cout << "A_2[0] = " << A_2[0] << std::endl;
    
    std::cout << "Fw_0[0] = " << Fw_0[0] << std::endl;
    std::cout << "Fw_2[0] = " << Fw_2[0] << std::endl;
    
    std::cout << "Fw_0[1] = " << Fw_0[1] << std::endl;
    std::cout << "Fw_2[1] = " << Fw_2[1] << std::endl;
    
    if (dataFileType != density_field) {
        std::cout << "Applying correction for CIC binning..." << std::endl;
        CICbinningCorrection((fftw_complex *) A_0.data(), N, L, kx, ky, kz);
        CICbinningCorrection((fftw_complex *) A_2.data(), N, L, kx, ky, kz);
        CICbinningCorrection((fftw_complex *) Fw_0.data(), N, L, kx, ky, kz);
        CICbinningCorrection((fftw_complex *) Fw_2.data(), N, L, kx, ky, kz);
    }
    
    std::cout << "A_0[0] = " << A_0[0] << std::endl;
    std::cout << "A_2[0] = " << A_2[0] << std::endl;
    
    std::cout << "Fw_0[0] = " << Fw_0[0] << std::endl;
    std::cout << "Fw_2[0] = " << Fw_2[0] << std::endl;
    
    std::cout << "Fw_0[1] = " << Fw_0[1] << std::endl;
    std::cout << "Fw_2[1] = " << Fw_2[1] << std::endl;
    
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
    
    std::cout << gal_bk_nbw.z << std::endl;
    
    std::cout << "Computing the bispectrum monopole..." << std::endl;
    double start = omp_get_wtime();
    std::vector<double> B;
    std::vector<vec3<double>> k_trip;
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
                                                                     gal_bk_nbw, ran_bk_nbw, 2, k_min, delta_k,
                                                                     alpha);
                        SN_2[bispecBin] = shotNoise;
                        fout << k_trip[bispecBin].x << " " << k_trip[bispecBin].y << " " << k_trip[bispecBin].z;
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
    
    return 0;
}
