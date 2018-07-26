#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <omp.h>
#include "include/file_io.h"
#include "include/cic.h"
#include "include/cosmology.h"
#include "include/transformers.h"
#include "include/tpods.h"
#include "include/shells.h"
#include "include/harppi.h"
#include "include/power.h"

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
    
    std::vector<double> delta(N.x*N.y*2.0*(N.z/2 + 1));
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
        
        readFile(p.gets("randomsFile"), ran, N, L, r_min, cosmo, ran_pk_nbw, ran_bk_nbw, ranFileType);
        readFile(p.gets("dataFile"), gal, N, L, r_min, cosmo, gal_pk_nbw, gal_bk_nbw, dataFileType);
        
        alpha = gal_pk_nbw.x/ran_pk_nbw.x;
        
        #pragma omp parallel for
        for (size_t i = 0; i < N.x; ++i) {
            for (size_t j = 0; j < N.y; ++j) {
                for (size_t k = 0; k < N.z; ++k) {
                    int index1 = k + N.z*(j + N.y*i);
                    int index2 = k + 2*(N.z/2 + 1)*(j + N.y*i);
                    
                    delta[index2] = gal[index1] - alpha*ran[index1];
                }
            }
        }
    }
    std::cout << "Done!" << std::endl;
    
    std::vector<double> kx = fft_freq(N.x, L.x);
    std::vector<double> ky = fft_freq(N.y, L.y);
    std::vector<double> kz = fft_freq(N.z, L.z);
    
    std::cout << "Fourier transforming overdensity field..." << std::endl;
    fip_r2c(delta, N, p.gets("wisdomFile"), omp_get_max_threads());
    std::cout << "Applying correction for CIC binning..." << std::endl;
    CICbinningCorrection((fftw_complex *) delta.data(), N, L, kx, ky, kz);
    
    double V_f = get_V_f(L);
    
    std::vector<double> shell(N.x*N.y*2*(N.z/2 + 1));
    
    // Frequency range 0.04 <= k <= 0.168
    double k_min = p.getd("k_min");
    double k_max = p.getd("k_max");
    double delta_k = p.getd("delta_k");
    int num_k_bins = int((k_max - k_min)/delta_k);
    
    std::cout << "Computing the power spectrum..." << std::endl;
    std::vector<double> P(num_k_bins);
    std::vector<int> N_k(num_k_bins);
    double PkShotNoise = gal_pk_nbw.y + alpha*ran_bk_nbw.y;
    binFrequencies((fftw_complex *)delta.data(), P, N_k, N, kx, ky, kz, delta_k, k_min, k_max,
                   PkShotNoise);
    
    std::cout << "Selecting frequency shells, inverse transforming, and outputting to files..." << std::endl;
    std::vector<double> ks;
    for (int i = 0; i < num_k_bins; ++i) {
        double k = k_min + (i + 0.5)*delta_k;
        get_shell((fftw_complex *)shell.data(), (fftw_complex *)delta.data(), kx, ky, kz, k, delta_k,
                  N);
        bip_c2r(shell, N, p.gets("wisdomFile"), omp_get_max_threads());
        std::string shellFile = filename(p.gets("shellBase"), 2, i, p.gets("shellExt"));
        writeShellFile(shellFile, shell, N);
        ks.push_back(k);
    }
    
    std::cout << "Computing the bispectrum monopole..." << std::endl;
    std::vector<double> B;
    std::vector<vec3<double>> k_trip;
    get_bispectrum(ks, P, gal_bk_nbw, ran_bk_nbw, N, alpha, B, k_trip, p.gets("shellBase"), 
                   p.gets("shellExt"));
    writeBispectrumFile(p.gets("outFile"), k_trip, B);
    
    return 0;
}
