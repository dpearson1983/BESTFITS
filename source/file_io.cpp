#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <CCfits/CCfits>
#include <gsl/gsl_integration.h>
#include "../include/file_io.h"
#include "../include/tpods.h"
#include "../include/galaxy.h"
#include "../include/cosmology.h"

void getDR12Gals(std::string file, std::vector<galaxy> &gals) {
    std::unique_ptr<CCfits::FITS> pInfile(new CCfits::FITS(file, CCfits::Read, false));
    
    CCfits::ExtHDU &table = pInfile->extension(1);
    long start = 1L;
    long end = table.rows();
    
    std::vector<double> ra;
    std::vector<double> dec;
    std::vector<double> red;
    std::vector<double> nz;
    std::vector<double> w_fkp;
    std::vector<double> w_sys;
    
    table.column("RA").read(ra, start, end);
    table.column("DEC").read(dec, start, end);
    table.column("Z").read(red, start, end);
    table.column("NZ").read(nz, start, end);
    table.column("WEIGHT_FKP").read(w_fkp, start, end);
    table.column("WEIGHT_SYSTOT").read(w_sys, start, end);
    
    for (size_t i = 0; i < ra.size(); ++i) {
        galaxy gal(ra[i], dec[i], red[i], nz[i], w_sys[i]*w_fkp[i]);
        gals.push_back(gal);
    }
}

void getDR12Rans(std::string file, std::vector<galaxy> &gals) {
//     std::cout << "      Reading randoms from a DR12 fits file..." << std::endl;
    std::unique_ptr<CCfits::FITS> pInfile(new CCfits::FITS(file, CCfits::Read, false));
    
    CCfits::ExtHDU &table = pInfile->extension(1);
    long start = 1L;
    long end = table.rows();
    
    std::vector<double> ra;
    std::vector<double> dec;
    std::vector<double> red;
    std::vector<double> nz;
    std::vector<double> w_fkp;
    
    table.column("RA").read(ra, start, end);
    table.column("DEC").read(dec, start, end);
    table.column("Z").read(red, start, end);
    table.column("NZ").read(nz, start, end);
    table.column("WEIGHT_FKP").read(w_fkp, start, end);
    
//     std::cout << "      Number of randoms = " << ra.size() << std::endl;
    
//     std::cout << "      Adding randoms to a vector..." << std::endl;
    for (size_t i = 0; i < ra.size(); ++i) {
//         std::cout << "\r" << i;
        galaxy gal(ra[i], dec[i], red[i], nz[i], w_fkp[i]);
        gals.push_back(gal);
    }
//     std::cout << "\n" << "      Number of randoms = " << gals.size() << std::endl;
}

vec3<double> getRMin(std::vector<galaxy> &gals, cosmology &cosmo, vec3<double> &L) {
    vec3<double> r_min = {1E17, 1E17, 1E17};
    vec3<double> r_max = {-1E17, -1E17, -1E17};
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
//     std::cout << "      Finding cartesian coordinates..." << std::endl;
    for (size_t i = 0; i < gals.size(); ++i) {
//         std::cout << "\r" << i;
        vec3<double> pos = gals[i].get_unshifted_cart(cosmo, ws);
        if (pos.x < r_min.x) r_min.x = pos.x;
        if (pos.y < r_min.y) r_min.y = pos.y;
        if (pos.z < r_min.z) r_min.z = pos.z;
        if (pos.x > r_max.x) r_max.x = pos.x;
        if (pos.y > r_max.y) r_max.y = pos.y;
        if (pos.z > r_max.z) r_max.z = pos.z;
    }
    gsl_integration_workspace_free(ws);
    L.x = r_max.x - r_min.x;
    L.y = r_max.y - r_min.y;
    L.z = r_max.z - r_min.z;
    
    double length = L.x;
    if (L.y > length) length = L.y;
    if (L.z > length) length = L.z;
    
    int num = int(pow(2,int(log2(length/3.5)) + 1));
    length = num*3.5;
    
    r_min.x -= (length - L.x)/2.0;
    r_min.y -= (length - L.y)/2.0;
    r_min.z -= (length - L.z)/2.0;
    
    L.x = L.y = L.z = length;
    
    return r_min;
}

void readDR12(std::string file, std::vector<double> &delta, vec3<int> N, vec3<double> L, 
              vec3<double> r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw) {
    std::vector<galaxy> gals;
    getDR12Gals(file, gals);
    
//     vec3<double> r_min = getRMin(gals, cosmo, L);
    
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < gals.size(); ++i) {
        gals[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
    }
    gsl_integration_workspace_free(ws);
}

void readDR12Ran(std::string file, std::vector<double> &delta, vec3<int> N, vec3<double> &L, 
              vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw) {
    std::vector<galaxy> rans;
    getDR12Rans(file, rans);
    
    r_min = getRMin(rans, cosmo, L);
    
//     std::cout << "      Binning the randoms..." << std::endl;
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < rans.size(); ++i) {
//         std::cout << "\r" << i;
        rans[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
    }
    gsl_integration_workspace_free(ws);
    std::cout << std::endl;
}

void readPatchy(std::string file, std::vector<double> &delta, vec3<int> N, vec3<double> L, 
                vec3<double> r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw) {
    std::vector<galaxy> gals;
    
    std::ifstream fin(file);
    while (!fin.eof()) {
        double ra, dec, red, mstar, nbar, bias, veto_flag, fiber_collision;
        fin >> ra >> dec >> red >> mstar >> nbar >> bias >> veto_flag >> fiber_collision;
        double w_fkp = (veto_flag*fiber_collision)/(1.0 + nbar*10000.0);
        galaxy gal(ra, dec, red, nbar, w_fkp);
        gals.push_back(gal);
    }
    
//     vec3<double> r_min = getRMin(gals, cosmo, L);
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < gals.size(); ++i) {
        gals[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
    }
    gsl_integration_workspace_free(ws);
}

void readPatchyRan(std::string file, std::vector<double> &delta, vec3<int> N, vec3<double> &L, 
                vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw) {
    std::vector<galaxy> rans;
    
    std::ifstream fin(file);
    while (!fin.eof()) {
        double ra, dec, red, nbar, bias, veto_flag, fiber_collision;
        fin >> ra >> dec >> red >> nbar >> bias >> veto_flag >> fiber_collision;
        double w_fkp = (veto_flag*fiber_collision)/(1.0 + nbar*10000.0);
        galaxy ran(ra, dec, red, nbar, w_fkp);
        rans.push_back(ran);
    }
    
    r_min = getRMin(rans, cosmo, L);
    
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < rans.size(); ++i) {
        rans[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
    }
    gsl_integration_workspace_free(ws);
}

void setFileType(std::string typeString, FileType &type) {
    if (typeString == "DR12") {
        type = dr12;
    } else if (typeString == "patchy") {
        type = patchy;
    } else if (typeString == "DR12_ran") {
        type = dr12_ran;
    } else if (typeString == "patchy_ran") {
        type = patchy_ran;
    } else {
        std::stringstream err_msg;
        err_msg << "Unrecognized or unsupported file type.\n";
        throw std::runtime_error(err_msg.str());
    }
}

void readFile(std::string file, std::vector<double> &delta, vec3<int> N, vec3<double> &L, 
              vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw,
              FileType type) {
    switch(type) {
        case dr12:
            readDR12(file, delta, N, L, r_min, cosmo, pk_nbw, bk_nbw);
            break;
        case patchy:
            readPatchy(file, delta, N, L, r_min, cosmo, pk_nbw, bk_nbw);
            break;
        case dr12_ran:
            readDR12Ran(file, delta, N, L, r_min, cosmo, pk_nbw, bk_nbw);
            break;
        case patchy_ran:
            readPatchyRan(file, delta, N, L, r_min, cosmo, pk_nbw, bk_nbw);
        default:
            std::stringstream err_msg;
            err_msg << "Unrecognized or unsupported file type.\n";
            throw std::runtime_error(err_msg.str());
            break;
    }
}

void writeBispectrumFile(std::string file, std::vector<vec3<double>> &ks, std::vector<double> &B) {
    std::ofstream fout(file);
    for (size_t i = 0; i < B.size(); ++i) {
        fout << ks[i].x << " " << ks[i].y << " " << ks[i].z << " " << B[i] << "\n";
    }
    fout.close();
}

void writeShellFile(std::string file, std::vector<double> &shell, vec3<int> N) {
    size_t N_tot = N.x*N.y*N.z;
    std::ofstream fout(file, std::ios::out|std::ios::binary);
    for (int i = 0; i < N.x; ++i) {
        for (int j = 0; j < N.y; ++j) {
            for (int k = 0; k < N.z; ++k) {
                int index = k + 2*(N.z/2 + 1)*(j + N.y*i);
                shell[index] /= N_tot;
                fout.write((char *) &shell[index], sizeof(double));
            }
        }
    }
    fout.close();
}

void writePowerSpectrumFile(std::string file, std::vector<double> &ks, std::vector<double> &P) {
    std::ofstream fout(file);
    for (int i = 0; i < ks.size(); ++i) {
        fout << ks[i] << " " << P[i] << "\n";
    }
    fout.close();
}

std::string filename(std::string base, int digits, int num, std::string ext) {
    std::stringstream file;
    file << base << std::setw(digits) << std::setfill('0') << num << "." << ext;
    return file.str();
}
