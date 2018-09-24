#ifndef _FILE_IO_H_
#define _FILE_IO_H_

#include <string>
#include <vector>
#include "tpods.h"
#include "cosmology.h"

// 
enum FileType{
    unsupported,
    dr12,
    patchy,
    dr12_ran,
    patchy_ran,
    density_field
};

bool FileExists(const std::string& name);

void setFileType(std::string typeString, FileType &type);

void readFile(std::string file, std::vector<double> &delta, std::vector<double> &delta2, vec3<int> N, 
              vec3<double> &L, vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw,
              double z_min, double z_max, FileType type);

void writeBispectrumFile(std::string file, std::vector<vec4<double>> &ks, std::vector<double> &B);

void writeShellFile(std::string file, std::vector<double> &shell, vec3<int> N);

void writePowerSpectrumFile(std::string file, std::vector<double> &ks, std::vector<double> &P);

std::string filename(std::string base, int digits, int num, std::string ext);

std::string triangleFilename(vec3<double> L, double k_min, double k_max);

void writeTriangleFile(std::vector<size_t> &N_tri, vec3<double> L, double k_min, double k_max);

void writeCovarianceFile(std::string file, std::vector<std::vector<double>> &covariance);

#endif
