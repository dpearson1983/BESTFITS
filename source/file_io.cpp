#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <CCfits/CCfits>
#include <gsl/gsl_integration.h>
#include "../include/file_io.h"
#include "../include/tpods.h"
#include "../include/cic.h"
#include "../include/gadgetReader.h"
#include "../include/galaxy.h"
#include "../include/cosmology.h"

bool FileExists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

void getDR12Gals(std::string file, std::vector<galaxy> &gals, double z_min, double z_max) {
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
    std::vector<double> w_rf;
    std::vector<double> w_cp;
    
    table.column("RA").read(ra, start, end);
    table.column("DEC").read(dec, start, end);
    table.column("Z").read(red, start, end);
    table.column("NZ").read(nz, start, end);
    table.column("WEIGHT_FKP").read(w_fkp, start, end);
    table.column("WEIGHT_SYSTOT").read(w_sys, start, end);
    table.column("WEIGHT_NOZ").read(w_rf, start, end);
    table.column("WEIGHT_CP").read(w_cp, start, end);
    
    for (size_t i = 0; i < ra.size(); ++i) {
        if (red[i] >= z_min && red[i] < z_max) {
            galaxy gal(ra[i], dec[i], red[i], nz[i], w_sys[i]*w_fkp[i]*(w_rf[i] + w_cp[i] - 1));
            gals.push_back(gal);
        }
    }
}

void getDR12Rans(std::string file, std::vector<galaxy> &gals, double z_min, double z_max) {
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
    
    for (size_t i = 0; i < ra.size(); ++i) {
        if (red[i] >= z_min && red[i] < z_max) {
            galaxy gal(ra[i], dec[i], red[i], nz[i], w_fkp[i]);
            gals.push_back(gal);
        }
    }
}

vec3<double> getRMin(std::vector<galaxy> &gals, cosmology &cosmo, vec3<int> N, vec3<double> &L) {
    vec3<double> r_min = {1E17, 1E17, 1E17};
    vec3<double> r_max = {-1E17, -1E17, -1E17};
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < gals.size(); ++i) {
        vec3<double> pos = gals[i].get_unshifted_cart(cosmo, ws);
        if (pos.x < r_min.x) r_min.x = pos.x;
        if (pos.y < r_min.y) r_min.y = pos.y;
        if (pos.z < r_min.z) r_min.z = pos.z;
        if (pos.x > r_max.x) r_max.x = pos.x;
        if (pos.y > r_max.y) r_max.y = pos.y;
        if (pos.z > r_max.z) r_max.z = pos.z;
    }
    gsl_integration_workspace_free(ws);
    
    std::ofstream fout("grid_properties.log");
    
    L.x = r_max.x - r_min.x;
    L.y = r_max.y - r_min.y;
    L.z = r_max.z - r_min.z;
    
    fout << "Minimum box dimensions:\n";
    fout << "   L.x = " << L.x << "\n";
    fout << "   L.y = " << L.y << "\n";
    fout << "   L.z = " << L.z << "\n";
    
    double resx = L.x/N.x;
    double resy = L.y/N.y;
    double resz = L.z/N.z;
    
    fout << "\nRaw resolutions:\n";
    fout << " res.x = " << resx << "\n";
    fout << " res.y = " << resy << "\n";
    fout << " res.z = " << resz << "\n";
    
    double resolution = resx;
    if (resy > resolution) resolution = resy;
    if (resz > resolution) resolution = resz;
    
    resolution = floor(resolution*2 + 0.5)/2;
    
    fout << "\nAdopted resolution:\n";
    fout << "   resolution = " << resolution << "\n";
    
    // Centering the sample
    r_min.x -= (resolution*N.x - L.x)/2.0;
    r_min.y -= (resolution*N.y - L.y)/2.0;
    r_min.z -= (resolution*N.z - L.z)/2.0;
    
    fout << "\nMinimum box position:\n";
    fout << "   r_min.x = " << r_min.x << "\n";
    fout << "   r_min.y = " << r_min.y << "\n";
    fout << "   r_min.z = " << r_min.z << "\n";
    
    L.x = resolution*N.x;
    L.y = resolution*N.y;
    L.z = resolution*N.z;
    
    fout << "\nFinal box dimensions:\n";
    fout << "   L.x = " << L.x << "\n";
    fout << "   L.y = " << L.y << "\n";
    fout << "   L.z = " << L.z << "\n";
    fout.close();
    
    return r_min;
}

void readDR12(std::string file, std::vector<double> &delta, std::vector<double> &delta2, vec3<int> N, 
              vec3<double> L, vec3<double> r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw, 
              double z_min, double z_max) {
    std::vector<galaxy> gals;
    getDR12Gals(file, gals, z_min, z_max);
    
    vec3<double> p_dummynbw = {0.0, 0.0, 0.0};
    vec3<double> b_dummynbw = {0.0, 0.0, 0.0};
    
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < gals.size(); ++i) {
        gals[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
        gals[i].set_weight(gals[i].get_weight()*gals[i].get_weight());
        gals[i].bin(delta2, N, L, r_min, cosmo, p_dummynbw, b_dummynbw, ws);
    }
    gsl_integration_workspace_free(ws);
}

void readDR12Ran(std::string file, std::vector<double> &delta, std::vector<double> &delta2, vec3<int> N, 
                 vec3<double> &L,  vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, 
                 vec3<double> &bk_nbw, double z_min, double z_max) {
    std::vector<galaxy> rans;
    getDR12Rans(file, rans, z_min, z_max);
    
    r_min = getRMin(rans, cosmo, N, L);
    
    vec3<double> p_dummynbw = {0.0, 0.0, 0.0};
    vec3<double> b_dummynbw = {0.0, 0.0, 0.0};
    
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < rans.size(); ++i) {
        rans[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
        rans[i].set_weight(rans[i].get_weight()*rans[i].get_weight());
        rans[i].bin(delta2, N, L, r_min, cosmo, p_dummynbw, b_dummynbw, ws);
    }
    gsl_integration_workspace_free(ws);
}

void readPatchy(std::string file, std::vector<double> &delta, std::vector<double> &delta2, vec3<int> N, 
                vec3<double> L, vec3<double> r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw,
                double z_min, double z_max) {
    std::vector<galaxy> gals;
    
    std::ifstream fin(file);
    while (!fin.eof()) {
        double ra, dec, red, mstar, nbar, bias, veto_flag, fiber_collision;
        fin >> ra >> dec >> red >> mstar >> nbar >> bias >> veto_flag >> fiber_collision;
        if (red >= z_min && red < z_max) {
            double w_fkp = (veto_flag*fiber_collision)/(1.0 + nbar*10000.0);
            galaxy gal(ra, dec, red, nbar, w_fkp);
            gals.push_back(gal);
        }
    }
    fin.close();
    
    vec3<double> p_dummynbw = {0.0, 0.0, 0.0};
    vec3<double> b_dummynbw = {0.0, 0.0, 0.0};
    
//     vec3<double> r_min = getRMin(gals, cosmo, L);
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < gals.size(); ++i) {
        gals[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
        gals[i].set_weight(gals[i].get_weight()*gals[i].get_weight());
        gals[i].bin(delta2, N, L, r_min, cosmo, p_dummynbw, b_dummynbw, ws);
    }
    gsl_integration_workspace_free(ws);
}

void readPatchyRan(std::string file, std::vector<double> &delta, std::vector<double> &delta2, vec3<int> N, 
                   vec3<double> &L, vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, 
                   vec3<double> &bk_nbw, double z_min, double z_max) {
    std::vector<galaxy> rans;
    
    std::ifstream fin(file);
    while (!fin.eof()) {
        double ra, dec, red, nbar, bias, veto_flag, fiber_collision;
        fin >> ra >> dec >> red >> nbar >> bias >> veto_flag >> fiber_collision;
        if (red >= z_min && red < z_max) {
            double w_fkp = (veto_flag*fiber_collision)/(1.0 + nbar*10000.0);
            galaxy ran(ra, dec, red, nbar, w_fkp);
            rans.push_back(ran);
        }
    }
    fin.close();
    
    r_min = getRMin(rans, cosmo, N, L);
    
    vec3<double> p_dummynbw = {0.0, 0.0, 0.0};
    vec3<double> b_dummynbw = {0.0, 0.0, 0.0};
    
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(10000000);
    for (size_t i = 0; i < rans.size(); ++i) {
        rans[i].bin(delta, N, L, r_min, cosmo, pk_nbw, bk_nbw, ws);
        rans[i].set_weight(rans[i].get_weight()*rans[i].get_weight());
        rans[i].bin(delta2, N, L, r_min, cosmo, p_dummynbw, b_dummynbw, ws);
    }
    gsl_integration_workspace_free(ws);
}

// TODO: Setup implementation to read in a sum or weights squared density field.
void readDensityField(std::string file, std::vector<double> &delta, std::vector<double> &delta2, vec3<int> N, 
                      vec3<double> &L, vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, 
                      vec3<double> &bk_nbw, double z_min, double z_max) {
    if (delta.size() != N.x*N.y*N.z) {
        delta.resize(N.x*N.y*N.z);
    }
    std::ifstream fin(file, std::ios::in|std::ios::binary);
    fin.read((char *) &L, 3*sizeof(double));
    fin.read((char *) &r_min, 3*sizeof(double));
    fin.read((char *) &pk_nbw, 3*sizeof(double));
    fin.read((char *) &bk_nbw, 3*sizeof(double));
    fin.read((char *) delta.data(), N.x*N.y*N.z*sizeof(double));
    fin.close();
}

void readGadget2(std::string file, std::vector<double> &delta, std::vector<double> &delta2, vec3<int> N, vec3<double> L,
                 vec3<double> r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw, double z_min,
                 double z_max) {
    gadget_header header;
    read_gadget_snapshot(file, header, delta, delta2, N, L, pk_nbw, bk_nbw);
}

void readGadget2_ran(std::string file, std::vector<double> &delta, std::vector<double> &delta2, vec3<int> N, 
                     vec3<double> &L, vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw,
                     double z_min, double z_max) {
    gadget_header header;
    read_gadget_header(file, header);
    
    L.x = header.boxSize/1000.0;
    L.y = header.boxSize/1000.0;
    L.z = header.boxSize/1000.0;
    
    r_min.x = 0.0;
    r_min.y = 0.0;
    r_min.z = 0.0;
    
    std::cout << "(" << N.x << ", " << N.y << ", " << N.z << ")" << std::endl;
    
    double dV = (L.x*L.y*L.z)/(N.x*N.y*N.z);
    
    unsigned int totalParticles = 0;
    for (int i = 0; i < 6; ++i) 
        totalParticles += header.N_tot[i];
    
    double nbar = totalParticles/(L.x*L.y*L.z);
    double n_avg = nbar*dV;
    
    for (size_t i = 0; i < delta.size(); ++i) {
        delta[i] = n_avg;
        delta2[i] = n_avg;
        
        pk_nbw.x += n_avg;
        pk_nbw.y += n_avg;
        pk_nbw.z += nbar*n_avg;
        
        bk_nbw.x += n_avg;
        bk_nbw.y += nbar*n_avg;
        bk_nbw.z += nbar*nbar*n_avg;
    }
}

void readLNKNLog(std::string file, std::vector<double> &delta, std::vector<double> &delta2, vec3<int> N, vec3<double> L,
                 vec3<double> r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw, double z_min,
                 double z_max) {
    std::ifstream fin(file);
    while(!fin.eof()) {
        vec3<double> pos;
        double nbar, b, w;
        fin >> pos.x >> pos.y >> pos.z >> nbar >> b >> w;
        if (!fin.eof()) {
            pk_nbw.x += w;
            pk_nbw.y += w*w;
            pk_nbw.z += nbar*w*w;
            
            bk_nbw.x += w*w*w;
            bk_nbw.y += nbar*w*w*w;
            bk_nbw.z += nbar*nbar*w*w*w;
            
            std::vector<double> weights;
            std::vector<size_t> indices;
            
            getCICInfo(pos, N, L, indices, weights);
            
            for (int i = 0; i < indices.size(); ++i) {
                delta[indices[i]] += weights[i]*w;
                delta2[indices[i]] += weights[i]*w*w;
            }
        }
    }
    fin.close();
}

void readLNKNLog_ran(std::string file, std::vector<double> &delta, std::vector<double> &delta2, vec3<int> N, 
                     vec3<double> &L, vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw, 
                     double z_min, double z_max) {
    vec3<double> r_max = {0.0, 0.0, 0.0};
    
    r_min.x = 0.0;
    r_min.y = 0.0;
    r_min.z = 0.0;
    
    std::ifstream fin(file);
    fin >> L.x >> L.y >> L.z;
    while(!fin.eof()) {
        vec3<double> pos;
        double nbar, b, w;
        fin >> pos.x >> pos.y >> pos.z >> nbar >> b >> w;
        if (!fin.eof()) {
            if (pos.x > r_max.x) r_max.x = pos.x;
            if (pos.y > r_max.y) r_max.y = pos.y;
            if (pos.z > r_max.z) r_max.z = pos.z;
            
            pk_nbw.x += w;
            pk_nbw.y += w*w;
            pk_nbw.z += nbar*w*w;
            
            bk_nbw.x += w*w*w;
            bk_nbw.y += nbar*w*w*w;
            bk_nbw.z += nbar*nbar*w*w*w;
            
            std::vector<double> weights;
            std::vector<size_t> indices;
            
            getCICInfo(pos, N, L, indices, weights);
            
            for (int i = 0; i < indices.size(); ++i) {
                delta[indices[i]] += weights[i]*w;
                delta2[indices[i]] += weights[i]*w*w;
            }
        }
    }
    fin.close();
    
    L.x = r_max.x;
    L.y = r_max.y;
    L.z = r_max.z;
}

void setFileType(std::string typeString, FileType &type) {
    std::cout << "Setting file type " << typeString << std::endl;
    if (typeString == "DR12") {
        type = dr12;
    } else if (typeString == "patchy") {
        type = patchy;
    } else if (typeString == "DR12_ran") {
        type = dr12_ran;
    } else if (typeString == "patchy_ran") {
        type = patchy_ran;
    } else if (typeString == "density_field") {
        type = density_field;
    } else if (typeString == "gadget2") {
        type = gadget2;
    } else if (typeString == "gadget2_ran") {
        type = gadget2_ran;
    } else if (typeString == "lnknlog") {
        type = lnknlog;
    } else if (typeString == "lnknlog_ran") {
        type = lnknlog_ran;
    } else {
        std::stringstream err_msg;
        err_msg << "Unrecognized or unsupported file type.\n";
        throw std::runtime_error(err_msg.str());
    }
}

void readFile(std::string file, std::vector<double> &delta, std::vector<double> &delta2, vec3<int> N, 
              vec3<double> &L, vec3<double> &r_min, cosmology &cosmo, vec3<double> &pk_nbw, vec3<double> &bk_nbw,
              double z_min, double z_max, FileType type) {
    switch(type) {
        case dr12:
            std::cout << "Reading file type: DR12" << std::endl;
            readDR12(file, delta, delta2, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
            break;
        case patchy:
            std::cout << "Reading file type: patchy" << std::endl;
            readPatchy(file, delta, delta2, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
            break;
        case dr12_ran:
            std::cout << "Reading file type: DR12_ran" << std::endl;
            readDR12Ran(file, delta, delta2, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
            break;
        case patchy_ran:
            std::cout << "Reading file type: patchy_ran" << std::endl;
            readPatchyRan(file, delta, delta2, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
        case density_field:
            std::cout << "Reading file type: density_field" << std::endl;
            readDensityField(file, delta, delta2, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
            break;
        case gadget2:
            std::cout << "Reading file type: gadget2" << std::endl;
            readGadget2(file, delta, delta2, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
            break;
        case gadget2_ran:
            std::cout << "Reading file type: gadget2_ran" << std::endl;
            readGadget2_ran(file, delta, delta2, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
            break;
        case lnknlog:
            std::cout << "Reading file type: lnknlog" << std::endl;
            readLNKNLog(file, delta, delta2, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
            break;
        case lnknlog_ran:
            std::cout << "Reading file type: lnknlog_ran" << std::endl;
            readLNKNLog_ran(file, delta, delta2, N, L, r_min, cosmo, pk_nbw, bk_nbw, z_min, z_max);
            break;
        default:
            std::stringstream err_msg;
            err_msg << "Unrecognized or unsupported file type.\n";
            throw std::runtime_error(err_msg.str());
            break;
    }
}

void writeBispectrumFile(std::string file, std::vector<vec4<double>> &ks, std::vector<double> &B) {
    std::ofstream fout(file);
    fout.precision(15);
    for (size_t i = 0; i < B.size(); ++i) {
        fout << ks[i].w << " " << ks[i].x << " " << ks[i].y << " " << ks[i].z << " " << B[i] << "\n";
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
    fout.precision(15);
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

std::string triangleFilename(vec3<double> L, double k_min, double k_max) {
    std::stringstream file;
    file << "numTri-" << L.x << "x" << L.y << "x" << L.z << "-" << k_min << "-" << k_max << ".dat";
    return file.str();
}

void writeTriangleFile(std::vector<size_t> &N_tri, vec3<double> L, double k_min, double k_max) {
    std::string file = triangleFilename(L, k_min, k_max);
    std::ofstream fout(file);
    for (int i = 0; i < N_tri.size(); ++i)
        fout << N_tri[i] << "\n";
    fout.close();
}
    
void writeCovarianceFile(std::string file, std::vector<std::vector<double>> &covariance) {
    std::ofstream fout(file);
    for (int i = 0; i < covariance.size(); ++i) {
        for (int j = 0; j < covariance[i].size(); ++j) {
            fout.width(20);
            fout << covariance[i][j];
        }
        fout << "\n";
    }
}
