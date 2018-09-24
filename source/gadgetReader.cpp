#include "../include/gadgetReader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "../include/tpods.h"
#include "../include/cic.h"

std::vector<std::string> part_names = {"Gas", "Halo", "Disk", "Bulge", "Star", "Boundary"};

void read_gadget_header(std::string snapshot, gadget_header &header) {
    std::ifstream fin(snapshot, std::ios::in|std::ios::binary);
    int blockSize;
    
    // Read all the crap in the header of the file
    fin.read((char *) &blockSize, sizeof(int));
    fin.read((char *) &header, sizeof(gadget_header));
    
    fin.close();
    print_gadget_header(header);
}

void read_gadget_snapshot(std::string snapshot, gadget_header &header, 
                          std::vector<std::vector<vec3<float>>> &pos,
                          std::vector<std::vector<vec3<float>>> &vel) {
    std::ifstream fin(snapshot, std::ios::in|std::ios::binary);
    
    int blockSize;
    
    // Read all the crap in the header of the file
    fin.read((char *) &blockSize, sizeof(int));
    fin.read((char *) &header, sizeof(gadget_header));
    fin.read((char *) &blockSize, sizeof(int));
    
    // Get the size of the position block
    int posBlockSize;
    fin.read((char *) &posBlockSize, sizeof(int));
    
    int theoPosBlockSize = 0;
    for (int i = 0; i < 6; ++i)
        theoPosBlockSize += header.N_p[i]*sizeof(vec3<float>);
    
    if (posBlockSize != theoPosBlockSize) {
        print_gadget_header(header);
        std::stringstream errMessage;
        errMessage << "The reported size of the position block does not appear to be correct.";
        throw std::runtime_error(errMessage.str());
    }
    
    // Read in particle positions
    for (int i = 0; i < 6; ++i) {
        if (header.N_p[i] > 0) {
            for (unsigned int j = 0; j < header.N_p[i]; ++j) {
                vec3<float> part;
                fin.read((char *) &part, sizeof(vec3<float>));
                pos[i].push_back(part);
            }
        }
    }
    
    // Verify end of position block
    fin.read((char *) &blockSize, sizeof(int));
    
    // Get the size of the velocity block
    int velBlockSize;
    fin.read((char *) &velBlockSize, sizeof(int));
    
    int theoVelBlockSize = 0;
    for (int i = 0; i < 6; ++i)
        theoVelBlockSize += header.N_p[i]*sizeof(vec3<float>);
    
    if (velBlockSize != theoVelBlockSize) {
        print_gadget_header(header);
        std::stringstream errMessage;
        errMessage << "The reported size of the velocity block does not appear to be correct.";
        throw std::runtime_error(errMessage.str());
    }
    
    // Read in the particle velocities
    for (int i = 0; i < 6; ++i) {
        if (header.N_p[i] > 0) {
            for (unsigned int j = 0; j < header.N_p[i]; ++j) {
                vec3<float> part;
                fin.read((char *) &part, sizeof(vec3<float>));
                vel[i].push_back(part);
            }
        }
    }
    
    fin.close();
    print_gadget_header(header);
}

void read_gadget_snapshot(std::string snapshot, gadget_header &header, 
                          std::vector<std::vector<vec3<float>>> &pos,
                          std::vector<std::vector<vec3<float>>> &vel, std::vector<std::vector<int>> &ids) {
    std::ifstream fin(snapshot, std::ios::in|std::ios::binary);
    
    int blockSize;
    
    // Read all the crap in the header of the file
    fin.read((char *) &blockSize, sizeof(int));
    fin.read((char *) &header, sizeof(gadget_header));
    fin.read((char *) &blockSize, sizeof(int));
    
    // Get the size of the position block
    int posBlockSize;
    fin.read((char *) &posBlockSize, sizeof(int));
    
    int theoPosBlockSize = 0;
    for (int i = 0; i < 6; ++i)
        theoPosBlockSize += header.N_p[i]*sizeof(vec3<float>);
    
    if (posBlockSize != theoPosBlockSize) {
        print_gadget_header(header);
        std::stringstream errMessage;
        errMessage << "The reported size of the position block does not appear to be correct.";
        throw std::runtime_error(errMessage.str());
    }
    
    // Read in particle positions
    for (int i = 0; i < 6; ++i) {
        if (header.N_p[i] > 0) {
            for (unsigned int j = 0; j < header.N_p[i]; ++j) {
                vec3<float> part;
                fin.read((char *) &part, sizeof(vec3<float>));
                pos[i].push_back(part);
            }
        }
    }
    
    // Verify end of position block
    fin.read((char *) &blockSize, sizeof(int));
    
    // Get the size of the velocity block
    int velBlockSize;
    fin.read((char *) &velBlockSize, sizeof(int));
    
    int theoVelBlockSize = 0;
    for (int i = 0; i < 6; ++i)
        theoVelBlockSize += header.N_p[i]*sizeof(vec3<float>);
    
    if (velBlockSize != theoVelBlockSize) {
        print_gadget_header(header);
        std::stringstream errMessage;
        errMessage << "The reported size of the velocity block does not appear to be correct.";
        throw std::runtime_error(errMessage.str());
    }
    
    // Read in the particle velocities
    for (int i = 0; i < 6; ++i) {
        if (header.N_p[i] > 0) {
            for (unsigned int j = 0; j < header.N_p[i]; ++j) {
                vec3<float> part;
                fin.read((char *) &part, sizeof(vec3<float>));
                vel[i].push_back(part);
            }
        }
    }
    
    // Verify the end of the velocity block
    fin.read((char *) &blockSize, sizeof(int));
    
    // Get the size of the particle ID block
    int idBlockSize;
    fin.read((char *) &idBlockSize, sizeof(int));
    
    int theoIDBlockSize = 0;
    for (int i = 0; i < 6; ++i)
        theoIDBlockSize += header.N_p[i]*sizeof(int);
    
    if (idBlockSize != theoIDBlockSize) {
        print_gadget_header(header);
        std::stringstream errMessage;
        errMessage << "The reported size of the particle ID block does not appear to be correct.";
        throw std::runtime_error(errMessage.str());
    }
    
    // Read in the particle IDs
    for (int i = 0; i < 6; ++i) {
        if (header.N_p[i] > 0) {
            for (unsigned int j = 0; j < header.N_p[i]; ++j) {
                int ID;
                fin.read((char *) &ID, sizeof(int));
                ids[i].push_back(ID);
            }
        }
    }
    
    fin.close();
    print_gadget_header(header);
}

void read_gadget_snapshot(std::string snapshot, gadget_header &header, std::vector<double> &delta, 
                          std::vector<double> &delta2, vec3<int> &N, vec3<double> &L, vec3<double> &pk_nbw,
                          vec3<double> &bk_nbw) {
    int blockSize;
    
    std::ifstream fin(snapshot, std::ios::in|std::ios::binary);
    
    // Read all the crap in the header
    fin.read((char *) &blockSize, sizeof(int));
    fin.read((char *) &header, sizeof(gadget_header));
    fin.read((char *) &blockSize, sizeof(int));
    
    // Get the size of the position block
    int posBlockSize;
    fin.read((char *) &posBlockSize, sizeof(int));
    
    int theoPosBlockSize = 0;
    for (int i = 0; i < 6; ++i)
        theoPosBlockSize += header.N_p[i]*sizeof(vec3<float>);
    
    if (posBlockSize != theoPosBlockSize) {
        print_gadget_header(header);
        std::stringstream errMessage;
        errMessage << "The reported size of the position block does not appear to be correct.";
        throw std::runtime_error(errMessage.str());
    }
    
    unsigned int totalParticles = 0;
    for (int i = 0; i < 6; ++i)
        totalParticles += header.N_tot[i];
    
    double length = header.boxSize/1000.0;
    double nbar = totalParticles/(length*length*length);
    
    // Read in particle positions
    for (int i = 0; i < 6; ++i) {
        if (header.N_p[i] > 0) {
            for (unsigned int j = 0; j < header.N_p[i]; ++j) {
                vec3<float> part;
                fin.read((char *) &part, sizeof(vec3<float>));
                vec3<double> pos = {(double)part.x/1000.0, (double)part.y/1000.0, (double)part.z/1000.0};
                std::vector<size_t> indices;
                std::vector<double> weights;
                
                pk_nbw.x += 1.0;
                pk_nbw.y += 1.0;
                pk_nbw.z += nbar;
                
                bk_nbw.x += 1.0;
                bk_nbw.y += nbar;
                bk_nbw.z += nbar*nbar;
                
                getCICInfo(pos, N, L, indices, weights);
                for (int i = 0; i < indices.size(); ++i) {
                    delta[indices[i]] += weights[i];
                    delta2[indices[i]] += weights[i]*weights[i];
                }
            }
        }
    }
    
    // Verify end of position block
    fin.read((char *) &blockSize, sizeof(int));
    
    fin.close();
    print_gadget_header(header);
}

void print_gadget_header(gadget_header &header) {
    std::cout << "Snapshot Details:\n";
    std::cout << "      a = " << header.time << " (z = " << header.redshift << ")\n";
    std::cout << "      L = " << header.boxSize << " h^-1 kpc\n";
    std::cout << "Omega_0 = " << header.Omega_0 << "\n";
    std::cout << "Omega_L = " << header.Omega_L << "\n";
    std::cout << "    h_0 = " << header.h_0 << "\n\n";
    std::cout << "Particle Numbers:\n";
    for (int i = 0; i < 6; ++i)
        std::cout << "    Number of " << part_names[i] << " particles: " << header.N_p[i] << " of " 
                  << header.N_tot[i] << "\n";
    std::cout << "\nParticle Masses:\n";
    for (int i = 0; i < 6; ++i)
        std::cout << "    M_" << part_names[i] << " = " << header.M_p[i] << "\n";
}
