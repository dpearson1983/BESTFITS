#ifndef _COSMOLOGY_H_
#define _COSMOLOGY_H_

#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>

class cosmology{
    double Om_M, Om_L, Om_b, Om_c, tau, T_CMB, h;
    gsl_interp_accel *acc;
    gsl_spline *r2z;
    
    double E(double z);
    
    static double E_inv(double z, void *params);
    
    static double rd_int(double z, void *params);
    
    static double rz(double z, void *params);
    
    double Theta();
    
    double k_eq();
    
    public:
        cosmology(double H_0 = 70.0, double OmegaM = 0.3, double OmegaL = 0.7, double Omegab = 0.04, 
                  double Omegac = 0.26, double Tau = 0.066, double TCMB = 2.718);
        
        ~cosmology();
        
        double Omega_M();
        
        double Omega_L();
        
        double Omega_bh2();
        
        double Omega_ch2();
        
        double h_param();
        
        double H0();
        
        double H(double z);
        
        double D_A(double z);
        
        double D_V(double z);
        
        double z_eq();
        
        double z_d();
        
        double R(double z);
        
        double r_d();
        
        double comoving_distance(double z, gsl_integration_workspace *w);
        
        double get_redshift_from_comoving_distance(double r);

};

#endif
