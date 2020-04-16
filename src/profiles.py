import numpy as np
from scipy import integrate
from scipy.interpolate import UnivariateSpline
import numba

from astropy.io import fits
from astropy import units as u
from astropy import constants as c

km2Mpc  = (u.km).to(u.Mpc)
Msun2kg = (u.Msun).to(u.kg)
m2Mpc   = (u.m).to(u.Mpc)
Mpc2m   = (u.Mpc).to(u.m)
G       = c.G.value
sigma_T = c.sigma_T.value
mp      = c.m_p.to('kg').value

def GammaTau(x, c500=1.156):
    """
    Electron density cNFW profile
    Eq. 11 from Louis+17
    ! x = r/R500
    """
    x0 = 0.02 * 0.5 # R200/R500 ~ 1/2
    return ( (x+x0)*c500*(1+x*c500)**2)**(-1)

def gTau(x, c500=1.156, theta500=1):
    int_num = lambda xz: GammaTau(np.sqrt(xz**2+x**2), c500=c500)  
    num = 2 * integrate.quad(int_num, 0, np.inf, epsabs=0, epsrel=1e-3)[0]
    int_den = lambda xr: GammaTau(xr, c500=c500)  
    den = 4 * np.pi * theta500**2 * integrate.quad(int_den, 0, 1, epsabs=0, epsrel=1e-3)[0]
    return num/den

def cNFW(bkd, theta, z, M=1e14, c500=1.177, fb=0.16, fH=0.76, mass_def=500):
    d_A     = bkd.angular_diameter_distance(z) # [Mpc]
    rho_c_z = 3.*(bkd.hubble_parameter(z)*km2Mpc)**2/(8.*np.pi*G)# [kg/m^3]
    r500 = (((M*Msun2kg/(500*4.*np.pi/3.))/rho_c_z)**(1./3.))*m2Mpc # [Mpc]
    rs = r500/c500
    r0 = 0.01 * r500 
    R = np.radians(theta/60.) * d_A # [Mpc]
    print(rho_c_z, r500, d_A, np.rad2deg(rs/d_A)*60)
    delta_c = (500/3.)*(c500**3.)/(np.log(1.+c500)-c500/(1.+c500))

    Lmax = np.inf#1000*r500
    
    norm_for_int = 2. * rho_c_z * rs**3. * sigma_T * delta_c * fb * (1+fH) / (2*mp) * Mpc2m#[kg/m^3 Mpc^3 m^2 kg^-1]
    tau = np.zeros_like(R)
    
    for ii in range(len(R)):
#         print(R[ii])
        def integrand(r):
            return  1 / ( (r0 + r) * (rs + r)**2. ) * ( r / np.sqrt(r**2. - R[ii]**2.))
        tau[ii] = integrate.quad(integrand, R[ii], Lmax, epsabs=0, epsrel=1e-4)[0]

    return norm_for_int*tau

def cNFW_grid(bkd, z, M=1e14, c500=1.177, reso=0.2, theta_max=10,fb=0.16, fH=0.76, mass_def=500):
    theta_x,theta_y = np.meshgrid(np.arange(-theta_max,theta_max+reso,reso), np.arange(-theta_max,theta_max+reso,reso))
    theta   = np.sqrt(theta_x**2+theta_y**2) # arcmin
    d_A     = bkd.angular_diameter_distance(z) # [Mpc]
    rho_c_z = 3.*(bkd.hubble_parameter(z)*km2Mpc)**2/(8.*np.pi*G)# [kg/m^3]
    r500    = (((M*Msun2kg/(500*4.*np.pi/3.))/rho_c_z)**(1./3.))*m2Mpc # [Mpc]
    rs      = r500/c500
    r0      = 0.01 * r500 
    R       = np.radians(theta/60.) * d_A # [Mpc]
    R       = R.flatten()
#     print(rho_c_z, r500, d_A, np.rad2deg(rs/d_A)*60)

    Lmax = np.inf#5*r500
    delta_c = (500/3.)*(c500**3.)/(np.log(1.+c500)-c500/(1.+c500))
    
    norm_for_int = 2. * rho_c_z * rs**3. * sigma_T * delta_c * fb * (1+fH) / (2*mp) * Mpc2m#[kg/m^3 Mpc^3 m^2 kg^-1]
    tau = np.zeros_like(R)
    
    for ii in range(len(R)):
#         print(R[ii])
        def integrand(r):
            return  1 / ( (r0 + r) * (rs + r)**2. ) * ( r / np.sqrt(r**2. - R[ii]**2.))
        tau[ii] = integrate.quad(integrand, R[ii], Lmax, epsabs=0, epsrel=1e-3)[0]

    tau = tau.reshape(theta.shape)*norm_for_int

    return tau

def M500toR500(bkd, z, M):
    d_A     = bkd.angular_diameter_distance(z) # [Mpc]
    rho_c_z = 3.*(bkd.hubble_parameter(z)*km2Mpc)**2/(8.*np.pi*G)# [kg/m^3]
    r500 = (((M*Msun2kg/(500*4.*np.pi/3.))/rho_c_z)**(1./3.))*m2Mpc # [Mpc]
    return r500