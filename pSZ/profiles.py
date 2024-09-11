import numpy as np
from scipy import integrate
import scipy.optimize

from astropy import units as u
from astropy import constants as c

# Conversion factors
km2Mpc = (u.km).to(u.Mpc)
Msun2kg = (u.Msun).to(u.kg)
m2Mpc = (u.m).to(u.Mpc)
Mpc2m = (u.Mpc).to(u.m)

# Physical constants
G = c.G.value  # Gravitational constant [m^3 kg^-1 s^-2]
sigma_T = c.sigma_T.value  # Thomson cross-section [m^2]
mp = c.m_p.to('kg').value  # Proton mass [kg]


def GammaTau(x, c500=1.156):
    """
    Electron density cNFW profile, as given by Eq. 11 from Louis+17.
    
    Args:
        x (float): Dimensionless radius r/R500.
        c500 (float, optional): Concentration parameter (default: 1.156).
        
    Returns:
        float: The electron density profile at the given radius.
    """
    x0 = 0.02 * 0.5  # R200/R500 ~ 1/2
    return ((x + x0) * c500 * (1 + x * c500)**2)**(-1)


def gTau(x, c500=1.156, theta500=1):
    """
    Calculates the dimensionless gTau parameter.
    
    Args:
        x (float): Dimensionless radius r/R500.
        c500 (float, optional): Concentration parameter (default: 1.156).
        theta500 (float, optional): Angular scale related to R500 (default: 1 arcmin).
        
    Returns:
        float: gTau parameter.
    """
    # Numerator for the integral
    int_num = lambda xz: GammaTau(np.sqrt(xz**2 + x**2), c500=c500)
    num = 2 * integrate.quad(int_num, 0, np.inf, epsabs=0, epsrel=1e-3)[0]
    
    # Denominator for the integral
    int_den = lambda xr: GammaTau(xr, c500=c500)
    den = 4 * np.pi * theta500**2 * integrate.quad(int_den, 0, 1, epsabs=0, epsrel=1e-3)[0]
    
    return num / den



def cNFW(bkd, theta, z, M=1e14, c500=1.177, fb=0.16, fH=0.76, mass_def=500):
    """
    Computes the cNFW optical depth profile for a galaxy cluster.
    
    Args:
        bkd: Background cosmology object with necessary cosmological functions.
        theta (float): Angular radius in arcminutes.
        z (float): Redshift of the cluster.
        M (float, optional): Mass of the cluster in solar masses (default: 1e14).
        c500 (float, optional): Concentration parameter (default: 1.177).
        fb (float, optional): Baryon fraction (default: 0.16).
        fH (float, optional): Hydrogen mass fraction (default: 0.76).
        mass_def (int, optional): Overdensity definition (default: 500).
        
    Returns:
        np.ndarray: Optical depth profile at the given angular radii.
    """
    # Derived quantities
    d_A = bkd.angular_diameter_distance(z)  # Angular diameter distance [Mpc]
    rho_c_z = 3. * (bkd.hubble_parameter(z) * km2Mpc)**2 / (8. * np.pi * G)  # Critical density [kg/m^3]
    r500 = (((M * Msun2kg / (500 * 4. * np.pi / 3.)) / rho_c_z)**(1. / 3.)) * m2Mpc  # R500 [Mpc]
    rs = r500 / c500
    R = np.radians(theta / 60.) * d_A  # Physical radius corresponding to theta [Mpc]
    
    delta_c = (500 / 3.) * (c500**3.) / (np.log(1. + c500) - c500 / (1. + c500))
    norm_for_int = (2. * rho_c_z * rs**3. * sigma_T * delta_c * fb * (1 + fH) / (2 * mp) * Mpc2m)

    tau = np.zeros_like(R)
    for ii in range(len(R)):
        def integrand(r):
            return 1 / ((r + 0.01 * r500) * (r + rs)**2) * (r / np.sqrt(r**2 - R[ii]**2))
        
        tau[ii] = integrate.quad(integrand, R[ii], np.inf, epsabs=0, epsrel=1e-4)[0]

    return norm_for_int * tau


def cNFW_grid(bkd, z, M=1e14, c500=1.177, reso=0.2, theta_max=10, fb=0.16, fH=0.76, mass_def=500):
    """
    Computes a 2D grid of the cNFW optical depth profile for a galaxy cluster.

    Args:
        bkd: Background cosmology object with necessary cosmological functions.
        z (float): Redshift of the cluster.
        M (float, optional): Mass of the cluster in solar masses (default: 1e14).
        c500 (float, optional): Concentration parameter (default: 1.177).
        reso (float, optional): Resolution of the grid in arcminutes (default: 0.2).
        theta_max (float, optional): Maximum angular radius in arcminutes (default: 10).
        fb (float, optional): Baryon fraction (default: 0.16).
        fH (float, optional): Hydrogen mass fraction (default: 0.76).
        mass_def (int, optional): Overdensity definition (default: 500).

    Returns:
        np.ndarray: 2D optical depth grid.
    """
    theta_x, theta_y = np.meshgrid(np.arange(-theta_max, theta_max + reso, reso),
                                   np.arange(-theta_max, theta_max + reso, reso))
    theta = np.sqrt(theta_x**2 + theta_y**2)  # Angular radii in arcminutes

    # Flatten the grid for easier integration
    d_A = bkd.angular_diameter_distance(z)
    R = np.radians(theta / 60.) * d_A  # Convert to physical radii [Mpc]
    R = R.flatten()

    # Compute necessary quantities
    rho_c_z = 3. * (bkd.hubble_parameter(z) * km2Mpc)**2 / (8. * np.pi * G)
    r500 = (((M * Msun2kg / (500 * 4. * np.pi / 3.)) / rho_c_z)**(1. / 3.)) * m2Mpc
    rs = r500 / c500
    delta_c = (500 / 3.) * (c500**3.) / (np.log(1. + c500) - c500 / (1. + c500))
    
    norm_for_int = (2. * rho_c_z * rs**3. * sigma_T * delta_c * fb * (1 + fH) / (2 * mp) * Mpc2m)
    tau = np.zeros_like(R)

    for ii in range(len(R)):
        def integrand(r):
            return 1 / ((r + 0.01 * r500) * (r + rs)**2) * (r / np.sqrt(r**2 - R[ii]**2))

        tau[ii] = integrate.quad(integrand, R[ii], np.inf, epsabs=0, epsrel=1e-3)[0]

    tau = tau.reshape(theta.shape) * norm_for_int
    return tau


def M500toR500(bkd, z, M):
    """
    Converts a given mass M500 to radius R500 at redshift z.
    
    Args:
        bkd: Background cosmology object with necessary cosmological functions.
        z (float): Redshift of the cluster.
        M (float): Cluster mass M500 in solar masses.

    Returns:
        float: The radius R500 in Mpc.
    """
    d_A = bkd.angular_diameter_distance(z)  # Angular diameter distance [Mpc]
    rho_c_z = 3. * (bkd.hubble_parameter(z) * km2Mpc)**2 / (8. * np.pi * G)  # Critical density [kg/m^3]
    r500 = (((M * Msun2kg / (500 * 4. * np.pi / 3.)) / rho_c_z)**(1. / 3.)) * m2Mpc  # R500 [Mpc]
    return r500


def R5002M500(theory, z, Mmin=1e13, Mmax=1e16):
    """
    Finds the mass M500 that corresponds to a radius R500 at redshift z using root finding.
    
    This function solves for the mass M500 that gives a specific R500 (which equals 1 in arbitrary units)
    by using Brent's method to find the root of the equation M500toR500(theory.bkd, z, M) - 1 = 0.

    Args:
        theory: A theory object that contains cosmological background information.
        z (float): Redshift of the cluster.
        Mmin (float, optional): Minimum mass for root-finding (default: 1e13 solar masses).
        Mmax (float, optional): Maximum mass for root-finding (default: 1e16 solar masses).

    Returns:
        float: The mass M500 (in solar masses) that corresponds to R500 at redshift z.
    """
    return scipy.optimize.brentq(lambda M: M500toR500(theory.bkd, z, M=M) - 1, Mmin, Mmax)

def M5002Theta500(theory, z, M500):
    """
    Converts the mass M500 to an angular scale theta500 (in arcminutes) at redshift z.
    
    This function computes the angular size of R500 in arcminutes, given a mass M500, 
    by calculating the corresponding physical radius R500 and then converting it to 
    an angular size using the angular diameter distance.

    Args:
        theory: A theory object that contains cosmological background information.
        z (float): Redshift of the cluster.
        M500 (float): Mass M500 in solar masses.

    Returns:
        float: Angular scale theta500 in arcminutes.
    """
    R500 = M500toR500(theory.bkd, z, M=M500)  # Physical radius in Mpc
    d_A = theory.bkd.angular_diameter_distance(z)  # Angular diameter distance in Mpc
    return np.rad2deg(R500 / d_A) * 60  # Convert angular size to arcminutes
