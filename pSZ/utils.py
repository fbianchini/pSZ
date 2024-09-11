import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from matplotlib import rc, rcParams
from astropy.coordinates import SkyCoord
import astropy.units as u
import scipy.ndimage as ndimage

# Conversion constants
arcmin2rad = np.pi / 180. / 60.
rad2arcmin = 1. / arcmin2rad

def GaussSmooth(map, fwhm, reso, order=0):
    """
    Smooth the input map with a Gaussian beam, specified by its FWHM (in arcmin).
    
    Args:
        map (np.ndarray): The input map to smooth.
        fwhm (float): Full width at half maximum of the Gaussian beam in arcminutes.
        reso (float): Pixel resolution of the map in arcminutes.
        order (int, optional): The order of the filter (default is 0, which corresponds to Gaussian smoothing).
    
    Returns:
        np.ndarray: The smoothed map.
    """
    sigma = fwhm / np.sqrt(8 * np.log(2)) / reso  # Convert FWHM to sigma in pixels
    return ndimage.gaussian_filter(map, sigma=sigma, order=order)

def GenerateRndPointsS2(size, ra_min=0, ra_max=360, dec_min=-90, dec_max=90):
    """
    Generate random points uniformly distributed on the sphere within the specified RA/DEC range.
    
    Args:
        size (int): The number of random points to generate.
        ra_min (float, optional): Minimum RA value in degrees (default: 0).
        ra_max (float, optional): Maximum RA value in degrees (default: 360).
        dec_min (float, optional): Minimum DEC value in degrees (default: -90).
        dec_max (float, optional): Maximum DEC value in degrees (default: 90).
    
    Returns:
        tuple: Arrays of RA and DEC values of the random points.
    """
    ras = np.rad2deg(np.random.uniform(np.deg2rad(ra_min), np.deg2rad(ra_max), size))
    decs = np.rad2deg(np.arcsin(np.random.uniform(np.sin(np.deg2rad(dec_min)), np.sin(np.deg2rad(dec_max)), size)))
    return ras, decs

def Equatorial2Galactic(ra, dec):
    """
    Convert equatorial coordinates (RA/DEC) to galactic coordinates (l/b).
    
    Args:
        ra (float or np.ndarray): Right ascension in degrees.
        dec (float or np.ndarray): Declination in degrees.
    
    Returns:
        tuple: Galactic longitude (l) and latitude (b) in degrees.
    """
    c_eq = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    c_gal = c_eq.transform_to('galactic')
    return c_gal.l.degree, c_gal.b.degree
