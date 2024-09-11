import numpy as np
from scipy.special import spherical_jn
from scipy import integrate
from scipy.interpolate import UnivariateSpline

import healpy as hp
import camb

class pSZ:
    """
    Class to calculate and store quantities related to the CMB quadrupole and the polarized Sunyaev-Zel'dovich (pSZ) effect.
    
    Attributes:
    ----------
    params : dict
        Cosmological parameters for the CAMB initialization.
    cltt : np.array
        CMB temperature power spectrum.
    growth_eta : UnivariateSpline
        Growth factor as a function of conformal time.
    der_growtg_eta : UnivariateSpline
        Derivative of the growth factor with respect to conformal time.
    """
    
    def __init__(self, params=None, smica_path=None):
        """
        Initializes the pSZ class with cosmological parameters and CAMB results.
        
        Parameters:
        ----------
        params : dict, optional
            Cosmological parameters for CAMB. If None, default parameters are used.
        smica_path : str, optional
            Path to the SMICA map FITS file. If not provided, a default path is used.
        """
        # Set cosmological parameters
        def_cosmo = {'H0': 67.5, 'ombh2': 0.022, 'omch2': 0.122, 'mnu': 0.06, 'omk': 0, 'tau': 0.06, 'As': 2e-9, 'ns': 0.965}
        if params is None:
            params = def_cosmo.copy()
        else:
            for key, val in def_cosmo.items():
                params.setdefault(key, val)
        self.params = params.copy()

        # Initialize CAMB for distances, growth factor derivatives, and CMB TT spectrum
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=self.params['H0'], ombh2=self.params['ombh2'], omch2=self.params['omch2'],
                                mnu=self.params['mnu'], omk=self.params['omk'], tau=self.params['tau'])
        self.pars.InitPower.set_params(As=params['As'], ns=params['ns'], r=0)
        self.pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_results(self.pars)
        self.bkd = camb.get_background(self.pars)
        self.eta_star = self.bkd.conformal_time(1089)  # conformal time at recombination

        cmb = results.get_cmb_power_spectra(self.pars, lmax=1000, raw_cl=True)
        self.TCMBmuK = 2.726e6  # CMB temperature in microKelvin
        self.cltt = np.copy(cmb['lensed_scalar'][:, 0]) * self.TCMBmuK**2  # muK^2

        # Redshift range and growth factor calculations
        self.zmin = 0
        self.zmax = 8
        self.nz = 300
        self.z = np.linspace(self.zmin, self.zmax, self.nz)
        ev = results.get_redshift_evolution(1e-2, self.z, ['growth'])
        self.growth_eta = UnivariateSpline(self.bkd.conformal_time(self.z[::-1]), ev[:, 0][::-1], ext=3)
        self.der_growtg_eta = self.growth_eta.derivative()

        # SMICA map initialization
        if smica_path is None:
            self.smica_path ='./data/smica_nside512.fits'
        self.smica_init = False

    def integrand_ISW(self, etap, eta, k, ell=2):
        """
        Calculates the integrand for the ISW effect in the transfer function relating the gravitational potential to CMB temperature.
  
        Parameters:
        ----------
        etap : float
            Conformal time during the photon travel.
        eta : float
            Conformal time at observation.
        k : float
            Wavenumber.
        ell : int, optional
            Multipole index, default is 2.
        
        Returns:
        -------
        float
            Value of the integrand.
			
        """
        return spherical_jn(ell, k * (eta - etap)) * self.der_growtg_eta(etap)

    def SW(self, eta, k, ell=2):
        """
        Calculates the Sachs-Wolfe (SW) term of the transfer function.

        Parameters:
        ----------
        eta : float
            Conformal time at observation.
        k : float
            Wavenumber.
        ell : int, optional
            Multipole index, default is 2.
        
        Returns:
        -------
        float
            SW term.
        """
        return spherical_jn(ell, k * (eta - self.eta_star)) / 3.

    def Delta(self, k, r, ell=2, terms='all', epsabs=0., epsrel=1e-2):
        """
        Calculates the transfer function Delta(k,r).
		!!!! If I *don't* put a factor 2 in front of the integrand_iSW I get the ~same plots as Louis+17, don't know why!
        
        Parameters:
        ----------
        k : float
            Wavenumber.
        r : float
            Radial distance.
        ell : int, optional
            Multipole index, default is 2.
        terms : str, optional
            Specify 'all', 'isw', or 'sw' to compute different terms of the transfer function.
        epsabs : float, optional
            Absolute error tolerance for integration.
        epsrel : float, optional
            Relative error tolerance for integration.
        
        Returns:
        -------
        float
            Transfer function value.
        """
        eta = self.bkd.tau0 - r
        if terms.lower() == 'all':
            return self.SW(eta, k, ell=ell) + integrate.quad(self.integrand_ISW, self.eta_star, eta, args=(eta, k, ell), epsabs=epsabs, epsrel=epsrel)[0]
        elif terms.lower() == 'isw':
            return integrate.quad(self.integrand_ISW, self.eta_star, eta, args=(eta, k, ell), epsabs=epsabs, epsrel=epsrel)[0]
        elif terms.lower() == 'sw':
            return self.SW(eta, k, ell=ell)
        else:
            raise ValueError("Invalid term specified: choose from 'all', 'isw', or 'sw'.")

    def f_ell(self, ell):
        """
        Calculates a factorial ratio for a given multipole moment.
        
        Parameters:
        ----------
        ell : int
            Multipole moment.
        
        Returns:
        -------
        float
            The factorial ratio.
        """
        return np.math.factorial(ell + 2) / np.math.factorial(ell - 2)

    def h_ell(self, k, z, ell=2, terms='all'):
        """
        Computes h_ell as defined in Seto & Pierpaoli (2005), Eq. 4. (https://arxiv.org/pdf/astro-ph/0502564.pdf)

        Parameters:
        ----------
        k : float
            Wavenumber.
        z : float
            Redshift.
        ell : int, optional
            Multipole index, default is 2.
        terms : str, optional
            Specify 'all', 'isw', or 'sw' to compute different terms of the transfer function.
        
        Returns:
        -------
        float
            h_ell value.
        """
        r = self.bkd.comoving_radial_distance(z)
        x = k * (self.bkd.tau0 - self.bkd.conformal_time(z))
        f_ell = np.sqrt(self.f_ell(ell)) * spherical_jn(ell, x) / (30 * x**2)
        return f_ell * self.Delta(k, r, ell=ell, terms=terms)


    def integrand_xi(self, k, r, rp, ell=2):
        """
        Integrand of the power spectrum of the polarization field generated in two redshift slices r and rp.

        See eq 3 from Louis+17 (https://arxiv.org/abs/1706.03719).

        Parameters:
        ----------
        k : float
            Wavenumber.
        r : float
            Radial distance to redshift z.
        rp : float
            Radial distance to redshift zp.
        ell : int, optional
            Multipole index, default is 2.

        Returns:
        -------
        float
            Value of the integrand.
        """
        fact = (81 / 100) * np.pi * self.f_ell(ell)
        xr = spherical_jn(ell, k * r) / (k * r)**2
        xrp = spherical_jn(ell, k * rp) / (k * rp)**2
        return fact * xr * xrp * self.Delta(k, r, ell=2) * self.Delta(k, rp, ell=2) * self.pars.scalar_power(k)/ k

    def integrand_zeta(self, k, r, ell=2):
        """
        Integrand of the cross-power spectrum between the polarization field at redshift slice r and the CMB temperature.

        See eq 4 from Louis+17 (https://arxiv.org/abs/1706.03719).

        Parameters:
        ----------
        k : float
            Wavenumber.
        r : float
            Radial distance to redshift z.
        ell : int, optional
            Multipole index, default is 2.

        Returns:
        -------
        float
            Value of the integrand.
        """
        fact = -27 / 25 * np.pi * np.sqrt(self.f_ell(ell))
        xr = spherical_jn(ell, k * r) / (k * r)**2
        return fact * xr  * self.Delta(k, r, ell=2) * self.Delta(k, 0, ell) * self.pars.scalar_power(k)/ k

    def xi(self, z, zp, ell=2, kmin=1e-5, kmax=1e-1, epsabs=0., epsrel=1e-2, limit=50):
        """
        The power spectrum of the polarization field generated in two redshift slices z and zp.

        Units are muK^2.

        See eq 3 from Louis+17 (https://arxiv.org/abs/1706.03719).

        Parameters:
        ----------
        z : float
            Redshift of the first slice.
        zp : float
            Redshift of the second slice.
        ell : int, optional
            Multipole index, default is 2.
        kmin : float, optional
            Minimum wavenumber.
        kmax : float, optional
            Maximum wavenumber.
        epsabs : float, optional
            Absolute error tolerance for integration.
        epsrel : float, optional
            Relative error tolerance for integration.
        limit : int, optional
            Limit on the number of subdivisions for integration.

        Returns:
        -------
        float
            Polarization power spectrum in muK^2.
        """
        r = self.bkd.comoving_radial_distance(z)
        rp = self.bkd.comoving_radial_distance(zp)
        return self.TCMBmuK**2 * integrate.quad(self.integrand_xi, kmin, kmax, args=(r, rp, ell), epsabs=epsabs, epsrel=epsrel, limit=limit)[0]

    def zeta(self, z, ell=2, kmin=1e-5, kmax=1e-1, epsabs=0., epsrel=1e-2, limit=50):
        """
        The cross-power spectrum between the polarization field at redshift slice z and the CMB temperature.

        Units are muK^2.

        See eq 4 from Louis+17 (https://arxiv.org/abs/1706.03719).

        Parameters:
        ----------
        z : float
            Redshift.
        ell : int, optional
            Multipole index, default is 2.
        kmin : float, optional
            Minimum wavenumber.
        kmax : float, optional
            Maximum wavenumber.
        epsabs : float, optional
            Absolute error tolerance for integration.
        epsrel : float, optional
            Relative error tolerance for integration.
        limit : int, optional
            Limit on the number of subdivisions for integration.

        Returns:
        -------
        float
            Cross-power spectrum in muK^2.
        """
        r = self.bkd.comoving_radial_distance(z)
        return self.TCMBmuK**2 * integrate.quad(self.integrand_zeta, kmin, kmax, args=(r, ell), epsabs=epsabs, epsrel=epsrel, limit=limit)[0]

    def R(self, z, ell=2, kmin=1e-5, kmax=1e-1, epsabs=0., epsrel=1e-2, limit=50):
        """
        Correlation coefficient between the polarization field at redshift slice r and the CMB temperature.

        Parameters:
        ----------
        z : float
            Redshift.
        ell : int, optional
            Multipole index, default is 2.
        kmin : float, optional
            Minimum wavenumber.
        kmax : float, optional
            Maximum wavenumber.
        epsabs : float, optional
            Absolute error tolerance for integration.
        epsrel : float, optional
            Relative error tolerance for integration.
        limit : int, optional
            Limit on the number of subdivisions for integration.

        Returns:
        -------
        float
            Correlation coefficient between the polarization field and the CMB temperature.
        """
        xi_ = self.xi(z, z, ell=ell, kmin=kmin, kmax=kmax, epsabs=epsabs, epsrel=epsrel, limit=limit)
        zeta_ = self.zeta(z, ell=ell, kmin=kmin, kmax=kmax, epsabs=epsabs, epsrel=epsrel, limit=limit)
        return np.abs(zeta_) / np.sqrt(xi_ * self.cltt[ell])

    def load_smica(self):
        """
        Loads the SMICA map and computes the spherical harmonic coefficients (alms).
        """
        self.smica = hp.read_map(self.smica_path)
        self.alms_smica = hp.map2alm(self.smica * 1e6)  # muK
        self.lmax_smica = hp.Alm.getlmax(self.alms_smica.size)
        self.smica_init = True

    def remote_quadrupole(self, z, lmax=5, nside=512, cmap='viridis', coord=['G', 'C'], plot=False, return_QU=False):
        """
        Returns Healpix maps of the remote quadrupole at redshift z correlated with the CMB local measurement.

        Assumes that the sum converges after ell = 5. B-modes of the polarization field are assumed to be zero.

        Parameters:
        ----------
        z : float
            Redshift.
        lmax : int, optional
            Maximum multipole index.
        nside : int, optional
            Healpix nside resolution.
        cmap : str, optional
            Color map for plotting.
        coord : list, optional
            Coordinate system for the map (e.g., ['G', 'C'] for Galactic and Celestial).
        plot : bool, optional
            If True, a map is plotted.
        return_QU : bool, optional
            If True, returns the Q and U maps.

        Returns:
        -------
        np.array or tuple of np.array
            Healpix map of the quadrupole (and optionally Q and U maps).
        """
        if not self.smica_init:
            self.load_smica()
        ells = np.arange(2, lmax + 1)
        zeta_tmp = np.asarray([self.zeta(z, ell) for ell in ells])  # muK^2
        plm_tmp = np.zeros(self.lmax_smica + 1)
        plm_tmp[2:lmax + 1] = zeta_tmp / self.cltt[2:lmax + 1]  # unitless
        I_tmp, Q_tmp, U_tmp = hp.alm2map([np.zeros_like(self.alms_smica), hp.almxfl(self.alms_smica, plm_tmp), np.zeros_like(self.alms_smica)], nside)  # muK
        if plot:
            hp.mollview(np.sqrt(Q_tmp**2 + U_tmp**2), cmap=cmap, coord=coord, title=f'z = {z}')
        if return_QU:
            return np.sqrt(Q_tmp**2 + U_tmp**2), Q_tmp, U_tmp
        else:
            return np.sqrt(Q_tmp**2 + U_tmp**2)

    def uncorrelated_quadrupole(self, z, lmax=5, nside=512, cmap='viridis', coord=['G', 'C'], plot=False, return_QU=False):
        """
        Returns Healpix maps of the polarization at redshift z uncorrelated with the CMB local measurement.

        Assumes that the sum converges after ell = 5. B-modes of the polarization field are assumed to be zero.

        Parameters:
        ----------
        z : float
            Redshift.
        lmax : int, optional
            Maximum multipole index.
        nside : int, optional
            Healpix nside resolution.
        cmap : str, optional
            Color map for plotting.
        coord : list, optional
            Coordinate system for the map (e.g., ['G', 'C'] for Galactic and Celestial).
        plot : bool, optional
            If True, a map is plotted.
        return_QU : bool, optional
            If True, returns the Q and U maps.

        Returns:
        -------
        np.array or tuple of np.array
            Healpix map of the quadrupole (and optionally Q and U maps).
        """
        if not self.smica_init:
            self.load_smica()
        ells = np.arange(2, lmax + 1)
        zeta_tmp = np.asarray([self.zeta(z, ell) for ell in ells])  # muK^2
        xi_tmp = np.asarray([self.xi(z, z, ell) for ell in ells])  # muK^2
        pl_tmp = np.zeros((self.lmax_smica + 1, 6))
        pl_tmp[2:lmax + 1, 3] = xi_tmp - zeta_tmp**2 / self.cltt[2:lmax + 1]  # muK^2
        I_tmp, Q_tmp, U_tmp = hp.synfast(pl_tmp.T, nside)
        if plot:
            hp.mollview(np.sqrt(Q_tmp**2 + U_tmp**2), cmap=cmap, coord=coord, title=f'z = {z}')
        if return_QU:
            return np.sqrt(Q_tmp**2 + U_tmp**2), Q_tmp, U_tmp
        else:
            return np.sqrt(Q_tmp**2 + U_tmp**2)
