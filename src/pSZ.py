import numpy as np
from scipy.special import spherical_jn, jv
from scipy import integrate
from scipy.interpolate import UnivariateSpline
import numba

from astropy.io import fits
from astropy import units as u
from astropy import constants as c
import healpy as hp

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

import camb
from camb import model, initialpower

def_cosmo = {'H0':67.5, 'ombh2':0.022, 'omch2':0.122, 'mnu':0.06, 'omk':0, 'tau':0.06, 'As':2e-9, 'ns':0.965}

class pSZ():
	"""
	Class to store the CMB quadrupole related calculations.
	"""

	def __init__(self, params=None, smica_path=None):
		# Setting cosmo params
		if params is None:
			params = def_cosmo.copy()
		else:
			for key, val in def_cosmo.iteritems():
				params.setdefault(key,val)
		self.params = params.copy()

		#Let's initialize CAMB for distances, growth factor ders, and CMB TT spectrum
		self.pars = camb.CAMBparams()
		self.pars.set_cosmology(H0=self.params['H0'], ombh2=self.params['ombh2'], omch2=self.params['omch2'], mnu=self.params['mnu'], omk=self.params['omk'], tau=self.params['tau'])
		self.pars.InitPower.set_params(As=params['As'], ns=params['ns'], r=0)
		self.pars.set_for_lmax(2500, lens_potential_accuracy=0);
		results = camb.get_results(self.pars)
		self.bkd = camb.get_background(self.pars)
		self.eta_star = self.bkd.conformal_time(1089) # conformal time at recombination

		cmb = results.get_cmb_power_spectra(self.pars,lmax=1000, raw_cl=True)
		self.TCMBmuK = 2.726e6
		self.cltt = np.copy(cmb['lensed_scalar'][:,0])*self.TCMBmuK**2 # muK^2
		
		self.zmin = 0
		self.zmax = 8
		self.nz   = 300
		self.z = np.linspace(self.zmin,self.zmax,self.nz)
		ev = results.get_redshift_evolution(1e-2, self.z, ['growth'])

		self.growth_eta = UnivariateSpline(self.bkd.conformal_time(self.z[::-1]), ev[:, 0][::-1], ext=3)
		self.der_growtg_eta = self.growth_eta.derivative()
	
		if smica_path is None:
			self.smica_path = '/Users/fbianchini/Research/pSZ/data/smica_nside512.fits'
		self.smica_init = False

	def integrand_ISW(self, etap, eta, k, ell=2):
		"""
		Integrand of the ISW term of the transfer function relating the gravitational potential in matter-domination, Psi(k), to the CMB temperature.

		Taken from RHS second term in eq 14 og Hall&Challinor14 https://arxiv.org/pdf/1407.5135.pdf.
		"""
		return spherical_jn(ell, k*(eta-etap)) * self.der_growtg_eta(etap)

	def SW(self, eta, k, ell=2):
		"""
		SW term of the transfer function relating the gravitational potential in matter-domination, Psi(k), CMB to the temperature.

		Taken from RHS first term in eq 14 og Hall&Challinor14 https://arxiv.org/pdf/1407.5135.pdf.
		"""
		return spherical_jn(ell, k*(eta-self.eta_star))/3.
	
	def Delta(self, k, r, ell=2, terms='all', epsabs=0., epsrel=1e-2):
		"""
		transfer function
		!!!! If I *don't* put a factor 2 in front of the integrand_iSW I get the ~same plots as Louis+17, don't know why!
		"""
		eta = self.bkd.tau0 - r
		if terms == 'all':
			return self.SW(eta, k, ell=ell) + integrate.quad(self.integrand_ISW, self.eta_star, eta, args=(eta, k, ell), epsabs=epsabs, epsrel=epsrel)[0]
		elif terms == 'isw' or terms == 'iSW':
			return integrate.quad(self.integrand_ISW, self.eta_star, eta, args=(eta, k, ell), epsabs=epsabs, epsrel=epsrel)[0]
		elif terms == 'sw' or terms == 'SW': 
			return self.SW(eta, k, ell=ell)
		else:
			raise NameError('term')

	def f_ell(self, ell):
		return np.math.factorial(ell+2)/(np.math.factorial(ell-2))
	
	def h_ell(self, k, z, ell=2, terms='all'):
		"""
		Eq 4 from Seto&Pierpaoli https://arxiv.org/pdf/astro-ph/0502564.pdf
		"""
		r = self.bkd.comoving_radial_distance(z)
		x = k*(self.bkd.tau0-self.bkd.conformal_time(z))
		f_ell = np.sqrt(np.math.factorial(ell+2)/(6*np.math.factorial(ell-2))) * spherical_jn(ell, x) / (30*x**2)
		return f_ell * self.Delta(k, r, ell=ell, terms=terms)
	
	def integrand_xi(self, k, r, rp, ell=2):
		"""
		Integrand of the power spectrum of the polarization field generated in two redshift slices r and rp.
		See eq 3 from Louis+17
		"""
		fact = 81/100 * np.pi * self.f_ell(ell) 
		xr  = spherical_jn(ell, k*r)/(k*r)**2
		xrp = spherical_jn(ell, k*rp)/(k*rp)**2
		return fact * xr * xrp * self.Delta(k, r, ell=2) * self.Delta(k, rp, ell=2) * self.pars.scalar_power(k)/ k
	
	def integrand_zeta(self, k, r, ell=2):
		"""
		Integrand of the cross-power spectrum between the polarization field at redshift slice r and the CMB temperature.
		See eq 4 from Louis+17
		"""
		fact = -27/25 * np.pi * self.f_ell(ell)**0.5
		xr  = spherical_jn(ell, k*r)/(k*r)**2
		return fact * xr  * self.Delta(k, r, ell=2) * self.Delta(k, 0, ell) * self.pars.scalar_power(k)/ k
	
	def xi(self, z, zp, ell=2, kmin=1e-5, kmax=1e-1, epsabs=0., epsrel=1e-2, limit=50):
		"""
		The power spectrum of the polarization field generated in two redshift slices z and zp. 
		Units are muK^2
		See eq 3 from Louis+17
		"""
		r  = self.bkd.comoving_radial_distance(z)
		rp = self.bkd.comoving_radial_distance(zp)
		return self.TCMBmuK**2 * integrate.quad(self.integrand_xi, kmin, kmax, args=(r,rp,ell), epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
	
	def zeta(self, z, ell=2, kmin=1e-5, kmax=1e-1, epsabs=0., epsrel=1e-2, limit=50):
		"""
		The cross-power spectrum between the polarization field at redshift slice z and the CMB temperature.
		Units are muK^2
		See eq 4 from Louis+17
		"""
		r  = self.bkd.comoving_radial_distance(z)
		return self.TCMBmuK**2 * integrate.quad(self.integrand_zeta, kmin, kmax, args=(r,ell), epsabs=epsabs, epsrel=epsrel, limit=limit)[0] 
	
	def R(self, z, ell=2, kmin=1e-5, kmax=1e-1, epsabs=0., epsrel=1e-2, limit=50):
		"""
		Correlation coefficient between the polarization field at redshift slice r and the CMB temperature.
		"""
		xi_ = self.xi(z, z, ell=ell, kmin=kmin, kmax=kmax, epsabs=epsabs, epsrel=epsrel, limit=limit) 
		zeta_ = self.zeta(z, ell=ell, kmin=kmin, kmax=kmax, epsabs=epsabs, epsrel=epsrel, limit=limit)
		return np.abs(zeta_)/np.sqrt(xi_*self.cltt[ell])

	def load_smica(self):
		self.smica = hp.read_map(self.smica_path)
		self.alms_smica = hp.map2alm(self.smica*1e6)  # muK
		self.lmax_smica = hp.Alm.getlmax(self.alms_smica.size)
		self.smica_init = True

	def remote_quadrupole(self, z, lmax=5, nside=512, cmap='jet', coord=['G','C'], plot=False, return_QU=False):
		"""
		Returns Healpix maps of the remote quadrupole at redshift z correlated with the CMB local measurement.
		We assume that the sum converges after ell = 5.
		Note that we also assume B-modes of the polarization field to be zero.
		"""
		if not self.smica_init: self.load_smica()
		ells = np.arange(2,lmax+1)
		zeta_tmp = np.asarray([self.zeta(z, ell) for ell in ells]) # muK^2
		plm_tmp = np.zeros(self.lmax_smica+1)
		plm_tmp[2:lmax+1] = zeta_tmp/self.cltt[2:lmax+1] # unitless
		I_tmp, Q_tmp, U_tmp = hp.alm2map([np.zeros_like(self.alms_smica),hp.almxfl(self.alms_smica,plm_tmp),np.zeros_like(self.alms_smica)], nside) # muK
		if plot: hp.mollview(np.sqrt(Q_tmp**2+U_tmp**2), cmap=cmap, coord=coord, title='z = %f'%z )

		if return_QU:
			return np.sqrt(Q_tmp**2+U_tmp**2), Q_tmp, U_tmp
		else:
			return np.sqrt(Q_tmp**2+U_tmp**2)

	def uncorrelated_quadrupole(self, z, lmax=5, nside=512, cmap='jet', coord=['G','C'], plot=False, return_QU=False):
		"""
		Returns Healpix maps of the polarization at redshift z uncorrelated with the CMB local measurement.
		We assume that the sum converges after ell = 5.
		Note that we also assume that B-modes of the polarization field to be zero.
		"""
		if not self.smica_init: self.load_smica()
		ells = np.arange(2,lmax+1)
		zeta_tmp = np.asarray([self.zeta(z, ell) for ell in ells]) # muK^2
		xi_tmp = np.asarray([self.xi(z, z, ell) for ell in ells]) # muK^2
		pl_tmp = np.zeros((self.lmax_smica+1,6)) 
		pl_tmp[2:lmax+1,3] = xi_tmp - zeta_tmp**2/self.cltt[2:lmax+1] # muK^2
		I_tmp, Q_tmp, U_tmp = hp.synfast(pl_tmp.T, nside)
		if plot: hp.mollview(np.sqrt(Q_tmp**2+U_tmp**2), cmap=cmap, coord=coord, title='z = %f'%z )

		if return_QU:
			return np.sqrt(Q_tmp**2+U_tmp**2), Q_tmp, U_tmp
		else:
			return np.sqrt(Q_tmp**2+U_tmp**2)
