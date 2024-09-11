import numpy as np  
import healpy as hp
import matplotlib.pyplot as plt

from utils import GenerateRndPointsS2, Equatorial2Galactic, GaussSmooth
from profiles import R5002M500, cNFW_grid


def GetNoiseMap(npix, delta_P, reso):
    """
    Generates a noise map with Gaussian-distributed random values.
    
    Args:
        npix (int): The number of pixels along one dimension of the map.
        delta_P (float): Noise level (standard deviation) in microKelvin.
        reso (float): Map resolution in arcminutes.
        
    Returns:
        np.ndarray: A noise map with shape (npix, npix).
    """
    return np.random.randn(npix, npix) * (delta_P / reso)

def GetP(data, tau, theta_max, R):
    """
    Computes the weighted sum of P over a region with angular radius theta_max.
    
    Args:
        data (np.ndarray): Input data map (e.g., Q or U component).
        tau (np.ndarray): Tau profile (already beam smoothed).
        theta_max (float): Maximum angular radius in arcminutes.
        R (np.ndarray): Radial distance map in degrees.
        
    Returns:
        float: The mean value of P over the specified region.
    """
    mask = R < theta_max / 60.  # Convert theta_max to degrees
    return np.sum((data * tau)[mask]) / np.sum(tau[mask]**2)


class pSZSims:
    def __init__(self, theory):
        """
        Initializes the pSZ simulation class.
        
        Args:
            theory: A theory object that contains cosmological background information.
        """
        self.theory = theory

    def GetVarAlphaManyBins(self, nclusters, zmin, zmax, nz, white_noise, beam, reso, theta_max=5, theta_max_profile=20, quad=None, marginalize_bkd=False):
        """
        Computes the variance of alpha across multiple redshift bins.
        
        Args:
            nclusters (int): Total number of clusters to simulate.
            zmin (float): Minimum redshift.
            zmax (float): Maximum redshift.
            nz (int): Number of redshift bins.
            white_noise (float): Level of white noise in microKelvin.
            beam (float): Beam size in arcminutes.
            reso (float): Resolution of the grid in arcminutes.
            theta_max (float, optional): Maximum angular radius for analysis (default: 5 arcminutes).
            theta_max_profile (float, optional): Maximum angular radius for the tau profile (default: 20 arcminutes).
            quad (tuple, optional): Quadruple data (Pc, Qc, Uc) for the remote quadrupole field. If None, use theory.
            marginalize_bkd (bool, optional): Whether to marginalize over the background (default: False).
        
        Returns:
            float: The variance of alpha.
        """
        dz = (zmax - zmin) / nz
        nclusters_bin = int(nclusters / nz)

        p_c = np.zeros((nz, 2 * nclusters_bin), dtype=complex)
        C_p_inv = np.zeros(2 * nz * nclusters_bin)

        for idz in range(nz):
            z_tmp = (0.5 * (np.linspace(zmin, zmax, nz+1)[1:] + np.linspace(zmin, zmax, nz+1)[:-1]))[idz]
            M500 = R5002M500(self.theory, z_tmp)
            theta_max_tmp = theta_max * np.rad2deg(1. / self.theory.bkd.angular_diameter_distance(z_tmp)) * 60

            # Generate tau profile and apply beam smoothing
            tau = cNFW_grid(self.theory.bkd, z_tmp, M=M500, reso=reso, theta_max=theta_max_profile)
            tau_beam = GaussSmooth(tau, beam, reso)

            N = tau.shape[0]
            ones = np.ones(N)
            inds = (np.arange(N) + 0.5 - N / 2.)
            R = np.outer(ones, inds) * reso / 60.  # Radial distance map in degrees

            # Compute variance of p
            tau2_mean = np.mean(tau_beam[R < theta_max_tmp / 60.]**2)
            tau_mean2 = np.mean(tau_beam[R < theta_max_tmp / 60.])**2
            npix = len(tau_beam[R < theta_max_tmp / 60.])

            if marginalize_bkd:
                var_p = 2 * (white_noise / reso)**2 / (npix * (tau2_mean - tau_mean2))
            else:
                var_p = 2 * (white_noise / reso)**2 / np.sum(tau_beam[R < theta_max_tmp / 60.]**2)

            C_p_inv[idz * 2 * nclusters_bin:(idz + 1) * 2 * nclusters_bin] = 1 / var_p

            # Generate random cluster locations
            ras_tmp, decs_tmp = GenerateRndPointsS2(nclusters_bin, ra_min=-50, ra_max=50, dec_min=-70, dec_max=-40)
            l_tmp, b_tmp = Equatorial2Galactic(ras_tmp, decs_tmp)

            if quad is None:
                Pc_tmp, Qc_tmp, Uc_tmp = self.theory.remote_quadrupole(z_tmp, nside=2048, return_QU=True)
            else:
                Pc_tmp, Qc_tmp, Uc_tmp = quad

            # Extract and process data for each cluster
            for idx in range(nclusters_bin):
                Qcorr_tmp = hp.gnomview(Qc_tmp, rot=[l_tmp[idx], b_tmp[idx]], reso=reso, xsize=tau.shape[0], return_projected_map=True); plt.close()
                Ucorr_tmp = hp.gnomview(Uc_tmp, rot=[l_tmp[idx], b_tmp[idx]], reso=reso, xsize=tau.shape[0], return_projected_map=True); plt.close()

                p_plus_tmp = Qcorr_tmp + 1j * Ucorr_tmp
                p_minus_tmp = Qcorr_tmp - 1j * Ucorr_tmp

                # Compute p_c for both p_plus and p_minus
                p_c[idz, idx] = GetP(p_plus_tmp, np.ones_like(Qcorr_tmp), theta_max_tmp, R)
                p_c[idz, idx + nclusters_bin] = p_c[idz, idx].conj()

        # Calculate variance of alpha
        C_p_inv = np.diag(C_p_inv)
        p_c = p_c.flatten()
        var_alpha = 1. / np.dot(p_c.conj().T, np.dot(C_p_inv, p_c))

        return var_alpha.real

    def GetVarAlpha(self, nclusters, z, M, white_noise, beam, reso, theta_max, theta_max_profile=20, quad=None):
        """
        Computes the variance of alpha for a given mass and redshift.
        
        Args:
            nclusters (int): Number of clusters.
            z (float): Redshift of the cluster.
            M (float): Mass of the cluster.
            white_noise (float): Level of white noise in microKelvin.
            beam (float): Beam size in arcminutes.
            reso (float): Resolution of the grid in arcminutes.
            theta_max (float): Maximum angular radius for analysis in arcminutes.
            theta_max_profile (float, optional): Maximum angular radius for the tau profile (default: 20 arcminutes).
            quad (tuple, optional): Quadruple data (Pc, Qc, Uc). If None, use theory.
        
        Returns:
            float: Variance of alpha.
        """
        tau = cNFW_grid(self.theory.bkd, z, M=M, reso=reso, theta_max=theta_max_profile)
        tau_beam = GaussSmooth(tau, beam, reso)

        N = tau.shape[0]
        ones = np.ones(N)
        inds = (np.arange(N) + 0.5 - N / 2.)
        R = np.outer(ones, inds) * reso / 60.

        ras_tmp, decs_tmp = GenerateRndPointsS2(nclusters, ra_min=-50, ra_max=50, dec_min=-70, dec_max=-40)
        l_tmp, b_tmp = Equatorial2Galactic(ras_tmp, decs_tmp)

        if quad is None:
            Pc_tmp, Qc_tmp, Uc_tmp = self.theory.remote_quadrupole(z, nside=2048, return_QU=True)
        else:
            Pc_tmp, Qc_tmp, Uc_tmp = quad

        var_p = 2 * (white_noise / reso)**2 / np.sum(tau_beam[R < theta_max / 60.]**2)
        p_c = np.zeros(2 * nclusters, dtype=complex)

        # Compute p_c for each cluster
        for idx in range(nclusters):
            Qcorr_tmp = hp.gnomview(Qc_tmp, rot=[l_tmp[idx], b_tmp[idx]], reso=reso, xsize=tau.shape[0], return_projected_map=True); plt.close()
            Ucorr_tmp = hp.gnomview(Uc_tmp, rot=[l_tmp[idx], b_tmp[idx]], reso=reso, xsize=tau.shape[0], return_projected_map=True); plt.close()

            p_plus_tmp = Qcorr_tmp + 1j * Ucorr_tmp
            p_minus_tmp = Qcorr_tmp - 1j * Ucorr_tmp

            p_c[idx] = GetP(p_plus_tmp, np.ones_like(Qcorr_tmp), theta_max, R)
            p_c[idx + nclusters] = p_c[idx].conj()

        C_p_inv = np.diag(np.ones(2 * nclusters) / var_p)
        var_alpha = 1. / np.dot(p_c.conj().T, np.dot(C_p_inv, p_c))

        return var_alpha.real

    def GetS2N(self, nclusters, z, M, white_noise, beam, reso, theta_max, theta_max_profile=20, nsims=1, quad=None):
        """
        Computes the signal-to-noise ratio (S/N) for a given set of cluster simulations.
        
        Args:
            nclusters (int): Number of clusters.
            z (float): Redshift of the cluster.
            M (float): Mass of the cluster.
            white_noise (float): Level of white noise in microKelvin.
            beam (float): Beam size in arcminutes.
            reso (float): Resolution of the grid in arcminutes.
            theta_max (float): Maximum angular radius for analysis in arcminutes.
            theta_max_profile (float, optional): Maximum angular radius for the tau profile (default: 20 arcminutes).
            nsims (int, optional): Number of simulations to run (default: 1).
            quad (tuple, optional): Quadruple data (Pc, Qc, Uc). If None, use theory.
        
        Returns:
            np.ndarray: Signal-to-noise ratios for each simulation.
        """
        tau = cNFW_grid(self.theory.bkd, z, M=M, reso=reso, theta_max=theta_max_profile)
        tau_beam = GaussSmooth(tau, beam, reso)

        N = tau.shape[0]
        ones = np.ones(N)
        inds = (np.arange(N) + 0.5 - N / 2.)
        R = np.outer(ones, inds) * reso / 60.

        if quad is None:
            Pc_tmp, Qc_tmp, Uc_tmp = self.theory.remote_quadrupole(z, nside=2048, return_QU=True)
        else:
            Pc_tmp, Qc_tmp, Uc_tmp = quad

        var_p = 2 * (white_noise / reso)**2 / np.sum(tau_beam[R < theta_max / 60.]**2)
        S2N = np.zeros(nsims)

        for isim in range(nsims):
            ras_tmp, decs_tmp = GenerateRndPointsS2(nclusters, ra_min=-50, ra_max=50, dec_min=-70, dec_max=-40)
            l_tmp, b_tmp = Equatorial2Galactic(ras_tmp, decs_tmp)

            p_c = np.zeros(2 * nclusters, dtype=complex)
            p_hat = np.zeros(2 * nclusters, dtype=complex)

            for idx in range(nclusters):
                Qcorr_tmp = hp.gnomview(Qc_tmp, rot=[l_tmp[idx], b_tmp[idx]], reso=reso, xsize=tau.shape[0], return_projected_map=True); plt.close()
                Ucorr_tmp = hp.gnomview(Uc_tmp, rot=[l_tmp[idx], b_tmp[idx]], reso=reso, xsize=tau.shape[0], return_projected_map=True); plt.close()

                Q_noise = GetNoiseMap(N, white_noise, reso)
                U_noise = GetNoiseMap(N, white_noise, reso)

                d_plus_tmp = (Qcorr_tmp + 1j * Ucorr_tmp) * tau_beam + (Q_noise + 1j * U_noise)
                d_minus_tmp = (Qcorr_tmp - 1j * Ucorr_tmp) * tau_beam + (Q_noise - 1j * U_noise)

                p_plus_tmp = Qcorr_tmp + 1j * Ucorr_tmp
                p_minus_tmp = Qcorr_tmp - 1j * Ucorr_tmp

                p_c[idx] = GetP(p_plus_tmp, np.ones_like(Qcorr_tmp), theta_max, R)
                p_c[idx + nclusters] = p_c[idx].conj()

                p_hat[idx] = GetP(d_plus_tmp, tau_beam, theta_max, R)
                p_hat[idx + nclusters] = p_hat[idx].conj()

            C_p_inv = np.diag(np.ones(2 * nclusters) / var_p)
            num = np.dot(p_c.conj().T, np.dot(C_p_inv, p_hat))
            den = np.dot(p_c.conj().T, np.dot(C_p_inv, p_c))

            S2N[isim] = num.real / den.real

        return S2N