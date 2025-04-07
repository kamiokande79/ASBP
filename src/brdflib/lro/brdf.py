#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 19:49:03 2024

@author: gianluca
"""
import colour
import drjit
import scipy
import numpy as np
from .HapkeModel import get_val
from ..coordinates import sph_to_dir, elevation, u2phi, u2theta
from ..coordinates import mi, MarginalContinuous2D2, Vector2f, Frame3f
from ..spectra import weighted_spectra


def get_backscatter(elevations: np.ndarray, parameters: dict[int, np.ndarray]) -> np.ndarray:
    """
    Backscatter reflectance distribution function (BRDF(omega_i, omega_i))

    Parameters
    ----------
    elevations : np.ndarray
        elevation angles (incident = observation).
    parameters : dict[int, np.ndarray]
        Hapke model parameters as a function of wavelength.

    Returns
    -------
    backscatter : np.ndarray
        backscatter BRDF.

    """
    backscatter = np.ndarray((len(elevations),))
    min_wavelenght = min(parameters.keys())
    max_wavelenght = max(parameters.keys())

    int_wavelengths = list(range(min_wavelenght, max_wavelenght + 1))
    for ic, theta in enumerate(elevations):
        spectrum = {}
        for wavelenght in parameters:
            spectrum[wavelenght] = get_val(theta_in=theta, theta_out=theta, phi_in=0.0, phi_out=0.0, parameters=parameters[wavelenght])
        spd_fun = scipy.interpolate.interp1d(list(spectrum.keys()), list(spectrum.values()))
        ref = scipy.integrate.quad(spd_fun, int_wavelengths[0], int_wavelengths[-1])[0] / (int_wavelengths[-1] - int_wavelengths[0])
        backscatter[ic] = ref
        # sd = colour.SpectralDistribution(spectrum)
        # sd_int = colour.SpectralDistribution(dict(zip(int_wavelengths, list(sd[int_wavelengths]))))
        # xyz = colour.sd_to_XYZ(sd_int) / 100.0
        # backscatter[ic] = xyz[1]
    backscatter[backscatter > 1.0] = 1.0
    return backscatter


def get_spectra(
    parameters: dict[int, np.ndarray],
    wavelengths: np.ndarray,
    ndf: np.ndarray,
    vndf: np.ndarray,
    sigma: np.ndarray,
    sampple_size: int,
    theta_in: np.ndarray,
    phi_in: np.ndarray,
    theta_max: float,
    jacobian: int,
    isotropic: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """


    Parameters
    ----------
    parameters : dict[int, np.ndarray]
        Hapke model parameters as a function of wavelength.
    wavelengths : np.ndarray
        wavelengths to be computed.
    ndf : np.ndarray
        microfacet normal distribution fuction.
    vndf : np.ndarray
        microfacet visible normal distribution fuction.
    sigma : np.ndarray
        microfacet projected area.
    sampple_size : int
        number of samples per coordinate.
    theta_in : np.ndarray
        incident elevation angles.
    phi_in : np.ndarray
        incident azimuthal angles.
    theta_max : float
        maximum elevation angle.
    jacobian : int
        1 if the spectra must be weighted by the jacobian, 0 otherwise. The default is 1.
    isotropic : bool, optional
        True if the materila is isotropic, False if it is anisotropic. The default is True.

    Returns
    -------
    valid : np.ndarray[resolution*resolution]
        uint8 array obtained packing the tensor of valid angles..
    merl_spectra : np.ndarray[n_phi_i, n_theta_i, n_wavelengths, resolution, resolution]
        tensor spectral representation of the material BRDF..
    merl_luminance : np.ndarray[[n_phi_i, n_theta_i, resolution, resolution]
        tensor lumiance representation of the material BRDF.

    """
    # set the sample size for spherical the coordinates
    n_phi = sampple_size
    n_theta = sampple_size
    n_tot = n_phi * n_theta

    # get the MarginalContinuous2D2 warp for the vndf
    params = [phi_in.tolist(), theta_in.tolist()]
    m_vndf = MarginalContinuous2D2(vndf, params, normalize=True)

    # create grid in spherical coordinates and map it onto the sphere
    u1, u2 = drjit.meshgrid(drjit.linspace(mi.Float, 0, 1, n_theta), drjit.linspace(mi.Float, 0, 1, n_phi))
    # create the sample vectors
    samples = Vector2f(u1, u2)

    # initialize zhe outputs
    n_phi_i = len(phi_in)
    n_theta_i = len(theta_in)
    n_wavelengths = len(wavelengths)
    lro_spectra = np.ndarray((n_phi_i, n_theta_i, n_wavelengths, n_phi, n_theta))
    lro_luminance = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))
    active = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))

    theta_out = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))
    phi_out = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))

    for ic, phi_i in enumerate(phi_in):
        for jc, theta_i in enumerate(theta_in):
            # specify an incident direction
            wi = sph_to_dir(theta_i, phi_i)

            # get half vector parameters by warping square samples via the vndf
            u_m, ndf_pdf = m_vndf.sample(samples, np.array([phi_i, theta_i]))
            u_m[1] = u_m[1] - drjit.floor(u_m[1])

            # get the half vector phi and theta
            phi_m = u2phi(u_m.y)
            theta_m = u2theta(u_m.x)

            if isotropic:
                phi_m += np.ones(n_tot) * phi_i

            # build the ahalf vectors wm and the observation vectors
            wm = sph_to_dir(theta_m, phi_m)
            wo = wm * 2.0 * drjit.dot(wi, wm) - wi

            # get the observation phi and theta
            theta_o = elevation(wo)
            phi_o = drjit.abs(drjit.atan2(wo.y, wo.x))  # * 0.995
            # phi_o[drjit.rad2deg(phi_o) > 179.5] = drjit.deg2rad(179.5)

            # consider only observation vector which are above the horizon and inclination< theta_max
            valid = Frame3f.cos_theta(wo) > 0
            valid &= drjit.rad2deg(theta_o) < drjit.rad2deg(theta_max)

            # initialize color components and luminance vectors
            luminance = np.zeros(n_tot)
            active[ic, jc] = np.reshape(np.array(valid), (n_phi, n_theta))
            theta_out[ic, jc] = np.reshape(theta_o, (n_phi, n_theta))
            phi_out[ic, jc] = np.reshape(phi_o, (n_phi, n_theta))

            mc = 0
            for kc in range(n_phi):
                for lc in range(n_theta):
                    spectrum = {}
                    for wavelenght, wl_parameters in parameters.items():
                        if valid[mc]:
                            # get the BRDF rgb values
                            spectrum[wavelenght] = get_val(
                                theta_in=theta_i,
                                theta_out=theta_o[mc],
                                phi_in=phi_i,
                                phi_out=phi_o[mc],
                                parameters=wl_parameters,
                            )
                        else:
                            # if below horizon or > theta_max
                            spectrum[wavelenght] = 0.0
                    sd = colour.SpectralDistribution(spectrum)
                    # interpolated the spectrum
                    sd_int = colour.SpectralDistribution(dict(zip(wavelengths, sd[wavelengths])))
                    # compute the tristimulus
                    xyz = colour.sd_to_XYZ(sd_int) / 100
                    # get the luminance from tristimulus
                    luminance[mc] = xyz[1]
                    # store the spectral distributio
                    lro_spectra[ic, jc, :, kc, lc] = sd[wavelengths]
                    mc += 1
            # store the luminance
            lro_luminance[ic, jc] = np.reshape(luminance, (n_phi, n_theta))

    # pack the bool(active) tensor into an uin8 array
    valid = np.packbits(active.flatten().astype(bool))
    if jacobian:
        # sclae the color tensor by the jacobian
        lro_spectra = weighted_spectra(lro_spectra, ndf, sigma, theta_in, phi_in, theta_out, phi_out, active.astype(dtype=bool))
    return valid, lro_spectra, lro_luminance
