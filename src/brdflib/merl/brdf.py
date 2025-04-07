#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 19:49:03 2024

@author: gianluca
"""
import colour
import drjit
import numpy as np
from .BRDFRead import lookup_brdf_val
from ..spectra import weighted_spectra
from ..coordinates import elevation, sph_to_dir, u2phi, u2theta
from ..coordinates import MarginalContinuous2D2, Frame3f, Vector2f, mi

COLORSPACE = "sRGB"


def get_backscatter(brdf: np.ndarray, elevations: np.ndarray) -> np.ndarray:
    """
    Backscatter reflectance distribution function (BRDF(omega_i, omega_i))

    Parameters
    ----------
    brdf : np.ndarray
        MERL brdf binary data..
    elevations : np.ndarray
        elevation angles (incident = observation).

    Returns
    -------
    backscatter : np.ndarray
        backscatter BRDF .

    """
    phi = 0.0
    backscatter = np.ndarray((len(elevations),))
    for ic, theta in enumerate(elevations):
        rgb = lookup_brdf_val(brdf=brdf, theta_in=theta, fi_in=phi, theta_out=theta, fi_out=phi)
        xyz = colour.RGB_to_XYZ(rgb, COLORSPACE)
        backscatter[ic] = xyz[1]
    backscatter[backscatter > 1.0] = 1.0
    return backscatter


def get_rgb(
    brdf: np.ndarray,
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
    Evaluate the material spectra or each incdent vector and for sampple_size x sampple_size observation vectors.


    Parameters
    ----------
    brdf : np.ndarray
        MERL brdf binary data.
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
        DESCRIPTION.
    merl_rgb : np.ndarray[n_phi_i, n_theta_i, 3, resolution, resolution]
        tensor RGB representation of the material BRDF.
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
    merl_rgb = np.ndarray((n_phi_i, n_theta_i, 3, n_phi, n_theta))
    merl_luminance = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))
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
            red = np.zeros(n_tot)
            green = np.zeros(n_tot)
            blue = np.zeros(n_tot)
            luminance = np.zeros(n_tot)

            for kc in range(n_tot):
                if valid[kc]:
                    # get the BRDF rgb values
                    rgb = lookup_brdf_val(brdf=brdf, theta_in=theta_i, fi_in=phi_i, theta_out=theta_o[kc], fi_out=phi_o[kc])
                else:
                    # if below horizon or > theta_max
                    rgb = (0.0, 0.0, 0.0)
                # compute the tristimulus
                xyz = colour.RGB_to_XYZ(rgb, COLORSPACE)
                # get the relative luminance
                y = xyz[1]
                # keep the physics correct
                if y > 1:
                    # normalize the luminace if y > 1
                    xyz /= y
                    # recompute the rgb
                    rgb = colour.XYZ_to_RGB(xyz, COLORSPACE)

                # store colors and luminance
                luminance[kc] = xyz[1]
                red[kc], green[kc], blue[kc] = rgb

            # reshape the vecotrs to match the correct dimensions
            active[ic, jc] = np.reshape(np.array(valid), (n_phi, n_theta))
            theta_out[ic, jc] = np.reshape(theta_o, (n_phi, n_theta))
            phi_out[ic, jc] = np.reshape(phi_o, (n_phi, n_theta))

            merl_rgb[ic, jc, 0, :, :] = np.reshape(red, (n_phi, n_theta))
            merl_rgb[ic, jc, 1, :, :] = np.reshape(green, (n_phi, n_theta))
            merl_rgb[ic, jc, 2, :, :] = np.reshape(blue, (n_phi, n_theta))
            merl_luminance[ic, jc] = np.reshape(luminance, (n_phi, n_theta))

    # pack the bool(active) tensor into an uin8 array
    valid = np.packbits(active.flatten().astype(bool))
    if jacobian:
        # sclae the color tensor by the jacobian
        merl_rgb = weighted_spectra(merl_rgb, ndf, sigma, theta_in, phi_in, theta_out, phi_out, active.astype(dtype=bool))
    return valid, merl_rgb, merl_luminance


def get_spectra(
    brdf: np.ndarray,
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
    brdf : np.ndarray
        MERL brdf binary data.
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
    merl_spectra = np.ndarray((n_phi_i, n_theta_i, n_wavelengths, n_phi, n_theta))
    merl_luminance = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))
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
                    if valid[mc]:
                        # get the BRDF rgb values
                        rgb = lookup_brdf_val(brdf=brdf, theta_in=theta_i, fi_in=phi_i, theta_out=theta_o[mc], fi_out=phi_o[mc])
                    else:
                        # if below horizon or > theta_max
                        rgb = (0.0, 0.0, 0.0)
                    # compute the tristimulus
                    xyz = colour.RGB_to_XYZ(rgb, COLORSPACE)
                    # get the spectral representation of the rgb values
                    # clip=True ensure that reflectance is physical and energy is conserved
                    sd = colour.XYZ_to_sd(xyz, method="Otsu 2018", clip=True)
                    # re-evaluate tristimulus from the spectral distribution
                    xyz = colour.sd_to_XYZ(sd) / 100.0
                    # get the luminance from tristimulus
                    luminance[mc] = xyz[1]
                    # store the spectral distributio
                    merl_spectra[ic, jc, :, kc, lc] = sd[wavelengths]
                    mc += 1
            # store the luminance
            merl_luminance[ic, jc] = np.reshape(luminance, (n_phi, n_theta))

    # pack the bool(active) tensor into an uin8 array
    valid = np.packbits(active.flatten().astype(bool))
    if jacobian:
        # sclae the color tensor by the jacobian
        merl_spectra = weighted_spectra(merl_spectra, ndf, sigma, theta_in, phi_in, theta_out, phi_out, active.astype(dtype=bool))
    return valid, merl_spectra, merl_luminance
