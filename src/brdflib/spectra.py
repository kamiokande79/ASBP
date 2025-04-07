#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 13:48:48 2024

@author: gianluca
"""
from typing import Optional
import drjit
import numpy as np
from .coordinates import Vector2f, MarginalContinuous2D0
from .coordinates import dir_to_sph, sph_to_dir, theta2u, phi2u


def weighted_spectra(
    spectra: np.ndarray,
    ndf: np.ndarray,
    sigma: np.ndarray,
    theta_i: np.ndarray,
    phi_i: np.ndarray,
    theta_o: np.ndarray,
    phi_o: np.ndarray,
    active: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Spectra 5D array weighted with the Jacobian of the parametrization.

    Ref. https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Dupuy2018Adaptive.pdf

    Original code: https://github.com/scalingsolution/brdf-fitting/tree/main
    by Ewan Schafer

    Parameters
    ----------
    spectra : np.ndarray
        spectra 5D array.
    ndf : np.ndarray
        microfacet normal distibution.
    sigma : np.ndarray
        projected area of the microfacet.
    theta_i : np.ndarray
        incident elevation angles.
    phi_i : np.ndarray
        incident azimuthal angles.
    theta_o : np.ndarray
        observation elevation angles.
    phi_o : np.ndarray
        incident azimuthal angles.
    active : Optional[np.ndarray], optional
        array of valid angles. The default is None.

    Returns
    -------
    scaled_spectra : np.ndarray
        spectra 5D array weighted with the Jacobian of the parametrization.

    """
    m_ndf = MarginalContinuous2D0(ndf, normalize=False)
    m_sigma = MarginalContinuous2D0(sigma, normalize=False)
    scaled_spectra = np.zeros(spectra.shape)
    n_phi = spectra.shape[-2]
    n_theta = spectra.shape[-1]

    for i in range(phi_i.size):
        for j in range(theta_i.size):
            # Incient direction
            wi = sph_to_dir(theta_i[j], phi_i[i])
            # np.array fix the difference of float format between the 2 angles
            u_wi = Vector2f(np.array(theta2u(theta_i[j]), phi2u(phi_i[i])))
            # Outgoing direction
            wo = sph_to_dir(theta_o[i, j].flatten(), phi_o[i, j].flatten())
            # Phase direction
            m = drjit.normalize(wi + wo)
            theta_m, phi_m = dir_to_sph(m)
            u_m = Vector2f(theta2u(theta_m), phi2u(phi_m))
            # Scale by inverse jacobian
            jacobian = m_ndf.eval(u_m) / (4 * m_sigma.eval(u_wi))
            scaled_spectra[i, j] = spectra[i, j] / np.reshape(jacobian, (n_phi, n_theta))
    if active is not None:
        n_wavelenths = spectra.shape[2]
        for i in range(n_wavelenths):
            scaled_spectra[:, :, i][~active] = 0
    return scaled_spectra
