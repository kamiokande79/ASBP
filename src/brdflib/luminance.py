#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:48:04 2024

@author: gianluca
"""
import numpy as np
from scipy.integrate import trapezoid


def spectra_to_luminance(
    spectra: np.ndarray, wavelengths: np.ndarray, theta_i: np.ndarray, phi_i: np.ndarray, visible: bool = True
) -> np.ndarray:
    """
    Evaluate the luminance 4D array from the spectrum 5D array.

    Ref. https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Dupuy2018Adaptive.pdf

    Original code: https://github.com/scalingsolution/brdf-fitting/tree/main
    by Ewan Schafer

    Parameters
    ----------
    spectra : np.ndarray
        5D array containing the spectral data, shape[n_phi_i, n_theta_i, n_wavelengths, n_phi, n_theta].
    wavelengths : np.ndarray
        acquired wavelengths.
    theta_i : np.ndarray
        sampled incident elevations.
    phi_i : np.ndarray
        sampled incident azimuths.
    visible : bool, optional
        limit the integration to the visible spectrum (<= 830nm). The default is True.

    Returns
    -------
    luminance : spectra
        luminance 4D array, shape[n_phi_i, n_theta_i, n_phi, n_theta].

    """
    n_phi_i = spectra.shape[0]
    n_theta_i = spectra.shape[1]
    n_phi_o = spectra.shape[3]
    n_theta_o = spectra.shape[4]
    luminance = np.zeros((n_phi_i, n_theta_i, n_phi_o, n_theta_o))
    x = wavelengths
    if visible:
        pc = wavelengths <= 830
        x = x[pc]
    span = x.max() - x.min()
    for ic in range(n_phi_i):
        for jc in range(n_theta_i):
            y = spectra[ic, jc][pc] if visible else spectra[ic, jc]
            y.reshape(-1, *y.shape[-2:])
            integral = trapezoid(y, x, axis=0) / span
            luminance[ic, jc] = np.reshape(integral, (n_phi_o, n_theta_o))
    return luminance


def normalize_2D2(func, theta_i, phi_i):
    from mitsuba import MarginalContinuous2D2, Vector2f

    params = [phi_i.tolist(), theta_i.tolist()]
    m_func = MarginalContinuous2D2(func, params, normalize=True)

    n_phi_o = func.shape[2]
    n_theta_o = func.shape[3]

    normalized = np.zeros(func.shape)
    # Create uniform samples
    u_0 = np.linspace(0, 1, n_theta_o)
    u_1 = np.linspace(0, 1, n_phi_o)

    samples = Vector2f(np.tile(u_0, n_phi_o), np.repeat(u_1, n_theta_o))

    for i in range(phi_i.size):
        for j in range(theta_i.size):
            # Warp uniform samples by VNDF distribution (G1 mapping)
            normalized[i, j] = np.reshape(
                m_func.eval(samples, [float(phi_i[i]), float(theta_i[j])]),
                (n_phi_o, n_theta_o),
            )
    return normalized
