#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:33:04 2024

@author: gianluca
"""
import numpy as np
import pandas as pd
from scipy.special import roots_legendre
from scipy.interpolate import interp1d
from scipy.integrate import quad, dblquad

# from matplotlib.pyplot import plot
from brdf import (
    integrate_spectrum,
    outgoing_direction,
    projected_area,
    visible_ndf,
    vndf_intp2sample,
    weight_measurements,
)
from HapkeLRO import get_val
import mitsuba

from visualize import plot_tensor, read_tensor
from visualize import write_tensor

# Set the any mitsuba variant
mitsuba.set_variant("cuda_ad_spectral")


DEG2RAD = np.pi / 180.0


def normalize_ndf(ndf, angles):
    N = len(angles)
    a = min(angles)
    b = max(angles)
    h = (b - a) / 2.0

    nodes, weights = roots_legendre(N)
    fun_ndf = interp1d(angles, ndf[0])
    x = h * nodes + 0.5 * (b + a)
    k = h * 2 * np.pi * np.dot(weights, fun_ndf(x) * np.cos(x) * np.sin(x))
    return ndf / k


def read_hapke_parameters(filename, albedo):
    df = pd.read_csv(filename)
    parameters = np.ndarray((7, 10))  # [[None]*9 for ic in range(7)]
    for ic in range(7):
        kc0 = ic * 9
        for jc in range(9):
            kc = kc0 + jc
            parameters[ic, jc] = df.iloc[kc, 1].item()
        # set porosity factor K = 1 - phi
        parameters[ic, jc] = 1 - parameters[ic, jc]
        # set albedo
        parameters[ic, jc + 1] = albedo
    return parameters


def average(x, f):
    a = min(x)
    b = max(x)
    func = interp1d(x, f)
    avg = quad(func, a, b)[0] / (b - a)
    return avg


def interp_parameters(wl_int, wl, params):
    interp_params = np.zeros(10)
    for ic in range(10):
        f_param = interp1d(wl, params[:, ic], fill_value='extrapolate')
        interp_params[ic] = f_param(wl_int)
    return interp_params


if __name__ == "__main__":
    # parameters = (
    #     0.0544,
    #     0.0578,
    #     0.0526,
    #     0.21,
    #     3.29 * np.exp(-17.4 * (0.21**2)) - 0.908,
    #     23.4 * np.pi / 180.0,
    #     0.057,
    #     1.96,
    #     0.23,
    #     1.0,
    #     np.pi / 0.0562,  # 0.1,  # albedo at 0.1, (rho / pi)**-1
    # )
    import sys

    hapke_parameters = sys.argv[1]
    spd_filename = sys.argv[2]
    # w     | Single scattering albedo
    # b     | Henyey-Greenstein double-lobed single particle phase function parameter
    # c     | Henyey-Greenstein double-lobed single particle phase function parameter
    # Bc0   | Amplitude of Coherent Backscatter Opposition Effect (CBOE) - fixed at 0.0
    # hc    | Angular width of CBOE - fixed at 1.0
    # Bs0   | Amplitude of Shadow Hiding Opposition Effect (SHOE)
    # hs    | Angular width of SHOE
    # theta | Effective value of the photometric roughness - fixed at 23.657
    # phi   | Filling factor - fixed at 1.0
    # WAC_HAPKEPARAMMAP_321NM | 321 nm (WAC band 1) |      9
    # WAC_HAPKEPARAMMAP_360NM | 360 nm (WAC band 2) |      9
    # WAC_HAPKEPARAMMAP_415NM | 415 nm (WAC band 3) |      9
    # WAC_HAPKEPARAMMAP_566NM | 566 nm (WAC band 4) |      9
    # WAC_HAPKEPARAMMAP_604NM | 604 nm (WAC band 5) |      9
    # WAC_HAPKEPARAMMAP_643NM | 643 nm (WAC band 6) |      9
    # WAC_HAPKEPARAMMAP_689NM | 689 nm (WAC band 7) |      9
    WAC_LAMBDA = [321, 360, 415, 566, 604, 643, 689]
    # Directional-hemispherical reflectance, a.k.a. black_sky_albedo
    black_sky_albedo = 1.0  # np.pi / 0.0562
    # prepare parameters
    parameters = read_hapke_parameters(hapke_parameters, black_sky_albedo)
    parameters[:, 7] = parameters[:, 7] * DEG2RAD
    Bs0_avg = average(WAC_LAMBDA, parameters[:, 5])
    LAMBDA_AVG = interp1d(parameters[:, 5], WAC_LAMBDA)(Bs0_avg).item()
    avg_parameters = interp_parameters(LAMBDA_AVG, WAC_LAMBDA, parameters)
    # compute the BRDF
    N = 128
    theta = np.linspace(0, np.pi / 2, N)
    back_scatter = np.zeros(N)
    for ic, t in enumerate(theta):
        back_scatter[ic] = get_val(t, t, 0.0, 0.0, avg_parameters, bsdf=True)
    # linear extrpolation for 90 deg
    m = (back_scatter[-2] - back_scatter[-3]) / (theta[-2] - theta[-3])
    back_scatter[-1] = m * (theta[-1] - theta[-2]) + back_scatter[-2]

    ndf = np.vstack((back_scatter, back_scatter))
    ndf = normalize_ndf(ndf, theta)
    sigma = projected_area(ndf, isotropic=True, projected=True)

    theta_i = (
        np.array([0.0, 26.8, 38.6, 48.3, 57.2, 65.9, 75.0, 90.0]) * DEG2RAD
    )
    n_theta_i = len(theta_i)
    phi_i = np.array([0.0])

    Dvis = visible_ndf(ndf, sigma, theta_i, phi_i, isotropic=True)
    vndf = vndf_intp2sample(Dvis)

    n_wavelengths = 195
    wavelengths = np.linspace(360, 750, n_wavelengths)
    # wavelengths = np.linspace(360, 1000, n_wavelengths)

    reflactance = pd.read_csv(spd_filename)
    f_reflactance = interp1d(
        reflactance["Wavelengths"], reflactance["Reflactance"]
    )
    rho = f_reflactance(wavelengths)
    # Compute sample positions
    R = 32
    theta_o_c, phi_o_c, active, invalid = outgoing_direction(
        R,
        R,
        vndf,
        theta_i,
        phi_i,
        isotropic=True,
        theta_max=85.0 * DEG2RAD,
        all=True,
    )
    # Out of bounds rays
    valid = np.packbits(invalid.flatten())

    spectra_no_jac = np.zeros(
        (1, n_theta_i, n_wavelengths, R, R), dtype=np.float32
    )
    dhr = np.zeros(n_wavelengths)
    for sc in range(n_wavelengths):
        _wl_ = wavelengths[sc]
        wl_parameters = interp_parameters(
            _wl_, WAC_LAMBDA, parameters
        )
        dhr[sc] = dblquad(
            lambda y, x: get_val(0, x, 0, y, wl_parameters)
            * np.cos(x)
            * np.sin(x),
            0,
            np.pi / 2,
            0,
            2 * np.pi,
        )[0]

    blast_factor = 1.126
    fi_in = 0
    for ic, theta_in in enumerate(theta_i):
        for jc in range(R):
            for kc in range(R):
                theta_out = theta_o_c[0][ic][jc][kc]
                fi_out = phi_o_c[0][ic][jc][kc]
                # val = get_val(
                #     theta_in,
                #     theta_out,
                #     fi_in,
                #     fi_out,
                #     avg_parameters,
                #     bsdf=True,
                # )
                for sc in range(n_wavelengths):
                    _wl_ = wavelengths[sc]
                    wl_parameters = interp_parameters(
                        _wl_, WAC_LAMBDA, parameters
                    )

                    val = get_val(
                        theta_in,
                        theta_out,
                        fi_in,
                        fi_out,
                        wl_parameters,
                        bsdf=True,
                    )
                    spectra_no_jac[0][ic][sc][jc][kc] = val
                    # spectra_no_jac[0][ic][sc][jc][kc] = (
                    #     blast_factor * rho[sc] * val / dhr[sc]
                    # )

    sym_active = np.array(active)
    for ic in range(8):
        for jc in range(16):
            sym_active[0][ic][15 - jc, :] = active[0][ic][jc + 16, :]
    sym_active = sym_active.astype(bool)
    spectra = weight_measurements(
        spectra_no_jac,
        ndf,
        sigma,
        theta_i,
        phi_i,
        theta_o_c,
        phi_o_c,
        sym_active,
    )
    # to be verified
    luminance = integrate_spectrum(
        spectra, wavelengths, theta_i, phi_i, lum=True
    )
    tensor = {
        "version": np.array([1, 0], dtype=np.uint32),
        "description": np.ones(41, dtype=np.uint8),
        "phi_i": phi_i.astype(dtype=np.float32),
        "theta_i": theta_i.astype(dtype=np.float32),
        "sigma": sigma.astype(dtype=np.float32),
        "ndf": ndf.astype(dtype=np.float32),
        "vndf": vndf.astype(dtype=np.float32),
        "luminance": luminance.astype(dtype=np.float32),
        "spectra": spectra.astype(dtype=np.float32),
        "wavelengths": wavelengths.astype(dtype=np.float32),
        "jacobian": np.array([1], dtype=np.uint8),
        "valid": valid.astype(dtype=np.uint8),
    }

    write_tensor("10084_hapke_lro_original_extrapolated.bsdf", **tensor)
