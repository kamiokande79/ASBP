#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 10:42:57 2024

@author: gianluca
"""
import numpy as np
from brdflib.merl.brdf import get_backscatter, get_rgb, get_spectra
from brdflib.coordinates import u2theta
from brdflib.ndf import eval_ndf, eval_vndf, projected_area, weighted_vndf
from brdflib.merl.BRDFRead import read_brdf


# COLORSPACE = "CIE RGB"
# COLORSPACE = "sRGB"


# def get_backscatter(brdf: np.ndarray, elevations: np.ndarray) -> np.ndarray:
#     """
#     Backscatter reflectance distribution function (BRDF(omega_i, omega_i))

#     Parameters
#     ----------
#     brdf : np.ndarray
#         MERL brdf binary data..
#     elevations : np.ndarray
#         elevation angles (incident = observation).

#     Returns
#     -------
#     backscatter : np.ndarray
#         backscatter BRDF .

#     """
#     phi = 0.0
#     backscatter = np.ndarray((len(elevations),))
#     for ic, theta in enumerate(elevations):
#         rgb = lookup_brdf_val(brdf=brdf, theta_in=theta, fi_in=phi, theta_out=theta, fi_out=phi)
#         xyz = colour.RGB_to_XYZ(rgb, COLORSPACE)
#         backscatter[ic] = xyz[1]
#     backscatter[backscatter > 1.0] = 1.0
#     return backscatter


# def get_rgb(
#     brdf: np.ndarray,
#     ndf: np.ndarray,
#     vndf: np.ndarray,
#     sigma: np.ndarray,
#     theta_in: np.ndarray,
#     phi_in: np.array,
#     theta_max: float,
#     jacobian: int = 1,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Evaluate the material spectra or each incdent vector and for 32 x 32 observation vectors.

#     Parameters
#     ----------
#     brdf : np.ndarray
#         MERL brdf binary data.
#     ndf : np.nd_array
#         microfacet normal distribution fuction.
#     vndf : np.nd_array
#         microfacet visible normal distribution fuction.
#     sigma : np.ndarray
#         microfacet projected areas.
#     theta_in : np.ndarray
#         incident elevation angles.
#     phi_in : np.array
#         incident azimuthal angles.
#     theta_max : float
#         maximum elevation angle.
#     jacobian : int, optional
#         1 if the spectra must be weighted by the jacobian, 0 otherwise. The default is 1.

#     Returns
#     -------
#     valid: np.ndarray
#         valid angles (not below horizon).
#     merl_spectra: np.ndarray
#         spectra 5D array [n_phi_i, n_theta_i, n_wavelengths, 32, 32].

#     """
#     n_phi_in = len(phi_in)
#     n_theta_in = len(theta_in)
#     n_colors = 3
#     n_dim = 32
#     theta_out, phi_out, active, invalid = outgoing_direction(
#         n_phi=n_dim,
#         n_theta=n_dim,
#         vndf=vndf,  # vndf_intp2sample(vndf),
#         theta_i=theta_i,
#         phi_i=phi_i,
#         theta_max=theta_max,
#         full=True,
#     )
#     valid = np.packbits(invalid.flatten())
#     merl_rgb = np.ndarray((n_phi_in, n_theta_in, n_colors, n_dim, n_dim))
#     merl_luminance = np.ndarray((n_phi_in, n_theta_in, n_dim, n_dim))

#     # fix simmetry issue of active array
#     for ic in range(n_theta_in):
#         active[0, ic, 15::-1, :] = active[0, ic, 16::, :]

#     theta_o_max = np.pi / 2.0
#     phi_o_max = np.pi
#     for ic, t_in in enumerate(theta_in):
#         t_in = t_in
#         for jc in range(n_dim):
#             for kc in range(n_dim):
#                 t_out = theta_out[0, ic, jc, kc]
#                 # force the symmetry from isotropy
#                 p_out = abs(phi_out[0, ic, jc, kc])*0.998
#                 if p_out > phi_o_max:
#                     p_out = phi_o_max*0.998
#                 if t_out > theta_o_max:
#                     t_out = theta_o_max
#                 rgb = lookup_brdf_val(brdf=brdf, theta_in=t_in, fi_in=0.0, theta_out=t_out, fi_out=p_out)
#                 xyz = colour.RGB_to_XYZ(rgb, COLORSPACE)
#                 merl_rgb[0, ic, :, jc, kc] = np.array(rgb)
#                 merl_luminance[0, ic, jc, kc] = xyz[1]
#     if jacobian:
#         return (
#             valid,
#             weighted_spectra(merl_rgb, ndf, sigma, theta_in, phi_in, theta_out, phi_out, active),
#             merl_luminance,
#         )
#     else:
#         return valid, merl_rgb, merl_luminance


# def get_spectra(
#     brdf: np.ndarray,
#     wavelengths: np.ndarray,
#     ndf: np.ndarray,
#     vndf: np.ndarray,
#     sigma: np.ndarray,
#     theta_in: np.ndarray,
#     phi_in: np.array,
#     theta_max: float,
#     jacobian: int = 1,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Evaluate the material spectra or each incdent vector and for 32 x 32 observation vectors.

#     Parameters
#     ----------
#     brdf : np.ndarray
#         MERL brdf binary data.
#     wavelengths : np.ndarray
#         wavelengths to be computed.
#     ndf : np.nd_array
#         microfacet normal distribution fuction.
#     vndf : np.nd_array
#         microfacet visible normal distribution fuction.
#     sigma : np.ndarray
#         microfacet projected areas.
#     theta_in : np.ndarray
#         incident elevation angles.
#     phi_in : np.array
#         incident azimuthal angles.
#     theta_max : float
#         maximum elevation angle.
#     jacobian : int, optional
#         1 if the spectra must be weighted by the jacobian, 0 otherwise. The default is 1.

#     Returns
#     -------
#     valid: np.ndarray
#         valid angles (not below horizon).
#     merl_spectra: np.ndarray
#         spectra 5D array [n_phi_i, n_theta_i, n_wavelengths, 32, 32].

#     """
#     n_phi_in = len(phi_in)
#     n_theta_in = len(theta_in)
#     n_wavelengths = len(wavelengths)
#     n_dim = 32
#     theta_out, phi_out, active, invalid = outgoing_direction(
#         n_phi=n_dim,
#         n_theta=n_dim,
#         vndf=vndf,  # vndf_intp2sample(vndf),
#         theta_i=theta_i,
#         phi_i=phi_i,
#         theta_max=theta_max,
#         full=True,
#     )
#     valid = np.packbits(invalid.flatten())
#     merl_spectra = np.ndarray((n_phi_in, n_theta_in, n_wavelengths, n_dim, n_dim))

#     # fix simmetry issue of active array
#     for ic in range(n_theta_in):
#         active[0, ic, 15::-1, :] = active[0, ic, 16::, :]

#     theta_o_max = np.pi / 2.0
#     phi_o_max = np.pi
#     for ic, t_in in enumerate(theta_in):
#         t_in = t_in
#         for jc in range(n_dim):
#             for kc in range(n_dim):
#                 t_out = theta_out[0, ic, jc, kc]
#                 # force the symmetry from isotropy
#                 p_out = abs(phi_out[0, ic, jc, kc])
#                 if p_out > phi_o_max:
#                     p_out = phi_o_max
#                 if t_out > theta_o_max:
#                     t_out = theta_o_max
#                 rgb = lookup_brdf_val(brdf=brdf, theta_in=t_in, fi_in=0.0, theta_out=t_out, fi_out=p_out)
#                 xyz = colour.RGB_to_XYZ(rgb, COLORSPACE)
#                 sd = colour.XYZ_to_sd(xyz, method="Otsu 2018", clip=True)
#                 merl_spectra[0, ic, :, jc, kc] = sd[wavelengths]
#     if jacobian:
#         return valid, weighted_spectra(merl_spectra, ndf, sigma, theta_in, phi_in, theta_out, phi_out, active)
#     else:
#         return valid, merl_spectra


# def get_rgb(
#     brdf: np.ndarray,
#     ndf: np.ndarray,
#     vndf: np.ndarray,
#     sigma: np.ndarray,
#     sampple_size: int,
#     theta_in: np.ndarray,
#     phi_in: np.ndarray,
#     theta_max: float,
#     jacobian: int,
#     isotropic: bool = True,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Evaluate the material spectra or each incdent vector and for sampple_size x sampple_size observation vectors.


#     Parameters
#     ----------
#     brdf : np.ndarray
#         MERL brdf binary data.
#     ndf : np.ndarray
#         microfacet normal distribution fuction.
#     vndf : np.ndarray
#         microfacet visible normal distribution fuction.
#     sigma : np.ndarray
#         microfacet projected area.
#     sampple_size : int
#         number of samples per coordinate.
#     theta_in : np.ndarray
#         incident elevation angles.
#     phi_in : np.ndarray
#         incident azimuthal angles.
#     theta_max : float
#         maximum elevation angle.
#     jacobian : int
#         1 if the spectra must be weighted by the jacobian, 0 otherwise. The default is 1.
#     isotropic : bool, optional
#         True if the materila is isotropic, False if it is anisotropic. The default is True.

#     Returns
#     -------
#     valid : np.ndarray[resolution*resolution]
#         uint8 array obtained packing the tensor of valid angles.
#     merl_rgb : np.ndarray[n_phi_i, n_theta_i, 3, resolution, resolution]
#         tensor RGB representation of the material BRDF.
#     merl_luminance : np.ndarray[[n_phi_i, n_theta_i, resolution, resolution]
#         tensor lumiance representation of the material BRDF.

#     """
#     # set the sample size for spherical the coordinates
#     n_phi = sampple_size
#     n_theta = sampple_size
#     n_tot = n_phi * n_theta

#     # get the MarginalContinuous2D2 warp for the vndf
#     params = [phi_in.tolist(), theta_in.tolist()]
#     m_vndf = MarginalContinuous2D2(vndf, params, normalize=True)

#     # create grid in spherical coordinates and map it onto the sphere
#     u1, u2 = drjit.meshgrid(drjit.linspace(mi.Float, 0, 1, n_theta), drjit.linspace(mi.Float, 0, 1, n_phi))
#     # create the sample vectors
#     samples = Vector2f(u1, u2)

#     n_phi_i = len(phi_in)
#     n_theta_i = len(theta_in)
#     merl_rgb = np.ndarray((n_phi_i, n_theta_i, 3, n_phi, n_theta))
#     merl_luminance = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))
#     active = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))

#     theta_out = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))
#     phi_out = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))

#     for ic, phi_i in enumerate(phi_in):
#         for jc, theta_i in enumerate(theta_in):
#             # Specify an incident direction
#             wi = sph_to_dir(theta, phi_i)
#             # u_wi = Vector2f(theta2u(theta), phi2u(phi))
#             u_m, ndf_pdf = m_vndf.sample(samples, np.array([phi_i, theta_i]))

#             u_m[1] = u_m[1] - drjit.floor(u_m[1])

#             phi_m = u2phi(u_m.y)
#             theta_m = u2theta(u_m.x)

#             if isotropic:
#                 phi_m += np.ones(n_tot) * phi_i

#             wm = sph_to_dir(theta_m, phi_m)
#             wo = wm * 2.0 * drjit.dot(wi, wm) - wi

#             theta_o = elevation(wo)
#             phi_o = drjit.abs(drjit.atan2(wo.y, wo.x))  # * 0.995
#             phi_o[drjit.rad2deg(phi_o) > 179.5] = drjit.deg2rad(179.5)

#             valid = Frame3f.cos_theta(wo) > 0
#             valid &= drjit.rad2deg(theta_o) < drjit.rad2deg(theta_max)

#             red = np.zeros(n_tot)
#             green = np.zeros(n_tot)
#             blue = np.zeros(n_tot)
#             luminance = np.zeros(n_tot)

#             for kc in range(n_tot):
#                 if valid[kc]:
#                     rgb = lookup_brdf_val(
#                         brdf=brdf, theta_in=theta_i, fi_in=phi_i, theta_out=theta_o[kc], fi_out=phi_o[kc]
#                     )
#                 else:
#                     rgb = (0.0, 0.0, 0.0)
#                 xyz = colour.RGB_to_XYZ(rgb, COLORSPACE)
#                 y = xyz[1]
#                 if y > 1:
#                     xyz /= y
#                     rgb = colour.XYZ_to_RGB(xyz, COLORSPACE)

#                 luminance[kc] = xyz[1]
#                 red[kc], green[kc], blue[kc] = rgb

#             active[ic, jc] = np.reshape(np.array(valid), (n_phi, n_theta))
#             theta_out[ic, jc] = np.reshape(theta_o, (n_phi, n_theta))
#             phi_out[ic, jc] = np.reshape(phi_o, (n_phi, n_theta))

#             merl_rgb[ic, jc, 0, :, :] = np.reshape(red, (n_phi, n_theta))
#             merl_rgb[ic, jc, 1, :, :] = np.reshape(green, (n_phi, n_theta))
#             merl_rgb[ic, jc, 2, :, :] = np.reshape(blue, (n_phi, n_theta))
#             merl_luminance[ic, jc] = np.reshape(luminance, (n_phi, n_theta))

#     valid = np.packbits(active.flatten().astype(bool))
#     if jacobian:
#         merl_rgb = weighted_spectra(merl_rgb, ndf, sigma, theta_i, phi_i, theta_out, phi_out, active.astype(dtype=bool))
#     return valid, merl_rgb, merl_luminance


# def get_spectra(
#     brdf: np.ndarray,
#     wavelengths: np.ndarray,
#     ndf: np.ndarray,
#     vndf: np.ndarray,
#     sigma: np.ndarray,
#     sampple_size: int,
#     theta_i: np.ndarray,
#     phi_i: np.ndarray,
#     theta_max: float,
#     jacobian: int,
#     isotropic: bool = True,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """


#     Parameters
#     ----------
#     brdf : np.ndarray
#         MERL brdf binary data.
#     wavelengths : np.ndarray
#         wavelengths to be computed.
#     ndf : np.ndarray
#         microfacet normal distribution fuction.
#     vndf : np.ndarray
#         microfacet visible normal distribution fuction.
#     sigma : np.ndarray
#         microfacet projected area.
#     sampple_size : int
#         number of samples per coordinate.
#     theta_in : np.ndarray
#         incident elevation angles.
#     phi_in : np.ndarray
#         incident azimuthal angles.
#     theta_max : float
#         maximum elevation angle.
#     jacobian : int
#         1 if the spectra must be weighted by the jacobian, 0 otherwise. The default is 1.
#     isotropic : bool, optional
#         True if the materila is isotropic, False if it is anisotropic. The default is True.

#     Returns
#     -------
#     valid : np.ndarray[resolution*resolution]
#         uint8 array obtained packing the tensor of valid angles..
#     merl_spectra : np.ndarray[n_phi_i, n_theta_i, n_wavelengths, resolution, resolution]
#         tensor spectral representation of the material BRDF..
#     merl_luminance : np.ndarray[[n_phi_i, n_theta_i, resolution, resolution]
#         tensor lumiance representation of the material BRDF.

#     """
#     n_phi = sampple_size
#     n_theta = sampple_size

#     params = [phi_i.tolist(), theta_i.tolist()]
#     m_vndf = MarginalContinuous2D2(vndf, params, normalize=True)

#     # Create grid in spherical coordinates and map it onto the sphere
#     u1, u2 = drjit.meshgrid(drjit.linspace(mi.Float, 0, 1, n_theta), drjit.linspace(mi.Float, 0, 1, n_phi))

#     n_tot = n_phi * n_theta
#     samples = Vector2f(u1, u2)

#     n_phi_i = len(phi_i)
#     n_theta_i = len(theta_i)
#     n_wavelengths = len(wavelengths)
#     merl_spectra = np.ndarray((n_phi_i, n_theta_i, n_wavelengths, n_phi, n_theta))
#     merl_luminance = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))
#     active = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))

#     theta_out = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))
#     phi_out = np.ndarray((n_phi_i, n_theta_i, n_phi, n_theta))

#     for ic, phi in enumerate(phi_i):
#         for jc, theta in enumerate(theta_i):
#             # Specify an incident direction
#             wi = sph_to_dir(theta, phi)
#             # u_wi = Vector2f(theta2u(theta), phi2u(phi))
#             u_m, ndf_pdf = m_vndf.sample(samples, np.array([phi, theta]))

#             u_m[1] = u_m[1] - drjit.floor(u_m[1])

#             phi_m = u2phi(u_m.y)
#             theta_m = u2theta(u_m.x)

#             if isotropic:
#                 phi_m += np.ones(n_tot) * phi

#             wm = sph_to_dir(theta_m, phi_m)
#             wo = wm * 2.0 * drjit.dot(wi, wm) - wi

#             theta_o = elevation(wo)
#             phi_o = drjit.abs(drjit.atan2(wo.y, wo.x))  # * 0.995
#             # phi_o[drjit.rad2deg(phi_o) > 179.5] = drjit.deg2rad(179.5)

#             valid = Frame3f.cos_theta(wo) > 0
#             valid &= drjit.rad2deg(theta_o) < drjit.rad2deg(theta_max)

#             luminance = np.zeros(n_tot)

#             active[ic, jc] = np.reshape(np.array(valid), (n_phi, n_theta))
#             theta_out[ic, jc] = np.reshape(theta_o, (n_phi, n_theta))
#             phi_out[ic, jc] = np.reshape(phi_o, (n_phi, n_theta))

#             mc = 0
#             for kc in range(n_phi):
#                 for lc in range(n_theta):
#                     if valid[mc]:
#                         rgb = lookup_brdf_val(
#                             brdf=brdf, theta_in=theta, fi_in=phi, theta_out=theta_o[mc], fi_out=phi_o[mc]
#                         )
#                     else:
#                         rgb = (0.0, 0.0, 0.0)
#                     xyz = colour.RGB_to_XYZ(rgb, COLORSPACE)
#                     sd = colour.XYZ_to_sd(xyz, method="Otsu 2018", clip=True)
#                     xyz = colour.sd_to_XYZ(sd)
#                     luminance[mc] = xyz[1]
#                     merl_spectra[ic, jc, :, kc, lc] = sd[wavelengths]
#                     mc += 1

#             merl_luminance[ic, jc] = np.reshape(luminance, (n_phi, n_theta))

#     valid = np.packbits(active.flatten().astype(bool))
#     if jacobian:
#         merl_spectra = weighted_spectra(
#             merl_spectra, ndf, sigma, theta_i, phi_i, theta_out, phi_out, active.astype(dtype=bool)
#         )
#     return valid, merl_spectra, merl_luminance


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from scipy.signal import savgol_filter
    from matplotlib import pyplot as plt
    from visualize import write_tensor, plot_tensor

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="MERL binary file to load", type=Path)
    parser.add_argument(
        "--rgb",
        dest="color",
        help="store the RGB file instead of spectral",
        action="store_const",
        const="rgb",
        default="spectral",
    )
    args = parser.parse_args()

    rgb = False
    if args.color == "rgb":
        rgb = True

    filename: Path = args.filename
    merl_brdf = read_brdf(filename)

    tensor = {}
    tensor["version"] = np.array([1, 0], dtype=np.uint8)
    tensor["description"] = np.ones(30, dtype=np.uint8)

    n_angles = 128
    theta = u2theta(np.linspace(0.0, 1.0, n_angles))
    theta_i = np.deg2rad(np.array([0.0, 28.0, 40.0, 50.0, 60.0, 69.0, 79, 90.0], dtype=np.float64))
    phi_i = np.array([0.0], dtype=np.float32)

    tensor["phi_i"] = phi_i
    tensor["theta_i"] = theta_i.astype(np.float32)

    print("Getting the backscatter reflectance distribution function ...")
    retro = get_backscatter(merl_brdf, theta)
    retro_filtered = savgol_filter(retro, 10, 2)
    print("Evaluating the microfacet normal distribution function ...")
    ndf = eval_ndf(theta=theta, backscatter=retro_filtered * np.cos(theta))

    def forward(x):
        return x ** (1 / 2)

    def inverse(x):
        return x**2

    plt.xscale("function", functions=(forward, inverse))
    plt.xlim(min(np.rad2deg(theta)), max(np.rad2deg(theta)))
    plt.plot(np.rad2deg(theta), ndf)
    plt.show()
    print("Evaluating the microfacet projected areas ...")
    sigma = projected_area(np.stack([ndf, ndf]), isotropic=True)
    plt.plot(np.rad2deg(theta), sigma[0])
    plt.show()
    print("Evaluating the microfacet visible normal distribution function ...")
    vndf = weighted_vndf(eval_vndf(ndf, sigma[0], theta_i, phi_i))
    plot_tensor(vndf)

    tensor["sigma"] = sigma.astype(dtype=np.float32)
    plt.plot(np.rad2deg(theta), sigma[0])
    plt.show()
    tensor["ndf"] = np.stack([ndf.astype(dtype=np.float32), ndf.astype(dtype=np.float32)])
    tensor["vndf"] = vndf.astype(dtype=np.float32)

    theta_max = np.deg2rad(89.0)
    jacobian = 1
    sample_size = 32
    if rgb:
        print("Evaluating the material rgb and luminance ...")
        valid, color, luminance = get_rgb(
            merl_brdf, tensor["ndf"], vndf, tensor["sigma"], sample_size, theta_i, phi_i, theta_max, jacobian
        )
        tensor["rgb"] = color.astype(dtype=np.float32)
    else:
        print("Evaluating the material spectra and luminance...")
        wavelengths_i = np.linspace(360.0, 780.0, 85, dtype=np.float32)
        tensor["wavelengths"] = wavelengths_i
        valid, spectra, luminance = get_spectra(
            merl_brdf,
            wavelengths_i,
            tensor["ndf"],
            vndf,
            tensor["sigma"],
            sample_size,
            theta_i,
            phi_i,
            theta_max,
            jacobian,
        )
        tensor["spectra"] = spectra.astype(dtype=np.float32)
        # visible = True

    tensor["luminance"] = luminance.astype(dtype=np.float32)
    tensor["jacobian"] = np.array([jacobian], dtype=np.uint8)
    tensor["valid"] = valid.astype(dtype=np.uint8)
    print("Saving the bsdf file ...")
    if rgb:
        out_filename = filename.parent / (filename.stem + "_rgb.bsdf")
    else:
        out_filename = filename.parent / (filename.stem + "_spec.bsdf")
    write_tensor(str(out_filename.absolute()), **tensor)
