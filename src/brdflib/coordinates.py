#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:40:01 2024

@author: gianluca
"""
import drjit
import numpy as np
import mitsuba as mi

# Minimal divison coefficient
EPSILON = 1e-4

mi.set_variant("llvm_ad_rgb")
Vector3f = mi.Vector3f
MarginalContinuous2D0 = mi.MarginalContinuous2D0
MarginalContinuous2D2 = mi.MarginalContinuous2D2
Vector2f = mi.Vector2f
Vector3f = mi.Vector3f
Frame3f = mi.Frame3f


def u2theta(u: float | np.ndarray) -> float | np.ndarray:
    """
    Transform the unit square side to spherical coordinate elevation angle theta.

    Ref. https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Dupuy2018Adaptive.pdf

    Parameters
    ----------
    u : float | np.ndarray
        float or array of floats of the set [0, 1].

    Returns
    -------
    float | np.ndarray
        float or array of floats of the set [0, PI/2] representing the
        spherical coordinate elevation angle theta.

    """
    # return np.square(u) * (np.pi / 2.0)
    return drjit.sqr(u) * (drjit.pi / 2.0)


def u2phi(u: float | np.ndarray) -> float | np.ndarray:
    """
    Transform the unit square side to spherical coordinate azimuthal angle phi.

    Ref. https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Dupuy2018Adaptive.pdf

    Parameters
    ----------
    u : float | np.ndarray
        float or array of floats of the set [0, 1].

    Returns
    -------
    float | np.ndarray
        float or array of floats of the set [-PI, PI] representing the
        spherical coordinate azimuthal angle phi.

    """
    return (2.0 * u - 1.0) * drjit.pi


def theta2u(theta: float | np.ndarray) -> float | np.ndarray:
    """
    Transform the spherical coordinate elevation angle theta to the unit square side.

    Ref. https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Dupuy2018Adaptive.pdf

    Parameters
    ----------
    theta : float | np.ndarray
        elevetion angle theta, a float or array of floats of the set [0, PI/2].

    Returns
    -------
    float | np.ndarray
        float or array of floats of the set [0, 1] representing the
        side of a unit square.

    """
    # return np.sqrt(theta * (2.0 / np.pi))
    return drjit.sqrt(theta * (2. / drjit.pi))


def phi2u(phi: float | np.ndarray) -> float | np.ndarray:
    """
    Transform the spherical coordinate azimuthal angle phi to the unit square side.

    Ref. https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Dupuy2018Adaptive.pdf

    Parameters
    ----------
    phi : float | np.ndarray
        azimuthal angle phi, float or array of floats of the set [-PI, PI].

    Returns
    -------
    float | np.ndarray
        float or array of floats of the set [0, 1] representing the
        side of a unit square.

    """
    # return 0.5 * (phi / np.pi + 1)
    return (phi + drjit.pi) * drjit.inv_two_pi


def elevation(omega: Vector3f):
    """
    Evaluate the elevetion angle from the omega vector.

    Original code: https://github.com/scalingsolution/brdf-fitting/tree/main
    by Ewan Schafer

    Parameters
    ----------
    omega : mitsuba.Vector3f
        omega 3D incident/observation vectors.

    Returns
    -------
    drjit.Float
        elevation angles.

    """

    dist = drjit.sqrt(drjit.sqr(omega.x) + drjit.sqr(omega.y) + drjit.sqr(omega.z - 1.0))
    return 2.0 * drjit.safe_asin(0.5 * dist)


def dir_to_sph(omega: Vector3f):
    """
    Map Euclidean to spherical coordinates.

    Original code: https://github.com/scalingsolution/brdf-fitting/tree/main
    by Ewan Schafer

    Parameters
    ----------
    omega : mitsuba.Vector3f
        omega 3D incident/observation vectors.

    Returns
    -------
    theta : drjit.Float
        elevation angles.
    phi : drjit.Float
        azimuthal angles.

    """
    # Convert: Cartesian coordinates -> Spherical
    theta = elevation(omega)
    phi = drjit.atan2(omega.y, omega.x)
    return theta, phi


def sph_to_dir(theta, phi):
    """
    Map spherical to Euclidean coordinates

    Parameters
    ----------
    theta : np.ndarray
        elevation angles, i.e. angles between the vectors ant the Z axis.
    phi : np.ndarray
        azimuthal angles, i.e. angles between the vectors projected on the xy plane and the X axis.

    Returns
    -------
    omega : mitsuba.Vector3f
        unit 3D omega incident/observation vectors.

    """
    try:
        st, ct = drjit.sincos(theta)
    except TypeError:
        st, ct = drjit.sincos(drjit.llvm.ad.Float64(theta))
    try:
        sp, cp = drjit.sincos(phi)
    except TypeError:
        sp, cp = drjit.sincos(drjit.llvm.ad.Float64(phi))
    return mi.Vector3f(cp * st, sp * st, ct)
