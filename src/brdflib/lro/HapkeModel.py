import numpy as np
from numba import jit


# @jit(nopython=True)
def Bs(g, hs):
    return 1.0 / (1.0 + (1.0 / hs) * np.tan(g / 2))


# @jit(nopython=True)
def Bc(g, hc):
    A = np.tan(g / 2) / hc
    B = 1 - np.exp(-A) / A
    return (1 + B) / 2 / (1 + A) ** 2


# @jit(nopython=True)
def P(g, b, c):
    f1 = 0.5 * (1 + c) * (1 - b**2) / (1.0 - 2.0 * b * np.cos(g) + b**2) ** 1.5
    f2 = 0.5 * (1 - c) * (1 - b**2) / (1.0 + 2.0 * b * np.cos(g) + b**2) ** 1.5
    # f1 = (1.0 - c) * (
    #     (1.0 - b**2) / (1.0 + 2.0 * b * np.cos(g) + b**2) ** 1.5
    # )
    # f2 = c * ((1.0 - b**2) / (1.0 - 2.0 * b * np.cos(g) + b**2) ** 1.5)
    return f1 + f2


# @jit(nopython=True)
def H(x, w):
    r0 = (1.0 - (1.0 - w) ** 0.5) / (1.0 + (1.0 - w) ** 0.5)
    return (1.0 - w * x * (r0 + 0.5 * (1.0 - 2.0 * r0 * x) * np.log((1.0 + x) / x))) ** (-1)


# @jit(nopython=True)
def roughness(t, g, i, e, psi):
    f = np.exp(-2.0 * np.tan(0.5 * psi))

    pi = np.pi

    ci, ce = np.cos(i), np.cos(e)
    si, se = np.sin(i), np.sin(e)

    tt = np.tan(t)
    ctt = 1.0 / tt
    ctt2 = ctt**2

    ti = np.tan(i)
    if ti == 0.0:
        cti = np.inf
    else:
        cti = 1.0 / ti
    cti2 = cti**2

    te = np.tan(e)
    if te == 0.0:
        cte = np.inf
    else:
        cte = 1.0 / te
    cte2 = cte**2

    cp = np.cos(psi)
    sp2 = np.sin(psi / 2.0) ** 2

    A = 1.0 / np.sqrt(1.0 + pi * tt**2)

    u0p0 = A * (ci + si * tt * np.exp(-ctt2 * cti2 / pi) / (2.0 - np.exp(-2.0 * ctt * cti / pi)))
    up0 = A * (ce + se * tt * np.exp(-ctt2 * cte2 / pi) / (2.0 - np.exp(-2.0 * ctt * cte / pi)))

    u0 = ci
    u = ce

    if i <= e:
        B = np.exp(-ctt2 * cte2 / pi) - sp2 * np.exp(-ctt2 * cti2 / pi)
        B0 = cp * np.exp(-ctt2 * cte2 / pi) + sp2 * np.exp(-ctt2 * cti2 / pi)
        C = 2 - np.exp(-2.0 * ctt * cte / pi) - psi * np.exp(-2.0 * ctt * cti / pi) / pi

        up = A * (ce + se * tt * B / C)
        u0p = A * (ci + si * tt * B0 / C)

        S = (up / up0) * (u0 / u0p0) * A / (1.0 - f + f * A * (u0 / u0p0))
    else:
        B = cp * np.exp(-ctt2 * cti2 / pi) + sp2 * np.exp(-ctt2 * cte2 / pi)
        B0 = np.exp(-ctt2 * cti2 / pi) - sp2 * np.exp(-ctt2 * cte2 / pi)
        C = 2 - np.exp(-2.0 * ctt * cti / pi) - psi * np.exp(-2.0 * ctt * cte / pi) / pi

        up = A * (ce + se * tt * B / C)
        u0p = A * (ci + si * tt * B0 / C)

        S = (up / up0) * (u0 / u0p0) * A / (1.0 - f + f * A * (u / up0))

    return u0p, up, S


# @jit(nopython=True)
def hapke_brdf(
    i: float,
    e: float,
    psi: float,
    b: float,
    c: float,
    Bs0: float,
    hs: float,
    Bc0: float,
    hc: float,
    w: float,
    t: float,
    K: float = 1.0,
) -> float:
    """
    Evaluate the BRDF usig the Hapke model.

    Ref. https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JE004580

    Parameters
    ----------
    i : float
        incident elevation angle.
    e : float
        observation elevation angle.
    psi : float
        azimuthal angle between e and i (phi_e - phi_i).
    b : float
        Henyey–Greenstein double-lobed single particle phase function b parameter.
    c : float
        Henyey–Greenstein double-lobed single particle phase function c parameter.
    Bs0 : float
        amplitude of Shadow Hiding Opposition Effect (SHOE).
    hs : float
        angular width of SHOE.
    Bc0 : float
        amplitude of Coherent Backscatter Opposition Effect (CBOE).
    hc : float
        angular width of CBOE.
    w : float
        single scattering albedo
    t : float
        Effective value of the photometric roughness.
    K : float, optional
        Porosity factor. The default is 1.0.

    Returns
    -------
    float
        BRDF.

    """
    t = np.deg2rad(t)
    # phase angle
    cos_g = np.cos(e) * np.cos(i) + np.sin(e) * np.sin(i) * np.cos(psi)
    if cos_g > 1.0:
        cos_g = 1.0
    if cos_g < -1.0:
        cos_g = -1.0
    g = np.abs(np.arccos(cos_g))
    # effective cosines u0 and u and shadowing function S
    u0, u, S = roughness(t, g, i, e, psi)
    # Lommel–Seeliger law
    LS = K * w * u0 / (4.0 * np.pi * (u0 + u)) / np.cos(i)
    # IMSA function
    M = H(u0 / K, w) * H(u / K, w) - 1.0
    # return the BRDF
    # return LS * ((1.0 + Bs0 * Bs(g, hs)) * P(g, b, c) + M) * (1 + Bc0 * Bc(g, hc)) * S
    return LS * ((1.0 + Bs0 * Bs(g, hs)) * P(g, b, c) + M) * S


# @jit(nopython=True)
def get_val(
    theta_in: float,
    theta_out: float,
    phi_in: float,
    phi_out: float,
    parameters: tuple[float],
    # bsdf: bool = True,
) -> None:
    # w     | Single scattering albedo
    # b     | Henyey-Greenstein double-lobed single particle phase function parameter
    # c     | Henyey-Greenstein double-lobed single particle phase function parameter
    # Bc0   | Amplitude of Coherent Backscatter Opposition Effect (CBOE) - fixed at 0.0
    # hc    | Angular width of CBOE - fixed at 1.0
    # Bs0   | Amplitude of Shadow Hiding Opposition Effect (SHOE)
    # hs    | Angular width of SHOE
    # theta | Effective value of the photometric roughness - fixed at 23.657
    # phi   | Filling factor - fixed at 1.0
    # w = parameters[0]
    # b = parameters[1]
    # c = parameters[2]
    # Bc0 = parameters[3]
    # hc = parameters[4]
    # Bs0 = parameters[5]
    # hs = parameters[6]
    # theta = parameters[7]
    # K = parameters[8]

    # get Hapke parameters
    w, b, c, Bc0, hc, Bs0, hs, theta, phi = parameters
    a = 2.0 / 3.0
    phi_max = (1 / 1.209) ** (1 / a)
    if phi == 0.0:
        K = 1.0
    elif phi < phi_max:
        K = -np.log(1 - 1.209 * phi**a) / 1.209 / phi**a
    else:
        raise ValueError(f" filling factor mast be lower than {phi_max}")
    # make psi vary from 0 to 2*pi
    if phi_out < 0.0:
        phi_out += 2.0 * np.pi
    if phi_in < 0.0:
        phi_in += 2.0 * np.pi
    psi = phi_out - phi_in
    # if bsdf:
    #     if psi < 0:
    #         psi = abs(psi)
    #     if psi > np.pi:
    #         psi = np.pi
    # else:
    #     if psi < 0:
    #         psi += 2 * np.pi
    #     if psi > np.pi:
    #         psi = 2 * np.pi - psi
    if psi < 0:
        psi += 2 * np.pi
    if psi > np.pi:
        psi = 2 * np.pi - psi
    val = hapke_brdf(theta_in, theta_out, psi, b, c, Bs0, hs, Bc0, hc, w, theta, K)
    return val
