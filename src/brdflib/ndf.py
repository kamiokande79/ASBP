"""
"""

# from scipy.special import roots_legendre
# from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import drjit
import numpy as np

from .coordinates import u2theta, u2phi, theta2u, phi2u, sph_to_dir
from .coordinates import MarginalContinuous2D0, Vector2f, Vector3f


def power_iteration(A: np.ndarray, num_iterations: int) -> np.ndarray:
    """
    Power iteration algorithm to evaluate the eigenvector associated to the maximum eigenvalue.

    Ref. https://en.wikipedia.org/wiki/Power_iteration

    Parameters
    ----------
    A : np.ndarray
        matrix from which evaluate the eigenvector.
    num_iterations : int
        number of iterations.

    Returns
    -------
    b_k : np.ndarray.
        eigenvector associated to the maximum eigenvalue of A.

    """
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_iterations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


def dot(x: np.ndarray, y: np.ndarray) -> float | np.ndarray:
    """
    Evaluate the max(0, dot(x, y)).

    Parameters
    ----------
    x : np.ndarray
        a 3D vector or an array of 3D vectors.
    y : np.ndarray
        another 3D vector or an array of 3D vectors..

    Returns
    -------
    float | np.ndarray
        return the max(0, dot(x, y)).

    """
    value = np.dot(x, y)
    if isinstance(value, np.ndarray):
        value[value < 0.0] = 0.0
        return value
    return np.max([0.0, value])


# def omega_vectors(theta: np.ndarray, phi: float) -> np.ndarray:
#     """
#     Evaluate omega unit vectors.

#     Parameters
#     ----------
#     angles : np.ndarray
#         DESCRIPTION.
#     phi : float
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     n_angles = len(theta)
#     # omega vectors
#     omega = np.ndarray((3, n_angles))
#     for ic, angle in enumerate(theta):
#         omega_ic = np.array(
#             [
#                 np.sin(angle) * np.cos(phi),
#                 np.sin(angle) * np.sin(phi),
#                 np.cos(angle),
#             ]
#         )
#         omega_ic /= np.linalg.norm(omega_ic)
#         omega[:, ic] = omega_ic

#     return omega


def normalize_ndf(theta: np.ndarray, ndf: np.ndarray) -> np.ndarray:
    """
    Normalize the NDF.

    Ref. p. 34, https://hal.science/tel-01291974/file/TH2015DupuyJonathan2.pdf

    Parameters
    ----------
    theta : np.ndarray
        spherical coordinate theta [0, PI/2].
    ndf : np.ndarray
        isotropic microfacet normal distribution fucntion.

    Returns
    -------
    np.ndarray
        normalized NDF.

    """
    k = trapezoid(2 * np.pi * ndf * np.sin(theta) * np.cos(theta), x=theta)

    return ndf / k


def eval_ndf(theta: np.ndarray, backscatter: np.ndarray) -> np.ndarray:
    """
    Evaluate microfacet normal distribution function from backscatter function.

    Parameters
    ----------
    theta : np.ndarray
        elevation angles [rad].
    backscatter : np.ndarray
        cosine-weighted retro-reflection function.

    Returns
    -------
    np.ndarray
        microfacet normal distribution function.

    """
    n_angles = len(theta)

    u = np.linspace(0.0, 1.0, n_angles)
    phi = u2phi(u)
    sin_phi_h, cos_phi_h = (np.sin(phi), np.cos(phi))
    omega_o = np.vstack((np.sin(theta), np.zeros(n_angles), np.cos(theta)))

    weights = np.ones(n_angles)

    K = np.ndarray((n_angles, n_angles))
    for jc in range(n_angles):
        omega_h = np.vstack(
            (
                cos_phi_h * omega_o[0, jc],
                sin_phi_h * omega_o[0, jc],
                omega_o[2, jc] * np.ones(n_angles),
            )
        )
        omega_h /= np.linalg.norm(omega_h)
        tmp = np.dot(omega_o.T, omega_h)
        integral = trapezoid(tmp, x=phi, axis=1)
        K[:, jc] = weights[jc] * backscatter * integral * omega_o[0, jc]

    return normalize_ndf(theta, power_iteration(K, 4))


def sphere_surface_patch(r, dtheta, dphi):
    # Hemisphere surface area
    h = 2 * np.pi * np.square(r)
    # Elevation slice
    el_s = np.cos(dtheta[0]) - np.cos(dtheta[1])
    # Azimimuth slice
    az_s = (dphi[1] - dphi[0]) / (2 * np.pi)
    return h * el_s * az_s


def projected_area(D, isotropic, projected=True):
    # Check dimensions of micro-facet model
    sigma = np.zeros(D.shape)

    # Construct projected surface area interpolant data structure
    m_D = MarginalContinuous2D0(D, normalize=False)

    # Create uniform samples and warp by G2 mapping
    if isotropic:
        n_theta = n_phi = D.shape[1]
    else:
        n_phi = D.shape[0]
        n_theta = D.shape[1]
    theta = u2theta(np.linspace(0, 1, n_theta))
    phi = u2phi(np.linspace(0, 1, n_phi))

    # Temporary values for surface area calculation
    theta_mean = np.zeros(n_theta + 1)
    for i in range(n_theta - 1):
        theta_mean[i + 1] = (theta[i + 1] - theta[i]) / 2.0 + theta[i]
    theta_mean[-1] = theta[-1]
    theta_mean[0] = theta[0]

    """
    Surface area portion of unit sphere.
    Conditioning better for non vectorized approach.
    a  = sphere_surface_patch(1, theta_next, Vector2f(phi[0], phi[1]))
    """
    a = np.zeros(n_theta)
    for i in range(n_theta):
        a[i] = sphere_surface_patch(1, theta_mean[i : i + 2], phi[-3:-1])

    # Calculate constants for integration
    for j in range(n_phi):
        # Interpolation points
        o = sph_to_dir(theta, phi[j])
        # o = np.array(sph_to_dir(theta, phi[j]))
        # Postion for NDF samples
        u0 = theta2u(drjit.llvm.ad.Float64(theta))
        u1 = np.ones(n_theta) * phi2u(phi[j])
        if j == 0:
            omega = o
            u_0 = u0
            u_1 = u1
            area = a / 2
        else:
            # omega = np.concatenate((omega, o))
            omega = np.concatenate((omega, o), axis=1)
            u_0 = np.concatenate((u_0, u0))
            u_1 = np.concatenate((u_1, u1))
            if j == n_phi - 1:
                area = np.concatenate((area, a / 2))
            else:
                area = np.concatenate((area, a))
    sample = Vector2f(u_0, u_1)
    D_s = m_D.eval(sample)
    omega = Vector3f(omega)

    P = 1.0
    # Calculate projected area of micro-facets
    for i in range(sigma.shape[0] - 1):
        for j in range(sigma.shape[1]):
            # Get projection factor from incident and outgoind direction
            if projected:
                # Incident direction
                omega_i = sph_to_dir(theta[j], phi[i])
                P = drjit.max([0, drjit.dot(omega, omega_i)])

            # Integrate over sphere
            F = P * D_s
            sigma[i, j] = drjit.dot(F, area)[0]

        if projected:
            # Normalise
            sigma[i] = sigma[i] / sigma[i, 0]

    # TODO: Check for anisotropic case
    if isotropic:
        sigma[1] = sigma[0]

    return sigma


# def projected_area(theta: np.ndarray, ndf: np.ndarray) -> np.ndarray:
#     """
#     Projected area of the facets sigma(omega).

#     Ref. https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Dupuy2018Adaptive.pdf


#     Parameters
#     ----------
#     theta : np.ndarray
#         spherical cooedinates theta.
#     ndf : np.ndarray
#         isotropic microfacet normal distribution function.

#     Returns
#     -------
#     None.

#     """
#     n_angles = len(theta)
#     u = np.linspace(0.0, 1.0, n_angles)

#     # trapezoidal integration rule weights
#     weights = np.ones(n_angles)
#     weights[0] = 0.5
#     weights[-1] = 0.5

#     # integration varables theta_m, phi_m (omega_m)
#     phi_m = u2phi(u)
#     sin_theta_m, cos_theta_m = (np.sin(theta), np.cos(theta))
#     sin_phi_m, cos_phi_m = (np.sin(phi_m), np.cos(phi_m))
#     dtheta_m = np.pi / 2 / n_angles
#     # omega_o with phi_o = 0.0
#     omega_o = np.vstack((np.sin(theta), np.zeros(n_angles), np.cos(theta)))
#     sigma = np.ndarray((n_angles,))
#     for ic in range(n_angles):
#         integral = np.zeros(n_angles)
#         for jc in range(n_angles):
#             # build the integral variable omega_m from theta_m and phi_m
#             omega_m = np.vstack(
#                 (
#                     cos_phi_m * sin_theta_m[jc],
#                     sin_phi_m * sin_theta_m[jc],
#                     cos_theta_m[jc] * np.ones(n_angles),
#                 )
#             )
#             omega_m /= np.linalg.norm(omega_m)
#             # integrate over phi_m for each theta_m
#             integral[jc] = trapezoid(dot(omega_o[:, ic], omega_m), x=phi_m)
#         # integrate over dtheta_m
#         # because NDF is isotropic does not depends from phi_m
#         sigma[ic] = np.sum(integral * ndf * weights * sin_theta_m * dtheta_m)

#     # n_phi = 360
#     # dphi_m = 2 * np.pi / float(n_phi)
#     # dtheta_m = np.pi / 2 / n_angles
#     # weights = np.ones(n_phi)
#     # weights[0] = 0.5
#     # weights[-1] = 0.5
#     # omega_o = np.stack((np.sin(thetas), np.zeros(n_angles), np.cos(thetas)))
#     # sigma = np.ndarray((n_angles,))
#     # for ic in range(n_angles):
#     #     sigma[ic] = 0.0
#     #     for jc in range(n_angles):
#     #         theta_m = thetas[jc]
#     #         sin_theta_m, cos_theta_m = (np.sin(theta_m), np.cos(theta_m))
#     #         integral = 0.0
#     #         for kc in range(n_phi):
#     #             phi_m = kc * dphi_m
#     #             sin_phi_m, cos_phi_m = (np.sin(phi_m), np.cos(phi_m))
#     #             omega_m = np.array(
#     #                 [
#     #                     cos_phi_m * sin_theta_m,
#     #                     sin_phi_m * sin_theta_m,
#     #                     cos_theta_m,
#     #                 ]
#     #             )
#     #             omega_m /= np.linalg.norm(omega_m)
#     #             integral += dot(omega_m, omega_o[:, ic]) * weights[kc]
#     #         integral *= dphi_m
#     #         sigma[ic] += (
#     #             integral * ndf[jc] * weights[jc] * sin_theta_m * dtheta_m
#     #         )

#     # normalize and return sigma
#     return sigma / sigma[0]

# def compute_p22_smith(angles: np.ndarray, ndf: np.ndarray):
#     n_angles = len(angles)
#     fun_ndf = interp1d(angles, ndf)
#     n_phi = 360
#     dphi_h = 2 * np.pi / float(n_phi)
#     weights = np.ones(n_phi)
#     weights[0] = 0.5
#     weights[-1] = 0.5
#     # thetas = 0.5 * np.pi * np.linspace(0.0, 1.0, n_angles)**2
#     # phis = 2 * np.pi * np.linspace(0.0, 1.0, n_angles) - np.pi
#     # omega_o = np.vstack((np.sin(thetas), np.zeros(n_angles), np.cos(thetas)))

#     # sin_phi, cos_phi = (np.sin(phis), np.cos(phis))
#     thetas_o = np.ndarray((n_angles,))
#     K = np.ndarray((n_angles, n_angles))
#     for ic in range(n_angles):
#         tmp = float(ic) / float(n_angles)
#         theta = tmp * np.sqrt(np.pi * 0.5)
#         theta_o = theta * theta
#         thetas_o[ic] = theta_o
#         cos_theta_o = np.cos(theta_o)
#         sin_theta_o = np.sin(theta_o)
#         omega_o = np.array([sin_theta_o, 0.0, cos_theta_o])
#         for jc in range(n_angles):
#             tmp = float(jc) / float(n_angles)
#             theta = tmp * np.sqrt(np.pi * 0.5)
#             theta_m = theta * theta
#             sin_theta_m = np.sin(theta_m)
#             cos_theta_m = np.cos(theta_m)
#             integral = 0.0
#             for kc in range(n_phi):
#                 phi_m = kc * dphi_h
#                 sin_phi_m, cos_phi_m = (np.sin(phi_m), np.cos(phi_m))
#                 omega_m = np.array(
#                     [
#                         cos_phi_m * sin_theta_m,
#                         sin_phi_m * sin_theta_m,
#                         cos_theta_m,
#                     ]
#                 )
#                 omega_m /= np.linalg.norm(omega_m)
#                 integral += dot(omega_m, omega_o) * weights[kc]

#             integral *= dphi_h
#             K[ic, jc] = fun_ndf(theta_o) * integral * sin_theta_m

#     return thetas_o, normalize_ndf(thetas_o, power_iteration(K, 4))


def eval_vndf(
    ndf: np.ndarray, sigma: np.ndarray, theta_i: np.ndarray, phi_i: np.ndarray, isotropic: bool = True
) -> np.ndarray:
    """
    Compute visible (bidirectional) micro-facet Normal Distribution Function (NDF).

    Ref. https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Dupuy2018Adaptive.pdf

    Original code: https://github.com/scalingsolution/brdf-fitting/tree/main
    by Ewan Schafer

    Parameters
    ----------
    ndf : np.ndarray
        micro-facet Normal Distribution Function.
    sigma : np.ndarray
        projected area of micro-facets.
    theta_i : np.ndarray
        sampled incident elevations.
    phi_i : np.ndarray
        sampled incident azimuths.
    isotropic : bool, optional
        True if the material is isotropic, False otherwise. The default is True.

    Returns
    -------
    Dvis : np.ndarray
        Visible NDF.

    """
    # adjust the input format
    D = np.stack([ndf, ndf])
    sigma = np.stack([sigma, sigma])
    # Construct projected surface area interpolant data structure
    m_sigma = MarginalContinuous2D0(sigma, normalize=False)

    # Create uniform samples and warp by G2 mapping
    if isotropic:
        n_theta = n_phi = D.shape[1]
    else:
        n_phi = D.shape[0]
        n_theta = D.shape[1]

    # Check dimensions of micro-facet model
    Dvis = np.zeros((phi_i.size, theta_i.size, n_phi, n_theta))

    theta = u2theta(np.linspace(0, 1, n_theta))
    phi = u2phi(np.linspace(0, 1, n_phi))

    # Calculate projected area of micro-facets
    for ic in range(Dvis.shape[0]):  # incident elevation
        for jc in range(Dvis.shape[1]):  # incident azimuth
            # Postion for sigma samples
            sample = Vector2f(drjit.scalar.ArrayXf(theta2u(theta_i[jc]), phi2u(phi_i[ic])))
            sigma_i = m_sigma.eval(sample)

            # Incident direction
            omega_i = sph_to_dir(theta_i[jc], phi_i[ic])

            for kc in range(Dvis.shape[2]):  # observation azimuth
                # Observation direction
                omega = sph_to_dir(theta, phi[kc])

                # NDF at observation directions
                if isotropic:
                    D_m = D[0]
                else:
                    D_m = D[kc]
                # Bidirectional NDF
                Dvis[ic, jc, kc, :] = np.array(drjit.max([0, drjit.dot(omega, omega_i)]) * D_m / sigma_i)
    return Dvis


def weighted_vndf(vndf: np.ndarray) -> np.ndarray:
    """
    Distribution of visible normals weighted with the Jacobian of the parametrization.

    Ref. https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Dupuy2018Adaptive.pdf

    Original code: https://github.com/scalingsolution/brdf-fitting/tree/main
    by Ewan Schafer

    Parameters
    ----------
    vndf : np.ndarray
        distribution of visible normals.

    Returns
    -------
    vndf_corrected: np.ndarray
        distribution of visible normals weighted with the Jacobian of the parametrization.

    """
    # Check dimensions of micro-facet model
    vndf_weighted = np.zeros(vndf.shape)

    # Create uniform samples and warp by G2 mapping
    n_theta = vndf.shape[3]
    theta_m = u2theta(np.linspace(0, 1, n_theta))

    # Apply Jacobian correction factor to interpolants
    for lc in range(n_theta):
        jc = np.sqrt(8 * np.power(np.pi, 3) * theta_m[lc]) * np.sin(theta_m[lc])
        vndf_weighted[:, :, :, lc] = vndf[:, :, :, lc] * jc / (n_theta * n_theta)
    return vndf_weighted


# def ndf_intp2sample(D_intp):
#     # Check dimensions of micro-facet model
#     D_sampler = np.zeros(D_intp.shape)

#     # Create uniform samples and warp by G2 mapping
#     n_theta = D_intp.shape[1]
#     theta_m = u2theta(np.linspace(0, 1, n_theta))

#     # Apply Jacobian correction factor to interpolants
#     for l in range(n_theta):
#         jc = np.sqrt(8 * np.power(np.pi, 3) * theta_m[l]) * np.sin(theta_m[l])
#         D_sampler[:, l] = D_intp[:, l] * jc / n_theta
#     return D_sampler


# def vndf_intp2sample(Dvis_intp):
#     # Check dimensions of micro-facet model
#     Dvis_sampler = np.zeros(Dvis_intp.shape)

#     # Create uniform samples and warp by G2 mapping
#     n_theta = Dvis_intp.shape[3]
#     theta_m = u2theta(np.linspace(0, 1, n_theta))

#     # Apply Jacobian correction factor to interpolants
#     for l in range(n_theta):
#         jc = np.sqrt(8 * np.power(np.pi, 3) * theta_m[l]) * np.sin(theta_m[l])
#         Dvis_sampler[:, :, :, l] = Dvis_intp[:, :, :, l] * jc / (n_theta * n_theta)
#     return Dvis_sampler
