""" Copyright 2005 Mitsubishi Electric Research Laboratories All Rights Reserved.

Permission to use, copy and modify this software and its documentation without
fee for educational, research and non-profit purposes, is hereby granted, provided
that the above copyright notice and the following three paragraphs appear in all copies.

To request permission to incorporate this software into commercial products contact:
Vice President of Marketing and Business Development;
Mitsubishi Electric Research Laboratories (MERL), 201 Broadway, Cambridge, MA 02139 or
<license@merl.com>.

IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
ITS DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED
HEREUNDER IS ON AN "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS OR MODIFICATIONS. """

from pathlib import Path
from typing import Optional
import numpy as np
from numba import jit


EPS = np.finfo(float).eps

BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D = 360

RED_SCALE = 1.0 / 1500.0
GREEN_SCALE = 1.15 / 1500.0
BLUE_SCALE = 1.66 / 1500.0
M_PI = np.pi

DEG2RAD = M_PI / 180
RAD2DEG = 180 / M_PI


@jit(nopython=True)
def rot_matrix_along(axis: np.ndarray, angle: float) -> np.ndarray:
    a1, a2, a3 = axis  # unpack axis
    c = np.cos(angle)
    s = np.sin(angle)
    rc = 1 - c
    rot_mat = np.array(
        [
            [c + a1 * a1 * rc, a1 * a2 * rc - a3 * s, a1 * a3 * rc + a2 * s],
            [a1 * a2 * rc + a3 * s, c + a2 * a2 * rc, a2 * a3 * rc - a1 * s],
            [a1 * a3 * rc - a2 * s, a2 * a3 * rc + a1 * s, c + a3 * a3 * rc],
        ]
    )
    return rot_mat


@jit(nopython=True)
def rotate_vector(
    vector: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    # rotate vector along one axis
    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)

    out = vector * cos_ang
    out += axis * np.dot(vector, axis) * (1.0 - cos_ang)
    out += np.cross(axis, vector) * sin_ang

    return out


@jit(nopython=True)
def angle2vec(theta: float, fi: float) -> np.ndarray:
    vec = np.array(
        [np.sin(theta) * np.cos(fi), np.sin(theta) * np.sin(fi), np.cos(theta)]
    )
    return vec


@jit(nopython=True)
def normalize(vector: np.ndarray) -> np.ndarray:
    norm = vector / np.linalg.norm(vector)
    norm[np.abs(norm) <= EPS] = 0.0
    return norm


@jit(nopython=True)
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    (n_a, m_a) = a.shape
    (n_b,) = b.shape
    c = np.zeros(n_b)
    if m_a == n_b:
        for ic in range(n_b):
            for jc in range(m_a):
                c[ic] += a[ic, jc] * b[jc]
    return c


@jit(nopython=True)
def std_coords_to_half_diff_coords2(
    theta_in: float, fi_in: float, theta_out: float, fi_out: float
) -> float:
    w_i = normalize(angle2vec(theta_in, fi_in))
    w_o = normalize(angle2vec(theta_out, fi_out))

    half = normalize((w_i + w_o) / 2)
    theta_half = np.arccos(half[2])
    fi_half = np.arctan2(half[1], half[0])

    bi_normal = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    Rn = rot_matrix_along(normal, -fi_half)
    Rb = rot_matrix_along(bi_normal, -theta_half)
    # diff = normalize(np.matmul(Rb, np.matmul(Rn, w_i)))
    diff = normalize(matmul(Rb, matmul(Rn, w_i)))
    theta_diff = np.arccos(diff[2])
    fi_diff = np.arctan2(diff[1], diff[0])

    return theta_half, fi_half, theta_diff, fi_diff


@jit(nopython=True)
def half_diff_to_std_coords_coords2(
    theta_half: float, fi_half: float, theta_diff: float, fi_diff: float
) -> float:
    half = normalize(angle2vec(theta_half, fi_half))
    diff = normalize(angle2vec(theta_diff, fi_diff))

    bi_normal = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    Rn = rot_matrix_along(normal, fi_half)
    Rb = rot_matrix_along(bi_normal, theta_half)

    # w_i = normalize(np.matmul(Rn, np.matmul(Rb, diff)))
    w_i = normalize(matmul(Rn, matmul(Rb, diff)))

    theta_in = np.arccos(w_i[2])
    fi_in = np.arctan2(w_i[1], w_i[0])

    w_o = normalize(2 * np.dot(half, w_i) * half - w_i)
    theta_out = np.arccos(w_o[2])
    fi_out = np.arctan2(w_o[1], w_o[0])

    return theta_in, fi_in, theta_out, fi_out


@jit(nopython=True)
def std_coords_to_half_diff_coords(
    theta_in: float, fi_in: float, theta_out: float, fi_out: float
) -> float:
    # double& theta_half,double& fi_half,double& theta_diff,double& fi_diff )
    # convert standard coordinates to half vector/difference vector coordinates
    # compute in vector
    in_vec_z = np.cos(theta_in)
    proj_in_vec = np.sin(theta_in)
    in_vec_x = proj_in_vec * np.cos(fi_in)
    in_vec_y = proj_in_vec * np.sin(fi_in)
    in_vec = np.array([in_vec_x, in_vec_y, in_vec_z])
    in_vec /= np.linalg.norm(in_vec)
    in_vec[np.abs(in_vec) <= EPS] = 0.0

    # compute out vector
    out_vec_z = np.cos(theta_out)
    proj_out_vec = np.sin(theta_out)
    out_vec_x = proj_out_vec * np.cos(fi_out)
    out_vec_y = proj_out_vec * np.sin(fi_out)
    out_vec = np.array([out_vec_x, out_vec_y, out_vec_z])
    out_vec /= np.linalg.norm(out_vec)
    out_vec[np.abs(out_vec) <= EPS] = 0.0

    # compute halfway vector
    half = (in_vec + out_vec) / 2.0
    half /= np.linalg.norm(half)
    half[np.abs(half) <= EPS] = 0.0
    # compute  theta_half, fi_half
    theta_half = np.arccos(half[2])
    fi_half = np.arctan2(half[1], half[0])

    bi_normal = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    # compute diff vector
    diff = rotate_vector(in_vec, normal, -fi_half)
    diff = rotate_vector(diff, bi_normal, -theta_half)
    diff /= np.linalg.norm(diff)
    diff[np.abs(diff) <= EPS] = 0.0
    # compute  theta_diff, fi_diff
    theta_diff = np.arccos(diff[2])
    fi_diff = np.arctan2(diff[1], diff[0])

    return theta_half, fi_half, theta_diff, fi_diff


@jit(nopython=True)
def half_diff_to_std_coords_coords(
    theta_diff: float, fi_diff: float, theta_half: float, fi_half: float
) -> float:
    # double& theta_half,double& fi_half,double& theta_diff,double& fi_diff )
    # convert standard coordinates to half vector/difference vector coordinates
    # compute diff vector
    diff_vec_z = np.cos(theta_diff)
    proj_in_vec = np.sin(theta_diff)
    diff_vec_x = proj_in_vec * np.cos(fi_diff)
    diff_vec_y = proj_in_vec * np.sin(fi_diff)
    diff = np.array([diff_vec_x, diff_vec_y, diff_vec_z])
    diff /= np.linalg.norm(diff)
    diff[np.abs(diff) <= EPS] = 0.0

    # compute half vector
    half_vec_z = np.cos(theta_half)
    proj_out_vec = np.sin(theta_half)
    half_vec_x = proj_out_vec * np.cos(fi_half)
    half_vec_y = proj_out_vec * np.sin(fi_half)
    half = np.array([half_vec_x, half_vec_y, half_vec_z])
    half /= np.linalg.norm(half)
    half[np.abs(half) <= EPS] = 0.0

    bi_normal = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])

    # compute inin_vec vector
    in_vec = rotate_vector(diff, bi_normal, theta_half)
    in_vec = rotate_vector(in_vec, normal, fi_half)
    in_vec /= np.linalg.norm(in_vec)
    in_vec[np.abs(in_vec) <= EPS] = 0.0
    # compute  theta_in, fi_in
    theta_in = np.arccos(in_vec[2])
    fi_in = np.arctan2(in_vec[1], in_vec[0])

    # compute out vector
    out_vec = 2.0 * half - in_vec
    out_vec /= np.linalg.norm(out_vec)
    out_vec[np.abs(out_vec) <= EPS] = 0.0
    # compute  theta_half, fi_half
    theta_out = np.arccos(out_vec[2])
    fi_out = np.arctan2(out_vec[1], out_vec[0])

    return theta_in, fi_in, theta_out, fi_out


# Lookup theta_half index
# This is a non-linear mapping!
# In:  [0 .. pi/2]
# Out: [0 .. 89]
@jit(nopython=True)
def theta_half_index(theta_half: float) -> int:
    if theta_half <= 0.0:
        return 0
    theta_half_deg = (theta_half / M_PI) * BRDF_SAMPLING_RES_THETA_H * 2
    ret_val = int(
        np.around(np.sqrt(theta_half_deg * BRDF_SAMPLING_RES_THETA_H))
    )
    if ret_val < 0:
        ret_val = 0
    if ret_val >= BRDF_SAMPLING_RES_THETA_H:
        ret_val = BRDF_SAMPLING_RES_THETA_H - 1
    return ret_val


@jit(nopython=True)
def theta_diff_index(theta_diff: float) -> int:
    # Lookup theta_diff index
    # In:  [0 .. pi/2]
    # Out: [0 .. 89]
    tmp = int(np.around(theta_diff / (M_PI * 0.5) * BRDF_SAMPLING_RES_THETA_D))
    if tmp < 0:
        return 0
    elif tmp < (BRDF_SAMPLING_RES_THETA_D - 1):
        return tmp
    else:
        return int(BRDF_SAMPLING_RES_THETA_D - 1)


# Lookup phi_diff index
@jit(nopython=True)
def phi_diff_index(phi_diff: float) -> int:
    # Because of reciprocity, the BRDF is unchanged under
    # phi_diff -> phi_diff + M_PI
    if phi_diff < 0.0:
        phi_diff += M_PI

    # In: phi_diff in [0 .. pi]
    # Out: tmp in [0 .. 179]
    tmp = int(np.around(phi_diff / M_PI * BRDF_SAMPLING_RES_PHI_D / 2))
    if tmp < 0:
        return 0
    elif tmp < (BRDF_SAMPLING_RES_PHI_D / 2 - 1):
        return tmp
    else:
        return int(BRDF_SAMPLING_RES_PHI_D / 2 - 1)


@jit(nopython=True)
def lookup_brdf_val(
    brdf: np.ndarray,
    theta_in: float,
    fi_in: float,
    theta_out: float,
    fi_out: float,
) -> float:
    # Given a pair of incoming/outgoing angles, look up the BRDF.
    # double& red_val,double& green_val,double& blue_val
    # Convert to halfangle / difference angle coordinates
    theta_half, fi_half, theta_diff, fi_diff = std_coords_to_half_diff_coords2(
        theta_in, fi_in, theta_out, fi_out
    )
    # Find index.
    # Note that phi_half is ignored, since isotropic BRDFs are assumed
    ind = int(
        phi_diff_index(fi_diff)
        + theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2
        + theta_half_index(theta_half)
        * BRDF_SAMPLING_RES_PHI_D
        / 2
        * BRDF_SAMPLING_RES_THETA_D
    )

    red_val = brdf[ind] * RED_SCALE
    delta_ind = int(
        BRDF_SAMPLING_RES_THETA_H
        * BRDF_SAMPLING_RES_THETA_D
        * BRDF_SAMPLING_RES_PHI_D
        / 2
    )
    green_val = brdf[ind + delta_ind] * GREEN_SCALE
    delta_ind = (
        BRDF_SAMPLING_RES_THETA_H
        * BRDF_SAMPLING_RES_THETA_D
        * BRDF_SAMPLING_RES_PHI_D
    )
    blue_val = brdf[ind + delta_ind] * BLUE_SCALE

    if (red_val < 0.0) or (green_val < 0.0) or (blue_val < 0.0):
        print("Below horizon.")

    return red_val, green_val, blue_val


@jit(nopython=True)
def write_brdf_val(
    brdf: np.ndarray,
    theta_diff: float,
    fi_diff: float,
    theta_half: float,
    red: float,
    green: float,
    blue: float,
) -> float:
    # Given a pair of incoming/outgoing angles, look up the BRDF.
    # double& red_val,double& green_val,double& blue_val

    # Find index.
    # Note that phi_half is ignored, since isotropic BRDFs are assumed
    ind = int(
        phi_diff_index(fi_diff)
        + theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2
        + theta_half_index(theta_half)
        * BRDF_SAMPLING_RES_PHI_D
        / 2
        * BRDF_SAMPLING_RES_THETA_D
    )

    brdf[ind] = red / RED_SCALE
    delta_ind = int(
        BRDF_SAMPLING_RES_THETA_H
        * BRDF_SAMPLING_RES_THETA_D
        * BRDF_SAMPLING_RES_PHI_D
        / 2
    )
    brdf[ind + delta_ind] = green / GREEN_SCALE
    delta_ind = (
        BRDF_SAMPLING_RES_THETA_H
        * BRDF_SAMPLING_RES_THETA_D
        * BRDF_SAMPLING_RES_PHI_D
    )
    brdf[ind + delta_ind] = blue / BLUE_SCALE


def read_brdf(filename: Path) -> Optional[np.ndarray]:
    # Read BRDF data
    f = open(filename, "rb")
    if f is None:
        return None

    # int dims[3];
    # f.read(dims, sizeof(int), 3, f);
    dims = np.array(np.int32)
    dims = np.fromfile(file=f, dtype=np.int32, count=3)
    n = dims[0] * dims[1] * dims[2]
    if (
        n
        != BRDF_SAMPLING_RES_THETA_H
        * BRDF_SAMPLING_RES_THETA_D
        * BRDF_SAMPLING_RES_PHI_D
        / 2
    ):
        print("Dimensions don't match")
        f.close()
        return None

    # brdf = (double*) malloc (sizeof(double)*3*n)
    # fread(brdf, sizeof(double), 3*n, f)
    brdf = np.array(np.float64)
    brdf = np.fromfile(file=f, dtype=np.float64, count=3 * n)
    f.close()
    return brdf


@jit(nopython=True)
def get_indeces(
    theta_in: float, phi_in: float, theta_out: float, phi_out: float
) -> float:
    """get brdf index from standard angles"""
    theta_half, fi_half, theta_diff, fi_diff = std_coords_to_half_diff_coords(
        theta_in, phi_in, theta_out, phi_out
    )
    theta_half_i = theta_half_index(theta_half)
    theta_diff_i = theta_diff_index(theta_diff)
    phi_diff_i = phi_diff_index(fi_diff)
    return theta_half_i, theta_diff_i, phi_diff_i


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="MERL binary file to load", type=Path)
    args = parser.parse_args()
    # read brdf
    brdf = read_brdf(args.filename)

    print("theta_in,phi_in,theta_out,phi_out,r,g,b")
    n = 90
    for ic in range(n):
        theta_in = ic * 0.5 * M_PI / (n - 1)
        for jc in range(1):
            phi_in = jc * 2.0 * M_PI / (2 * n - 1)
            for kc in range(n):
                theta_out = kc * 0.5 * M_PI / (n - 1)
                for lc in range(2 * n):
                    phi_out = lc * 2.0 * M_PI / (2 * n - 1)
                    red, green, blue = lookup_brdf_val(
                        brdf, theta_in, phi_in, theta_out, phi_out
                    )
                    print(
                        f"{theta_in},{phi_in},{theta_out},{phi_out},"
                        + f"{red},{green},{blue}"
                    )
